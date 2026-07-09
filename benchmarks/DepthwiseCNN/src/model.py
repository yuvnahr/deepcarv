"""
benchmarks/DepthwiseCNN/src/model.py
------------------------------------
PyTorch implementation of the **DepthwiseCNN** family of file-fragment
classifiers, drawn from:

    "File Fragment Type Classification Using Light-Weight
     Convolutional Neural Networks"

Three variants are supported:

- **DSC**    – Depthwise Separable Convolution baseline
- **DSC-SE** – DSC with per-block Squeeze-and-Excitation gates
- **M-DSC**  – Modified DSC: depthwise first-conv, GroupNorm, ReLU,
               head Dropout

Public interface
----------------
    from benchmarks.DepthwiseCNN.src.model import build_depthwisecnn

    model = build_depthwisecnn(num_classes=75, variant="dsc-se")
    log_probs = model(x)   # x: [B, L] int64, log-probs: [B, num_classes]

All variants accept raw byte sequences as ``torch.long`` tensors with
values in ``[0, 255]``.  The embedding layer is part of the forward pass
so no external pre-processing is required.

Paper ambiguities and design decisions
---------------------------------------
1. **Residual in InceptionBlock** – The paper describes a residual/shortcut
   path drawn in Figure 4 but does not give explicit details on how the
   shortcut is resized when channels change.  Here we use a 1×1 Conv1d
   (no bias, no norm) on the shortcut whenever in_channels != out_channels,
   matching standard ResNet practice.

2. **MaxPool on shortcut** – Figure 4 shows the shortcut being added to the
   pooled inception output, implying the shortcut must also be pooled.
   We apply the same MaxPool1d(4, 4) to the shortcut when pool=True.

3. **SE reduction ratio** – The paper does not state an explicit ratio for
   SE blocks.  We use reduction=4 uniformly (standard MobileNetV3 value).
   This is annotated where relevant.

4. **M-DSC first conv** – The paper states the first conv is depthwise in
   M-DSC.  This means it maps 32→32 with groups=32 (depth-only, no point-
   wise mixing), equivalent to a per-channel filter.  A pointwise step is
   not added here because the paper does not mention one.

5. **GroupNorm group count** – GroupNorm requires channels % num_groups == 0.
   We use num_groups=8 which divides evenly into all channel widths used
   (32, 64, 128).

6. **Dropout placement (M-DSC)** – The paper places dropout before the final
   classifier.  We apply it after global average pooling and before the
   1×1 classifier conv, mirroring MobileNetV3's head structure.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Literal aliases for documented variant names
# ---------------------------------------------------------------------------

Variant = Literal["dsc", "dsc-se", "m-dsc"]
NormType = Literal["batch", "group"]
ActType = Literal["hardswish", "relu"]

# Number of groups for GroupNorm; all channel widths used (32, 64, 128) are
# divisible by 8.
_GROUP_NORM_GROUPS: int = 8

# SE reduction ratio – see design decision #3 in module docstring.
_SE_REDUCTION: int = 4


# ---------------------------------------------------------------------------
# Primitive building blocks
# ---------------------------------------------------------------------------


class SeparableConv1d(nn.Module):
    """Depthwise-separable 1-D convolution (depthwise then pointwise).

    Parameters
    ----------
    in_channels:
        Number of input channels.
    out_channels:
        Number of output channels produced by the pointwise step.
    kernel_size:
        Kernel size of the depthwise convolution.
    stride:
        Stride of the depthwise convolution (pointwise stride is always 1).
    padding:
        Padding applied to the depthwise convolution.

    Notes
    -----
    Neither the depthwise nor the pointwise layer uses a bias because they
    are followed immediately by a normalisation layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise then pointwise convolution.

        Parameters
        ----------
        x:
            Input feature map of shape ``[B, C_in, L]``.

        Returns
        -------
        torch.Tensor
            Output of shape ``[B, C_out, L']``.
        """
        return self.pointwise(self.depthwise(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for 1-D feature maps.

    Learns per-channel attention weights via a two-layer bottleneck (squeeze
    → excite), then re-scales the input feature map channel-wise.

    Parameters
    ----------
    in_channels:
        Number of input (and output) channels.
    reduction:
        Bottleneck reduction factor.  ``in_channels`` must be divisible by
        ``reduction``.

    Notes
    -----
    Following the original SE-Net paper we use a sigmoid gate (not softmax)
    so that channels can be independently up- or down-weighted.
    """

    def __init__(self, in_channels: int, reduction: int = _SE_REDUCTION) -> None:
        super().__init__()
        squeezed = max(1, in_channels // reduction)
        self.fc1 = nn.Linear(in_channels, squeezed, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(squeezed, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Re-calibrate ``x`` with learned channel weights.

        Parameters
        ----------
        x:
            Feature map of shape ``[B, C, L]``.

        Returns
        -------
        torch.Tensor
            Channel-re-weighted feature map of the same shape ``[B, C, L]``.
        """
        # Squeeze: global average pool to [B, C]
        y: torch.Tensor = x.mean(dim=2)
        # Excite: bottleneck FC → sigmoid gate → [B, C, 1]
        y = self.sigmoid(self.fc2(self.relu(self.fc1(y)))).unsqueeze(-1)
        return x * y  # broadcast over L


# ---------------------------------------------------------------------------
# Normalisation and activation factories
# ---------------------------------------------------------------------------


def _build_norm(norm_type: NormType, channels: int) -> nn.Module:
    """Return a normalisation layer for the given channel width.

    Parameters
    ----------
    norm_type:
        ``"batch"`` → ``BatchNorm1d``; ``"group"`` → ``GroupNorm``.
    channels:
        Number of feature channels.

    Returns
    -------
    nn.Module
        Instantiated normalisation layer (no weights shared between calls).

    Raises
    ------
    ValueError
        If ``norm_type`` is not recognised.
    """
    if norm_type == "batch":
        return nn.BatchNorm1d(channels)
    if norm_type == "group":
        # _GROUP_NORM_GROUPS must divide ``channels``; see module docstring #5
        return nn.GroupNorm(_GROUP_NORM_GROUPS, channels)
    raise ValueError(f"Unknown norm_type '{norm_type}'.  Expected 'batch' or 'group'.")


def _build_act(act_type: ActType) -> nn.Module:
    """Return an activation layer.

    Parameters
    ----------
    act_type:
        ``"hardswish"`` or ``"relu"``.

    Returns
    -------
    nn.Module
        Instantiated, stateless activation module.

    Raises
    ------
    ValueError
        If ``act_type`` is not recognised.
    """
    if act_type == "hardswish":
        return nn.Hardswish()
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unknown act_type '{act_type}'.  Expected 'hardswish' or 'relu'.")


# ---------------------------------------------------------------------------
# Inception block
# ---------------------------------------------------------------------------


class InceptionBlock(nn.Module):
    """Multi-scale inception block using depthwise-separable convolutions.

    Three parallel separable convolution branches (kernel sizes 11, 19, 27)
    capture features at small, medium, and large temporal scales.  Their
    outputs are element-wise summed and then, optionally, max-pooled.  A
    residual shortcut is added before the activation.

    Parameters
    ----------
    in_channels:
        Number of channels fed into this block.
    out_channels:
        Number of channels produced by each branch (and the block output).
    norm_type:
        Normalisation strategy applied after each branch convolution.
    act_type:
        Activation applied after the residual add.
    pool:
        If ``True``, apply ``MaxPool1d(kernel_size=4, stride=4)`` to both
        the summed branch output and the shortcut before adding them.

    Notes
    -----
    See design decision #1 and #2 in the module docstring for the rationale
    behind the residual implementation when channels or lengths change.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: NormType = "batch",
        act_type: ActType = "hardswish",
        pool: bool = True,
    ) -> None:
        super().__init__()
        self._pool = pool

        # Three parallel separable convolutions — all use symmetric "same"
        # padding so that output length equals input length before any pooling.
        self.branch_11 = SeparableConv1d(in_channels, out_channels, kernel_size=11, padding=5)
        self.branch_19 = SeparableConv1d(in_channels, out_channels, kernel_size=19, padding=9)
        self.branch_27 = SeparableConv1d(in_channels, out_channels, kernel_size=27, padding=13)

        # Per-branch normalisation (applied after each convolution, before summing)
        self.norm_11 = _build_norm(norm_type, out_channels)
        self.norm_19 = _build_norm(norm_type, out_channels)
        self.norm_27 = _build_norm(norm_type, out_channels)

        # Post-residual activation
        self.act = _build_act(act_type)

        # Optional spatial downsampling — applied to both branches and shortcut
        self.max_pool: nn.Module = nn.MaxPool1d(kernel_size=4, stride=4) if pool else nn.Identity()

        # Shortcut projection: 1×1 conv when channel dimensions change
        # (design decision #1).  No norm on the shortcut — mirrors ResNet.
        self.shortcut: nn.Module = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the inception block output.

        Parameters
        ----------
        x:
            Input feature map of shape ``[B, C_in, L]``.

        Returns
        -------
        torch.Tensor
            Output feature map of shape ``[B, C_out, L']`` where
            ``L' = L // 4`` if ``pool=True`` else ``L' = L``.
        """
        # --- Multi-scale branches ---
        b11 = self.norm_11(self.branch_11(x))
        b19 = self.norm_19(self.branch_19(x))
        b27 = self.norm_27(self.branch_27(x))
        out: torch.Tensor = b11 + b19 + b27  # element-wise sum, [B, C_out, L]

        # --- Optional spatial pooling on summed branches (design decision #2) ---
        out = self.max_pool(out)  # [B, C_out, L'] (or same if Identity)

        # --- Residual shortcut (also pooled to match spatial dim) ---
        shortcut = self.max_pool(self.shortcut(x))

        # --- Add and activate ---
        return self.act(out + shortcut)


# ---------------------------------------------------------------------------
# Full DepthwiseCNN model
# ---------------------------------------------------------------------------


class DepthwiseCNNModel(nn.Module):
    """Lightweight file-fragment classifier using depthwise-separable convolutions.

    All three variants (DSC, DSC-SE, M-DSC) are implemented within this
    single module; the ``variant`` argument selects the configuration.

    Architecture summary
    --------------------
    ::

        [B, L] int64
          └─ Embedding(256, embed_dim=32)       → [B, 32, L]
          └─ Conv1d(32→32, k=19, s=2) + Norm + Act  → [B, 32, L/2]
          └─ InceptionBlock(32→64,  pool=True)   → [B, 64, L/8]
              └─ [DSC-SE only] SEBlock(64)
          └─ InceptionBlock(64→64,  pool=False)  → [B, 64, L/8]
              └─ [DSC-SE only] SEBlock(64)
          └─ InceptionBlock(64→128, pool=True)   → [B, 128, L/32]
              └─ [DSC-SE only] SEBlock(128)
          └─ GlobalAveragePool                   → [B, 128, 1]
          └─ [M-DSC only] Dropout(p=dropout_p)
          └─ Conv1d(128→num_classes, k=1)        → [B, num_classes, 1]
          └─ log_softmax                         → [B, num_classes]

    Parameters
    ----------
    num_classes:
        Number of output classes (e.g. 75 for FFT-75 Scenario 1).
    variant:
        One of ``"dsc"``, ``"dsc-se"``, or ``"m-dsc"``.
    dropout_p:
        Dropout probability for the M-DSC head.  Ignored by DSC / DSC-SE.

    Inputs
    ------
    x : torch.Tensor
        Integer tensor of shape ``[B, L]`` with values in ``[0, 255]``
        representing raw byte sequences.  ``L`` may be 512 or 4 096.

    Outputs
    -------
    torch.Tensor
        Log-probabilities of shape ``[B, num_classes]``.  Compatible with
        ``nn.NLLLoss`` and the framework's ``FragmentClassifier.loss()``
        default implementation.
    """

    # All valid string identifiers for each variant
    VALID_VARIANTS: frozenset[str] = frozenset({"dsc", "dsc-se", "m-dsc"})

    def __init__(
        self,
        num_classes: int,
        variant: Variant = "dsc",
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()

        if variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'.  "
                f"Choose from {sorted(self.VALID_VARIANTS)}."
            )

        self.variant: Variant = variant
        self.num_classes: int = num_classes

        # Derive per-variant configuration
        norm_type: NormType = "group" if variant == "m-dsc" else "batch"
        act_type: ActType = "relu" if variant == "m-dsc" else "hardswish"
        use_se: bool = variant == "dsc-se"

        # ------------------------------------------------------------------
        # 1. Byte embedding
        # Paper: embedding dimension 32, vocabulary = 256 byte values.
        # ------------------------------------------------------------------
        self.embedding = nn.Embedding(num_embeddings=256, embedding_dim=32)

        # ------------------------------------------------------------------
        # 2. First convolution
        # DSC / DSC-SE: standard Conv1d(32→32, k=19, stride=2).
        # M-DSC:        depthwise Conv1d(32→32, k=19, stride=2, groups=32).
        #               Design decision #4: no pointwise mixing step is added
        #               because the paper does not mention one here.
        # ------------------------------------------------------------------
        if variant == "m-dsc":
            self.conv1: nn.Module = nn.Conv1d(
                32, 32,
                kernel_size=19, stride=2, padding=9,
                groups=32, bias=False,           # purely depthwise
            )
        else:
            self.conv1 = nn.Conv1d(
                32, 32,
                kernel_size=19, stride=2, padding=9,
                bias=False,
            )
        self.norm1: nn.Module = _build_norm(norm_type, 32)
        self.act1: nn.Module = _build_act(act_type)

        # ------------------------------------------------------------------
        # 3. Three inception blocks
        # Block 1: 32→64, pool  (downsamples by 4)
        # Block 2: 64→64, no pool
        # Block 3: 64→128, pool (downsamples by 4)
        # ------------------------------------------------------------------
        self.block1 = InceptionBlock(32, 64, norm_type=norm_type, act_type=act_type, pool=True)
        self.se1: nn.Module = SEBlock(64) if use_se else nn.Identity()

        self.block2 = InceptionBlock(64, 64, norm_type=norm_type, act_type=act_type, pool=False)
        self.se2: nn.Module = SEBlock(64) if use_se else nn.Identity()

        self.block3 = InceptionBlock(64, 128, norm_type=norm_type, act_type=act_type, pool=True)
        self.se3: nn.Module = SEBlock(128) if use_se else nn.Identity()

        # ------------------------------------------------------------------
        # 4. Classification head
        # GlobalAveragePool → [optional Dropout] → 1×1 Conv1d → log_softmax
        # Dropout is only present in M-DSC (design decision #6).
        # ------------------------------------------------------------------
        self.dropout: nn.Module = nn.Dropout(p=dropout_p) if variant == "m-dsc" else nn.Identity()
        # 1×1 convolution acts as the final linear classifier over 128 features
        self.classifier = nn.Conv1d(128, num_classes, kernel_size=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the model.

        Parameters
        ----------
        x:
            Raw byte sequence tensor of shape ``[B, L]``, dtype ``torch.long``,
            with values in the range ``[0, 255]``.

        Returns
        -------
        torch.Tensor
            Log-probabilities of shape ``[B, num_classes]``.

        Shape trace (example: B=4, L=512, num_classes=75)
        ---------------------------------------------------
        After embedding:    [4, 32, 512]
        After conv1:        [4, 32, 256]   (stride=2)
        After block1 (pool):[4, 64,  64]   (÷4)
        After block2 (no p):[4, 64,  64]
        After block3 (pool):[4, 128, 16]   (÷4)
        After GAP:          [4, 128,  1]
        After classifier:   [4, 75,   1]
        After squeeze:      [4, 75]
        """
        # 1. Embedding: [B, L] → [B, L, 32] → [B, 32, L] (channels-first)
        x = self.embedding(x).transpose(1, 2)  # [B, 32, L]

        # 2. First conv block
        x = self.act1(self.norm1(self.conv1(x)))  # [B, 32, L//2]

        # 3. Inception blocks (with optional SE gates)
        x = self.se1(self.block1(x))   # [B, 64,  L//8]
        x = self.se2(self.block2(x))   # [B, 64,  L//8]
        x = self.se3(self.block3(x))   # [B, 128, L//32]

        # 4. Global average pool → [B, 128, 1]
        x = x.mean(dim=2, keepdim=True)

        # 5. Head dropout (M-DSC only; Identity otherwise)
        x = self.dropout(x)

        # 6. 1×1 classifier → log-softmax
        x = self.classifier(x).squeeze(-1)  # [B, num_classes]
        return F.log_softmax(x, dim=1)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Include variant and num_classes in repr string."""
        return f"variant={self.variant!r}, num_classes={self.num_classes}"


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_depthwisecnn(
    num_classes: int,
    variant: Variant = "dsc",
    dropout_p: float = 0.2,
) -> DepthwiseCNNModel:
    """Construct and return a :class:`DepthwiseCNNModel`.

    This is the **single public entry point** for instantiating any variant.
    The adapter layer calls this factory; model code inside the benchmark
    package should prefer this function over constructing
    ``DepthwiseCNNModel`` directly.

    Parameters
    ----------
    num_classes:
        Number of output classes.  Must be ≥ 1.
    variant:
        Architecture variant — ``"dsc"``, ``"dsc-se"``, or ``"m-dsc"``.
    dropout_p:
        Dropout probability for the M-DSC head (ignored for DSC / DSC-SE).

    Returns
    -------
    DepthwiseCNNModel
        Freshly initialised model, moved to no particular device.

    Examples
    --------
    >>> model = build_depthwisecnn(num_classes=75, variant="dsc-se")
    >>> x = torch.randint(0, 256, (4, 512))   # batch of 4 × 512-byte fragments
    >>> log_probs = model(x)                   # [4, 75]
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be ≥ 1, got {num_classes}.")
    return DepthwiseCNNModel(num_classes=num_classes, variant=variant, dropout_p=dropout_p)
