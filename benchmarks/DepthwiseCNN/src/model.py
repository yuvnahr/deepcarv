"""
benchmarks/DepthwiseCNN/src/model.py
--------------------------------------
Implementation of the lightweight CNN family from:

    "File Fragment Type Classification Using Light-Weight Convolutional Neural Networks"

Three variants are supported:

    DSC    — Depthwise Separable Convolution baseline
    DSC-SE — DSC + Squeeze-and-Excitation blocks (SE)
    M-DSC  — Multi-scale Depthwise Separable Convolution (Inception-style branches)

Architecture overview (from §III of the paper)
----------------------------------------------

Input: raw byte sequence  [B, fragment_size]  (values 0–255)

Stage 1 — Byte Embedding
    nn.Embedding(256, embed_dim)            → [B, L, embed_dim]
    Permute to [B, embed_dim, L]            (treat channels-first for Conv1d)

Stage 2 — Inception/DSC Blocks  (num_blocks times)
    Each block is one of:
        DSC block   : depthwise_conv → BN → Hardswish → pointwise_conv → GroupNorm
        DSC-SE block: DSC block + SE sub-block (channel attention)
        M-DSC block : three parallel DSC branches (kernels 3, 7, 11) → cat → project

Stage 3 — Head
    Global average pooling  [B, channels, L] → [B, channels]
    Dropout
    Linear(channels, num_classes)
    log_softmax

Architectural assumptions (paper ambiguities documented here)
-------------------------------------------------------------
* embed_dim (§III-A): The paper shows an "embedding layer" without specifying width.
  We default to 64, matching common practice for byte-level models at this scale.
  ASSUMPTION: embed_dim = 64.

* num_blocks (§III-B): The paper does not give an explicit number for the
  DepthwiseCNN stack.  ByteRCNN uses 3 blocks; we use 4 to match the depth
  shown in the paper's figure.  ASSUMPTION: num_blocks = 4.

* channels (§III-B): Paper shows channel progression 64 → 128 → 256 → 256.
  ASSUMPTION: [64, 128, 256, 256].

* GroupNorm groups (§III-B): Paper specifies GroupNorm after the pointwise
  convolution.  We use groups = max(1, channels // 16), clamped so that
  channels % groups == 0.  ASSUMPTION: num_groups = channels // 16.

* Kernel sizes for M-DSC (§III-C): Paper specifies three depthwise branches
  with kernel sizes 3, 7, and 11.  Padding is set so spatial dimension is
  unchanged.  ASSUMPTION: padding = kernel_size // 2.

* SE reduction ratio (§III-B, DSC-SE): Paper mentions SE blocks without
  specifying the reduction ratio r.  Standard SE uses r = 16; we adopt that.
  ASSUMPTION: se_reduction = 16.

* Dropout probability: Paper trains with dropout; no value given.
  ASSUMPTION: p_dropout = 0.5 (matches ByteRCNN baseline convention).

* Stride: All depthwise convolutions use stride=1; downsampling is via
  max-pooling after every other block.  ASSUMPTION: max-pool after blocks 1
  and 3 (halving the sequence length twice).

This file is intentionally model-only.  No training logic lives here.
All training is handled by the DeepCarv generic Trainer via the adapter.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Variant = Literal["dsc", "dsc_se", "m_dsc"]


# ---------------------------------------------------------------------------
# Building-block helpers
# ---------------------------------------------------------------------------


def _groupnorm(channels: int) -> nn.GroupNorm:
    """GroupNorm with num_groups = channels // 16 (paper assumption).

    Clamps so that channels % groups == 0.
    """
    num_groups = max(1, channels // 16)
    # Walk down until divisible
    while channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups, channels)


def _se_block(channels: int, reduction: int = 16) -> nn.Sequential:
    """Squeeze-and-Excitation block (channel attention).

    SE-block:  GAP → FC(r) → ReLU → FC(C) → Sigmoid → scale

    Returned as a nn.Sequential that takes [B, C, L] → [B, C, L].
    Since nn.Sequential is linear (no branching), we wrap it in SeBlock
    so the residual multiplication is explicit.
    """
    mid = max(1, channels // reduction)
    return nn.Sequential(
        nn.AdaptiveAvgPool1d(1),        # [B, C, 1]
        nn.Flatten(1),                  # [B, C]
        nn.Linear(channels, mid, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(mid, channels, bias=False),
        nn.Sigmoid(),
    )


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for 1-D feature maps [B, C, L]."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, mid, bias=False)
        self.fc2 = nn.Linear(mid, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        s = self.pool(x).squeeze(-1)           # [B, C]
        s = F.relu(self.fc1(s), inplace=True)  # [B, mid]
        s = torch.sigmoid(self.fc2(s))          # [B, C]
        return x * s.unsqueeze(-1)              # [B, C, L]


# ---------------------------------------------------------------------------
# DSC block
# ---------------------------------------------------------------------------


class DSCBlock(nn.Module):
    """Depthwise Separable Convolution block (DSC).

    Architecture (paper §III-B):
        depthwise_conv(kernel_size, groups=in_ch) → BN → Hardswish
        pointwise_conv(1×1)                       → GroupNorm
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(in_channels)
        self.act = nn.Hardswish()
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.gn = _groupnorm(out_channels)

        # Residual projection when channel dims differ
        self.proj: nn.Module
        if in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = self.depthwise(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.pointwise(out)
        out = self.gn(out)
        return out + residual


# ---------------------------------------------------------------------------
# DSC-SE block
# ---------------------------------------------------------------------------


class DSCSEBlock(nn.Module):
    """DSC block augmented with a Squeeze-and-Excitation attention sub-block.

    Architecture (paper §III-B, DSC-SE variant):
        DSCBlock → SEBlock
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.dsc = DSCBlock(in_channels, out_channels, kernel_size=kernel_size)
        self.se = SEBlock(out_channels, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dsc(x)
        out = self.se(out)
        return out


# ---------------------------------------------------------------------------
# M-DSC block (Inception-style multi-scale)
# ---------------------------------------------------------------------------


class MDSCBlock(nn.Module):
    """Multi-scale Depthwise Separable Convolution block.

    Architecture (paper §III-C):
        Three parallel depthwise-separable branches with kernel sizes 3, 7, 11
        → concatenate along channel axis
        → 1×1 projection back to out_channels

    The three branches each produce out_channels // 3 channels internally,
    then the concatenation (3 × out_channels // 3 ≈ out_channels) is projected
    to exactly out_channels via a 1×1 pointwise conv.

    ASSUMPTION (§III-C ambiguity): The paper does not specify per-branch channel
    widths.  We allocate out_channels // 3 per branch so the concatenated
    representation has ~out_channels features before projection.  Integer rounding
    means the first branch gets the remainder: widths = (w + r, w, w) where
    w = out_channels // 3 and r = out_channels % 3.
    """

    _KERNELS: tuple[int, int, int] = (3, 7, 11)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        # Per-branch channel allocation
        base_w = out_channels // 3
        remainder = out_channels % 3
        branch_widths = [base_w + remainder, base_w, base_w]

        self.branches = nn.ModuleList()
        for k, bw in zip(self._KERNELS, branch_widths):
            self.branches.append(
                nn.Sequential(
                    # depthwise
                    nn.Conv1d(
                        in_channels, in_channels,
                        kernel_size=k, padding=k // 2,
                        groups=in_channels, bias=False,
                    ),
                    nn.BatchNorm1d(in_channels),
                    nn.Hardswish(),
                    # pointwise to branch width
                    nn.Conv1d(in_channels, bw, kernel_size=1, bias=False),
                    _groupnorm(bw),
                )
            )

        # Concatenated channels == sum(branch_widths) == out_channels
        concat_channels = sum(branch_widths)
        self.project = nn.Conv1d(concat_channels, out_channels, kernel_size=1, bias=False)
        self.project_gn = _groupnorm(out_channels)

        # Residual projection
        self.res_proj: nn.Module
        if in_channels != out_channels:
            self.res_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_proj(x)
        branches = [branch(x) for branch in self.branches]
        out = torch.cat(branches, dim=1)   # [B, concat_ch, L]
        out = self.project(out)
        out = self.project_gn(out)
        return out + residual


# ---------------------------------------------------------------------------
# Main DepthwiseCNN model
# ---------------------------------------------------------------------------


class DepthwiseCNN(nn.Module):
    """Lightweight 1-D CNN for file-fragment classification.

    Supports three variants: 'dsc', 'dsc_se', 'mdsc'.

    Parameters
    ----------
    num_classes : int
        Number of output classes (75 for FFT-75).
    fragment_size : int
        Input length in bytes (512 or 4096).
    variant : str
        One of 'dsc', 'dsc_se', 'm_dsc'.
    embed_dim : int
        Width of the byte embedding.  ASSUMPTION: 64.
    channels : list[int]
        Channel dimension after each block stage.
        ASSUMPTION: [64, 128, 256, 256].
    kernel_size : int
        Depthwise kernel size for DSC/DSC-SE blocks.
        ASSUMPTION: 3 (standard). Ignored for M-DSC (uses 3, 7, 11).
    se_reduction : int
        SE block reduction ratio.  ASSUMPTION: 16.
    p_dropout : float
        Dropout probability before the classifier head.  ASSUMPTION: 0.5.
    """

    def __init__(
        self,
        num_classes: int,
        fragment_size: int = 512,
        variant: Variant = "dsc",
        embed_dim: int = 64,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        se_reduction: int = 16,
        p_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 256]

        self.num_classes = num_classes
        self.fragment_size = fragment_size
        self.variant = variant
        _channels = [embed_dim] + list(channels)

        # ---- Stage 1: Byte Embedding ------------------------------------
        self.embedding = nn.Embedding(256, embed_dim)

        # ---- Stage 2: Block stack ----------------------------------------
        blocks: list[nn.Module] = []
        for i in range(len(channels)):
            in_ch = _channels[i]
            out_ch = _channels[i + 1]

            if variant == "dsc":
                blocks.append(DSCBlock(in_ch, out_ch, kernel_size=kernel_size))
            elif variant == "dsc_se":
                blocks.append(DSCSEBlock(in_ch, out_ch, kernel_size=kernel_size,
                                         se_reduction=se_reduction))
            elif variant == "m_dsc":
                blocks.append(MDSCBlock(in_ch, out_ch))
            else:
                raise ValueError(
                    f"Unknown DepthwiseCNN variant '{variant}'. "
                    "Choose 'dsc', 'dsc_se', or 'm_dsc'."
                )

            # ASSUMPTION: max-pool (stride 2) after every other block
            # (blocks 0 and 2, i.e. after the 1st and 3rd block).
            # This halves the sequence length twice total.
            if i in (0, 2):
                blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.blocks = nn.Sequential(*blocks)

        # ---- Stage 3: Head -----------------------------------------------
        final_ch = channels[-1]
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=p_dropout)
        self.classifier = nn.Linear(final_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Raw byte indices, shape [B, fragment_size], dtype torch.long.

        Returns
        -------
        torch.Tensor
            Log-probabilities, shape [B, num_classes].
        """
        # x: [B, L]  (byte indices 0–255)
        x = self.embedding(x)            # [B, L, embed_dim]
        x = x.permute(0, 2, 1)          # [B, embed_dim, L]  (channels-first)
        x = self.blocks(x)               # [B, final_ch, L']
        x = self.pool(x).squeeze(-1)     # [B, final_ch]
        x = self.dropout(x)
        logits = self.classifier(x)      # [B, num_classes]
        return F.log_softmax(logits, dim=1)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def build_depthwisecnn(
    num_classes: int,
    fragment_size: int = 512,
    variant: str = "dsc",
    embed_dim: int = 64,
    channels: list[int] | None = None,
    kernel_size: int = 3,
    se_reduction: int = 16,
    p_dropout: float = 0.5,
) -> DepthwiseCNN:
    """Construct a DepthwiseCNN from hyperparameters.

    This is the single factory used by the benchmark adapter.  All defaults
    reproduce the paper's architecture as described in §III.
    """
    return DepthwiseCNN(
        num_classes=num_classes,
        fragment_size=fragment_size,
        variant=variant,          # type: ignore[arg-type]
        embed_dim=embed_dim,
        channels=channels,
        kernel_size=kernel_size,
        se_reduction=se_reduction,
        p_dropout=p_dropout,
    )
