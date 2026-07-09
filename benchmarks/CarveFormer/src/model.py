"""
benchmarks/CarveFormer/src/model.py
====================================
Faithful implementation of **CarveFormer** (Guzhov & Wirth, ECCWS 2025),
*Transformer-Based File Fragment Type Classification for File Carving in
Digital Forensics*.

Paper architecture (Section 3.2 + Figure 1)
-------------------------------------------
The original Swin Transformer V2 begins with a 2D convolutional patch-embed
layer that maps a 3-channel RGB image into 96-channel patch tokens, followed
by LayerNorm. CarveFormer's single architectural change is to **replace that
image conv/norm front-end with a byte embedding layer of dimension 96**, then
**reshape the embedded byte sequence into a 2D structure** that matches the
token grid the Swin V2 backbone expects. Everything after the front-end is the
stock Swin Transformer V2 Tiny backbone (ImageNet1k-pretrained), followed by a
classification head sized to the number of FFT-75 classes.

Why this maps cleanly onto timm's SwinV2-Tiny
---------------------------------------------
The stock ``swinv2_tiny_window8_256`` model does::

    image [B, 3, 256, 256]
      -> patch_embed: Conv2d(3, 96, kernel=4, stride=4)
      -> [B, 64, 64, 96]        # a 64x64 grid of 96-d patch tokens
      -> layers (Swin V2 stages)
      -> [B, 8, 8, 768]
      -> head (global pool + Linear -> num_classes)

CarveFormer keeps ``layers``, ``norm``, and ``head`` **unchanged** and swaps
only ``patch_embed`` for a module that produces the *same* ``[B, H, W, 96]``
token grid directly from raw bytes:

    bytes [B, L]
      -> Embedding(256, 96)      # one 96-d vector per byte value
      -> [B, L, 96]
      -> reshape to [B, H, W, 96] with H * W == L
      -> (unchanged Swin V2 stages / norm / head)

For **L = 4096** this yields a 64x64 grid, which is *exactly* the native
SwinV2-Tiny token grid — a clean, paper-faithful match. For **L = 512** the
paper does not state the exact grid; we use a documented default of 16x32
(see :func:`default_grid_for_length`). The reshape is deliberately isolated in
one helper so the grid can be changed later without touching the rest of the
model.

Framework contract
------------------
``forward()`` returns **log-probabilities** of shape ``[B, num_classes]``
(``log_softmax``), matching ``src.core.interfaces.FragmentClassifier`` whose
default loss is ``nll_loss``. This model itself stays independent of the
benchmark framework (no imports from ``src`` of the parent repo); the thin
adapter in ``adapter.py`` (Phase 2) is what plugs it into the registry.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _SwinBackbone(Protocol):
    """Minimal typed view of the timm SwinV2 backbone methods CarveFormer uses.

    timm models are plain ``nn.Module``s, whose ``__getattr__`` is typed to
    return ``Tensor | Module``; casting to this Protocol lets the type checker
    see ``forward_features`` / ``forward_head`` as tensor-returning callables.
    """

    embed_dim: int
    patch_embed: nn.Module

    def forward_features(self, x: torch.Tensor) -> torch.Tensor: ...

    def forward_head(self, x: torch.Tensor) -> torch.Tensor: ...

# The stock timm backbone whose post-patch-embed stages CarveFormer reuses.
_DEFAULT_BACKBONE = "swinv2_tiny_window8_256"
# Byte vocabulary: values 0..255 inclusive.
_BYTE_VOCAB_SIZE = 256
# Paper embedding dimension for the byte front-end.
_DEFAULT_EMBED_DIM = 96
# SwinV2-Tiny's native patch size (conv stride). The byte-token grid times this
# gives the ``img_size`` we must construct the backbone with, so the stages'
# window-attention masks match our grid.
_NATIVE_PATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CarveFormerConfig:
    """Configuration for :class:`CarveFormer`.

    Every value the paper leaves adjustable is exposed here rather than
    hardcoded, so the same class serves all FFT-75 scenarios and both
    fragment sizes by configuration only.

    Attributes
    ----------
    num_classes:
        Number of output classes (75/11/25/5/2/2 depending on FFT-75 scenario).
    fragment_size:
        Byte-fragment length; 512 or 4096 for FFT-75.
    embed_dim:
        Byte-embedding dimension. Paper value is 96 (also the SwinV2-Tiny
        stage-0 channel count, which must match for the backbone to accept
        the tokens).
    backbone:
        timm model name for the Swin V2 Tiny backbone.
    pretrained:
        Whether to initialise the backbone from ImageNet1k weights (paper
        default True). Set False for offline/CI use.
    grid_hw:
        Optional explicit ``(H, W)`` token grid. If None, a sensible default
        is chosen for the fragment size (see :func:`default_grid_for_length`).
    drop_rate:
        Dropout on the classifier head (paper mentions dropout before the
        classifier; default 0.0 keeps behaviour neutral unless configured).
    """

    num_classes: int
    fragment_size: int = 512
    embed_dim: int = _DEFAULT_EMBED_DIM
    backbone: str = _DEFAULT_BACKBONE
    pretrained: bool = True
    grid_hw: tuple[int, int] | None = None
    drop_rate: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Paper-specific adaptation helpers (isolated so they are easy to audit/tune)
# ---------------------------------------------------------------------------
def default_grid_for_length(length: int) -> tuple[int, int]:
    """Return the default ``(H, W)`` token grid for a byte-fragment length.

    The paper reshapes the embedded byte sequence into a 2D structure so the
    Swin V2 backbone can consume it. For 4096 bytes the natural, paper-aligned
    grid is 64x64 (identical to SwinV2-Tiny's native 64x64 patch grid). For
    512 bytes the paper does not specify the grid; we default to 16x32.

    Both dimensions are chosen so ``H * W == length`` and so that H and W are
    divisible by the Swin window/downsampling factors (multiples of 8), which
    keeps the four downsampling stages well-defined.

    Parameters
    ----------
    length:
        Fragment length in bytes (e.g. 512 or 4096).

    Returns
    -------
    tuple[int, int]
        The ``(H, W)`` grid such that ``H * W == length``.

    Raises
    ------
    ValueError
        If no factorisation into two multiples of 8 is available and the
        length is not one of the known FFT-75 sizes.
    """
    known = {
        512: (16, 32),   # documented default; 16 and 32 are multiples of 8
        4096: (64, 64),  # matches SwinV2-Tiny native token grid exactly
    }
    if length in known:
        return known[length]

    # Fallback: nearest-square factorisation into two integers whose product
    # is `length`, preferring factors divisible by 8 where possible.
    root = int(math.isqrt(length))
    for h in range(root, 0, -1):
        if length % h == 0:
            w = length // h
            return (h, w)
    raise ValueError(f"Cannot factor fragment length {length} into a 2D grid.")


class ByteEmbeddingPatchifier(nn.Module):
    """Replacement for Swin's image ``patch_embed``.

    Turns a batch of raw byte fragments ``[B, L]`` (integer values 0..255)
    into the ``[B, H, W, embed_dim]`` token grid the Swin V2 backbone's stages
    expect — the exact tensor layout produced by the original conv patch-embed,
    but sourced from a learned byte embedding instead of an image convolution.

    This is CarveFormer's core architectural adaptation (paper Section 3.2).
    """

    def __init__(self, embed_dim: int, grid_hw: tuple[int, int]) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_h, self.grid_w = grid_hw
        self.embedding = nn.Embedding(_BYTE_VOCAB_SIZE, embed_dim)
        # LayerNorm mirrors the norm that follows the original patch-embed.
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed and reshape bytes into a 2D token grid.

        Parameters
        ----------
        x:
            Byte tensor of shape ``[B, L]`` with integer values in ``[0, 255]``
            and ``L == grid_h * grid_w``.

        Returns
        -------
        torch.Tensor
            Token grid of shape ``[B, grid_h, grid_w, embed_dim]``.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected byte input [B, L], got shape {tuple(x.shape)}.")
        b, length = x.shape
        expected = self.grid_h * self.grid_w
        if length != expected:
            raise ValueError(
                f"Fragment length {length} does not match configured grid "
                f"{self.grid_h}x{self.grid_w} (= {expected}). Adjust grid_hw."
            )
        emb = self.embedding(x.long())          # [B, L, embed_dim]
        emb = self.norm(emb)                    # matches post-patch-embed norm
        grid = emb.view(b, self.grid_h, self.grid_w, self.embed_dim)  # [B,H,W,C]
        return grid


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class CarveFormer(nn.Module):
    """CarveFormer: SwinV2-Tiny with a byte-embedding front-end.

    See the module docstring for the full paper mapping. This class is
    intentionally framework-agnostic; the benchmark adapter wraps it.
    """

    def __init__(self, config: CarveFormerConfig) -> None:
        super().__init__()
        self.config = config

        grid_hw = config.grid_hw or default_grid_for_length(config.fragment_size)
        self.grid_hw = grid_hw

        # The Swin V2 stages bake in resolution-dependent window-attention
        # masks at construction time, derived from ``img_size`` / patch_size.
        # Since we bypass the conv patch-embed and feed our own token grid, we
        # must build the backbone with an ``img_size`` whose native patch grid
        # equals our (H, W) grid. timm's SwinV2-Tiny uses patch_size 4, so
        # img_size = (H * 4, W * 4).
        patch = _NATIVE_PATCH_SIZE
        img_size = (grid_hw[0] * patch, grid_hw[1] * patch)
        backbone = _build_backbone(
            backbone=config.backbone,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            drop_rate=config.drop_rate,
            img_size=img_size,
        )
        backbone_embed_dim = int(cast(_SwinBackbone, backbone).embed_dim)
        if backbone_embed_dim != config.embed_dim:
            # The byte embedding must match the backbone's stage-0 channel
            # count, otherwise the Swin stages cannot consume the tokens.
            raise ValueError(
                f"embed_dim ({config.embed_dim}) must equal the backbone's "
                f"stage-0 dim ({backbone_embed_dim}) for '{config.backbone}'."
            )

        # Swap the image patch-embed for the byte-embedding patchifier.
        self.patchify = ByteEmbeddingPatchifier(config.embed_dim, grid_hw)
        backbone.patch_embed = nn.Identity()  # bytes are already patchified
        self.backbone: nn.Module = backbone

        logger.info(
            "CarveFormer built: fragment_size=%d grid=%dx%d embed_dim=%d "
            "backbone=%s pretrained=%s num_classes=%d",
            config.fragment_size, grid_hw[0], grid_hw[1], config.embed_dim,
            config.backbone, config.pretrained, config.num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of byte fragments.

        Parameters
        ----------
        x:
            Byte tensor ``[B, fragment_size]`` with integer values 0..255.

        Returns
        -------
        torch.Tensor
            Log-probabilities of shape ``[B, num_classes]`` (``log_softmax``),
            matching the framework's ``FragmentClassifier`` contract.
        """
        tokens = self.patchify(x)               # [B, H, W, embed_dim]
        # ``self.backbone`` is an nn.Module; cast to the typed Protocol so the
        # type checker resolves forward_features/forward_head (nn.Module's
        # __getattr__ otherwise returns a ``Tensor | Module`` union).
        backbone = cast(_SwinBackbone, self.backbone)
        features = backbone.forward_features(tokens)   # Swin stages -> [B,H',W',C]
        logits = backbone.forward_head(features)       # [B, num_classes]
        return F.log_softmax(logits, dim=1)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the model's parameter count."""
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)


# ---------------------------------------------------------------------------
# Backbone construction (isolated so timm specifics live in one place)
# ---------------------------------------------------------------------------
def _build_backbone(
    backbone: str,
    num_classes: int,
    pretrained: bool,
    drop_rate: float,
    img_size: tuple[int, int],
) -> nn.Module:
    """Create the Swin V2 Tiny backbone via timm.

    ``img_size`` is chosen by the caller so the backbone's native patch grid
    equals the byte-token grid CarveFormer feeds in. When ``img_size`` differs
    from the pretrained default (256), timm interpolates the pretrained
    relative-position tables to the new resolution automatically.

    Falls back to random initialisation (with a warning) if pretrained
    weights cannot be downloaded — this keeps offline/CI smoke tests runnable
    without changing the architecture.
    """
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "CarveFormer requires the 'timm' package for the Swin V2 backbone. "
            "Install it with `pip install timm`."
        ) from exc

    kwargs: dict[str, Any] = dict(
        num_classes=num_classes,
        drop_rate=drop_rate,
        img_size=img_size,
    )
    try:
        model = timm.create_model(backbone, pretrained=pretrained, **kwargs)
    except Exception as exc:  # noqa: BLE001 - surface offline weight failures
        if pretrained:
            logger.warning(
                "Failed to load pretrained weights for '%s' (%s); falling back "
                "to random initialisation. Set pretrained=False to silence.",
                backbone, exc,
            )
            model = timm.create_model(backbone, pretrained=False, **kwargs)
        else:
            raise
    return model


def build_carveformer(
    num_classes: int,
    fragment_size: int = 512,
    embed_dim: int = _DEFAULT_EMBED_DIM,
    backbone: str = _DEFAULT_BACKBONE,
    pretrained: bool = True,
    grid_hw: tuple[int, int] | None = None,
    drop_rate: float = 0.0,
    **extra: Any,
) -> CarveFormer:
    """Factory for :class:`CarveFormer`.

    This is the clean entry point the adapter (Phase 2) calls. Keeping a
    free-function factory (rather than only the class constructor) mirrors the
    ``build_bytercnn`` pattern already used elsewhere in DeepCarv.

    Parameters
    ----------
    num_classes:
        Number of FFT-75 classes for the target scenario.
    fragment_size:
        512 or 4096.
    embed_dim:
        Byte-embedding dimension (paper: 96; must match backbone stage-0 dim).
    backbone:
        timm SwinV2-Tiny model name.
    pretrained:
        Load ImageNet1k weights (paper default True).
    grid_hw:
        Optional explicit ``(H, W)`` token grid override.
    drop_rate:
        Classifier-head dropout.

    Returns
    -------
    CarveFormer
        The constructed model.
    """
    config = CarveFormerConfig(
        num_classes=num_classes,
        fragment_size=fragment_size,
        embed_dim=embed_dim,
        backbone=backbone,
        pretrained=pretrained,
        grid_hw=grid_hw,
        drop_rate=drop_rate,
        extra=dict(extra),
    )
    return CarveFormer(config)
