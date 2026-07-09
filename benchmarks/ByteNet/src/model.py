"""
benchmarks/ByteNet/src/model.py
--------------------------------
Full PyTorch implementation of ByteNet (ByteResNet and ByteFormer) from:

    "ByteNet: Rethinking Multimedia File Fragment Classification through
    Visual Perspectives" — Liu et al., 2023.

Architecture overview
---------------------
1. Byte2Image — converts a raw 1-D byte sector into a 2-D grayscale image
   by bit-shifting (intrabyte exposure) and stacking intrabyte n-grams.

2. ByteNet (dual-branch network):
   a. Byte Branch Feature Extraction (BBFE) — a single fully-connected
      layer that memorises byte co-occurrence ("magic bytes").
   b. Image Branch Feature Extraction (IBFE) — a deep hierarchical
      network (4 stages) operating on the Byte2Image output.
   c. Feature Fusion — concat(xsf, xdf) → FC → num_classes.

Two IBFE variants are provided:
   - ByteResNet  : n-gram embedding + ResNet blocks
   - ByteFormer  : patch embedding + PoolFormer blocks

Input contract (from FragmentDataset)
--------------------------------------
   x : torch.Tensor  shape [B, fragment_size]  dtype int64  values 0-255

Output contract (FragmentClassifier)
--------------------------------------
   log_probs : torch.Tensor  shape [B, num_classes]  (log-softmax)

Paper implementation details (§IV-B, page 6-7)
------------------------------------------------
ByteResNet:
  - n-gram embedding channel dim C1 = 96
  - stage layers Li = [2, 2, 2, 2]
  - channel dims Ci = [64, 128, 256, 512]

ByteFormer:
  - patch embedding dim C1 = 64 (512B) or 96 (4096B)
  - patch size P = 8
  - stage layers Li = [6, 6, 18, 6]
  - channel dims Ci = [64, 128, 320, 512]

4096-byte sectors:
  Segmented into 8 × 512-byte chunks. Each chunk is independently
  converted to a grayscale image; the 8 images are concatenated along
  the channel dimension, giving an 8-channel input to the IBFE.

This file is independent of the benchmark framework — it only depends
on PyTorch. The adapter in adapter.py wraps this for the registry.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default n-gram size (paper §IV-B)
DEFAULT_NGRAM_N: int = 16
# Byte range
_BYTE_MAX: float = 255.0
# Normalisation (approximated grayscale mean/std)
_NORM_MEAN: float = 0.5
_NORM_STD: float = 0.5
# PoolFormer MLP expansion ratio (MetaFormer standard)
_POOL_EXPAND: int = 4
# Chunk size used for 4096-byte sectors
_CHUNK_SIZE: int = 512
_CHUNKS_PER_4096: int = 8   # 4096 // 512


# ---------------------------------------------------------------------------
# Byte2Image
# ---------------------------------------------------------------------------


class Byte2Image(nn.Module):
    """Convert a raw 1-D byte sector to a 2-D grayscale image.

    Steps (paper §III-B):
    1. Intrabyte information exposure: bit-shift the Ns-byte sequence 7
       times to obtain 8 rows total (the original plus 7 shifted copies).
    2. Stack rows column-wise → byte matrix ∈ [Ns, 8].
    3. Intrabyte n-gram: slide a window of width n over the height
       dimension → image of shape [H, W] where H = Ns-n+1, W = 8*n.
    4. Normalise to float in [0, 1] then standardise.

    For fragment_size=4096, the input is pre-segmented into 8 × 512-byte
    chunks by the caller (ByteNet.forward).  This module operates on a
    single chunk (fragment_size=512) and produces a 1-channel image.

    Parameters
    ----------
    ngram_n : int
        n-gram size (default 16, as in the paper).
    fragment_size : int
        Number of bytes in the input sector (512 for a single chunk).
    """

    def __init__(self, ngram_n: int = DEFAULT_NGRAM_N, fragment_size: int = 512) -> None:
        super().__init__()
        self.ngram_n = ngram_n
        self.fragment_size = fragment_size

        H = fragment_size - ngram_n + 1
        W = 8 * ngram_n
        self._out_shape: tuple[int, int] = (H, W)  # (H, W)

    @property
    def out_h(self) -> int:
        return self._out_shape[0]

    @property
    def out_w(self) -> int:
        return self._out_shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, Ns]  int64 or float, values 0-255.

        Returns
        -------
        img : Tensor [B, 1, H, W]  float32, normalised.
        """
        B, Ns = x.shape
        # ---- Step 1: Intrabyte exposure via bit-shifting -----------------
        # x0 = x, xi = xi-1 << 1  (& 0xFF keeps within 8 bits)
        rows = [x]
        for _ in range(7):
            rows.append((rows[-1] << 1) & 0xFF)
        # byte_matrix: [B, Ns, 8]
        byte_mat = torch.stack(rows, dim=2).float()   # [B, Ns, 8]

        # ---- Step 2: Intrabyte n-gram ------------------------------------
        n = self.ngram_n
        # Unfold over the height (Ns) dimension with window=n, stride=1
        # unfold: [B, H, n, 8]  where H = Ns - n + 1
        unfolded = byte_mat.unfold(1, n, 1)           # [B, H, n, 8]
        # Reshape to [B, H, n*8] = [B, H, W]
        H, W = unfolded.shape[1], n * 8
        img = unfolded.reshape(B, H, W)               # [B, H, W]

        # ---- Step 3: Normalise ------------------------------------------
        img = img / _BYTE_MAX                         # [0, 1]
        img = (img - _NORM_MEAN) / _NORM_STD

        # ---- Add channel dim --------------------------------------------
        return img.unsqueeze(1)                       # [B, 1, H, W]


# ---------------------------------------------------------------------------
# Embedding layers
# ---------------------------------------------------------------------------


class NGramEmbedding(nn.Module):
    """N-gram embedding layer for ByteResNet (paper §III-D, Fig. 6a).

    Applies a 'wide' convolution (kernel width = image width W) to each row
    of the intrabyte n-gram image, producing K = embed_dim embeddings per
    row.  The resulting [B, K, H, 1] feature is transposed to
    [B, K, H, 1], followed by a 7×7 Conv2d and a stride-2 max-pool to
    produce x0 for the deep feature extraction stages.

    Parameters
    ----------
    embed_dim : int
        Number of output channels C1 (96 for ByteResNet per the paper).
    in_h : int
        Height of the Byte2Image output (Ns - n + 1).
    in_w : int
        Width of the Byte2Image output (8 * n).
    out_channels : int
        Output channels after the 7×7 conv (= stage-1 channel dim = 64).
    """

    def __init__(
        self,
        embed_dim: int,
        in_h: int,
        in_w: int,
        out_channels: int = 64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels   # output channel dim for ImageBranch detection
        self.in_w = in_w

        # Wide conv: kernel (1, W) over [B, 1, H, W] → [B, K, H, 1]
        self.wide_conv = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=(1, in_w),
            bias=True,
        )
        # 7×7 conv to blend n-gram info; input has shape [B, K, H, 1]
        # We treat K as channels and H as height.
        self.conv7 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Stride-2 pool to halve H before stage-1
        self.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, 1, H, W]  (output of Byte2Image, possibly multi-channel)

        Returns
        -------
        out : Tensor [B, C1, H', 1]
        """
        # If input has more than 1 channel (4096B mode, 8 channels),
        # process first channel only for n-gram embedding — the full
        # multi-channel handling is done in ImageBranch.
        if x.shape[1] > 1:
            # For multi-channel mode we sum-pool across channels first
            x = x.mean(dim=1, keepdim=True)

        xemb = self.wide_conv(x)          # [B, K, H, 1]
        x0 = self.conv7(xemb)             # [B, C1, H, 1]
        x0 = self.pool(x0)                # [B, C1, H//2, 1]
        return x0


class PatchEmbedding(nn.Module):
    """Patch embedding layer for ByteFormer (paper §III-D, Fig. 6b).

    Divides the input image into non-overlapping P×P patches and projects
    each to a C1-dimensional feature, resulting in a 2-D feature map
    x0 ∈ [B, C1, H//P, W//P].  Positional embedding is added (learnable).

    Parameters
    ----------
    embed_dim : int
        Patch embedding channel dimension C1.
    patch_size : int
        Patch size P (default 8, per the paper).
    in_channels : int
        Number of input image channels (1 for 512B, 8 for 4096B mode).
    in_h : int
        Image height H.
    in_w : int
        Image width W.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: int = 8,
        in_channels: int = 1,
        in_h: int = 497,
        in_w: int = 128,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Conv2d with stride uses floor division: out = floor((in - k) / s) + 1
        # For kernel_size == stride == patch_size: out = floor(in / patch_size)
        H_out = in_h // patch_size
        W_out = in_w // patch_size
        self._out_h = H_out
        self._out_w = W_out

        # Conv2d stride=P maps [B, C_in, H, W] → [B, C1, H//P, W//P]
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.norm = nn.LayerNorm(embed_dim)
        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, H_out, W_out))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, C_in, H, W]

        Returns
        -------
        out : Tensor [B, C1, H//P, W//P]
        """
        B = x.shape[0]
        x0 = self.proj(x)                 # [B, C1, H', W']
        x0 = x0 + self.pos_embed

        # LayerNorm over channel dim (applied in spatial form)
        B, C, H, W = x0.shape
        x0 = x0.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [BHW, C]
        x0 = self.norm(x0)
        x0 = x0.reshape(B, H, W, C).permute(0, 3, 1, 2)    # [B, C, H, W]
        return x0


# ---------------------------------------------------------------------------
# Feature extraction blocks
# ---------------------------------------------------------------------------


class ResNetBlock(nn.Module):
    """Standard residual block for ByteResNet (paper §III-D, Fig. 6a).

    Two 3×3 convolutions with batch normalisation and a shortcut.  For
    ``in_channels != out_channels`` or ``stride != 1``, the shortcut
    uses a 1×1 conv to match dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class PoolFormerBlock(nn.Module):
    """PoolFormer block for ByteFormer (paper §III-D, Fig. 6b; [46]).

    Two residual sub-blocks:
    1. Token mixer: average pooling (stride-1, 3×3) — replaces self-attention.
    2. Channel MLP: two 1×1 convolutions with GELU activation.

    Layer normalisation is applied before each sub-block (pre-norm design).
    """

    def __init__(self, channels: int, expand_ratio: int = _POOL_EXPAND) -> None:
        super().__init__()
        hidden = channels * expand_ratio
        self.norm1 = LayerNorm2d(channels)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)
        self.norm2 = LayerNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixer sub-block
        x = x + self.pool(self.norm1(x)) - self.norm1(x)
        # Channel MLP sub-block
        x = x + self.mlp(self.norm2(x))
        return x


class LayerNorm2d(nn.Module):
    """Channel-last LayerNorm operating on BCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)   # [B, H, W, C]
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)  # [B, C, H, W]


# ---------------------------------------------------------------------------
# RVFE (Residual Visual Feature Extractor) stage
# ---------------------------------------------------------------------------


class RVFEStage(nn.Module):
    """One stage of the Residual Visual Feature Extractor (paper §III-C).

    Contains L feature extraction blocks followed by a downsampling
    convolution (stride-2 3×3 conv), except for the last stage which
    uses Global Average Pooling instead.

    Parameters
    ----------
    block_type : 'resnet' | 'poolformer'
    in_channels, out_channels : int
    num_blocks : int  — L_i in the paper
    is_last : bool  — if True, replace downsample with GAP
    """

    def __init__(
        self,
        block_type: Literal["resnet", "poolformer"],
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        is_last: bool = False,
    ) -> None:
        super().__init__()
        self.is_last = is_last

        blocks: list[nn.Module] = []
        for i in range(num_blocks):
            if block_type == "resnet":
                blocks.append(ResNetBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                ))
            else:  # poolformer
                if i == 0 and in_channels != out_channels:
                    blocks.append(nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    ))
                blocks.append(PoolFormerBlock(out_channels))
        self.blocks = nn.Sequential(*blocks)

        if is_last:
            self.down = nn.AdaptiveAvgPool2d(1)   # GAP
        else:
            self.down = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.down(x)
        if self.is_last:
            x = x.flatten(1)   # [B, C]
        return x


# ---------------------------------------------------------------------------
# Byte Branch (BBFE)
# ---------------------------------------------------------------------------


class ByteBranch(nn.Module):
    """Shallow byte branch feature extraction (BBFE, paper §III-C).

    A single fully-connected layer that captures global byte co-occurrence
    (analogous to 'magic byte' detection).

    Input : [B, fragment_size] int64/float byte values
    Output: [B, out_dim] float
    """

    def __init__(self, fragment_size: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(fragment_size, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.float() / _BYTE_MAX)   # normalise input to [0,1]


# ---------------------------------------------------------------------------
# Image Branch (IBFE)
# ---------------------------------------------------------------------------


class ImageBranch(nn.Module):
    """Deep image branch feature extraction (IBFE, paper §III-C, Fig. 5b).

    Parameters
    ----------
    block_type : 'resnet' | 'poolformer'
    stage_layers : sequence of 4 ints  — L_i per stage
    channels : sequence of 4 ints  — C_i per stage
    embedding : NGramEmbedding | PatchEmbedding
    """

    def __init__(
        self,
        block_type: Literal["resnet", "poolformer"],
        embedding: nn.Module,
        stage_layers: Sequence[int],
        channels: Sequence[int],
    ) -> None:
        super().__init__()
        assert len(stage_layers) == 4 and len(channels) == 4

        self.embedding = embedding

        # Determine the embedding output channel dimension.
        # NGramEmbedding exposes .out_channels (= channels[0] by construction).
        # PatchEmbedding exposes .embed_dim (= C1, may differ from channels[0]).
        # Prefer .out_channels, fall back to .embed_dim, then channels[0].
        embed_out_ch: int = (
            getattr(embedding, 'out_channels', None)
            or getattr(embedding, 'embed_dim', None)
            or channels[0]
        )

        stages = []
        in_ch_for_stage = embed_out_ch  # stage 0 receives embedding output
        for i in range(4):
            out_ch = channels[i]
            stages.append(RVFEStage(
                block_type=block_type,
                in_channels=in_ch_for_stage,
                out_channels=out_ch,
                num_blocks=stage_layers[i],
                is_last=(i == 3),
            ))
            # After stage i, inter_downs[i] maps channels[i]→channels[i+1].
            # So stage i+1 receives channels[i+1] channels as input.
            # (For the last stage, i==3 and no inter_down is applied.)
            in_ch_for_stage = channels[i + 1] if i < 3 else out_ch
        self.stages = nn.ModuleList(stages)

        # Downsampling convolutions between stages (not used for last stage)
        downs = []
        for i in range(3):
            downs.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ))
        self.inter_downs = nn.ModuleList(downs)

        self.out_dim = channels[-1]

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        img : Tensor [B, C_in, H, W]  (output of Byte2Image, possibly stacked)

        Returns
        -------
        features : Tensor [B, channels[-1]]
        """
        x = self.embedding(img)         # [B, C0, H', W']
        for i, stage in enumerate(self.stages):
            x = stage.blocks(x)
            if i < 3:
                x = self.inter_downs[i](x)
            else:
                # Last stage: GAP → flatten
                x = stage.down(x).flatten(1)
        return x


# ---------------------------------------------------------------------------
# ByteNet (full dual-branch network)
# ---------------------------------------------------------------------------


class ByteNet(nn.Module):
    """End-to-end dual-branch ByteNet classifier (paper §III-C, Fig. 5).

    Combines BBFE (byte branch) and IBFE (image branch) by concatenation,
    followed by a linear classifier.

    Parameters
    ----------
    num_classes : int
    fragment_size : int  — 512 or 4096
    block_type : 'resnet' | 'poolformer'
    embedding : nn.Module  — NGramEmbedding or PatchEmbedding
    stage_layers : sequence of 4 ints
    channels : sequence of 4 ints
    byte_branch_dim : int  — BBFE output dimension (default 512)
    ngram_n : int  — n-gram size (default 16)
    """

    def __init__(
        self,
        num_classes: int,
        fragment_size: int,
        block_type: Literal["resnet", "poolformer"],
        embedding: nn.Module,
        stage_layers: Sequence[int],
        channels: Sequence[int],
        byte_branch_dim: int = 512,
        ngram_n: int = DEFAULT_NGRAM_N,
    ) -> None:
        super().__init__()
        self.fragment_size = fragment_size
        self.ngram_n = ngram_n
        self.num_classes = num_classes

        # 4096B → 8×512B chunking
        self._use_chunking = (fragment_size == 4096)
        _chunk_fs = _CHUNK_SIZE if self._use_chunking else fragment_size

        # Byte2Image operates on single 512B chunks
        self.byte2image = Byte2Image(ngram_n=ngram_n, fragment_size=_chunk_fs)

        # Byte branch
        self.byte_branch = ByteBranch(fragment_size, byte_branch_dim)

        # Image branch
        self.image_branch = ImageBranch(
            block_type=block_type,
            embedding=embedding,
            stage_layers=stage_layers,
            channels=channels,
        )

        # Fusion classifier
        feat_dim = byte_branch_dim + channels[-1]
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, num_classes),
        )

    def _convert_to_image(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw byte tensor to image tensor.

        For fragment_size=512: [B, 512] → [B, 1, H, W]
        For fragment_size=4096: [B, 4096] → [B, 8, H, W]
          (8 chunks of 512, converted independently, stacked on channel dim)
        """
        if not self._use_chunking:
            return self.byte2image(x)  # [B, 1, H, W]
        else:
            # Split into 8 × 512-byte chunks
            chunks = x.split(_CHUNK_SIZE, dim=1)   # 8 × [B, 512]
            imgs = [self.byte2image(c) for c in chunks]  # 8 × [B, 1, H, W]
            return torch.cat(imgs, dim=1)           # [B, 8, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, fragment_size]  int64, values 0-255

        Returns
        -------
        log_probs : Tensor [B, num_classes]
        """
        # Byte branch
        xsf = self.byte_branch(x)                  # [B, F0]

        # Image branch
        img = self._convert_to_image(x)             # [B, C_in, H, W]
        xdf = self.image_branch(img)                # [B, C_last]

        # Feature fusion
        fused = torch.cat([xsf, xdf], dim=1)       # [B, F0 + C_last]
        logits = self.classifier(fused)             # [B, num_classes]
        return F.log_softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def build_bytenet_resnet(
    num_classes: int,
    fragment_size: int = 512,
    embed_dim: int = 96,
    stage_layers: Sequence[int] = (2, 2, 2, 2),
    channels: Sequence[int] = (64, 128, 256, 512),
    byte_branch_dim: int = 512,
    ngram_n: int = DEFAULT_NGRAM_N,
) -> ByteNet:
    """Build a ByteResNet model (paper §IV-B).

    Parameters
    ----------
    num_classes : int
    fragment_size : int  — 512 or 4096
    embed_dim : int  — n-gram embedding output channels C1 (default 96)
    stage_layers : tuple of 4 ints  — ResNet blocks per stage [2,2,2,2]
    channels : tuple of 4 ints  — channel dims [64,128,256,512]
    byte_branch_dim : int  — BBFE output dim (default 512)
    ngram_n : int  — n-gram size (default 16)
    """
    _chunk_fs = _CHUNK_SIZE if fragment_size == 4096 else fragment_size
    in_h = _chunk_fs - ngram_n + 1
    in_w = 8 * ngram_n

    # Multi-channel for 4096B mode
    in_channels_img = _CHUNKS_PER_4096 if fragment_size == 4096 else 1

    embedding = NGramEmbedding(
        embed_dim=embed_dim,
        in_h=in_h,
        in_w=in_w,
        out_channels=channels[0],
    )

    return ByteNet(
        num_classes=num_classes,
        fragment_size=fragment_size,
        block_type="resnet",
        embedding=embedding,
        stage_layers=stage_layers,
        channels=channels,
        byte_branch_dim=byte_branch_dim,
        ngram_n=ngram_n,
    )


def build_bytenet_former(
    num_classes: int,
    fragment_size: int = 512,
    embed_dim: int | None = None,   # None → auto: 64 (512B) or 96 (4096B)
    patch_size: int = 8,
    stage_layers: Sequence[int] = (6, 6, 18, 6),
    channels: Sequence[int] = (64, 128, 320, 512),
    byte_branch_dim: int = 512,
    ngram_n: int = DEFAULT_NGRAM_N,
) -> ByteNet:
    """Build a ByteFormer model (paper §IV-B).

    Parameters
    ----------
    num_classes : int
    fragment_size : int  — 512 or 4096
    embed_dim : int | None  — patch embedding dim C1; auto-set to 64 (512B)
                              or 96 (4096B) per paper
    patch_size : int  — patch size P (default 8)
    stage_layers : tuple of 4 ints  — PoolFormer blocks per stage [6,6,18,6]
    channels : tuple of 4 ints  — channel dims [64,128,320,512]
    byte_branch_dim : int  — BBFE output dim (default 512)
    ngram_n : int  — n-gram size (default 16)
    """
    if embed_dim is None:
        embed_dim = 96 if fragment_size == 4096 else 64

    _chunk_fs = _CHUNK_SIZE if fragment_size == 4096 else fragment_size
    in_h = _chunk_fs - ngram_n + 1
    in_w = 8 * ngram_n
    in_channels_img = _CHUNKS_PER_4096 if fragment_size == 4096 else 1

    embedding = PatchEmbedding(
        embed_dim=embed_dim,
        patch_size=patch_size,
        in_channels=in_channels_img,
        in_h=in_h,
        in_w=in_w,
    )

    return ByteNet(
        num_classes=num_classes,
        fragment_size=fragment_size,
        block_type="poolformer",
        embedding=embedding,
        stage_layers=stage_layers,
        channels=channels,
        byte_branch_dim=byte_branch_dim,
        ngram_n=ngram_n,
    )
