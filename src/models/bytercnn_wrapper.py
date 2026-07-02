"""
src/models/bytercnn_wrapper.py
-------------------------------
Clean PyTorch reimplementation of the ByteRCNN architecture.

Published reference:
    Boucetta et al. / Srivastava et al. (ByteRCNN)
    Original Keras implementation: https://github.com/kristian-fer/ByteRCNN

This module is isolated from the rest of the codebase.  It does NOT import
from any other src/ module.  The only dependencies are torch.

Frozen architecture (must not be changed for the benchmark):
    - Byte embedding dim : 16
    - BiGRU layers       : 2 stacked, bidirectional
    - GRU hidden units   : 64  → 128 per direction pair
    - CNN branches       : 4 parallel 1D branches
    - CNN kernel sizes   : [9, 27, 40, 65]
    - CNN filters        : 64 per branch
    - Dense 1            : 1024
    - Dense 2            : 512
    - Output             : softmax over num_classes (default 75)

Feature vector before Dense layers
    - BiGRU last hidden  : 2 * 64 = 128
    - 4 × CNN branches   : 4 * 64 = 256 (global-max-pooled per branch)
    - Total concat       : 128 + 256 = 384

Usage
-----
    from src.models.bytercnn_wrapper import build_bytercnn

    model = build_bytercnn()          # num_classes=75 (frozen default)
    model = build_bytercnn(num_classes=10)  # for experimentation only

    # Forward pass:
    # x : torch.LongTensor  shape [batch, 512]   (byte values 0–255)
    # out: torch.FloatTensor shape [batch, num_classes]  (log-softmax)
    logits = model(x)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Frozen benchmark hyperparameters
# ---------------------------------------------------------------------------
_DEFAULT_NUM_CLASSES: int = 75
_DEFAULT_EMBEDDING_DIM: int = 16
_DEFAULT_GRU_UNITS: int = 64
_DEFAULT_GRU_LAYERS: int = 2
_DEFAULT_KERNEL_SIZES: list[int] = [9, 27, 40, 65]
_DEFAULT_CNN_FILTERS: int = 64
_DEFAULT_DENSE_UNITS: list[int] = [1024, 512]

# Vocabulary: byte values 0–255 (256 tokens total)
_VOCAB_SIZE: int = 256


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class _CNNBranch(nn.Module):
    """Single 1-D CNN branch: Conv1d → ReLU → GlobalMaxPool."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        # 'same' padding keeps sequence length identical
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        h = F.relu(self.bn(self.conv(x)))  # [B, out_channels, L]
        h, _ = h.max(dim=2)                # [B, out_channels]  global-max-pool
        return h


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class ByteRCNN(nn.Module):
    """PyTorch reimplementation of ByteRCNN.

    The forward method takes raw byte sequences (Long tensors in [0, 255])
    and returns log-probabilities over *num_classes* classes.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Frozen benchmark = 75.
    embedding_dim : int
        Dimension of byte embeddings. Frozen = 16.
    gru_units : int
        Number of units per GRU direction. Frozen = 64.
    gru_layers : int
        Number of stacked BiGRU layers. Frozen = 2.
    kernel_sizes : list[int]
        Kernel sizes for the 4 parallel CNN branches. Frozen = [9,27,40,65].
    cnn_filters : int
        Number of filters per CNN branch. Frozen = 64.
    dense_units : list[int]
        Hidden sizes for the two dense layers. Frozen = [1024, 512].
    dropout : float
        Dropout rate applied after each dense layer.
    """

    def __init__(
        self,
        num_classes: int = _DEFAULT_NUM_CLASSES,
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
        gru_units: int = _DEFAULT_GRU_UNITS,
        gru_layers: int = _DEFAULT_GRU_LAYERS,
        kernel_sizes: list[int] = _DEFAULT_KERNEL_SIZES,
        cnn_filters: int = _DEFAULT_CNN_FILTERS,
        dense_units: list[int] = _DEFAULT_DENSE_UNITS,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.gru_units = gru_units

        # ---- Byte Embedding ----
        # Embedding table: 256 byte values → embedding_dim (16)
        self.embedding = nn.Embedding(
            num_embeddings=_VOCAB_SIZE,
            embedding_dim=embedding_dim,
            padding_idx=None,
        )

        # ---- Bidirectional GRU (stacked) ----
        # Input: [batch, seq_len, embedding_dim]
        # Output: [batch, seq_len, 2 * gru_units]
        # Hidden of the last layer: [2 * gru_layers, batch, gru_units]
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_units,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        gru_output_dim = 2 * gru_units  # 128

        # ---- Parallel CNN Branches ----
        # Input to CNN: sequence of embeddings, transposed to [B, emb_dim, L]
        self.cnn_branches = nn.ModuleList(
            [
                _CNNBranch(
                    in_channels=embedding_dim,
                    out_channels=cnn_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )
        cnn_output_dim = cnn_filters * len(kernel_sizes)  # 64 * 4 = 256

        # ---- Dense Head ----
        # Concatenated feature: gru_output_dim + cnn_output_dim = 384
        concat_dim = gru_output_dim + cnn_output_dim  # 384

        assert len(dense_units) == 2, "ByteRCNN requires exactly 2 dense layers."
        self.dense1 = nn.Linear(concat_dim, dense_units[0])     # 384 → 1024
        self.dense2 = nn.Linear(dense_units[0], dense_units[1]) # 1024 → 512
        self.output_layer = nn.Linear(dense_units[1], num_classes)  # 512 → 75

        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(dense_units[0])
        self.bn2 = nn.BatchNorm1d(dense_units[1])

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier/Glorot initialisation for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.LongTensor
            Shape [batch_size, 512].  Values in [0, 255].

        Returns
        -------
        torch.FloatTensor
            Shape [batch_size, num_classes].  Log-softmax probabilities.
            Use with nn.NLLLoss or convert to raw logits for CrossEntropyLoss.
        """
        # ---- Embedding: [B, 512] → [B, 512, 16] ----
        emb = self.embedding(x)  # [B, L, E]

        # ---- BiGRU ----
        # gru_out : [B, L, 2*H]
        # hidden  : [2*layers, B, H]  (last layer hidden states)
        gru_out, hidden = self.gru(emb)

        # Take the last time-step output from the top BiGRU layer:
        # hidden[-2] = forward last hidden, hidden[-1] = backward last hidden
        h_fwd = hidden[-2]  # [B, H]
        h_bwd = hidden[-1]  # [B, H]
        gru_feat = torch.cat([h_fwd, h_bwd], dim=1)  # [B, 2H = 128]

        # ---- Parallel CNN branches ----
        # CNN expects [B, C, L] → transpose embedding
        emb_t = emb.transpose(1, 2)  # [B, E, L]
        cnn_feats = [branch(emb_t) for branch in self.cnn_branches]
        cnn_feat = torch.cat(cnn_feats, dim=1)  # [B, 4*64 = 256]

        # ---- Concatenate ----
        feat = torch.cat([gru_feat, cnn_feat], dim=1)  # [B, 384]

        # ---- Dense layers ----
        feat = self.dropout(F.relu(self.bn1(self.dense1(feat))))  # [B, 1024]
        feat = self.dropout(F.relu(self.bn2(self.dense2(feat))))  # [B, 512]

        # ---- Output ----
        logits = self.output_layer(feat)           # [B, num_classes]
        return F.log_softmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (softmax, not log-softmax)."""
        return self.forward(x).exp()

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"embedding_dim={self.embedding_dim}, "
            f"gru_units={self.gru_units}"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_bytercnn(
    num_classes: int = _DEFAULT_NUM_CLASSES,
    embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
    gru_units: int = _DEFAULT_GRU_UNITS,
    gru_layers: int = _DEFAULT_GRU_LAYERS,
    kernel_sizes: list[int] | None = None,
    cnn_filters: int = _DEFAULT_CNN_FILTERS,
    dense_units: list[int] | None = None,
    dropout: float = 0.5,
) -> ByteRCNN:
    """Instantiate ByteRCNN with frozen benchmark defaults.

    Keyword arguments shadow the frozen defaults only if explicitly provided.
    In benchmark runs, always call ``build_bytercnn()`` with no arguments.
    """
    if kernel_sizes is None:
        kernel_sizes = list(_DEFAULT_KERNEL_SIZES)
    if dense_units is None:
        dense_units = list(_DEFAULT_DENSE_UNITS)

    return ByteRCNN(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        gru_units=gru_units,
        gru_layers=gru_layers,
        kernel_sizes=kernel_sizes,
        cnn_filters=cnn_filters,
        dense_units=dense_units,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    model = build_bytercnn()
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Dummy forward pass
    dummy_x = torch.randint(0, 256, (4, 512), dtype=torch.long)
    with torch.no_grad():
        out = model(dummy_x)
    print(f"\nForward pass OK — output shape: {out.shape}")
    assert out.shape == (4, 75), f"Expected (4, 75), got {out.shape}"
    print("Smoke-test PASSED.")
    sys.exit(0)
