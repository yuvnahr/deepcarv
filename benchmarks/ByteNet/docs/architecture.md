# ByteNet Architecture

## Overview

ByteNet is a dual-branch multimedia file fragment classifier from:

> **ByteNet: Rethinking Multimedia File Fragment Classification through Visual Perspectives**
> Wenyang Liu, Kejun Wu, Tianyi Liu, Yi Wang, Kim-Hui Yap, Lap-Pui Chau — 2023.

The core insight is that traditional 1-D byte-sequence classifiers only capture **interbyte** relationships and ignore **intrabyte** information (bit-level patterns within each byte that cross byte boundaries). ByteNet exposes this information through a visual representation (**Byte2Image**) and exploits it with a dual-branch network.

---

## 1. Byte2Image Representation

### 1.1 Intrabyte Exposure (bit-shifting)

Given a raw byte sector `x ∈ Z^Ns` (values 0–255), Byte2Image applies a sliding byte window with stride = 1 bit. This is implemented by bit-shifting the sequence 7 times:

```
x0 = x
xi = xi-1 << 1  (& 0xFF to keep 8-bit)     i = 1…7
```

This yields 8 byte sequences stacked into a byte matrix `xin ∈ Z^(Ns × 8)`.

Each row of this matrix represents an interbyte sequence at a specific bit offset; each column represents the intrabyte view of the original byte sequence.

### 1.2 Intrabyte N-grams

To increase the aspect ratio of the resulting image (needed by standard CNNs/Transformers which expect near-square inputs), intrabyte n-grams are used:

```
H = Ns - n + 1
W = 8 * n

xngram ∈ Z^(H × W)   [each element: byte value 0-255]
```

For the default `n = 16`, `Ns = 512`:
- H = 497
- W = 128
- Image shape: `[1, 497, 128]`

Adjacent bytes in the n-gram still differ by a 1-bit shift, preserving relative intrabyte order.

### 1.3 Normalisation

The byte matrix is cast to float32, divided by 255.0, then standardised with `μ = 0.5, σ = 0.5` (standard grayscale approximation).

### 1.4 4096-Byte Mode

For 4096-byte sectors, the paper segments the sector into **8 × 512-byte chunks**. Each chunk is independently converted to a grayscale image. The 8 images are concatenated along the channel dimension, producing an 8-channel `[8, 497, 128]` input to the image branch.

*Design note*: The paper states this strategy preserves the same processing steps as the 512-byte mode and avoids the large image (4081 × 128) that direct conversion would produce.

### 1.5 Image Augmentation (training only)

| Technique | Probability | Purpose |
|---|---|---|
| Normalisation | 1.0 | Standardise pixel values |
| Horizontal flip | 0.5 | Data diversity |
| Random erase | 0.5 | Prevent memorisation of repeated n-gram patterns |
| CutMix | 1.0 | Hard augmentation, explicitly mixes samples |
| Mixup | 0.8 | Soft augmentation, linearly interpolates samples |

CutMix and Mixup produce **soft labels**, requiring soft-label NLL loss during training.

---

## 2. ByteNet: Dual-Branch Network

### 2.1 Architecture Overview

```
Input x ∈ Z^Ns
     ├── Byte Branch (BBFE)
     │   └── FC(Ns → F0)  ──────────────────┐
     │                                       │
     └── Image Branch (IBFE)                 │
         ├── Byte2Image                      │
         ├── Embedding Layer                 │
         └── 4-Stage Hierarchical RVFE ──────┤
                                             ▼
                                  concat([xsf, xdf])
                                             │
                                         FC → log_softmax
```

### 2.2 Byte Branch Feature Extraction (BBFE)

A **single fully-connected layer** `FC: Ns → F0 (F0=512)` applied to normalised raw bytes. This branch memorises the co-occurrence of specific bytes ("magic bytes") that serve as strong classification signals for certain file types.

### 2.3 Image Branch Feature Extraction (IBFE)

A 4-stage hierarchical network. Each stage contains:
- An RVFE module (L_i feature extraction blocks)
- A downsampling operation (3×3 conv stride-2, except the last stage uses GAP)

The embedding layer precedes stage 1 and differs between the two variants.

### 2.4 Feature Fusion

```
xfused = concat(xsf, xdf)   [F0 + C4 dimensions]
output = log_softmax(FC(xfused))
```

---

## 3. ByteResNet Variant

### Embedding: N-gram Embedding Layer

A **wide convolution** (kernel width = W = 128) projects each row of the n-gram image into a dense embedding:

```
Conv2d(1, K=96, kernel_size=(1, W)) → [B, 96, H, 1]
Conv2d(96, 64, kernel_size=(7, 1)) + BN + ReLU → [B, 64, H, 1]
MaxPool2d((3,1), stride=(2,1)) → [B, 64, H//2, 1]
```

### Feature Extraction Block: ResNet Block

Standard two-layer residual block with BN and ReLU:
```
x → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → (+shortcut) → ReLU
```

### Architecture Parameters

| Stage | Blocks | Channels |
|---|---|---|
| 1 | 2 | 64 |
| 2 | 2 | 128 |
| 3 | 2 | 256 |
| 4 (GAP) | 2 | 512 |

| Param | Value |
|---|---|
| N-gram embed dim C1 | 96 |
| BBFE output dim F0 | 512 |
| Total fusion dim | 1024 |

---

## 4. ByteFormer Variant

### Embedding: Patch Embedding Layer

Standard ViT-style patch embedding:
```
Conv2d(in_channels, C1, kernel_size=P, stride=P) → [B, C1, H//P, W//P]
+ learnable positional embedding
+ LayerNorm
```

For 512-byte sectors: `C1 = 64`, `P = 8`
For 4096-byte sectors: `C1 = 96`, `P = 8`

### Feature Extraction Block: PoolFormer Block

From MetaFormer (Yu et al., 2022 [46]). Replaces self-attention with average pooling:

```
Sub-block 1 (Token Mixer):
  x → LayerNorm → AvgPool(3×3, stride=1) → (+x)

Sub-block 2 (Channel MLP):
  x → LayerNorm → Conv1×1(C → 4C) → GELU → Conv1×1(4C → C) → (+x)
```

### Architecture Parameters

| Stage | Blocks | Channels |
|---|---|---|
| 1 | 6 | 64 |
| 2 | 6 | 128 |
| 3 | 18 | 320 |
| 4 (GAP) | 6 | 512 |

| Param | Value (512B) | Value (4096B) |
|---|---|---|
| Patch embed dim C1 | 64 | 96 |
| Patch size P | 8 | 8 |
| BBFE output dim F0 | 512 | 512 |
| Total fusion dim | 1024 | 1024 |

---

## 5. Training Details

| Setting | Value | Source |
|---|---|---|
| Optimizer | AdamW | §IV-B |
| Betas | (0.9, 0.999) | §IV-B |
| Weight decay | 0.01 | §IV-B |
| Batch size | 512 (FFT-75) | §IV-B |
| Total epochs | 50 | §IV-B |
| LR warmup | 5e-7 → 5e-4 linear, 2 epochs | §IV-B |
| LR decay | cosine → 0, remaining 48 epochs | §IV-B |
| n-gram n | 16 | §IV-B |
| Loss | NLL (soft-label for CutMix/Mixup) | §III-E |

---

## 6. Expected Performance (Paper §IV-D, Table III)

| Model | FFT-75 S1 (512B) | FFT-75 S1 (4096B) |
|---|---|---|
| ByteResNet | 71.0% | 82.1% |
| ByteFormer | 73.2% | 81.9% |
| FiFTy (baseline) | 65.6% | 77.5% |

---

## 7. Implementation Notes & Deviations

### 7.1 N-gram Embedding Interpretation
The paper describes K separate 1×W convolutions. Our implementation uses a single `nn.Conv2d(1, embed_dim, (1, W))` which is mathematically equivalent (K independent filters of size 1×W applied to a 1-channel image).

### 7.2 4096-Byte Channel Handling
When `fragment_size=4096`, the 8 stacked images form an 8-channel input. The NGramEmbedding averages over channels before applying the wide conv (since the wide conv is designed for 1-channel input). The PatchEmbedding in ByteFormer uses `in_channels=8` directly (the patch embedding Conv2d handles it naturally).

### 7.3 Image Normalisation
The paper uses image mean/standard deviation. We use the approximation `μ=0.5, σ=0.5`. If exact per-dataset statistics are needed, they can be precomputed from the training set and passed as config parameters.

### 7.4 Augmentation Implementation
CutMix and Mixup are applied at the **image level** (after Byte2Image conversion), not at the raw byte level. This is consistent with the paper's intent (§III-B, Fig. 4d). The byte branch (BBFE) always receives unaugmented raw bytes.

### 7.5 Scheduler
The generic `Trainer` does not support warmup+cosine. The ByteNet standalone `scripts/train.py` implements this using `torch.optim.lr_scheduler.SequentialLR` with `LinearLR` + `CosineAnnealingLR`.

### 7.6 No Architectural Simplifications
No significant simplifications were made. Both ByteResNet and ByteFormer are implemented as described in the paper, with distinct embedding layers and RVFE blocks. The distinction is preserved in code (not collapsed into a generic implementation).
