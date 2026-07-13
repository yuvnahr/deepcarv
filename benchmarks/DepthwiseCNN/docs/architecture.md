# DepthwiseCNN Architecture

Reproduced from:
> "File Fragment Type Classification Using Light-Weight Convolutional Neural Networks"

## Variants

Three model variants are implemented:

| Variant | Description |
|---------|-------------|
| `dsc`   | Depthwise Separable CNN (baseline) |
| `dsc_se`| DSC + Squeeze-and-Excitation channel attention |
| `m_dsc` | Multi-scale DSC with Inception-style parallel branches (kernels 3, 7, 11) |

## Architecture (§III)

```
Input: [B, L]  (raw byte indices 0-255)
  ↓
Embedding(256, embed_dim)  →  [B, L, embed_dim]
  ↓
Permute  →  [B, embed_dim, L]  (channels-first for Conv1d)
  ↓
Block 0  (DSC / DSC-SE / M-DSC)  →  MaxPool1d(2)
Block 1  (DSC / DSC-SE / M-DSC)
Block 2  (DSC / DSC-SE / M-DSC)  →  MaxPool1d(2)
Block 3  (DSC / DSC-SE / M-DSC)
  ↓
AdaptiveAvgPool1d(1)  →  [B, final_channels]
Dropout
Linear(final_channels, num_classes)
log_softmax
  ↓
Output: [B, num_classes]  (log-probabilities)
```

## DSC Block

```
depthwise_conv(kernel_size, groups=in_ch) → BatchNorm → Hardswish
pointwise_conv(1×1)                       → GroupNorm
+ residual projection (1×1 if in_ch != out_ch)
```

## DSC-SE Block

```
DSC Block → SEBlock(channels, reduction=16)
```

SEBlock: `GAP → FC(r) → ReLU → FC(C) → Sigmoid → scale`

## M-DSC Block

```
Three parallel DSC branches (kernels 3, 7, 11)
→ Concatenate along channel axis
→ 1×1 projection to out_channels
+ residual projection
```

## Default Hyperparameters

| Parameter     | Value       | Source |
|---------------|-------------|--------|
| embed_dim     | 64          | ASSUMPTION (paper implicit) |
| channels      | [64,128,256,256] | ASSUMPTION (paper figure) |
| kernel_size   | 3           | ASSUMPTION (standard) |
| se_reduction  | 16          | ASSUMPTION (standard SE) |
| p_dropout     | 0.5         | ASSUMPTION (matches ByteRCNN) |
| MaxPool       | after blocks 0, 2 | ASSUMPTION (2× downsampling) |

## Documented Assumptions

All implementation assumptions (where the paper is ambiguous) are documented
in `benchmarks/DepthwiseCNN/src/model.py` docstrings:

1. **embed_dim = 64**: Paper shows an embedding layer without specifying width.
2. **num_blocks = 4**: Paper does not give an explicit block count.
3. **channels = [64,128,256,256]**: Derived from paper figure.
4. **GroupNorm groups = channels // 16**: Standard GroupNorm sizing.
5. **M-DSC branch widths**: out_channels // 3 per branch (remainder to branch 0).
6. **SE reduction = 16**: Standard SE block default.
7. **Dropout = 0.5**: Matches ByteRCNN baseline convention.
8. **MaxPool after blocks 0 and 2**: 2× downsampling total.

## Integration

The model integrates with the DeepCarv framework via:

- **Registry key**: `depthwisecnn`
- **Benchmark adapter**: `benchmarks/DepthwiseCNN/src/adapter.py`
- **Framework adapter**: `src/models/adapters/depthwisecnn_adapter.py`
- **Model config**: `configs/models/depthwisecnn.yaml`
- **Training config**: `configs/training/depthwisecnn.yaml`
- **Experiment configs**: `configs/experiments/depthwisecnn_fft75_{512,4096}.yaml`

## Running the Benchmark

```bash
# Via the generic runner (recommended):
python -m src.core.runner --experiment depthwisecnn_fft75_512

# Via the benchmark script:
python benchmarks/DepthwiseCNN/scripts/train.py \
    --config configs/experiments/depthwisecnn_fft75_512.yaml

# Sanity run (2 epochs, 2000 samples):
python benchmarks/DepthwiseCNN/scripts/train.py \
    --data_dir data/FFT-75 --sanity

# DSC-SE variant:
python -m src.core.runner --experiment depthwisecnn_fft75_512 \
    --override model.variant=dsc_se

# M-DSC variant:
python -m src.core.runner --experiment depthwisecnn_fft75_512 \
    --override model.variant=m_dsc
```
