# DepthwiseCNN — Architecture Reference

> Implementation of: **File Fragment Type Classification Using Light-Weight Convolutional Neural Networks**
>
> Branch: `eval-depthwiseCNN`
> Status: Phase 4 complete — all tests pass, training/evaluation scripts verified.

---

## 1. Input representation

Raw byte sequences are treated as integer tokens.  No normalisation or byte-level encoding is applied before the model.

| Stage | Shape | Notes |
|-------|-------|-------|
| Raw input | `[B, L]` int64 | Values in `[0, 255]`; `L` ∈ {512, 4 096} |
| After `Embedding(256, 32)` | `[B, L, 32]` | Learnable 8 192-param lookup table |
| After transpose | `[B, 32, L]` | Channels-first for Conv1d |

The embedding is part of the model's `forward` pass — no preprocessing is required outside the model.

---

## 2. Architecture overview

All three variants share the same macro-structure:

```
Input [B, L] int64
  │
  ├─ Embedding(256, 32)              →  [B, 32, L]
  ├─ FirstConv + Norm + Act          →  [B, 32, L/2]     (stride=2)
  │
  ├─ InceptionBlock 1  (pool)        →  [B, 64,  L/8]    (÷4)
  │    └─ [DSC-SE] SEBlock(64)
  │
  ├─ InceptionBlock 2  (no pool)     →  [B, 64,  L/8]
  │    └─ [DSC-SE] SEBlock(64)
  │
  ├─ InceptionBlock 3  (pool)        →  [B, 128, L/32]   (÷4)
  │    └─ [DSC-SE] SEBlock(128)
  │
  ├─ GlobalAveragePool               →  [B, 128, 1]
  ├─ [M-DSC] Dropout(p=0.2)
  ├─ Conv1d(128 → num_classes, k=1)  →  [B, num_classes, 1]
  └─ log_softmax                     →  [B, num_classes]
```

**Concrete shape trace (B=1, L=512, num_classes=75, variant=DSC):**

| Layer | Output shape |
|-------|-------------|
| Embedding | `(1, 512, 32)` → transposed → `(1, 32, 512)` |
| FirstConv | `(1, 32, 256)` |
| InceptionBlock 1 (pool) | `(1, 64, 64)` |
| InceptionBlock 2 (no pool) | `(1, 64, 64)` |
| InceptionBlock 3 (pool) | `(1, 128, 16)` |
| GlobalAveragePool | `(1, 128, 1)` |
| Classifier Conv1d | `(1, 75, 1)` |
| log\_softmax + squeeze | `(1, 75)` |

---

## 3. Building blocks

### 3.1 `SeparableConv1d`

Standard depthwise-separable 1-D convolution: a **depthwise** Conv1d (one filter per channel, `groups=C`) followed immediately by a **pointwise** Conv1d (1×1, mixes channels).  Neither layer uses a bias because a normalisation layer follows in all call sites.

```
input [B, C_in, L]
  → Depthwise Conv1d(C_in, C_in, k=k, groups=C_in)  →  [B, C_in, L]
  → Pointwise Conv1d(C_in, C_out, k=1)               →  [B, C_out, L]
```

### 3.2 `SEBlock` (DSC-SE only)

Squeeze-and-Excitation over the channel dimension:

```
input [B, C, L]
  → GlobalAveragePool                 →  [B, C]
  → Linear(C, C//4)  → ReLU          →  [B, C//4]
  → Linear(C//4, C)  → Sigmoid       →  [B, C]
  → unsqueeze(-1)                     →  [B, C, 1]
  → multiply with input               →  [B, C, L]   (broadcast)
```

Reduction ratio = 4 (standard MobileNetV3 value — paper does not specify).

### 3.3 `InceptionBlock`

Three parallel `SeparableConv1d` branches with kernel sizes **11**, **19**, **27** (all "same"-padded), each followed by its own normalisation layer.  The three normalised outputs are **element-wise summed** (not concatenated), then optionally max-pooled.  A residual shortcut is added before the activation.

```
input [B, C_in, L]
  ├─ SepConv(k=11) → Norm  →  b11
  ├─ SepConv(k=19) → Norm  →  b19
  └─ SepConv(k=27) → Norm  →  b27
          ↓
  out = b11 + b19 + b27              (element-wise sum)
  out = MaxPool1d(4, 4)  [if pool]
  shortcut = project_if_needed(input)
  shortcut = MaxPool1d(4, 4)  [if pool]
  return Activation(out + shortcut)
```

The three InceptionBlocks in the network have the following channel/pool config:

| Block | C_in → C_out | pool | Spatial change |
|-------|-------------|------|----------------|
| Block 1 | 32 → 64 | ✓ | ÷4 |
| Block 2 | 64 → 64 | ✗ | none |
| Block 3 | 64 → 128 | ✓ | ÷4 |

---

## 4. Variant differences

| Property | DSC | DSC-SE | M-DSC |
|----------|-----|--------|-------|
| **First conv** | Standard Conv1d | Standard Conv1d | Depthwise Conv1d (groups=32) |
| **Normalisation** | BatchNorm1d | BatchNorm1d | GroupNorm(8 groups) |
| **Activation** | Hardswish | Hardswish | ReLU |
| **SE blocks** | — | After each InceptionBlock | — |
| **Head dropout** | — | — | Dropout(p=0.2) before classifier |

---

## 5. Parameter counts (num\_classes=75)

| Variant | Total | Δ vs DSC |
|---------|------:|-------:|
| **DSC** | **101,291** | baseline |
| **DSC-SE** | **113,579** | +12,288 (SE blocks) |
| **M-DSC** | **82,443** | −18,848 (depthwise first conv) |

**DSC layer breakdown:**

| Layer | Parameters |
|-------|----------:|
| Embedding(256, 32) | 8,192 |
| FirstConv + BatchNorm1d | 19,520 |
| InceptionBlock 1 | 10,400 |
| InceptionBlock 2 | 16,320 |
| InceptionBlock 3 | 37,184 |
| Classifier Conv1d(128→75) | 9,675 |
| **Total** | **101,291** |

**DSC-SE SE-block breakdown (additional parameters):**

| SE Block | Parameters |
|----------|----------:|
| SEBlock(64,  reduction=4) | 2,048 |
| SEBlock(64,  reduction=4) | 2,048 |
| SEBlock(128, reduction=4) | 8,192 |
| **SE total** | **12,288** |

---

## 6. Paper ambiguities and implementation decisions

The following decisions were made where the paper was ambiguous.  Each is annotated in `model.py` at the relevant line.

| # | Ambiguity | Decision |
|---|-----------|----------|
| 1 | **Shortcut when channels change** — Figure 4 shows a residual path but gives no implementation detail. | 1×1 Conv1d (no bias, no norm) on the shortcut when `in_channels ≠ out_channels`, matching ResNet practice. |
| 2 | **MaxPool on shortcut** — The shortcut must match the pooled branch spatial dimension. | The same `MaxPool1d(4, 4)` is applied to the shortcut when `pool=True`. |
| 3 | **SE reduction ratio** — Not stated in the paper. | Reduction = 4 (MobileNetV3 default). |
| 4 | **M-DSC first conv pointwise step** — The paper says the first conv is depthwise; it does not mention a following pointwise step. | No pointwise step is added — the first conv is `groups=32` only (purely depthwise). |
| 5 | **GroupNorm group count** — Not stated. Must divide all channel widths (32, 64, 128). | `num_groups=8` (divides all widths evenly). |
| 6 | **Dropout placement (M-DSC)** — The paper places dropout before the final classifier. | Applied after global average pooling and before the 1×1 Conv1d, matching MobileNetV3's head structure. |

---

## 7. Framework integration

The model is wired into the shared DeepCarv framework via:

| Component | Location |
|-----------|----------|
| Raw model | `benchmarks/DepthwiseCNN/src/model.py` |
| Adapter (`FragmentClassifier`) | `benchmarks/DepthwiseCNN/src/adapter.py` |
| Registry hook | `src/models/registry.py` → `"depthwisecnn"` |
| Config | `benchmarks/DepthwiseCNN/configs/benchmark.yaml` |
| Training script | `benchmarks/DepthwiseCNN/scripts/train.py` |
| Evaluation script | `benchmarks/DepthwiseCNN/scripts/evaluate.py` |

**Framework output contract:**

- `forward(x)` → `log_softmax` probabilities `[B, num_classes]` (compatible with `nn.NLLLoss`)
- `loss(log_probs, y)` → scalar NLLLoss (inherited default, no override needed)
- `predict(x)` → argmax class indices `[B]` (inherited default)
- `num_classes` attribute set by base class
- `name` attribute: `"depthwisecnn_<variant>"`
- `state_dict` / `load_state_dict` — standard `nn.Module` (used by checkpointing)

---

## 8. Known limitations

- The paper reports ~79 % accuracy; this implementation has not yet been trained to verify the number.
- FLOPs profiling has not yet been performed to confirm the paper's 164 M FLOPs claim.
- The M-DSC depthwise-only first conv (no pointwise) may underfit relative to the paper if the authors actually used a full depthwise-separable first layer.  This assumption is annotated in `model.py` (decision #4) and can be changed with a one-line edit.
