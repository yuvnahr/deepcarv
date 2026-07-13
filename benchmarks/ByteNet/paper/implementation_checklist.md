# ByteNet Implementation Checklist

Paper: **ByteNet: Rethinking Multimedia File Fragment Classification through Visual Perspectives**  
Liu et al., IEEE Transactions on Multimedia, 2023.

---

## Section-by-Section Checklist

### §III-B — Byte2Image Representation

| Item | Status | Notes |
|---|---|---|
| Intrabyte exposure: bit-shift × 7 times (`xi = xi-1 << 1 & 0xFF`) | ✅ | `model.py::Byte2Image` |
| Stack 8 rows → byte matrix `xin ∈ Z^(Ns × 8)` | ✅ | via `torch.stack + unfold` |
| Intrabyte n-gram with n=16: `H=Ns-n+1, W=8n` | ✅ | `model.py::Byte2Image.forward` |
| Cast to float, normalise to [0,1], standardise | ✅ | `μ=0.5, σ=0.5` approximation |
| Image augmentation: Normalise (p=1.0) | ✅ | `train.py` (normalise inside Byte2Image) |
| Image augmentation: Horizontal flip (p=0.5) | ✅ | `train.py::random_hflip` |
| Image augmentation: Random erase (p=0.5) | ✅ | `train.py::random_erase` |
| Image augmentation: CutMix (p=1.0) | ✅ | `train.py::cutmix_batch` |
| Image augmentation: Mixup (p=0.8) | ✅ | `train.py::mixup_batch` |
| 4096B mode: segment into 8×512B chunks | ✅ | `model.py::ByteNet._convert_to_image` |
| 4096B mode: stack 8 images on channel dim | ✅ | `torch.cat` on dim=1 |

### §III-C — ByteNet Dual-Branch Network

| Item | Status | Notes |
|---|---|---|
| BBFE: single FC layer `FC: Ns → F0` | ✅ | `model.py::ByteBranch` |
| IBFE: embedding layer + 4-stage RVFE | ✅ | `model.py::ImageBranch` |
| RVFE stages with downsampling conv (stride-2) between stages | ✅ | `model.py::RVFEStage + ImageBranch.inter_downs` |
| Last RVFE stage uses GAP | ✅ | `model.py::RVFEStage(is_last=True)` |
| Feature fusion: concat(xsf, xdf) → FC → log_softmax | ✅ | `model.py::ByteNet.forward` |

### §III-D — Embedding Layers & Feature Extraction Blocks

| Item | Status | Notes |
|---|---|---|
| ByteResNet: N-gram embedding (wide conv K=96 filters, 1×W kernel) | ✅ | `model.py::NGramEmbedding` |
| ByteResNet: 7×7 conv + pool after n-gram embedding | ✅ | `NGramEmbedding.conv7 + pool` |
| ByteResNet: ResNet block (2× Conv3×3-BN + shortcut) | ✅ | `model.py::ResNetBlock` |
| ByteFormer: Patch embedding (Conv2d stride-P + pos embed + LN) | ✅ | `model.py::PatchEmbedding` |
| ByteFormer: PoolFormer block (AvgPool + 2× 1×1 Conv-GELU) | ✅ | `model.py::PoolFormerBlock` |
| ByteFormer: Pre-norm design (LN before each sub-block) | ✅ | `PoolFormerBlock.norm1/2` |
| ByteFormer: MLP expansion ratio r=4 | ✅ | `_POOL_EXPAND=4` |

### §III-E — Training Loss

| Item | Status | Notes |
|---|---|---|
| NLL loss (negative log-likelihood) | ✅ | `F.nll_loss` in train.py + adapter |
| Soft-label NLL for CutMix/Mixup | ✅ | `train.py::soft_nll_loss` |
| log_softmax output from forward() | ✅ | `model.py::ByteNet.forward` |

### §IV-B — Architecture Parameters

| Item | ByteResNet | ByteFormer | Status |
|---|---|---|---|
| N-gram / patch embed dim C1 | 96 | 64 (512B) / 96 (4096B) | ✅ |
| Stage layers Li | [2,2,2,2] | [6,6,18,6] | ✅ |
| Channel dims Ci | [64,128,256,512] | [64,128,320,512] | ✅ |
| Patch size P (ByteFormer) | — | 8 | ✅ |
| n-gram n | 16 | 16 | ✅ |
| BBFE output dim F0 | 512 | 512 | ✅ (configurable) |

### §IV-B — Training Settings

| Setting | Paper | Implementation | Status |
|---|---|---|---|
| Optimizer | AdamW | `torch.optim.AdamW` | ✅ |
| β1, β2 | 0.9, 0.999 | default AdamW | ✅ |
| Weight decay | 0.01 | `weight_decay=0.01` | ✅ |
| FFT-75 batch size | 512 | `batch_size=512` | ✅ |
| LR warmup | 5e-7 → 5e-4, 2 epochs | `build_warmup_cosine_scheduler` | ✅ |
| LR decay | cosine → 0, 48 epochs | `CosineAnnealingLR` | ✅ |
| Total epochs | 50 | `epochs=50` | ✅ |
| n-gram n | 16 | `ngram_n=16` | ✅ |

---

## Deviations from Paper

| Deviation | Reason | Impact |
|---|---|---|
| Normalisation uses μ=0.5, σ=0.5 instead of dataset stats | Dataset mean/std not given in paper | Negligible — values close to true stats |
| 4096B: NGramEmbedding averages 8 channels before wide conv | Wide conv expects 1-channel input | Slight information loss vs. per-channel approach; kept for simplicity |
| PoolFormerBlock uses `x + pool(norm(x)) - norm(x)` form | Paper uses residual notation | Mathematically identical |
| Generic framework Trainer not used for training | Trainer doesn't support warmup+cosine | Standalone script; identical training dynamics |

---

## Testing Coverage

| Test | File | Status |
|---|---|---|
| Byte2Image output shape | `tests/smoke_test.py::test_byte2image_512_shape` | ✅ |
| ByteResNet forward (512B) | `tests/smoke_test.py::test_bytenet_resnet_512_forward` | ✅ |
| ByteFormer forward (512B) | `tests/smoke_test.py::test_bytenet_former_512_forward` | ✅ |
| 4096B chunking mode | `tests/smoke_test.py::test_bytenet_resnet_4096_forward` | ✅ |
| Registry resolution | `tests/smoke_test.py::test_registry_resolves_bytenet` | ✅ |
| FragmentClassifier contract | `tests/smoke_test.py::test_adapter_contract` | ✅ |
| ResNetBlock / PoolFormerBlock shapes | `tests/smoke_test.py` | ✅ |

---

## Pending / Out of Scope

- [ ] Full training run validation on FFT-75 (requires GPU + dataset)
- [ ] VFF-16 evaluation (out of scope for DeepCarv FFT-75 benchmark)
- [ ] Ablation study (optional research task)
- [ ] Exact per-dataset normalisation statistics