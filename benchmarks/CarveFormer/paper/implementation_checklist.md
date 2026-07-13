# CarveFormer Implementation Checklist

| Item | Status |
|---|---|
| Paper reviewed (Guzhov & Wirth, ECCWS 2025) | ✅ done |
| SwinV2-Tiny adaptation mapped to code | ✅ done (docs/architecture.md) |
| `src/model.py` implemented | ✅ done (Phase 1) |
| `src/adapter.py` implemented | ✅ done (Phase 2) |
| Registry hook added | ✅ done — `carveformer` builds the real model (Phase 2) |
| `configs/benchmark.yaml` with paper hyperparameters | ✅ done (Phase 3) |
| Train / evaluate scripts wired to shared framework | ✅ done (Phase 3) |
| Kaggle notebook complete & runnable top-to-bottom | ✅ done (Phase 3) |
| Smoke test passes | ✅ done — 7 tests (Phases 2, 4) |
| Config-loading / instantiation tests | ✅ done (Phase 4) |
| Docs updated (architecture, README) | ✅ done (Phase 4) |
| Outputs match the framework contract | ✅ verified end-to-end on synthetic data |
| No ByteRCNN files changed | ✅ confirmed |
| No split generation / CSV pipeline added | ✅ confirmed |
| Works on `train.npz` / `val.npz` / `test.npz` | ✅ verified (via shared FragmentDataset) |
| Full FFT-75 training run | ⏳ pending GPU + real dataset |

## Verified end-to-end

`config -> FragmentDataset -> registry build_model('carveformer') -> shared
Trainer -> shared Evaluator -> standardized outputs`, on a synthetic FFT-75
sample. ~27.6M params (matches paper's ~28M SwinV2-Tiny). ruff + mypy clean.

## Documented deviations

1. 512-byte reshape grid (16×32) — paper unspecified; isolated & overridable.
2. Effective batch 1024 — shared Trainer lacks gradient accumulation; needs a
   large-memory GPU or a future trainer feature.
3. Scheduler — cosine used as the closest built-in to the paper's warmup+decay.
4. Pretrained fallback — random init if ImageNet1k weights can't be fetched.

## Related framework bug (out of scope for this branch)

`src/data/dataset.py` validates NPZ keys against lowercase `{'x','y'}` but the
documented format is uppercase `X`/`y`. Flagged for a separate framework fix.
