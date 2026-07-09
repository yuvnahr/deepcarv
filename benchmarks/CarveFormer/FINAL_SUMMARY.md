# CarveFormer — Final Integration Summary (Phase 5)

Branch: **`eval-CarveFormer`**. Status: **implemented, integrated, and
end-to-end verified** on synthetic FFT-75 data. Full-scale training on the real
FFT-75 dataset is the only remaining step (needs a GPU + the dataset).

## Files changed (additive; 15 files)

All changes are confined to `benchmarks/CarveFormer/` plus a single registry
hook. The shared framework (trainer, evaluator, dataset) and ByteRCNN are
provably untouched.

| File | What |
|---|---|
| `src/model.py` | CarveFormer model: SwinV2-Tiny + 96-d byte embedding + reshape; `build_carveformer` factory; returns `log_softmax`. |
| `src/adapter.py` | `CarveFormerAdapter(FragmentClassifier)`; config-driven. |
| `src/__init__.py` | Fixed invalid scaffold docstring so the package imports. |
| `../../src/models/adapters/carveformer_adapter.py` | Registry hook: loads the real adapter, graceful stub fallback. Key `carveformer` unchanged. |
| `configs/benchmark.yaml` | One config; paper hyperparameters; 512/4096 switch. |
| `scripts/train.py` | Thin wrapper over shared `Trainer` + `Evaluator`. |
| `scripts/evaluate.py` | Thin wrapper over shared `Evaluator`. |
| `notebooks/carveformer_fft75_kaggle.ipynb` | End-to-end Kaggle runner. |
| `tests/smoke_test.py` | 7 offline tests (model, adapter, registry, config, grids). |
| `docs/architecture.md` | Architecture, mapping, assumptions, deviations. |
| `README.md`, `model.yaml`, `paper/*.md` | Docs, metadata, checklist, notes, Phase 0 recon. |

## Architecture implemented

Swin Transformer V2 Tiny (timm `swinv2_tiny_window8_256`, ImageNet1k-pretrained)
with the image conv/norm patch-embed replaced by `Embedding(256, 96)` +
LayerNorm, the embedded byte sequence reshaped into the `[B, H, W, 96]` token
grid the Swin stages expect (4096→64×64 native; 512→16×32 documented default).
Classification head → `log_softmax`. ~27.6M parameters (matches paper's ~28M).

## Verified

- `build_model('carveformer', num_classes=75, fragment_size=512)` builds a real
  27.6M-param model (previously raised `NotImplementedError`).
- Forward works for 512 (16×32) and 4096 (64×64); output shape == num_classes;
  valid log-probabilities; backward produces gradients.
- End-to-end run on synthetic FFT-75: config → `FragmentDataset` → registry →
  shared `Trainer` (checkpointing, cosine LR, best-ckpt selection) → shared
  `Evaluator` → full standardized output set.
- ByteRCNN still builds and runs; ByteRCNN files and the shared trainer/
  evaluator/dataset are unmodified (empty git diff).
- 7/7 smoke tests pass; ruff + mypy clean on all source files.

## How to run local training

```bash
pip install "timm>=1.0.0"
python -m benchmarks.CarveFormer.scripts.train \
    --config benchmarks/CarveFormer/configs/benchmark.yaml \
    --data-dir data/FFT-75                 # add --fragment-size 4096 to switch
```

## How to run Kaggle training

Open `notebooks/carveformer_fft75_kaggle.ipynb`, attach the FFT-75 dataset, set
`FRAGMENT_SIZE`, run top-to-bottom (install → verify NPZ → sanity-train →
full-train → evaluate → save outputs).

## Deviations from the paper (documented, not silent)

1. **512-byte reshape grid 16×32** — paper unspecified; isolated & overridable.
2. **Effective batch 1024** — shared `Trainer` has no gradient accumulation, so
   `batch_size` is the real batch; the paper's effective 1024 needs a
   large-memory GPU or a future trainer feature. `grad_accum` documents intent.
3. **Scheduler** — cosine used as the closest built-in to the paper's
   warmup+decay.
4. **Pretrained fallback** — random init if ImageNet1k weights can't be fetched.

## Known limitations / next steps

- Full FFT-75 training run pending (GPU + real dataset). Sanity targets:
  Scenario #1 ≈ 72% @512 / 83% @4096.
- **Framework bug to fix separately (not CarveFormer):** `src/data/dataset.py`
  validates NPZ keys against lowercase `{'x','y'}` and reads `data['x']`, but
  the documented format is uppercase `X`/`y`. Real uppercase-`X` FFT-75 files
  will not load until this is fixed. The Kaggle notebook's verify cell accepts
  either case defensively.
- If reaching the paper's effective batch 1024 matters, add gradient
  accumulation to the shared `Trainer` (a framework enhancement, not a
  CarveFormer change).

## Merge readiness

Additive and compatible: no ByteRCNN changes, no split generation, no CSV
pipeline, no shared-framework behavior changes. Ready to merge into `dev` once
a full FFT-75 run confirms accuracy lands in the paper's range.
