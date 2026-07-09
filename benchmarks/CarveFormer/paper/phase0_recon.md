# CarveFormer — Phase 0 Recon & Design Lock

Status: **Phase 0 complete — no source code changed.** This document is the
implementation-oriented output of Phase 0 (repo recon + design lock). It is the
contract Phases 1–5 build against.

Target paper: *Transformer-Based File Fragment Type Classification for File
Carving in Digital Forensics* (Guzhov & Wirth, DFKI, ECCWS 2025). CarveFormer =
Swin Transformer V2 Tiny with its image front-end replaced by a 96-d byte
embedding, reshaped to a pseudo-2D tensor.

---

## A. Framework interfaces confirmed (from the DeepCarv master framework)

### Model registry — `src/models/registry.py`
- `MODEL_REGISTRY: dict[str, ModelFactory]` maps a name to a factory.
- `"carveformer"` already resolves — currently to `build_carveformer_adapter`,
  a `NotAvailableModel` stub that raises `NotImplementedError`. Phase 2 replaces
  the stub with the real adapter; the registry **key and factory signature stay
  identical** so existing configs keep working.
- Factory signature is fixed: `build_x_adapter(num_classes: int, **kwargs) -> FragmentClassifier`.

### Model contract — `src/core/interfaces.py::FragmentClassifier`
- Adapters subclass `FragmentClassifier(nn.Module, ABC)`.
- The trainer/evaluator call **only**: `forward(x) -> log-probabilities
  [B, num_classes]`, `loss(log_probs, y) -> scalar`, `.num_classes`, `.name`.
- The default `loss()` is `nll_loss` over log-probs, so **`forward()` must return
  `log_softmax`, not raw logits** (matching the existing ByteRCNN adapter), unless
  the adapter also overrides `loss()`.

### Trainer — `src/training/trainer.py`
- `TrainerConfig` fields: `epochs, lr, weight_decay, optimizer, scheduler,
  scheduler_factor, scheduler_patience, grad_clip, patience, monitor,
  monitor_mode, amp, device, seed`. `TrainerConfig.from_dict()` silently ignores
  unknown keys.
- Paper hyperparameters map directly: `optimizer="adamw"`, `lr=3.75e-4`,
  `weight_decay=0.05`, `epochs=50`.
- The paper's **effective batch size 1024** is a dataloader concern, not a
  `TrainerConfig` field. On limited-VRAM GPUs (Kaggle) this needs a real batch
  size plus gradient accumulation — expose both in config, never hardcode 1024.
- Trainer already provides AMP, grad clipping, early stopping, best-checkpoint
  selection, and resume. CarveFormer must **not** duplicate any of this.

### Dataset — `src/data/dataset.py::FragmentDataset`
- Required NPZ layout (already produced upstream — do not regenerate splits):
  ```
  {root_dir}/{fragment_size}/{train,val,test}.npz
  # each: X [N, fragment_size] uint8, y [N] int
  ```
- `fragment_size ∈ {512, 4096}` is switchable by config only.

### Evaluator / output contract — `src/evaluation/evaluator.py`
- `Evaluator(model).evaluate_and_save(loader, out_dir, run_name)` writes the
  standardized set: `metrics.json`, `summary.json`, `predictions.csv`,
  `confusion_matrix.csv`, `per_class_metrics.csv`, `classification_report.txt`.
- CarveFormer must reuse this and must not invent a competing output format.

---

## B. File-by-file implementation checklist (Phases 1–5)

| File | Phase | Purpose |
|---|---|---|
| `benchmarks/CarveFormer/src/model.py` | 1 | Real CarveFormer nn.Module: 96-d byte embedding → pseudo-2D reshape → SwinV2-Tiny → head → log_softmax. Framework-independent. |
| `benchmarks/CarveFormer/src/adapter.py` | 2 | Thin `FragmentClassifier` wrapper around `model.py`; config-driven fragment size / class count. |
| `src/models/adapters/carveformer_adapter.py` | 2 | Replace stub `build_carveformer_adapter` with the real factory delegating to the branch adapter. Registry key unchanged. |
| `benchmarks/CarveFormer/configs/benchmark.yaml` | 3 | All tunables (see §E). |
| `benchmarks/CarveFormer/scripts/train.py` | 3 | Thin entrypoint using the shared Trainer — no duplicated training logic. |
| `benchmarks/CarveFormer/scripts/evaluate.py` | 3 | Thin entrypoint using the shared Evaluator. |
| `benchmarks/CarveFormer/notebooks/` | 3 | End-to-end Kaggle notebook; 512/4096 by config only. |
| `benchmarks/CarveFormer/tests/smoke_test.py` | 2, 4 | Instantiate model+adapter, dummy forward, output-shape checks. |
| `benchmarks/CarveFormer/docs/architecture.md` | 4 | Exact architecture, assumptions, deviations. |
| `benchmarks/CarveFormer/paper/implementation_checklist.md` | 4 | Progress tracker. |
| `benchmarks/CarveFormer/README.md` | 4 | Usage. |

---

## C. Architecture mapping (paper → code)

| Paper element | Target in code |
|---|---|
| Byte embedding dim 96 replacing image conv+norm | `nn.Embedding(256, 96)` in `model.py` |
| Reshape embedded sequence → pseudo-2D | isolated, documented helper (e.g. `_reshape_to_2d`) — swappable |
| Swin Transformer V2 Tiny, ImageNet1k-pretrained | `timm` (`swinv2_tiny_window*`), already in requirements |
| Classification head → num_classes | head in `model.py`, then `log_softmax` |
| AdamW, lr 3.75e-4, wd 0.05, 50 epochs, eff. batch 1024 | `configs/benchmark.yaml` |

Paper-grounded sanity targets (FFT-75): Scenario #1 = 72.10% @512 / 82.99% @4096;
#2 = 90.62% @512; #3 = 93.44% @512 / 96.87% @4096. Faithful runnable benchmark in
the same range is the goal, not exact reproduction.

---

## D. Compatibility risks

1. **log_softmax vs logits** — head must emit log-probs (or adapter overrides
   `loss()`). Decision: emit `log_softmax`, matching ByteRCNN.
2. **timm pretrained weight download** — needs network (Kaggle yes, offline CI
   no). Make `pretrained` configurable with graceful random-init fallback so the
   CPU smoke test runs offline.
3. **Reshape for 512** — paper is explicit for 4096 (64×64) but not 512.
   Implement a documented reshape, keep it swappable (handover §11: don't guess
   silently).
4. **Effective batch 1024 on limited VRAM** — expose real batch size +
   gradient accumulation; never hardcode 1024.
5. **Pre-existing framework issues (not caused by this branch, but affect how it
   runs):** `src/core/runner.py` imports a non-existent `src.data.dataset_factory`,
   and `configs/datasets/fft75_512.yaml` is absent. CarveFormer's scripts should
   call `FragmentDataset` directly (as `train_bytercnn.py` does) rather than route
   through the broken runner, until that is fixed separately. Not in scope for
   this branch.

---

## E. Values that must be configurable (never hardcoded)

`fragment_size` (512/4096) · `num_classes` (75/11/25/5/2/2 by scenario) ·
`embedding_dim` (default 96) · reshape height/width · backbone name/variant ·
`pretrained` (bool) · `lr` · `weight_decay` · `batch_size` + `grad_accum`
(effective 1024) · `epochs` · dataset `root_dir` · output/checkpoint paths ·
Kaggle working paths · `seed`.

---

## F. Scaffold discrepancy noted

The handover states `benchmarks/CarveFormer/` "already exists and must be
preserved" — confirmed present on the `eval-CarveFormer` branch (this branch),
**not** on `master`. All work proceeds additively on top of the existing
scaffold; no scaffold file is deleted or renamed.

## G. Phase 0 success criteria — met
- [x] Paper architecture mapped to the repo
- [x] Implementation scope is clear (file-by-file checklist above)
- [x] No code changed
