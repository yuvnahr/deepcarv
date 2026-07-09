# DepthwiseCNN Implementation Checklist

Paper: **File Fragment Type Classification Using Light-Weight Convolutional Neural Networks**  
Branch: `eval-depthwiseCNN`

This checklist is the authoritative record of implementation progress.  Each item links to the relevant file or commit so reviewers can verify completion.

---

## Phase 0 — Scaffold and planning ✅

- [x] Paper read in full; architecture diagram and Table 1 reviewed
- [x] Codebase audited: `FragmentClassifier` contract, `Trainer`, `Evaluator`, `ModelRegistry` all understood
- [x] Integration plan written and approved ([`implementation_plan.md`](../../../.gemini/antigravity-ide/brain/1bf518a0-5cf1-4ebe-a613-b956c6d20cf6/implementation_plan.md))
- [x] FFT-75 NPZ loader confirmed intact — no CSV conversion required
- [x] Benchmark scaffold directories created (`src/`, `scripts/`, `tests/`, `docs/`, `configs/`, `paper/`, `notebooks/`, `assets/`)

---

## Phase 1 — Model architecture ✅

File: [`src/model.py`](../src/model.py)

- [x] `SeparableConv1d`: depthwise + pointwise, bias-free, type-hinted, docstrings
- [x] `SEBlock`: squeeze–excite with global avg pool, bottleneck FC pair, sigmoid gate
- [x] `_build_norm` factory: `BatchNorm1d` / `GroupNorm(8 groups)`
- [x] `_build_act` factory: `Hardswish` / `ReLU`
- [x] `InceptionBlock`: 3 parallel branches (k=11/19/27), per-branch norm, element-wise sum, optional `MaxPool1d(4,4)`, residual shortcut (1×1 Conv when channels change)
- [x] `DepthwiseCNNModel`: single class covers DSC / DSC-SE / M-DSC via `variant` arg
- [x] DSC variant: standard `Conv1d` first conv, `BatchNorm1d`, `Hardswish`
- [x] DSC-SE variant: SE blocks inserted after each InceptionBlock
- [x] M-DSC variant: depthwise-only first conv, `GroupNorm`, `ReLU`, `Dropout(p=0.2)` head
- [x] All 6 paper ambiguities documented as inline comments (decisions #1–#6)
- [x] `build_depthwisecnn` public factory with `ValueError` guards
- [x] `extra_repr` for clean `print(model)` output
- [x] All type hints (`Literal`, `Variant`, `NormType`, `ActType`)

---

## Phase 2 — Adapter and registry integration ✅

Files: [`src/adapter.py`](../src/adapter.py), [`src/models/registry.py`](../../../src/models/registry.py)

- [x] `DepthwiseCNNAdapter` subclasses `FragmentClassifier` (inherits `nn.Module`)
- [x] `forward(x)` delegates to `_model(x)` — returns log-probabilities `[B, num_classes]`
- [x] `loss()` — **not overridden**; base-class `NLLLoss` is correct (model outputs log-probs)
- [x] `predict()` — **not overridden**; inherited argmax is correct
- [x] `num_parameters(trainable_only)` overridden to count inner model parameters
- [x] `name` attribute: `"depthwisecnn_<variant>"`
- [x] `variant` attribute: set for introspection / checkpoint metadata
- [x] `extra_repr` for clean repr output
- [x] Framework contract surface documented in module docstring (all 10 trainer/evaluator call sites)
- [x] `build_depthwisecnn_adapter` factory — sole public entry point
- [x] Registry key `"depthwisecnn"` added to `src/models/registry.py`
- [x] Registry comment updated: `depthwisecnn` listed as implemented (not stub)
- [x] `variant` kwarg propagation documented in registry comment

---

## Phase 3 — Config, scripts, and Kaggle notebook ✅

Files: [`configs/benchmark.yaml`](../configs/benchmark.yaml), [`scripts/train.py`](../scripts/train.py), [`scripts/evaluate.py`](../scripts/evaluate.py), [`notebooks/kaggle.ipynb`](../notebooks/kaggle.ipynb)

### Config (`benchmark.yaml`)
- [x] `dataset.root_dir` — resolves via `DEEPCARV_DATA_ROOT` env var
- [x] `dataset.fragment_size` — the **one key** to change for 512 ↔ 4096 switch
- [x] `dataset.cache` — loads NPZ into RAM
- [x] `model.name` — `"depthwisecnn"` (registry key)
- [x] `model.kwargs.variant` — `dsc` / `dsc-se` / `m-dsc`
- [x] `model.kwargs.dropout_p` — forwarded to M-DSC head
- [x] All `TrainerConfig` fields (`epochs`, `lr`, `weight_decay`, `optimizer`, `scheduler`, `scheduler_factor`, `scheduler_patience`, `grad_clip`, `patience`, `monitor`, `monitor_mode`, `amp`, `device`, `seed`)
- [x] `evaluation.split` — default split for evaluate.py
- [x] `evaluation.batch_size` — separate inference batch size
- [x] `paths.run_outputs` — checkpoints + training curves
- [x] `paths.best_checkpoint` — used by evaluate.py default
- [x] `paths.last_checkpoint` — used for resume
- [x] `paths.eval_outputs` — evaluator output directory

### Training script (`train.py`)
- [x] No custom training loop — delegates to `Trainer.fit()`
- [x] CLI: `--config`, `--fragment-size`, `--variant`, `--epochs`, `--batch-size`, `--resume`, `--run-name`
- [x] CLI overrides shadow YAML values (YAML remains unmodified)
- [x] Model built via `build_model()` (registry path, no direct adapter import)
- [x] `RunLogger` used as context manager (config snapshot + eval summary written)
- [x] Seed set before data loading
- [x] Resume path forwarded to `trainer.resume_from()`

### Evaluation script (`evaluate.py`)
- [x] No custom evaluation logic — delegates to `Evaluator.evaluate_and_save()`
- [x] CLI: `--config`, `--checkpoint`, `--split`, `--out-dir`, `--fragment-size`, `--run-name`
- [x] Produces full standard output set (6 files)
- [x] Checkpoint metadata (epoch, val_acc) extracted and logged
- [x] Model built via `build_model()` (registry path)

### Kaggle notebook (`kaggle.ipynb`)
- [x] 7 sections — runnable top-to-bottom with no editing beyond § 0
- [x] § 0: single config cell (`FRAGMENT_SIZE`, `VARIANT`, `EPOCHS`, …)
- [x] § 1: Kaggle path layout, env vars, dep install, repo clone/pull
- [x] § 2: dual-source download (Kaggle input → gdown fallback)
- [x] § 3: validates all 3 NPZ splits (keys, shape, dtype, class count)
- [x] § 4: sanity pass — 5 epochs on 2 000 samples
- [x] § 5: full training via `train_main()`
- [x] § 6: evaluation via `eval_main()`; writes 6 output files
- [x] § 7: summary table, top/bottom-10 per-class F1, confusion matrix heatmap, benchmark card
- [x] Fragment-size switch is config-only (change `FRAGMENT_SIZE` in § 0)

---

## Phase 4 — Tests, docs, and quality hardening ✅

Files: [`tests/smoke_test.py`](../tests/smoke_test.py), [`docs/architecture.md`](../docs/architecture.md), [`README.md`](../README.md)

### Smoke tests (`smoke_test.py` — 75 tests, all passing)
- [x] Phase 1 model tests: forward shape × 3 variants × 2 sizes, factory variant attribute, invalid variant/num_classes guards
- [x] Phase 2 adapter tests: instantiation, `isinstance(FragmentClassifier)`, `name`, `variant`, `num_classes` attributes
- [x] Forward shape + log-prob validity × 3 variants × 2 sizes
- [x] `loss()` returns scalar, is differentiable (backward)
- [x] `predict()` returns `[B]` int64 in `[0, num_classes)`
- [x] `num_parameters()`: total ≥ trainable, value in expected range
- [x] Train/eval mode toggle (`model.train(True/False)`)
- [x] Eval-mode determinism (two consecutive forward passes identical)
- [x] `state_dict` round-trip (load into fresh instance, outputs identical)
- [x] Device transfer (`.to("cpu")` does not raise)
- [x] Config loading tests: benchmark.yaml has all required keys
- [x] Registry tests: key present, all 3 variants buildable, default variant, num_classes alignment, non-regression for existing models

### Documentation
- [x] `docs/architecture.md`: exact shape traces, verified param counts, per-layer breakdown, 6-row ambiguity table, framework contract
- [x] `README.md`: quickstart commands, variant/fragment-size switch table, architecture summary, data layout, Kaggle guide, eval output list, file structure
- [x] `paper/implementation_checklist.md`: this file — all phases, all items

### Code quality
- [x] All public classes and functions have type hints and docstrings
- [x] All paper ambiguities annotated as comments at point of decision
- [x] No changes to ByteRCNN, core trainer, evaluator, or dataset loader
- [x] No `register_model()` calls in scripts (registry entry added once in `registry.py`)
- [x] `model.py` and `adapter.py` are self-contained with no circular imports

---

## Pending (post-merge)

- [ ] Full training run on FFT-75 (requires GPU + dataset access)
- [ ] Verify ~79 % test accuracy on FFT-75 Scenario 1 (as reported in paper)
- [ ] FLOPs profiling to confirm ~164 M FLOPs claim
- [ ] CPU vs GPU latency profiling
- [ ] Revisit M-DSC first-conv decision (#4) if accuracy is below expectation