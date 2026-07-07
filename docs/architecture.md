# Architecture

The framework is a thin, model-agnostic layer around one frozen reference
implementation (ByteRCNN) and a growing set of future models. Its only job
is to make training/evaluating *any* registered model identical in shape:

```
        configs/experiments/<name>.yaml
                    │  (composes)
   ┌───────┬────────┴────────┬──────────────┐
   ▼       ▼                 ▼              ▼
dataset  model             training      evaluation
 .yaml    .yaml             .yaml          .yaml
   │       │                 │              │
   ▼       ▼                 ▼              ▼
src/data/          src/models/        src/training/    src/evaluation/
dataset_factory ─▶ registry.build_model ─▶ Trainer ─▶ Evaluator
   │                    │                    │              │
   ▼                    ▼                    ▼              ▼
NpzFragmentDataset  FragmentClassifier   checkpoints,   metrics.json,
(train/val/test)    (adapter-wrapped)    loss/acc plots  confusion matrix,
                                                          per-class report
```

`src/core/runner.py` is the only orchestrator that wires these together;
it is what `python -m src.core.runner --experiment ...` runs.

## The contract that makes this work

`src/core/interfaces.py:FragmentClassifier` is the single interface every
model adapter must implement:

- `forward(x) -> log_probs` of shape `[batch, num_classes]`
- `loss(log_probs, targets) -> scalar`
- `num_classes` attribute

The generic `Trainer` and `Evaluator` only ever call these three things.
They never import a specific model, never branch on model name, and never
know that ByteRCNN uses a BiGRU + 4 parallel CNN branches internally.

## Why adapters exist

`src/models/bytercnn_wrapper.py` is a general-purpose PyTorch module — it
predates the framework and has its own API (`build_bytercnn(...)`).
`src/models/adapters/bytercnn_adapter.py` is a **thin** wrapper that
translates that API into the `FragmentClassifier` contract, without
changing any of the wrapped model's behavior or hyperparameters. Future
models (CarveFormer, ByteNet, DeepCarv) get the same treatment: implement
the model, wrap it in an adapter, register it.

`benchmarks/ByteRCNN/` (the frozen submodule with the original Keras
implementation) is not imported by the framework at all. It exists as a
read-only reference; if a future need arises to cross-check the PyTorch
port against it, that would be a separate, explicitly-named script, never
part of the trainer/evaluator path.

## Stub models

`carveformer`, `bytenet`, and `deepcarv` are registered in
`src/models/registry.py` today, but their adapters
(`src/models/adapters/*_adapter.py`) raise `NotImplementedError` on
instantiation — they subclass `src/models/base.py:NotAvailableModel`. This
keeps the registry API stable (the name resolves, configs referencing it
don't need to change later) while making it obvious the model isn't
runnable yet.

## Extending the framework

Adding a new model requires exactly four steps, none of which touch the
trainer, evaluator, or runner:

1. Implement the model (as its own module, or wrap a frozen reference
   implementation under `benchmarks/<model>/` the same way ByteRCNN is
   isolated today).
2. Write `src/models/adapters/<model>_adapter.py` satisfying
   `FragmentClassifier`.
3. Point the existing registry key at the real adapter in
   `src/models/registry.py` (replacing the stub import).
4. Fill in `configs/models/<model>.yaml` with real hyperparameters, then
   run `python -m src.core.runner --experiment <any experiment using it>`.
