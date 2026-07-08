# Training

`src/training/trainer.py:Trainer` is the single, model-agnostic training
loop used for every registered model. It has no ByteRCNN-specific (or any
model-specific) logic — it depends only on the `FragmentClassifier`
interface and standard `torch.utils.data.DataLoader`s.

## What it does each epoch

1. Runs one training pass (forward, `model.loss(...)`, backward, optional
   gradient clipping, optimizer step — with AMP autocast if enabled and a
   CUDA device is available).
2. Runs one validation pass (no grad).
3. Steps the LR scheduler (`ReduceLROnPlateau` on the monitored metric, or
   `CosineAnnealingLR`, or none).
4. Saves `checkpoint_last.pt` every epoch and `checkpoint_best.pt`
   whenever the monitored metric improves.
5. Calls every registered callback's `on_epoch_end(epoch, metrics)`; if any
   returns `True` (e.g. `EarlyStopping`), training stops after that epoch.

## Configuring it

All settings live in `configs/training/*.yaml` (see
`src/training/trainer.py:TrainerConfig` for the full field list):

```yaml
epochs: 30
batch_size: 256          # consumed by the dataset factory, not the trainer directly
lr: 0.001
weight_decay: 0.0001
optimizer: adamw          # adamw | adam | sgd
scheduler: reduce_on_plateau   # reduce_on_plateau | cosine | null
grad_clip: 1.0
patience: 5                # early stopping patience
monitor: val_acc
monitor_mode: max
amp: true                  # only takes effect on a CUDA device
device: auto                # auto | cuda | cpu
```

## Resuming

```python
trainer = Trainer(model, config, run_dir)
start_epoch = trainer.resume_from(Path("outputs/some_run/checkpoint_last.pt"))
trainer.fit(train_loader, val_loader, start_epoch=start_epoch)
```

## Checkpoints

`src/training/checkpointing.py` is intentionally dumb: it saves/loads
`{"epoch", "model_state_dict", "optimizer_state_dict", "metrics", ...}`
dicts and nothing model-specific. Any adapter's `state_dict()` round-trips
through it unchanged.

## Adding a new model to training

Nothing here changes. Once a model is registered
(`src/models/registry.py`) and satisfies `FragmentClassifier`, it trains
through this exact same `Trainer` with no subclassing or branching.
