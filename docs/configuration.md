# Configuration

Experiment configs never repeat hyperparameters inline. An experiment
config only *names* four components, each resolved from its own file:

```yaml
# configs/experiments/bytercnn_fft75.yaml
name: bytercnn_fft75
dataset: fft75_512          # -> configs/datasets/fft75_512.yaml
model: bytercnn              # -> configs/models/bytercnn.yaml
training: default            # -> configs/training/default.yaml
evaluation: benchmark        # -> configs/evaluation/benchmark.yaml
```

`src/utils/config.py:load_experiment_config` resolves each reference and
returns one flat `ExperimentConfig` with `.dataset`, `.model`, `.training`,
`.evaluation` dicts. Nothing downstream (dataset factory, registry,
trainer, evaluator) ever reads a YAML file directly — they only receive
already-composed dicts.

## Switching benchmarks: 512 vs 4096 bytes

This is a config change only:

```yaml
# configs/datasets/fft75_512.yaml
fragment_size: 512
root_dir: "data/FFT-75"
```
```yaml
# configs/datasets/fft75_4096.yaml
fragment_size: 4096
root_dir: "data/FFT-75"
```

Both point at the same `root_dir`; `NpzFragmentDataset` appends
`str(fragment_size)` to find the right subfolder
(`data/FFT-75/512/train.npz` vs `data/FFT-75/4096/train.npz`).
`configs/experiments/bytercnn_fft75_4096.yaml` demonstrates swapping just
the `dataset:` reference.

## CLI overrides

For quick experiments without editing files:

```bash
python -m src.core.runner --experiment bytercnn_fft75 \
    --override training.epochs=5 \
    --override training.batch_size=64 \
    --override dataset.tiny_subset=2000
```

Each `--override key.path=value` is applied after composition
(`src/utils/config.py:apply_dotted_overrides`). Values are coerced to
`int`/`float`/`bool` on a best-effort basis; everything else stays a
string.

## Inline overrides in an experiment file

An experiment file can also carry inline overrides for a referenced
component without creating a whole new component file:

```yaml
training:
  name: default
  overrides:
    epochs: 5
    patience: 2
```

## Adding a new dataset, model, training, or eval variant

Drop a new YAML file in the matching `configs/<kind>/` folder and
reference it by filename (without `.yaml`) from an experiment config. No
code changes needed for a new hyperparameter set.
