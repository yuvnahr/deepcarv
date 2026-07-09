# ByteNet Benchmark

This module implements the **ByteNet** architecture for multimedia file fragment classification.

## Paper
**ByteNet: Rethinking Multimedia File Fragment Classification through Visual Perspectives**
*Wenyang Liu, Kejun Wu, Tianyi Liu, Yi Wang, Kim-Hui Yap, Lap-Pui Chau (2023)*
See `docs/architecture.md` and `paper/implementation_checklist.md` for deep dives into how the paper's claims are mapped to this codebase.

## Variants
- **ByteResNet**: Combines n-gram embedding with CNN-based ResNet blocks.
- **ByteFormer**: Combines patch embedding with PoolFormer (average pooling) blocks.

## Usage

### Local Environment
Train the benchmark on the FFT-75 dataset (requires pre-split NPZ files):
```bash
python benchmarks/ByteNet/scripts/train.py --data_dir data/FFT-75 --fragment_size 512 --variant bytenet_resnet
```

Evaluate a checkpoint:
```bash
python benchmarks/ByteNet/scripts/evaluate.py --checkpoint checkpoints/best_bytenet_resnet_512b.pt --data_dir data/FFT-75 --fragment_size 512 --variant bytenet_resnet --out_dir outputs/eval_results
```

Run smoke tests:
```bash
python -m pytest benchmarks/ByteNet/tests/smoke_test.py -v
```

### Kaggle Environment
A ready-to-run Kaggle notebook is located at `notebooks/kaggle_bytenet_fft75.ipynb`. Upload this notebook, set `KAGGLE_DATASET_SLUG` to point to the mounted FFT-75 dataset, and run all cells.

## Status
- [x] Paper reviewed
- [x] Architecture implemented
- [x] Adapter integrated
- [x] Config added
- [x] Smoke test passing
- [x] Benchmark training script implemented
- [x] Evaluation script implemented
