# Datasets

FFT-75 is expected as pre-split NPZ files:

```text
FFT-75/
├── 512/{train,val,test}.npz
└── 4096/{train,val,test}.npz
```

Each NPZ contains `X` and `y`.

Before training, run:

```bash
python -m src.dataset_tools.cli validate --root /path/to/FFT-75 --fragment-size 512
python -m src.dataset_tools.cli fingerprint --root /path/to/FFT-75 --fragment-size 512 --output dataset_fingerprint.json
```

Dataset fingerprints record SHA256 hashes for `train.npz`, `val.npz`, and
`test.npz`, plus sample counts, class counts, fragment size, timestamp, and
FFT-75 version. This lets future experiments verify they used the same data.
