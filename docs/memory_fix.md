# Memory Bottleneck: Root Cause and Fix

## TL;DR

The team could not train on the full FFT-75 dataset because `FragmentDataset`
**read the entire split into RAM and upcast every byte from 1 byte to 2**. At
4096-byte fragments that turned a 25 GB training split into ~50 GB resident
(~75 GB peak while both copies coexisted) against a ~13 GB Kaggle limit.

The data is now **memory-mapped** off disk and kept in its native `uint8`.
Resident memory is now near-flat regardless of dataset size, so full-scale
training on the complete FFT-75 is possible.

**This was not a fundamental hardware limit. It was an avoidable 2× upcast plus
a full-RAM load.**

---

## Root cause

`src/data/dataset.py` did this on every split:

```python
data = np.load(str(npz_path))     # materialises the WHOLE array in RAM
X = data["x"]
...
self._X = torch.from_numpy(X.astype(np.int16))   # <-- doubles it again
```

The comment in the old code said int16 was "4x less RAM than int64" — true, but
it was comparing against the wrong baseline. The values are byte values (0–255).
They already fit in **`uint8`**. Casting to `int16` doubled memory for zero
benefit.

### The arithmetic (FFT-75 Scenario 1, ~6.1 M training samples)

| Fragment size | Raw (uint8) | After `astype(int16)` | Peak during load | Kaggle RAM |
|---|---|---|---|---|
| 512 | 3.1 GB | 6.3 GB | ~9.4 GB | ~13 GB — *just* fits |
| **4096** | **25.2 GB** | **50.3 GB** | **~75.5 GB** | ~13 GB — **overflow** |

That is exactly the reported symptom: 512 was painful but survivable, 4096 was
impossible, and the workaround was to fall back to a smaller scenario. (This is
almost certainly why the ByteNet 4096 result is on 11 classes rather than 75 —
that run is therefore *not* comparable to the paper's Scenario-1 number.)

---

## The fix

### 1. Memory-map the NPZ instead of reading it into RAM

New module `src/data/npz_mmap.py`.

An `.npz` is a ZIP container of `.npy` members. When written with `np.savez`
(**not** `savez_compressed`), members are stored uncompressed, so the raw array
bytes sit contiguously in the file and can be mapped with `np.memmap`. The OS
then pages in only the slices actually read.

`FragmentDataset(..., mmap=True)` is now the default.

**Measured** (1.23 GB file, 300 k × 4096 samples, fresh process):

| Path | RSS after load |
|---|---|
| old (`cache=True`, RAM) | 1.21 GB |
| **new (`mmap=True`)** | **0.006 GB** |

~200× less resident memory, and it does not grow with the file. Extrapolated to
the real 25 GB 4096 split: **~0 GB instead of ~50–75 GB.**

Verified **bit-identical**: 500 random samples compared between the mmap path
and the old cached path — zero mismatches.

### 2. Stop the pointless upcast

Cached mode now stores `X` in native `uint8` and casts to `int64` per item in
`__getitem__` (negligible cost). Even without mmap, this halves memory.

### 3. Graceful fallback

`savez_compressed` files cannot be mmapped. `is_mmappable()` detects this and
`FragmentDataset` silently falls back to a normal load rather than crashing.
`rewrite_uncompressed()` is provided to convert such a file.

If a dataset reports `mmappable=False` in the notebook, it was saved compressed
— re-saving it uncompressed is what unlocks the memory benefit.

### 4. Case-insensitive NPZ keys

Real FFT-75 files appear with both `X`/`y` and `x`/`y`. The loader previously
hard-required lowercase (`set(data.files) != {"x","y"}` → raise), so
uppercase-`X` files failed outright. Both cases now load.

---

## Notebooks: gdown removed

Per the team's report, `gdown` was making things *worse*: it downloads a zip
**and** extracts it, so both copies occupy the instance, and the arrays get
pulled through RAM on the way in.

Both `benchmarks/CarveFormer/notebooks/` and `benchmarks/DepthwiseCNN/notebooks/`
now:

* read directly from the attached Kaggle datasets
  (`thegifman/fft-75-512-1`, `thegifman/fft-75-4096-1`),
* **symlink** them into the `{root}/{fragment_size}/{split}.npz` layout the
  framework expects rather than `shutil.copytree` (the DepthwiseCNN notebook was
  copying tens of GB),
* rely on mmap so the arrays are never resident.

Net effect: **no zip, no second copy, no RAM load.**

---

## Also fixed along the way

These were pre-existing breakages on `dev`, unrelated to the memory work but
blocking the test suite:

* **`src/core/runner.py` could not be imported** — it still imported the deleted
  `src.data.dataset_factory`. Restored as a thin, typed wrapper over
  `FragmentDataset` (with `mmap=True` by default).
* **3 test files failed to collect** for the same reason.
* **`tests/test_dataset.py`** imported the removed `src.data.npz_dataset`.
  Rewritten against the current API, plus new tests for mmap correctness,
  compressed-file fallback, and both key cases.
* **`tests/test_registry.py`** still asserted CarveFormer *raises*
  `NotImplementedError` — stale, since it is now implemented. Updated to assert
  the implemented models build, and that a genuinely unimplemented stub
  (`deepcarv`) still fails loudly.

**Result: 31/31 tests pass, `ruff` clean, `mypy` clean (81 files).**
(Before: 3 collection errors, broken runner.)

---

## What this does *not* fix

**Accuracy below the paper.** That is a separate problem and this change does
not address it — though it does remove the biggest confound, because you can now
train on the **full** dataset instead of a subset.

Two things worth checking once full-scale runs are possible:

1. **Re-run the 4096 benchmarks on Scenario 1 (75 classes).** If ByteNet's 4096
   result was produced on 11 classes, it is not comparable to the paper and the
   "gap" may be partly illusory.
2. **CarveFormer's effective batch size.** The paper uses **1024**; the shared
   `Trainer` has no gradient accumulation, so the real batch is whatever fits on
   the GPU (~64 on a T4). Large-batch training materially changes optimisation
   for transformers, and this is a plausible contributor to the accuracy gap.
   Adding gradient accumulation to the shared `Trainer` is the clean fix and is
   the recommended next step.

The models having no published weights means reproduction gaps are expected —
but train on the full data first, then judge the gap.
