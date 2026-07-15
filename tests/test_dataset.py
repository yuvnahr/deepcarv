import numpy as np

from src.data.dataset import FragmentDataset
from src.data.dataset_factory import build_datasets
from src.data.npz_mmap import is_mmappable, mmap_npz_member


def test_fragment_dataset_loads(tiny_npz_dataset):
    ds = FragmentDataset(root_dir=tiny_npz_dataset, split="train", fragment_size=512)
    assert len(ds) == 40
    x, y = ds[0]
    assert x.shape == (512,)
    assert y.ndim == 0


def test_dataset_factory_builds_all_splits(tiny_npz_dataset):
    datasets = build_datasets(
        {"format": "npz", "root_dir": str(tiny_npz_dataset), "fragment_size": 512}
    )
    assert set(datasets.keys()) == {"train", "val", "test"}
    assert len(datasets["train"]) == 40
    assert len(datasets["val"]) == 20


def test_dataset_accepts_upper_and_lower_case_keys(tmp_path):
    """The documented format is uppercase X/y; some files use lowercase x/y.

    Both must load — a case mismatch previously made real FFT-75 files fail.
    """
    rng = np.random.default_rng(0)
    for keyname in ("X", "x"):
        root = tmp_path / keyname / "FFT-75"
        frag = root / "512"
        frag.mkdir(parents=True)
        for split in ("train", "val", "test"):
            arrays = {
                keyname: rng.integers(0, 256, (8, 512), dtype=np.uint8),
                "y": rng.integers(0, 3, (8,), dtype=np.int64),
            }
            np.savez(frag / f"{split}.npz", **arrays)
        ds = FragmentDataset(root_dir=root, split="train", fragment_size=512)
        assert len(ds) == 8


def test_mmap_is_available_and_matches_cached(tiny_npz_dataset):
    """The memory-mapped path must return data identical to the cached path."""
    npz = tiny_npz_dataset / "512" / "train.npz"
    assert is_mmappable(npz), "uncompressed npz should be mmappable"

    cached = FragmentDataset(
        root_dir=tiny_npz_dataset, split="train", fragment_size=512,
        cache=True, mmap=False,
    )
    mapped = FragmentDataset(
        root_dir=tiny_npz_dataset, split="train", fragment_size=512, mmap=True,
    )
    assert mapped._mmapped is True
    assert cached._mmapped is False
    assert len(cached) == len(mapped)

    for i in range(len(cached)):
        xc, yc = cached[i]
        xm, ym = mapped[i]
        assert xc.tolist() == xm.tolist()
        assert int(yc) == int(ym)


def test_mmap_member_returns_none_for_compressed(tmp_path):
    """Compressed npz members can't be mmapped -> loader must fall back, not crash."""
    p = tmp_path / "compressed.npz"
    np.savez_compressed(
        p,
        x=np.zeros((4, 512), dtype=np.uint8),
        y=np.zeros((4,), dtype=np.int64),
    )
    assert is_mmappable(p) is False
    assert mmap_npz_member(p, "x") is None


def test_dataset_falls_back_when_compressed(tmp_path):
    """A compressed dataset still loads (via normal np.load), just without mmap."""
    root = tmp_path / "FFT-75"
    frag = root / "512"
    frag.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for split in ("train", "val", "test"):
        np.savez_compressed(
            frag / f"{split}.npz",
            x=rng.integers(0, 256, (6, 512), dtype=np.uint8),
            y=rng.integers(0, 3, (6,), dtype=np.int64),
        )
    ds = FragmentDataset(root_dir=root, split="train", fragment_size=512, mmap=True)
    assert ds._mmapped is False  # fell back gracefully
    assert len(ds) == 6
    x, _ = ds[0]
    assert x.shape == (512,)
