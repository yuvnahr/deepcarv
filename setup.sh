#!/usr/bin/env bash
# ===========================================================================
#  setup.sh — Linux / macOS environment setup for DeepCarv / ByteRCNN
# ===========================================================================

set -e

echo "====================================="
echo " DeepCarv — ByteRCNN FFT-75 Setup"
echo " (Linux / macOS)"
echo "====================================="

# ── Python detection ────────────────────────────────────────────────────────
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "ERROR: Python is not installed or not in PATH."
    exit 1
fi

echo
$PYTHON --version

# ── Virtual environment ──────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo
    echo "Creating virtual environment…"
    $PYTHON -m venv venv
else
    echo
    echo "Virtual environment already exists."
fi

echo
echo "Activating virtual environment…"
source venv/bin/activate

# ── Pip upgrade ──────────────────────────────────────────────────────────────
echo
echo "Upgrading pip…"
python -m pip install --upgrade pip setuptools wheel

# ── Dependencies ─────────────────────────────────────────────────────────────
echo
echo "Installing dependencies from requirements.txt…"
pip install -r requirements.txt

# ── Verification ─────────────────────────────────────────────────────────────
echo
echo "Verifying installation…"
python <<EOF
import torch, sys
print(f"Python  : {sys.version.split()[0]}")
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
import sklearn, pandas, numpy, yaml, gdown
print(f"sklearn : {sklearn.__version__}")
print(f"pandas  : {pandas.__version__}")
print(f"numpy   : {numpy.__version__}")
print(f"gdown   : {gdown.__version__}")
EOF

# ── Project directories ──────────────────────────────────────────────────────
echo
echo "Creating project directories…"

mkdir -p \
    checkpoints \
    logs \
    outputs/bytercnn_fft75 \
    results \
    cache \
    datasets \
    models \
    data/raw \
    data/splits \
    src/data \
    src/models \
    src/training \
    src/evaluation \
    src/utils \
    configs \
    notebooks \
    benchmarks/ByteRCNN

# ── Model smoke-test (no data needed) ────────────────────────────────────────
echo
echo "Running ByteRCNN architecture smoke-test…"
python -c "
import sys
sys.path.insert(0, '.')
from src.models.bytercnn_wrapper import build_bytercnn
import torch
m = build_bytercnn()
x = torch.randint(0, 256, (2, 512), dtype=torch.long)
with torch.no_grad():
    out = m(x)
assert out.shape == (2, 75), f'Bad shape: {out.shape}'
n = sum(p.numel() for p in m.parameters())
print(f'  ByteRCNN OK — output {tuple(out.shape)}, params={n:,}')
"

echo
echo "====================================="
echo " Setup Complete"
echo "====================================="
echo
echo "To activate the venv later:"
echo "  source venv/bin/activate"
echo
echo "Quick-start commands:"
echo "  # 1. Build frozen split (after downloading dataset):"
echo "  python -m src.data.build_fft75_split --raw_dir data/raw --splits_dir data/splits"
echo
echo "  # 2. Sanity check:"
echo "  python -m src.training.sanity_train_bytercnn --config configs/fft75_s1_512_bytercnn.yaml"
echo
echo "  # 3. Full training:"
echo "  python -m src.training.train_bytercnn --config configs/fft75_s1_512_bytercnn.yaml"
echo
echo "  # 4. Evaluation:"
echo "  python -m src.evaluation.evaluate_bytercnn --config configs/fft75_s1_512_bytercnn.yaml"