$ErrorActionPreference = "Stop"

Write-Host "====================================="
Write-Host " DeepCarv — ByteRCNN FFT-75 Setup"
Write-Host " (Windows / PowerShell)"
Write-Host "====================================="

# ── Python detection ────────────────────────────────────────────────────────
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python is not installed or not in PATH."
    exit 1
}

Write-Host "`nPython Version:"
python --version

# ── Virtual environment ──────────────────────────────────────────────────────
if (!(Test-Path ".\venv")) {
    Write-Host "`nCreating virtual environment…"
    python -m venv venv
}
else {
    Write-Host "`nVirtual environment already exists."
}

Write-Host "`nActivating virtual environment…"
& ".\venv\Scripts\Activate.ps1"

# ── Pip upgrade ──────────────────────────────────────────────────────────────
Write-Host "`nUpgrading pip…"
python -m pip install --upgrade pip setuptools wheel

# ── Dependencies ─────────────────────────────────────────────────────────────
Write-Host "`nInstalling dependencies from requirements.txt…"
pip install -r requirements.txt

# ── Verification ─────────────────────────────────────────────────────────────
Write-Host "`nVerifying installation…"
python -c "
import torch, sys
print(f'Python  : {sys.version.split()[0]}')
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')
import sklearn, pandas, numpy, yaml, gdown
print(f'sklearn : {sklearn.__version__}')
print(f'pandas  : {pandas.__version__}')
print(f'numpy   : {numpy.__version__}')
print(f'gdown   : {gdown.__version__}')
"

# ── Project directories ──────────────────────────────────────────────────────
Write-Host "`nCreating project directories…"

$dirs = @(
    "checkpoints",
    "logs",
    "outputs\bytercnn_fft75",
    "results",
    "cache",
    "datasets",
    "models",
    "data\raw",
    "data\splits",
    "src\data",
    "src\models",
    "src\training",
    "src\evaluation",
    "src\utils",
    "configs",
    "notebooks",
    "benchmarks\ByteRCNN"
)

foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created: $dir"
    }
}

# ── Model smoke-test ─────────────────────────────────────────────────────────
Write-Host "`nRunning ByteRCNN architecture smoke-test…"
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

Write-Host ""
Write-Host "====================================="
Write-Host " Setup Complete"
Write-Host "====================================="
Write-Host ""
Write-Host "To activate the venv later:"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Quick-start commands:"
Write-Host "  # 1. Verify dataset (after downloading + unzipping):"
Write-Host "  python -m src.data.verify_dataset --data_dir data/FFT-75 --fragment_size 512"
Write-Host ""
Write-Host "  # 2. Sanity check:"
Write-Host "  python -m src.training.sanity_train_bytercnn --config configs/fft75_s1_512_bytercnn.yaml"
Write-Host ""
Write-Host "  # 3. Full training:"
Write-Host "  python -m src.training.train_bytercnn --config configs/fft75_s1_512_bytercnn.yaml"
Write-Host ""
Write-Host "  # 4. Evaluation:"
Write-Host "  python -m src.evaluation.evaluate_bytercnn --config configs/fft75_s1_512_bytercnn.yaml"