$ErrorActionPreference = "Stop"

Write-Host "====================================="
Write-Host " Project Setup (Windows)"
Write-Host "====================================="

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python is not installed or not in PATH."
    exit 1
}

Write-Host "`nPython Version:"
python --version

if (!(Test-Path ".\venv")) {
    Write-Host "`nCreating virtual environment..."
    python -m venv venv
}
else {
    Write-Host "`nVirtual environment already exists."
}

Write-Host "`nActivating virtual environment..."
& ".\venv\Scripts\Activate.ps1"

Write-Host "`nUpgrading pip..."
python -m pip install --upgrade pip setuptools wheel

Write-Host "`nInstalling dependencies..."
pip install -r requirements.txt

Write-Host "`nVerifying installation..."

python -c "
import torch, sys
print(f'Python : {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
"

Write-Host "`nCreating project directories..."

$dirs = @(
    "checkpoints",
    "logs",
    "outputs",
    "results",
    "cache",
    "datasets",
    "models"
)

foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
}

Write-Host ""
Write-Host "====================================="
Write-Host " Setup Complete"
Write-Host "====================================="
Write-Host ""
Write-Host "To activate later:"
Write-Host ".\venv\Scripts\Activate.ps1"