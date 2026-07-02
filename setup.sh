#!/usr/bin/env bash

set -e

echo "====================================="
echo " Project Setup (Linux/macOS)"
echo "====================================="

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "Python is not installed."
    exit 1
fi

echo
$PYTHON --version

if [ ! -d "venv" ]; then
    echo
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
else
    echo
    echo "Virtual environment already exists."
fi

echo
echo "Activating virtual environment..."

source venv/bin/activate

echo
echo "Upgrading pip..."

python -m pip install --upgrade pip setuptools wheel

echo
echo "Installing dependencies..."

pip install -r requirements.txt

echo
echo "Verifying installation..."

python << EOF
import torch
import sys

print(f"Python : {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA   : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
EOF

echo
echo "Creating project directories..."

mkdir -p \
checkpoints \
logs \
outputs \
results \
cache \
datasets \
models

echo
echo "====================================="
echo " Setup Complete"
echo "====================================="
echo
echo "To activate later:"
echo "source venv/bin/activate"