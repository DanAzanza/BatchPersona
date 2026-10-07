#!/usr/bin/env bash
# ==============================================================================
# BatchPersona - One-Click Environment Setup & Self-Test (Linux / macOS)
# ==============================================================================
set -e

echo "==========================================================================="
echo "      BatchPersona - One-Click Environment Setup & Self-Test"
echo "==========================================================================="
echo ""

# 1. Locate Python 3
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Python 3 was not found in PATH."
    echo "Please install Python 3.10+ from https://www.python.org/downloads/ or via your package manager."
    exit 1
fi

PY_VER=$($PYTHON_CMD --version)
echo "[OK] Detected $PY_VER"

# 2. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating Python virtual environment in .venv..."
    $PYTHON_CMD -m venv .venv
    echo "[OK] Virtual environment created successfully."
else
    echo "[INFO] Virtual environment .venv already exists."
fi

VENV_PYTHON=".venv/bin/python"

# 3. Check and install dependencies
echo ""
if $VENV_PYTHON -c "import requests, websocket, PIL, numpy, pytest" >/dev/null 2>&1; then
    echo "[OK] All required Python dependencies are already installed."
else
    echo "[INFO] Installing production dependencies from requirements.txt..."
    $VENV_PYTHON -m pip install -r requirements.txt
    echo "[OK] Dependencies installed successfully."
fi

# 4. Generate initial synthetic test data if missing
echo ""
if [ ! -f "data/input_campaign/campaign_fashion_female.png" ]; then
    echo "[INFO] Generating initial synthetic lookbooks and model portraits..."
    $VENV_PYTHON scripts/generate_testdata.py --output-dir data --size 1024
    echo "[OK] Synthetic datasets generated in data/"
else
    echo "[OK] Campaign datasets already present in data/"
fi

# 5. Run automated self-test verification suite
echo ""
echo "[INFO] Running automated self-test verification suite..."
$VENV_PYTHON -m pytest tests -q
echo "[OK] All automated tests passed with 100% success."

# 6. Complete banner
echo ""
echo "==========================================================================="
echo "      [SUCCESS] BatchPersona Installation Completed Successfully!"
echo "==========================================================================="
echo "  To launch the interactive model replacement pipeline, run:"
echo "    ./run_pipeline.sh"
echo "==========================================================================="
echo ""
