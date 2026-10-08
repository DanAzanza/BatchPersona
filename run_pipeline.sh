#!/usr/bin/env bash
# ==============================================================================
# BatchPersona - Streamlined Interactive Pipeline Runner (Linux / macOS)
# ==============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Auto-detect Python runtime
PYTHON_EXE="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_EXE="venv/bin/python"
else
    echo "[NOTICE] Virtual environment .venv not found. Running installer first..."
    ./install.sh
    if [ -f ".venv/bin/python" ]; then
        PYTHON_EXE=".venv/bin/python"
    fi
fi

# Forward all execution directly to the Python orchestrator
exec "$PYTHON_EXE" -m scripts.run_pipeline "$@"
