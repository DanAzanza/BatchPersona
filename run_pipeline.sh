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

# Pass CLI arguments directly if provided
if [ $# -gt 0 ]; then
    "$PYTHON_EXE" scripts/batch_swapper.py "$@"
    exit $?
fi

# Default server fallback
DEFAULT_SERVER="${COMFYUI_SERVER:-127.0.0.1:8000}"

ensure_comfyui() {
    if $PYTHON_EXE -c "import urllib.request; urllib.request.urlopen('http://$DEFAULT_SERVER/system_stats', timeout=1.5)" >/dev/null 2>&1; then
        return 0
    fi

    echo ""
    echo "[NOTICE] ComfyUI server is not responding at $DEFAULT_SERVER."

    COMFY_CMD=""
    if command -v comfyui >/dev/null 2>&1; then
        COMFY_CMD="comfyui"
    elif [ -f "$HOME/ComfyUI/main.py" ]; then
        COMFY_CMD="$PYTHON_EXE $HOME/ComfyUI/main.py"
    fi

    if [ -z "$COMFY_CMD" ]; then
        echo "[WARNING] ComfyUI server is offline. Please start ComfyUI or choose Option [2] for Zero-GPU Dry-Run."
        return 1
    fi

    echo "[FOUND] Detected local ComfyUI: $COMFY_CMD"
    read -r -p "Do you want to launch ComfyUI now? [Y/n, default Y]: " LAUNCH_CHOICE
    if [ "$LAUNCH_CHOICE" = "n" ] || [ "$LAUNCH_CHOICE" = "N" ]; then
        echo "[INFO] Skipping auto-launch."
        return 1
    fi

    echo "[LAUNCHING] Starting ComfyUI in the background..."
    $COMFY_CMD &
    echo "[WAITING] Waiting for ComfyUI server to become ready at $DEFAULT_SERVER..."
    for i in $(seq 1 20); do
        if $PYTHON_EXE -c "import urllib.request; urllib.request.urlopen('http://$DEFAULT_SERVER/system_stats', timeout=1.5)" >/dev/null 2>&1; then
            echo "[OK] ComfyUI is online and ready!"
            return 0
        fi
        sleep 2
    done

    echo "[TIMEOUT] ComfyUI did not respond within 40 seconds."
    return 1
}

while true; do
    clear
    echo "==========================================================================="
    echo "      BatchPersona - Headless ComfyUI Model Replacement Pipeline"
    echo "==========================================================================="
    echo "  Python runtime: $PYTHON_EXE"
    echo "  Server address: $DEFAULT_SERVER"
    echo "==========================================================================="
    echo ""
    echo "  [1] Run Commercial Model Swap  (Campaign Lookbooks - Diverse Personas)"
    echo "  [2] Run Quick Dry-Run          (Instant Zero-GPU Verification, < 1s)"
    echo "  [3] Run Quality Gate and Tests (Pytest 55/55 + Coverage + Ruff)"
    echo "  [0] Exit"
    echo ""
    echo "==========================================================================="
    read -r -p "Select an option [1-3, 0]: " CHOICE

    case "$CHOICE" in
        1)
            if ! ensure_comfyui; then
                read -r -p "Press Enter to return to menu..."
                continue
            fi
            echo ""
            echo "[RUNNING] Executing Commercial Batch Model Swap across Campaign Lookbooks..."
            for campaign_file in data/input_campaign/*.png data/input_campaign/*.jpg data/input_campaign/*.jpeg data/input_campaign/*.webp; do
                if [ -f "$campaign_file" ]; then
                    echo ""
                    echo "==========================================================================="
                    echo "[CAMPAIGN] Processing Lookbook: $(basename "$campaign_file")"
                    echo "==========================================================================="
                    $PYTHON_EXE scripts/batch_swapper.py \
                        --server "$DEFAULT_SERVER" \
                        --campaign "$campaign_file" \
                        --models-dir data/input_models \
                        --output-dir data/output \
                        --workflow workflows/model_swap_qwen21_maskless_api.json \
                        --timeout 300.0
                fi
            done
            read -r -p "Press Enter to return to menu..."
            ;;
        2)
            if ! ensure_comfyui; then
                read -r -p "Press Enter to return to menu..."
                continue
            fi
            echo ""
            echo "[RUNNING] Executing Instant Composite Test Dry-Run (No GPU Required)..."
            if [ ! -f "data_synthetic/input_campaign/campaign_summer_lookbook.png" ]; then
                echo "[INFO] Generating isolated synthetic test dataset for dry-run..."
                $PYTHON_EXE scripts/generate_testdata.py --output-dir data_synthetic --size 896
            fi
            $PYTHON_EXE scripts/batch_swapper.py \
                --server "$DEFAULT_SERVER" \
                --campaign data_synthetic/input_campaign/campaign_summer_lookbook.png \
                --mask data_synthetic/input_campaign/campaign_summer_mask.png \
                --models-dir data_synthetic/input_models \
                --output-dir data_synthetic/output \
                --workflow workflows/model_swap_composite_api.json \
                --market-tag test \
                --timeout 30.0
            read -r -p "Press Enter to return to menu..."
            ;;
        3)
            echo ""
            echo "[RUNNING] Running Complete Quality Gate and CI Verification locally..."
            $PYTHON_EXE scripts/run_ci_locally.py
            read -r -p "Press Enter to return to menu..."
            ;;
        0)
            echo "Exiting BatchPersona. Goodbye!"
            exit 0
            ;;
        *)
            echo "[WARNING] Invalid selection: $CHOICE. Please choose between 0 and 3."
            sleep 2
            ;;
    esac
done
