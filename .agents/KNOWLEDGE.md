# KNOWLEDGE.md - BatchPersona Operational Memory

## Runtime & Platform Constraints
- **Platform**: Windows 11 / PowerShell 7+ host.
- **Python Runtime**:
  - Local host: Python 3.14.5 64-bit (`C:\Users\danie\AppData\Local\Python\pythoncore-3.14-64\python.exe`).
  - GitHub Actions CI: Python 3.12 across a multi-OS matrix (`ubuntu-latest` and `windows-latest`, defined in `.github/workflows/ci.yml`).
  - Project requires Python >= 3.10 (managed code-first in `pyproject.toml`).
- **Windows File Paths & Filesystem**:
  - Always use `pathlib.Path` with POSIX or raw strings when passing paths to APIs.
  - Never string-interpolate Windows backslashes into JSON text templates (causes `JSONDecodeError: Invalid \escape`).
  - Set `sys.stdout.reconfigure(encoding='utf-8')` to prevent `UnicodeEncodeError` in Windows consoles.
  - Workspace `G:\` is mounted on Google Drive for Desktop; automated tests must be self-healing against background cloud sync latency.
  - Development environment must be installed in editable mode (`pip install -e .`) so that `python -m batchpersona` and CLI entrypoints resolve deterministically without relying on ad-hoc `sys.path` hacks.
- **Cross-Platform Subprocess Attributes**:
  - `subprocess.STARTUPINFO`, `subprocess.STARTF_USESHOWWINDOW`, `subprocess.CREATE_NEW_CONSOLE`, and `subprocess.DETACHED_PROCESS` do not exist in Linux/POSIX CPython builds.
  - When referencing or simulating Windows-specific process flags in cross-platform code or tests, always resolve them defensively via `getattr(subprocess, "...", None)` to avoid `AttributeError` on POSIX hosts.

## Local ComfyUI Instance Details
- **Active Instance**: Comfy Desktop running headless API backend at `http://127.0.0.1:8000`.
- **Default Server Port**:
  - Unified default across all modules (`models.py`, `launcher.py`, `pipeline.py`, `orchestrator.py`) is `127.0.0.1:8000` (`DEFAULT_SERVER_ADDRESS`, tuned for Comfy Desktop).
  - Standalone portable or git ComfyUI installations typically run on port `8188`; pass `--server 127.0.0.1:8188` or export `COMFYUI_SERVER=127.0.0.1:8188`.
- **Hardware Acceleration**: CUDA 0 - NVIDIA GeForce RTX 4060 Ti (16 GB VRAM).
- **Model Storage**: `C:\Users\danie\ComfyUI-Shared\models`
  - Diffusion: `diffusion_models/qwen_image_2.1_int8_convrot.safetensors`
  - Text Encoder: `text_encoders/qwen3vl_8b_int8_convrot.safetensors`
  - VAE: `vae/qwen_image_2.1_vae_bf16.safetensors`
  - CLIP Vision: `clip_vision/dino_v3_L_naf_fp32.safetensors`
  - Background Removal: `background_removal/birefnet.safetensors`

## ComfyUI Headless API Patterns & Quirks
- **Prompt Format**: ComfyUI API endpoint `/prompt` expects a dictionary of node IDs mapping to `class_type` and `inputs`. It does NOT accept the UI graph format (`nodes` / `links`).
- **Asset Ingestion & `/upload/image` Behavior**:
  - Standard ComfyUI `LoadImage` node does not read arbitrary local file paths; images must be uploaded to ComfyUI's input directory via `POST /upload/image` multipart form, or standard input folder names must be referenced.
  - With `overwrite=false` (default), ComfyUI detects existing files and renames duplicates (e.g. `image_1.png`). Always resolve the uploaded asset using the server-returned name (`response.json()["name"]`), never assume the original filename was preserved.
  - With `overwrite=true`, ComfyUI silently replaces existing files with identical names in the server input directory.
- **WebSocket Protocol & Event Delivery**:
  - Endpoint: `ws://<server>/ws?clientId=<client_id>`.
  - **Subscribe Before Dispatch & Event-Drop on Cache**: ComfyUI does NOT buffer WebSocket events for uncollected or disconnected `client_id`s; events dispatched while disconnected are permanently discarded. Jobs whose node outputs are cached finish execution in milliseconds (< 10 ms). Connecting WebSocket *after* queuing `POST /prompt` risks missing the `executing` completion signal entirely, causing the client to hang until the socket timeout.
  - Filter out binary frames (binary preview images sent during KSampler execution) before calling `json.loads`.
  - Match `data.prompt_id == current_prompt_id` to prevent cross-talk on multi-client servers.
  - Completion signal: `executing` event with `data.node is None` and matching `prompt_id`.
  - Error detection: `execution_error` event contains node ID, exception type, and traceback.
  - Execution timeouts and socket intervals must strictly use `time.monotonic()` to maintain resilience against system clock/NTP adjustments.
- **Qwen 2.1 DiT Guidance & Prompt Best Practices**:
  - Qwen Image 2.1 uses rectified flow matching with Euler sampler and simple scheduler.
  - Native inference operates at `CFG = 1.0` with `negative_prompt = ""` (pure conditional velocity field evaluation, 25 steps = 25 passes).
  - Raising `CFG > 1.0` with detailed negative prompts triggers cross-attention crosstalk ("pink elephant" effect) where tokens from `<image2>` (e.g. casual tank tops, t-shirts) inadvertently contaminate the scene, stripping away campaign garments.
  - Strict attribute isolation ("Only transfer the face, facial features, hair, skin tone, and identity of the model in <image2>") and explicit preservation ("Keep all clothing, garments, tailoring, shoes, accessories, pose, and studio background from <image1> completely unchanged. Do not transfer any clothing or footwear from <image2>") prevents footwear and outfit leakage from target models while ensuring seamless identity transfer.
- **Deterministic Output Naming**:
  - Output files are generated strictly as `<campaign_stem>_<clean_tag>_<model_stem>.png` when `--market-tag` is provided and non-global (case-insensitive).
  - When `--market-tag` is `"global"` or empty, output files preserve backward compatibility as `<campaign_stem>_<model_stem>.png`.
  - The market tag is sanitized to lowercase alphanumeric characters with hyphens and underscores (`clean_market_tag`).
- **Attention Masking**: When applying `IPAdapterApply` for face/model identity swap on lookbook images, wire the inpainting mask into `attn_mask` to prevent identity bleeding into background and garments.
- **Memory Management**: Do NOT flush cache (`free_memory: True`) between successful batch iterations to preserve upstream node evaluation caching. Only call `free_memory(unload_models=True)` on failure/OOM, and in the `finally:` block when the entire batch concludes.
- **Smart Housekeeping & Windows NTFS File-Locking**:
  - Purging intermediate ComfyUI input/output assets requires bounded micro-retries (3 attempts, 50 ms exponential backoff) due to transient Windows NTFS handle locking (`WinError 32` / `PermissionError`) by ComfyUI or the Windows Search indexer.
  - Deletion guards must verify canonical paths (`target.resolve() != source.resolve()` and `target.resolve() != export.resolve()`) to prevent self-deletion when campaign or output directories coincide with ComfyUI directories.

## Available Workflow Templates
- `workflows/model_swap_qwen21_maskless_api.json`: **Flagship** multimodal maskless instruction-based editing workflow using native `TextEncodeQwenImage21` with V3 Autogrow (`images.image_1`, `images.image_2`) and Euler/Simple sampler at native CFG 1.0, preserving base garments, tailoring, shoes, pose, and background from `<image1>` while seamlessly transferring the model identity from `<image2>`.
- `workflows/model_swap_api.json`: SDXL inpainting fallback graph with IP-Adapter identity conditioning and attention mask.
- `workflows/model_swap_auto_mask_api.json`: Autonomous self-masking workflow using on-the-fly BiRefNet foreground segmentation (zero external masks required).
- `workflows/model_swap_composite_api.json`: Live integration verification workflow using `ImageCompositeMasked` (instant CPU/GPU pipeline execution).

## Verification & Test Commands
- Unified Cross-Platform Pipeline Orchestrator: `python -m batchpersona.pipeline` or `python -m scripts.run_pipeline` (CLI: `batchpersona-pipeline`)
- Automated ComfyUI Server Healthcheck & Launcher: `python -m batchpersona.launcher --check` / `--detect` / `--launch` (CLI: `batchpersona-launcher`)
- Fast test suite with coverage: `python -m pytest tests --cov=batchpersona --cov=scripts --cov-report=term-missing -v`
- Modular test suites (each strictly $< 600$ lines, zero monolithic test files):
  - `tests/test_models.py` - Domain data models, SwapperConfig validation, and market localization tags.
  - `tests/test_client.py` - Low-level HTTP multipart, WebSocket event parsing, and connection lifecycle.
  - `tests/test_template.py` - Prompt API graph validation, placeholder replacement, and override patching.
  - `tests/test_orchestrator.py` - Batch execution loop, pre-flight checks, and single-model processing.
  - `tests/test_cache.py` - Incremental `--skip-existing` caching, force overrides, and corrupt output handling.
  - `tests/test_cli.py` - Command-line parsing, environment overrides, progress bar, and summary reporting.
  - `tests/test_metadata.py` - PNG chunk inspection and provenance metadata processing.
  - `tests/test_generator.py` - Synthetic PIL image fixtures and dataset preprocessing.
  - `tests/test_housekeeper.py` - Sandboxed input/output asset cleanup and path traversal prevention.
  - `tests/test_launch_comfyui.py` - Headless server instance detection and command synthesis.
  - `tests/test_server_lifecycle.py` - Server process supervision, PID management, and safe termination.
  - `tests/test_run_pipeline.py` - Interactive console menu and end-to-end execution flow.
  - `tests/test_ci_parity.py` - Multi-stage GitHub Actions CI equivalence and backward-compatibility shims.
  - `tests/test_packaging.py` - Standalone distribution packaging, file whitelisting, Unix permissions, and SHA-256 generation.
- Full local CI Quality Gate: `python scripts/run_ci_locally.py` (CLI: `batchpersona-ci`)
- Deterministic release archive packaging: `python scripts/package_release.py --version <X.Y.Z>` (CLI: `batchpersona-package`)
- Code formatting & linting: `python -m ruff check .` and `python -m ruff format --check .`
- Static type checking: `python -m pyright src scripts tests`
- Procedural synthetic test data generation: `python scripts/generate_testdata.py` (CLI: `batchpersona-testdata`)
- Full-body aspect-ratio preserving dataset standardization: `python scripts/prepare_fullbody_dataset.py` (CLI: `batchpersona-prep`)
- Windows 1-Click Interactive Batch Runner: `run_pipeline.bat`
- Live batch execution on local ComfyUI instance:
  ```powershell
  python -m batchpersona.orchestrator --server 127.0.0.1:8000 --campaign data/input_campaign/campaign_fashion_female.png --models-dir data/input_models/ --output-dir data/output/
  # Or via console entrypoint if installed in editable mode:
  batchpersona --server 127.0.0.1:8000 --campaign data/input_campaign/campaign_fashion_female.png --models-dir data/input_models/ --output-dir data/output/
  ```

## Commit Scopes
- `[core]` - Pipeline core library, client, and configuration models in `src/batchpersona/` (`client.py`, `models.py`, `template.py`)
- `[orchestrator]` - Batch execution and interactive workflows in `src/batchpersona/orchestrator.py` and `src/batchpersona/pipeline.py`
- `[launcher]` - Server lifecycle, detection, and termination in `src/batchpersona/launcher.py`
- `[housekeeper]` - Workspace and server asset purging in `src/batchpersona/housekeeper.py`
- `[ci]` - Continuous integration, configs, and verification scripts in `src/batchpersona/ci.py`, `scripts/run_ci_locally.py`, `.github/`, and root configs (`pyproject.toml`)
- `[data]` - Test data generation and dataset preprocessing in `src/batchpersona/` (`generator.py`, `preprocess.py`) & `scripts/`
- `[workflow]` - ComfyUI API graph templates in `workflows/`
- `[test]` - Test fixtures and mock integration tests in `tests/`
- `[docs]` - Architecture documentation and repository README
