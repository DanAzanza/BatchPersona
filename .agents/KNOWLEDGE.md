# KNOWLEDGE.md - BatchPersona Operational Memory

## Runtime & Platform Constraints
- **Platform**: Windows 11 / PowerShell 7+ host.
- **Python Runtime**: Python 3.14.5 64-bit (`C:\Users\danie\AppData\Local\Python\pythoncore-3.14-64\python.exe`).
- **Dependencies**: `requests`, `websocket-client`, `pillow`, `pytest`, `pytest-mock`, `numpy`.
- **Windows File Paths & Filesystem**:
  - Always use `pathlib.Path` with POSIX or raw strings when passing paths to APIs.
  - Never string-interpolate Windows backslashes into JSON text templates (causes `JSONDecodeError: Invalid \escape`).
  - Set `sys.stdout.reconfigure(encoding='utf-8')` to prevent `UnicodeEncodeError` in Windows consoles.
  - Workspace `G:\` is mounted on Google Drive for Desktop; automated tests must be self-healing against background cloud sync latency.

## Local ComfyUI Instance Details
- **Active Instance**: Comfy Desktop running headless API backend at `http://127.0.0.1:8000`.
- **Hardware Acceleration**: CUDA 0 - NVIDIA GeForce RTX 4060 Ti (16 GB VRAM).
- **Model Storage**: `C:\Users\danie\ComfyUI-Shared\models`
  - Diffusion: `diffusion_models/qwen_image_2.1_int8_convrot.safetensors`
  - Text Encoder: `text_encoders/qwen3vl_8b_int8_convrot.safetensors`
  - VAE: `vae/qwen_image_2.1_vae_bf16.safetensors`
  - CLIP Vision: `clip_vision/dino_v3_L_naf_fp32.safetensors`
  - Background Removal: `background_removal/birefnet.safetensors`

## ComfyUI Headless API Patterns & Quirks
- **Prompt Format**: ComfyUI API endpoint `/prompt` expects a dictionary of node IDs mapping to `class_type` and `inputs`. It does NOT accept the UI graph format (`nodes` / `links`).
- **Asset Ingestion**: Standard ComfyUI `LoadImage` node does not read arbitrary local file paths; images must be uploaded to ComfyUI's input directory via `POST /upload/image` multipart form, or standard input folder names must be referenced.
- **WebSocket Protocol**:
  - Endpoint: `ws://<server>/ws?clientId=<client_id>`.
  - Connect WebSocket *before* submitting `POST /prompt` with `client_id` to avoid race conditions.
  - Filter out binary frames (binary preview images sent during KSampler execution) before calling `json.loads`.
  - Match `data.prompt_id == current_prompt_id` to prevent cross-talk on multi-client servers.
  - Completion signal: `executing` event with `data.node is None` and matching `prompt_id`.
  - Error detection: `execution_error` event contains node ID, exception type, and traceback.
- **Qwen 2.1 DiT Guidance & Prompt Best Practices**:
  - Qwen Image 2.1 uses rectified flow matching with Euler sampler and simple scheduler.
  - Native inference operates at `CFG = 1.0` with `negative_prompt = ""` (pure conditional velocity field evaluation, 25 steps = 25 passes).
  - Raising `CFG > 1.0` with detailed negative prompts triggers cross-attention crosstalk ("pink elephant" effect) where tokens from `<image2>` (e.g. casual tank tops, t-shirts) inadvertently contaminate the scene, stripping away campaign garments.
  - Strict attribute isolation ("Only transfer the face, facial features, hair, skin tone, and identity of the model in <image2>") and explicit preservation ("Keep all clothing, garments, tailoring, shoes, accessories, pose, and studio background from <image1> completely unchanged. Do not transfer any clothing or footwear from <image2>") prevents footwear and outfit leakage from target models while ensuring seamless identity transfer.
- **Deterministic Output Naming**:
  - Output files are generated strictly as `<campaign_stem>_<model_stem>.png`.
  - Never use static market tags (`campaign_global_*`) in batch runners, as subsequent lookbooks will silently overwrite earlier runs.
- **Dynamic Asset Ingestion**:
  - Both `data/input_campaign` and `data/input_models` dynamically scan for all valid image formats (`.png`, `.jpg`, `.jpeg`, `.webp`) without hardcoded file name prefixes.
- **Attention Masking**: When applying `IPAdapterApply` for face/model identity swap on lookbook images, wire the inpainting mask into `attn_mask` to prevent identity bleeding into background and garments.
- **Memory Management**: Send `POST /free` (`{"unload_models": False, "free_memory": True}`) between batch iterations to clear PyTorch CUDA caches without checkpoint reload latency.

## Available Workflow Templates
- `workflows/model_swap_qwen21_maskless_api.json`: **Flagship** multimodal maskless instruction-based editing workflow using native `TextEncodeQwenImage21` with V3 Autogrow (`images.image_1`, `images.image_2`) and Euler/Simple sampler at native CFG 1.0, preserving 100% of garments, shoes, pose, and background from `<image1>` while seamlessly transferring the model identity from `<image2>`.
- `workflows/model_swap_api.json`: SDXL inpainting fallback graph with IP-Adapter identity conditioning and attention mask.
- `workflows/model_swap_auto_mask_api.json`: Autonomous self-masking workflow using on-the-fly BiRefNet foreground segmentation (zero external masks required).
- `workflows/model_swap_composite_api.json`: Live integration verification workflow using `ImageCompositeMasked` (instant CPU/GPU pipeline execution).

## Verification & Test Commands
- Unified Cross-Platform Pipeline Orchestrator: `python -m scripts.run_pipeline`
- Automated ComfyUI Server Healthcheck & Launcher: `python scripts/launch_comfyui.py --check` / `--detect` / `--launch`
- Fast test suite with coverage: `python -m pytest tests --cov=scripts --cov-report=term-missing -v`
- Full local CI Quality Gate: `python scripts/run_ci_locally.py`
- Code formatting & linting: `python -m ruff check .` and `python -m ruff format --check .`
- Procedural synthetic test data generation: `python scripts/generate_testdata.py`
- Full-body aspect-ratio preserving dataset standardization: `python scripts/prepare_fullbody_dataset.py`
- Windows 1-Click Interactive Batch Runner: `run_pipeline.bat`
- Live batch execution on local ComfyUI instance:
  ```powershell
  python scripts/batch_swapper.py --server 127.0.0.1:8000 --campaign data/input_campaign/campaign_fashion_female.png --models-dir data/input_models/ --output-dir data/output/ --market-tag apac
  ```

## Commit Scopes
- `[core]` - Pipeline orchestration and client logic in `scripts/batch_swapper.py`
- `[data]` - Test data generation and dataset preprocessing in `scripts/`
- `[workflow]` - ComfyUI API graph templates in `workflows/`
- `[test]` - Test fixtures and mock integration tests in `tests/`
- `[docs]` - Architecture documentation and repository README
