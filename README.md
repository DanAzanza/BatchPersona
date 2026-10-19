# BatchPersona 🎭

**Resilient, headless Python orchestration pipeline and API/WebSocket client for ComfyUI batch production. Automates model and persona replacement in commercial advertising campaigns.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CI Quality Gate](https://github.com/DanAzanza/BatchPersona/actions/workflows/ci.yml/badge.svg)](https://github.com/DanAzanza/BatchPersona/actions/workflows/ci.yml)
[![Type Checker: Pyright](https://img.shields.io/badge/type%20checker-pyright%200%20errors-blue.svg)](pyproject.toml)
[![ComfyUI API](https://img.shields.io/badge/ComfyUI-REST%20%2F%20WebSocket-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Architecture: DiT](https://img.shields.io/badge/Diffusion-Qwen%20Image%202.1%20DiT-purple.svg)](docs/ARCHITECTURE.md)
[![Coverage: 91% Branch](https://img.shields.io/badge/test%20coverage-91%25%20branch-brightgreen.svg)](tests/)
[![Tests: 190 Passed](https://img.shields.io/badge/tests-190%20passed-success.svg)](tests/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🌟 Executive Overview

In multi-market advertising and e-commerce lookbook production, localizing campaigns for distinct demographics (APAC, LATAM, EMEA, North America) traditionally requires either costly reshoots or manual compositing by digital artists.

**BatchPersona** automates this end-to-end through a headless, resilient CLI pipeline. Given a single base campaign lookbook (photographed in studio with styling, tailoring, and lighting) and a directory of target reference model portraits, the pipeline sequentially swaps the subject's identity, face, skin tone, and hair while **preserving the apparel, tailoring, accessories, and studio background with high semantic fidelity**.

```
Base Campaign Asset (Full-Body Lookbook)
          +
Target Model Portraits (Diverse Identities)
          │
          ▼
   [BatchPersona Engine] ── Headless ComfyUI REST + WebSocket
          │
          ├── Dynamic Prompt Graph Injection
          ├── VRAM-Guarded Sequential Execution (OOM Prevention)
          ├── Asynchronous WebSocket Event Tracking
          └── Atomic Asset Export & Validation
          │
          ▼
Localized Commercial Assets (data/output/)
```

---

## 📸 Visual Showcase & Results

The pipeline generates authentic, photorealistic model replacements while maintaining pixel-level consistency on apparel tailoring, folds, textures, and studio lighting:

### Example 1: Female Fashion Lookbook Localization (APAC Market)

![APAC Market Localization](docs/images/showcase_banner_female_apac.png)

### Example 2: Male Fashion Lookbook Localization (Global / West African Persona)

![West Africa Market Localization](docs/images/showcase_banner_male_west_africa.png)

---

### Side-by-Side Model Replacement Matrix

| 1. Base Campaign (`<image1>`) | 2. Target Persona (`<image2>`) | 3. BatchPersona Output (Qwen 2.1 DiT) | Market Focus |
| :---: | :---: | :---: | :---: |
| <img src="docs/images/campaign_female.png" width="220" alt="Base Female Lookbook"> | <img src="docs/images/model_east_asia.png" width="220" alt="Target East Asian Model"> | <img src="docs/images/result_east_asia.png" width="220" alt="APAC Localized Result"> | **APAC** (East Asian Model, 100% Lookbook Kept) |
| <img src="docs/images/campaign_female.png" width="220" alt="Base Female Lookbook"> | <img src="docs/images/model_nordic.png" width="220" alt="Target Nordic Model"> | <img src="docs/images/result_nordic.png" width="220" alt="EMEA Localized Result"> | **EMEA** (Nordic Model, 100% Lookbook Kept) |
| <img src="docs/images/campaign_male.png" width="220" alt="Base Male Lookbook"> | <img src="docs/images/model_west_africa.png" width="220" alt="Target West African Model"> | <img src="docs/images/result_west_africa.png" width="220" alt="Global Localized Result"> | **GLOBAL** (West African Model, 100% Lookbook Kept) |
| <img src="docs/images/campaign_male.png" width="220" alt="Base Male Lookbook"> | <img src="docs/images/model_south_america.png" width="220" alt="Target South American Model"> | <img src="docs/images/result_south_america.png" width="220" alt="LATAM Localized Result"> | **LATAM** (South American Model, 100% Lookbook Kept) |

---

## 🏗️ Technical Architecture & Pipeline Design

> [!NOTE]
> For workflow parameter dynamics (Euler sampler, CFG 1.0, guidance dynamics), WebSocket state machine details, and memory profiling, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### 1. Maskless DiT Multimodal Conditioning vs. Traditional Inpainting

Traditional inpainting pipelines rely on binary segmentation masks and IP-Adapter / ControlNet stacks. While functional, binary masks frequently produce **seam artifacts, fringe halos, and color bleed** along garment edges, particularly with complex textures, hair strands, and delicate collars.

**BatchPersona leverages Qwen Image 2.1**, a native multimodal Diffusion Transformer (DiT):

| Parameter / Metric | Traditional SDXL + Inpaint Mask | Qwen 2.1 Maskless DiT (Flagship) | Autonomous Alpha Composite |
| :--- | :--- | :--- | :--- |
| **Workflow Template** | `model_swap_api.json` | `model_swap_qwen21_maskless_api.json` | `model_swap_auto_mask_api.json` |
| **Mask Requirement** | Mandatory 1-bit or 8-bit alpha mask | **None (Zero Mask Seams)** | None (BiRefNet / RemBg auto) |
| **Conditioning Method** | VAEEncodeForInpaint + Cross-Attn | Multimodal Cross-Attention (`<image1>`, `<image2>`) | Hard pixel compositing |
| **Garment Preservation** | Bound to mask boundary accuracy | Photorealistic semantic understanding | 100% (Alpha cutout over base) |
| **Identity Transfer** | Facial crop projection | Holistic identity (hair, skin tone, skull anatomy) | Head/Torso cutout only |
| **Hardware Overhead** | Moderate (~8 GB VRAM) | Higher compute, INT8-optimized (~12 GB VRAM) | Ultra-light (~3 GB VRAM) |

### 2. ComfyUI Headless Execution Protocol

When interfacing with ComfyUI headlessly, passing the interactive UI canvas JSON (`workflow.json`) causes execution failures because the server's execution graph engine requires the **Prompt API Schema** (node IDs mapping strictly to `class_type` and resolved input edges):

```mermaid
sequenceDiagram
    autonumber
    participant CLI as BatchSwapper (Python)
    participant REST as ComfyUI REST API (:8000/:8188)
    participant WS as ComfyUI WebSocket (/ws)
    participant GPU as PyTorch / CUDA Engine

    CLI->>REST: POST /upload/image (Base Campaign & Target Model)
    REST-->>CLI: 200 OK {"name": "virtual_asset.png"}
    CLI->>CLI: Patch Workflow Graph with Virtual Asset Names
    CLI->>WS: Connect WebSocket (clientId: UUID)
    CLI->>REST: POST /prompt {"prompt": graph, "client_id": UUID}
    REST-->>CLI: 200 OK {"prompt_id": "job_id_123"}
    
    loop Real-Time Event Loop
        WS-->>CLI: Frame: {"type": "status", "data": {...}}
        WS-->>CLI: Frame: {"type": "executing", "data": {"node": "KSampler"}}
        WS-->>CLI: Frame: {"type": "progress", "data": {"value": 15, "max": 25}}
        CLI->>CLI: Render ANSI Progress Bar / CI Milestones
    end

    WS-->>CLI: Frame: {"type": "executed", "data": {"output": {"images": [...]}}}
    CLI->>REST: GET /view?filename=...&type=output
    REST-->>CLI: Image Binary Stream (*.part staging)
    CLI->>CLI: Embed EU AI Act Provenance & Strip Workflow (process_png_provenance_metadata)
    CLI->>CLI: Atomic File Replacement (*.part -> destination.png)
    Note over CLI,GPU: Execution cache kept warm across iterations (no /free per item)
```

---

## 💻 Software Engineering Highlights

* **Layered Architecture & Strict Typing**: Modern static type annotations compliant with Python 3.10+, Pyright (0 errors / 0 warnings), immutable dataclasses, explicit generic collections, and strict separation of presentation, transport, and domain logic.
* **Fail-Fast Configuration**: `SwapperConfig` validates file existence, directory access, and timeout bounds immediately at initialization, avoiding runtime failures halfway through a batch.
* **Deadlock-Free WebSockets**: `track_execution` enforces non-blocking socket timeouts (`ws.settimeout(min(2.0, remaining))`) with overall wall-clock deadlines to prevent infinite hangs if the ComfyUI backend stalls.
* **Warm-Cache Execution & OOM Resilience**: BatchPersona keeps ComfyUI's execution cache warm across batch iterations to eliminate redundant re-encoding of base campaign lookbooks and text conditioning. Memory reclamation (`/free` with `unload_models=True`) is invoked reactively upon CUDA Out-Of-Memory (OOM) recovery or deterministically in the session teardown at the end of `run_batch()` to release all VRAM.
* **Smart Local Housekeeping**: Safely purges intermediate uploads and generated outputs from ComfyUI directories to prevent storage bloat, protected by canonical path checks, pre-existing asset guards, and Windows lock retries.
* **Atomic File Writes**: Image assets are streamed to unique temporary staging files (`.part`) and atomically replaced into the target directory to prevent truncated assets during crashes or network interruptions.
* **CI-Aware Progress Logging**: Automatically senses interactive TTY vs. headless CI environments, toggling between an in-place ANSI progress bar and clean 25% milestone logging to eliminate log pollution.

---

## 📂 Repository Structure

```text
BatchPersona/
├── .github/
│   └── workflows/ci.yml         # Linux (Ubuntu / Python 3.12) CI quality gate
├── data/
│   ├── input_campaign/          # Standardized 896x1152 fashion lookbooks
│   │   ├── campaign_fashion_female.png
│   │   └── campaign_fashion_male.png
│   ├── input_models/            # Diverse multi-ethnic full-body portraits
│   │   ├── model_east_asia_f01.png
│   │   ├── model_west_africa_m01.png
│   │   ├── model_nordic_f01.png
│   │   └── model_south_america_m01.png
│   └── output/                  # Localized campaign deliverables
├── docs/
│   ├── ARCHITECTURE.md          # 📐 Mathematical derivations & systems architecture
│   └── images/                  # Showcase banners and comparison matrix assets
├── workflows/
│   ├── model_swap_qwen21_maskless_api.json  # 🌟 Flagship Qwen 2.1 DiT API graph
│   ├── model_swap_api.json                  # SDXL Inpainting fallback graph
│   ├── model_swap_auto_mask_api.json        # RemBg/BiRefNet autonomous masking
│   └── model_swap_composite_api.json        # Lightweight compositor fallback
├── src/batchpersona/            # 📦 Core production package
│   ├── __init__.py              # Unified library exports & metadata
│   ├── __main__.py              # CLI module entrypoint (python -m batchpersona)
│   ├── ci.py                    # CI test orchestration and parity checks
│   ├── client.py                # ComfyUI REST & WebSocket state machine client (HTTP/HTTPS, WS/WSS)
│   ├── housekeeper.py           # Storage de-duplication & intermediate asset cleanup
│   ├── launcher.py              # Automated server detection, launch & shutdown
│   ├── metadata.py              # EU AI Act Article 50 provenance metadata & zero-recompression chunk processor
│   ├── models.py                # Strongly typed domain dataclasses & exceptions
│   ├── orchestrator.py          # Batch model swapping pipeline engine
│   ├── packaging.py             # Deterministic distribution archive builder
│   ├── pipeline.py              # Interactive management console & CLI runner
│   ├── preprocess.py            # Centering-aware dataset preprocessor (ImageOps.fit)
│   └── template.py              # Workflow graph loading, mutation & node validation
├── scripts/                     # 🔌 Backward-compatible CLI shims
│   ├── batch_swapper.py         # Forwarding shim -> batchpersona.orchestrator
│   ├── housekeeper.py           # Forwarding shim -> batchpersona.housekeeper
│   ├── launch_comfyui.py        # Forwarding shim -> batchpersona.launcher
│   ├── package_release.py       # Forwarding shim -> batchpersona.packaging
│   ├── prepare_fullbody_dataset.py # Forwarding shim -> batchpersona.preprocess
│   ├── run_ci_locally.py        # Forwarding shim -> batchpersona.ci
│   └── run_pipeline.py          # Forwarding shim -> batchpersona.pipeline
├── tests/                       # 🧪 Automated test suites
│   ├── conftest.py              # Shared test fixtures and ComfyUI client mocks
│   ├── test_cache.py            # Cache invalidation & skip-existing tests
│   ├── test_ci_parity.py        # CI parity, AST compatibility & packaging checks
│   ├── test_cli.py              # CLI argument parsing, flags & exit code tests
│   ├── test_client.py           # HTTP multipart, WebSocket framing, TLS & metadata tests
│   ├── test_housekeeper.py      # Storage housekeeping & safety guard tests
│   ├── test_launch_comfyui.py   # Server launcher & headless detection tests
│   ├── test_metadata.py         # EU AI Act Article 50 & zero-recompression chunk tests
│   ├── test_models.py           # SwapperConfig validation & market localization tests
│   ├── test_orchestrator.py     # Batch swapper, skip-existing & execution tests
│   ├── test_packaging.py        # Standalone archive & file integrity tests
│   ├── test_preprocess.py       # Centering-aware dataset preprocessor test fixtures
│   ├── test_run_pipeline.py     # Interactive runner orchestration & menu tests
│   ├── test_server_lifecycle.py # Server supervision, PID management & termination tests
│   └── test_template.py         # Workflow template validation & node patching tests
├── install.bat                  # ⚡ Windows 1-Click Environment Setup & Self-Test
├── install.sh                   # ⚡ Linux/macOS 1-Click Environment Setup & Self-Test
├── run_pipeline.bat             # 🚀 Windows Interactive Management & Batch Runner
├── run_pipeline.sh              # 🚀 Linux/macOS Interactive Management & Batch Runner
├── .env.example                 # Configuration template (COMFYUI_SERVER, timeout)
├── pyproject.toml               # Modern packaging, CLI scripts, pytest, ruff & pyright
├── requirements.txt             # Minimal production dependencies
├── .gitignore                   # Caches, virtual environments, and temporary artifacts
├── LICENSE                      # MIT License
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. One-Click Automated Setup (Zero Friction)

BatchPersona provides automated installers that configure a dedicated virtual environment, install dependencies, synthesize initial lookbooks, and run a self-test suite:

* **Windows**: Double-click **`install.bat`** (or execute `cmd /c install.bat`).
* **Linux / macOS**:
  ```bash
  chmod +x install.sh run_pipeline.sh
  ./install.sh
  ```

*Manual Setup Fallback:*
```bash
git clone https://github.com/DanAzanza/BatchPersona.git
cd BatchPersona
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Interactive Pipeline Runner

Launch the interactive management console to execute campaigns, run zero-GPU dry runs, or inspect tests:

* **Windows**: Double-click **`run_pipeline.bat`**
* **Linux / macOS**: Execute **`./run_pipeline.sh`**

```text
===========================================================================
      BatchPersona - Headless ComfyUI Model Replacement Pipeline
===========================================================================
  Python runtime: .venv\Scripts\python.exe
  Server address: 127.0.0.1:8000
===========================================================================

  [1] Run Commercial Model Swap  (Campaign Lookbooks - Diverse Personas)
  [2] Run Quick Dry-Run          (Instant Zero-GPU Verification, < 1s)
  [3] Run Quality Gate and Tests (Pytest + Coverage + Ruff)
  [4] Close ComfyUI Console      (Free GPU VRAM & Terminate Server)
  [0] Exit
```

Or pass arguments directly through the runner from your terminal:

```cmd
run_pipeline.bat --campaign data\input_campaign\campaign_fashion_female.png --models-dir data\input_models
```

### 3. Preprocessing Full-Body High-Fidelity Lookbooks

To standardize high-resolution campaign assets to the canonical 3:4 diffusion aspect ratio (`896x1152`) without squashing or cutting off heads:

```bash
python scripts/prepare_fullbody_dataset.py --source-dir path/to/raw_photos --output-dir data
```

---

## 🕹️ CLI Runner & Orchestration

### Headless Batch Execution

Run the model replacement pipeline against an active ComfyUI instance (via standard script invocation or package entrypoint `batchpersona`):

```bash
# Standard Python invocation:
python scripts/batch_swapper.py \
  --server 127.0.0.1:8000 \
  --campaign data/input_campaign/campaign_fashion_female.png \
  --models-dir data/input_models \
  --output-dir data/output \
  --workflow workflows/model_swap_qwen21_maskless_api.json \
  --market-tag apac \
  --timeout 300.0

# Or via module execution:
python -m batchpersona --campaign data/input_campaign/campaign_fashion_female.png --models-dir data/input_models

# Or via package console entrypoint (when installed with `pip install -e .`):
batchpersona --campaign data/input_campaign/campaign_fashion_female.png --models-dir data/input_models
```

### CLI Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--server` | `str` | `127.0.0.1:8000` | ComfyUI host address and port (`host:port`). Default is `127.0.0.1:8000` (Comfy Desktop); portable/git ComfyUI uses `8188`. |
| `--campaign` | `Path` | *Required* | Path to base campaign lookbook advertisement. |
| `--models-dir` | `Path` | *Required* | Directory containing target model reference portraits. |
| `--output-dir` | `Path` | `data/output` | Destination directory for localized campaign assets. |
| `--workflow` | `Path` | `workflows/model_swap_qwen21_maskless_api.json` | ComfyUI API prompt JSON template. |
| `--market-tag` | `str` | `global` | Campaign market identifier metadata (`apac`, `latam`, `emea`). When specified, output filenames follow `<campaign>_<tag>_<model>.png`. |
| `--mask` | `Path` | `None` | Path to optional manual segmentation mask. |
| `--filter` | `str` | `*` | Glob pattern to filter models inside `--models-dir` (e.g. `*east_asia*`). |
| `--timeout` | `float` | `300.0` | Execution timeout in seconds per individual asset. |
| `--skip-existing` | `flag` | `False` | Skip models whose output asset already exists in `--output-dir` (size > 0). |
| `--force` | `flag` | `False` | Force re-rendering even if output asset already exists. |
| `--seed` | `int` | `None` | Override diffusion sampler seed (0 to 2^64 - 1). |
| `--resolution` | `int` | `None` | Override model resolution (e.g. 1024, 1280). |
| `--prompt` | `str` | `None` | Override positive instruction prompt (preserves `<image1>` and `<image2>` tokens). |
| `--keep-workflow-metadata` | `flag` | `False` | Retain internal ComfyUI prompt and workflow graph chunks in exported PNG deliverables. |
| `--no-cleanup` | `flag` | `False` | Disable automatic purging of intermediate ComfyUI assets. |
| `--comfy-input-dir` | `Path` | `None` | Override ComfyUI input directory path for housekeeping. |
| `--comfy-output-dir` | `Path` | `None` | Override ComfyUI output directory path for housekeeping. |

---

## 🧪 Verification & Test Suite

The automated test suite provides **91% branch coverage (190 passing tests)** without requiring an active GPU or live ComfyUI instance by leveraging mocked WebSocket frame streams, mock REST sessions, and procedural image fixtures.

Run the complete test suite with coverage report:

```bash
python -m pytest tests --cov=batchpersona --cov=scripts --cov-report=term-missing -v
```

### GitHub Actions CI Parity (Run CI Locally)

Mirror the exact GitHub Actions multi-stage CI pipeline deterministically on your local machine with a single command:

```bash
python scripts/run_ci_locally.py
```

This sequentially runs:
1. `ruff check .` (PEP compliance static linting)
2. `ruff format --check .` (Code style formatting verification)
3. `pyright src scripts tests` (Static type analysis across src, scripts, tests)
4. `pytest tests --cov=batchpersona --cov=scripts --cov-report=term-missing -v` (Full unit & integration regression suite)
5. Automatic cleanup of test artifacts

### Linting & Static Typing Quality Gate

Verify compliance with strict PEP standards and static typing:

```bash
# Code style and linting (0 warnings)
python -m ruff check .
python -m ruff format --check .

# Static type analysis across src, scripts, and tests (0 errors, 0 warnings)
python -m pyright src scripts tests
```

---

## ⚖️ Legal, Ethical & AI Transparency Notice

### EU AI Act Compliance (Article 50) & Commercial Deliverable Sanitization

BatchPersona implements automated post-download PNG metadata processing (`src/batchpersona/metadata.py`) to satisfy EU AI Act transparency requirements while safeguarding proprietary pipeline graphs:

1. **Machine-Readable Provenance Metadata**:
   In accordance with Article 50 of the European Union Artificial Intelligence Act (effective August 2026) regarding synthetic media transparency, every exported PNG embeds standard machine-readable text chunks:
   - `Software`: `"BatchPersona"`
   - `AI-Generated`: `"true"`
   - `AI-System`: `"BatchPersona / ComfyUI"`
   - `AI-Compliance`: `"EU-AI-Act-Article-50-Synthetic-Media"`
   - `Market-Tag`: `<market_tag>` (e.g. `"apac"`, `"latam"`, `"emea"`, or `"global"`)

2. **Proprietary Workflow & Prompt Stripping**:
   By default, ComfyUI embeds complete internal execution graphs and node parameters into saved PNGs (`prompt` and `workflow` text chunks). BatchPersona automatically strips these proprietary chunks from downloaded deliverables so that client deliverables do not leak internal node structures, custom model names, or prompt engineering IP.
   - Use `--keep-workflow-metadata` if you explicitly wish to retain internal ComfyUI workflow chunks for debugging or archiving.

### Model Rights & Informed Consent
* Replacing human likenesses in advertising campaigns requires explicit, documented contractual consent from all participating human models whose physical appearance or photographic likeness is utilized as conditioning input.
* Do not utilize likeness assets of individuals without verifiable commercial authorization.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.
