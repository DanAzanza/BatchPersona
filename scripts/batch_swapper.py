"""Production-grade headless batch-processing orchestrator for ComfyUI model replacement.

Coordinates REST API requests, WebSocket event tracking, dynamic workflow patching,
and asset ingestion/export for automated advertising campaign localization.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
import types
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

# Ensure repository root is on sys.path when executed directly as a script
REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure Windows terminal standard streams handle UTF-8 properly
if sys.platform == "win32":
    reconfig_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfig_stdout):
        reconfig_stdout(encoding="utf-8")
    reconfig_stderr = getattr(sys.stderr, "reconfigure", None)
    if callable(reconfig_stderr):
        reconfig_stderr(encoding="utf-8")

import requests  # noqa: E402
import websocket  # noqa: E402

from scripts.housekeeper import ComfyUIHousekeeper, create_housekeeper  # noqa: E402

LOGGER = logging.getLogger("batch_swapper")

# ANSI Color codes for high-visibility terminal output
COLOR_RESET: Final[str] = "\033[0m"
COLOR_CYAN: Final[str] = "\033[36m"
COLOR_GREEN: Final[str] = "\033[32m"
COLOR_YELLOW: Final[str] = "\033[33m"
COLOR_RED: Final[str] = "\033[31m"
COLOR_BOLD: Final[str] = "\033[1m"


class JobStatus(str, Enum):
    """Execution status of an individual model swap job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class PipelineError(Exception):
    """Base domain exception for ComfyUI pipeline failures."""


class ComfyAPIError(PipelineError):
    """Exception raised when ComfyUI REST API returns a non-200 status code."""


class ComfyExecutionError(PipelineError):
    """Exception raised when ComfyUI returns an execution_error WebSocket event."""


class ComfyTimeoutError(PipelineError):
    """Exception raised when an execution exceeds the maximum allocated duration."""


@dataclass(frozen=True)
class SwapperConfig:
    """Immutable configuration parameters for the batch swapper."""

    server_address: str
    campaign_path: Path
    models_dir: Path
    output_dir: Path
    workflow_path: Path
    mask_path: Path | None = None
    market_tag: str = "global"
    timeout_seconds: float = 300.0
    model_filter: str = "*"
    validate_paths: bool = True
    cleanup: bool = True
    comfy_input_dir: Path | None = None
    comfy_output_dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate configuration paths and parameters fail-fast."""
        if self.validate_paths:
            if not self.campaign_path.is_file():
                raise FileNotFoundError(f"Campaign image does not exist: {self.campaign_path}")
            if not self.models_dir.is_dir():
                raise NotADirectoryError(f"Models directory does not exist: {self.models_dir}")
            if not self.workflow_path.is_file():
                raise FileNotFoundError(f"Workflow file does not exist: {self.workflow_path}")
            if self.mask_path is not None and not self.mask_path.is_file():
                raise FileNotFoundError(f"Mask file does not exist: {self.mask_path}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"Timeout must be positive, got {self.timeout_seconds}")


@dataclass(frozen=True)
class OutputAsset:
    """Descriptor for an output image asset returned by ComfyUI."""

    filename: str
    subfolder: str
    image_type: str


@dataclass(frozen=True)
class JobResult:
    """Outcome and diagnostics of a model swap execution."""

    model_name: str
    output_path: Path | None
    status: JobStatus
    duration_seconds: float
    error_message: str | None = None


def configure_logging(level: int = logging.INFO) -> None:
    """Initialize structured console logging format."""
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


class WorkflowTemplate:
    """Handles loading, validation, and dynamic mutation of the ComfyUI API workflow graph."""

    def __init__(self, raw_graph: dict[str, Any]) -> None:
        self._validate_graph(raw_graph)
        self._graph = raw_graph

    @classmethod
    def load_from_file(cls, file_path: Path) -> WorkflowTemplate:
        """Load and parse workflow graph from a JSON file."""
        if not file_path.is_file():
            raise FileNotFoundError(f"Workflow file does not exist: {file_path}")
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    @staticmethod
    def _validate_graph(graph: dict[str, Any]) -> None:
        """Verify graph adheres to ComfyUI Prompt API schema."""
        if not isinstance(graph, dict) or not graph:
            raise ValueError("Workflow template must be a non-empty dictionary.")
        for node_id, node_def in graph.items():
            if not isinstance(node_def, dict):
                raise TypeError(f"Node '{node_id}' definition must be a dictionary.")
            if "class_type" not in node_def:
                raise ValueError(f"Node '{node_id}' missing required key 'class_type'.")
            if "inputs" not in node_def:
                raise ValueError(f"Node '{node_id}' missing required key 'inputs'.")

    def requires_mask(self) -> bool:
        """Check whether the workflow template expects an external mask image."""
        serialized = json.dumps(self._graph)
        return "{{MASK_IMG}}" in serialized

    def patch(
        self,
        campaign_image_name: str,
        model_image_name: str,
        output_prefix: str,
        mask_image_name: str | None = None,
    ) -> dict[str, Any]:
        """Produce a patched clone of the workflow graph with injected asset references."""
        if self.requires_mask() and mask_image_name is None:
            raise ValueError(
                "Workflow template requires an external inpainting mask ({{MASK_IMG}}), but none was provided or auto-paired."
            )

        patched: dict[str, Any] = json.loads(json.dumps(self._graph))

        # 1. Update any SaveImage node with the unique output prefix
        for node_def in patched.values():
            if isinstance(node_def, dict) and node_def.get("class_type") == "SaveImage":
                node_def.setdefault("inputs", {})["filename_prefix"] = output_prefix

        # 2. Update well-known node targets if present
        if "1" in patched and patched["1"].get("class_type") == "LoadImage":
            patched["1"]["inputs"]["image"] = campaign_image_name

        if (
            "2" in patched
            and patched["2"].get("class_type") == "LoadImage"
            and "3" in patched
            and patched["3"].get("class_type") == "LoadImage"
        ):
            if mask_image_name is not None:
                patched["2"]["inputs"]["image"] = mask_image_name
            patched["3"]["inputs"]["image"] = model_image_name
        elif (
            "2" in patched
            and patched["2"].get("class_type") == "LoadImage"
            and patched.get("3", {}).get("class_type") != "LoadImage"
        ):
            patched["2"]["inputs"]["image"] = model_image_name

        # 3. Support string template fallback substitution
        placeholders: dict[str, str] = {
            "{{CAMPAIGN_IMG}}": campaign_image_name,
            "{{MODEL_IMG}}": model_image_name,
            "{{OUTPUT_PREFIX}}": output_prefix,
        }
        if mask_image_name is not None:
            placeholders["{{MASK_IMG}}"] = mask_image_name

        self._substitute_placeholders_recursive(patched, placeholders)
        return patched

    def _substitute_placeholders_recursive(
        self,
        target: Any,
        placeholders: dict[str, str],
    ) -> None:
        """Recursively scan dict/list in-place and replace placeholder string tokens."""
        if isinstance(target, dict):
            for key, val in target.items():
                if isinstance(val, str) and val in placeholders:
                    target[key] = placeholders[val]
                elif isinstance(val, (dict, list)):
                    self._substitute_placeholders_recursive(val, placeholders)
        elif isinstance(target, list):
            for i, val in enumerate(target):
                if isinstance(val, str) and val in placeholders:
                    target[i] = placeholders[val]
                elif isinstance(val, (dict, list)):
                    self._substitute_placeholders_recursive(val, placeholders)


class ComfyUIClient:
    """REST and WebSocket client for headless interaction with ComfyUI."""

    def __init__(
        self,
        server_address: str,
        session: requests.Session | None = None,
        ws_factory: Callable[[str, float], Any] | None = None,
    ) -> None:
        self._server = server_address.rstrip("/")
        self._http_base = f"http://{self._server}"
        self._ws_base = f"ws://{self._server}"
        self._session = session or requests.Session()
        self._ws_factory = ws_factory or self._default_ws_connect

    @staticmethod
    def _default_ws_connect(url: str, timeout: float) -> Any:
        return websocket.create_connection(url, timeout=timeout)

    def check_connection(self, timeout: float = 3.0) -> bool:
        """Check if ComfyUI REST server is reachable and responsive."""
        try:
            resp = self._session.get(f"{self._http_base}/system_stats", timeout=timeout)
            return resp.status_code == 200
        except (requests.RequestException, OSError):
            return False

    def upload_image(self, file_path: Path, image_type: str = "input") -> str:
        """Upload image to ComfyUI /upload/image endpoint. Returns stored filename."""
        if not file_path.is_file():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        url = f"{self._http_base}/upload/image"
        with open(file_path, "rb") as f:
            files = {"image": (file_path.name, f, "image/png")}
            data = {"type": image_type, "overwrite": "true"}
            response = self._session.post(url, files=files, data=data, timeout=30.0)

        if response.status_code != 200:
            raise ComfyAPIError(f"Upload failed ({response.status_code}): {response.text}")

        res_json = response.json()
        uploaded_name = res_json.get("name", file_path.name)
        LOGGER.info("Uploaded %s -> ComfyUI virtual asset: %s", file_path.name, uploaded_name)
        return str(uploaded_name)

    def queue_prompt(self, prompt: dict[str, Any], client_id: str) -> str:
        """Submit prompt graph to ComfyUI /prompt endpoint. Returns prompt_id."""
        url = f"{self._http_base}/prompt"
        payload = {"prompt": prompt, "client_id": client_id}
        response = self._session.post(url, json=payload, timeout=30.0)

        if response.status_code != 200:
            raise ComfyAPIError(f"Failed to queue prompt ({response.status_code}): {response.text}")

        res_json = response.json()
        prompt_id = res_json.get("prompt_id")
        if not prompt_id:
            raise ComfyAPIError(f"Missing prompt_id in response: {res_json}")

        return str(prompt_id)

    def download_image(self, asset: OutputAsset, destination: Path) -> None:
        """Download output asset from ComfyUI /view endpoint directly to destination path."""
        url = f"{self._http_base}/view"
        params = {
            "filename": asset.filename,
            "subfolder": asset.subfolder,
            "type": asset.image_type,
        }
        response = self._session.get(url, params=params, stream=True, timeout=60.0)
        if response.status_code != 200:
            raise ComfyAPIError(
                f"Failed to download asset ({response.status_code}): {response.text}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_dest = destination.with_suffix(f"{destination.suffix}.part")
        try:
            with open(temp_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            # Atomic replacement avoiding Windows file lock race conditions
            temp_dest.replace(destination)
        finally:
            if temp_dest.exists():
                with contextlib.suppress(OSError):
                    temp_dest.unlink()

    def interrupt_execution(self) -> None:
        """Send interrupt signal to ComfyUI to halt running GPU work."""
        url = f"{self._http_base}/interrupt"
        try:
            self._session.post(url, timeout=5.0)
            LOGGER.info("Dispatched /interrupt signal to ComfyUI server.")
        except (requests.RequestException, OSError) as e:
            LOGGER.warning("Could not send /interrupt signal: %s", e)

    def free_memory(self, unload_models: bool = False) -> None:
        """Trigger CUDA cache freeing and optional checkpoint unloads between executions."""
        url = f"{self._http_base}/free"
        try:
            self._session.post(
                url,
                json={"unload_models": unload_models, "free_memory": True},
                timeout=5.0,
            )
        except (requests.RequestException, OSError) as e:
            LOGGER.debug("Could not trigger /free memory signal: %s", e)

    def close(self) -> None:
        """Close underlying HTTP session connection pools."""
        self._session.close()

    def __enter__(self) -> ComfyUIClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.close()

    def track_execution(
        self,
        prompt_id: str,
        client_id: str,
        timeout_seconds: float,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> list[OutputAsset]:
        """Listen to WebSocket events until prompt completes, fails, or times out."""
        ws_url = f"{self._ws_base}/ws?clientId={client_id}"
        ws = self._ws_factory(ws_url, timeout_seconds)
        start_time = time.time()
        output_assets: list[OutputAsset] = []

        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    raise ComfyTimeoutError(
                        f"Execution exceeded timeout limit ({timeout_seconds:.1f}s)"
                    )

                # Periodic socket timeout to prevent indefinite hangs on stalled servers
                remaining = max(0.1, timeout_seconds - elapsed)
                settimeout_fn = getattr(ws, "settimeout", None)
                if callable(settimeout_fn):
                    settimeout_fn(min(2.0, remaining))

                try:
                    raw_frame = ws.recv()
                except (
                    websocket.WebSocketTimeoutException,
                    TimeoutError,
                    OSError,
                ) as t_err:
                    if time.time() - start_time > timeout_seconds:
                        raise ComfyTimeoutError(
                            f"Execution exceeded timeout limit ({timeout_seconds:.1f}s)"
                        ) from t_err
                    continue

                if isinstance(raw_frame, bytes):
                    # Skip binary latent preview frames
                    continue

                try:
                    event = json.loads(raw_frame)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                event_data = event.get("data", {})

                # Ensure event pertains to the current execution
                if event_data.get("prompt_id") != prompt_id:
                    continue

                if event_type == "execution_error":
                    node_id = event_data.get("node_id")
                    node_type = event_data.get("node_type")
                    exc_msg = event_data.get("exception_message")
                    raise ComfyExecutionError(
                        f"ComfyUI node [{node_id}:{node_type}] failed: {exc_msg}"
                    )

                if event_type == "progress":
                    current_step = event_data.get("value", 0)
                    max_step = event_data.get("max", 1)
                    node_id = event_data.get("node", "sampler")
                    if on_progress:
                        on_progress(current_step, max_step, str(node_id))

                if event_type == "executed":
                    output_data = event_data.get("output", {})
                    images = output_data.get("images", [])
                    for img_info in images:
                        output_assets.append(
                            OutputAsset(
                                filename=img_info.get("filename", ""),
                                subfolder=img_info.get("subfolder", ""),
                                image_type=img_info.get("type", "output"),
                            )
                        )

                if event_type == "executing" and event_data.get("node") is None:
                    # None signifies workflow execution completed
                    break

        finally:
            try:
                ws.close()
            except (websocket.WebSocketException, OSError) as ws_err:
                LOGGER.debug("Error while closing websocket: %s", ws_err)

        return output_assets


class BatchSwapper:
    """High-level batch pipeline orchestrator managing localized model replacement."""

    def __init__(
        self,
        config: SwapperConfig,
        client: ComfyUIClient | None = None,
        template: WorkflowTemplate | None = None,
        housekeeper: ComfyUIHousekeeper | None = None,
    ) -> None:
        self._config = config
        self._client = client or ComfyUIClient(config.server_address)
        self._template = template or WorkflowTemplate.load_from_file(config.workflow_path)
        self._housekeeper = housekeeper or create_housekeeper(
            server_address=config.server_address,
            enabled=config.cleanup,
            input_dir=config.comfy_input_dir,
            output_dir=config.comfy_output_dir,
        )

    def _resolve_mask_path(self) -> Path | None:
        """Resolve mask path explicitly or via automatic matching pattern."""
        if self._config.mask_path is not None and self._config.mask_path.is_file():
            return self._config.mask_path

        # Auto-pairing pattern: look for <campaign_stem>_mask.png in the campaign folder
        candidate = self._config.campaign_path.with_name(
            f"{self._config.campaign_path.stem}_mask.png"
        )
        if candidate.is_file():
            LOGGER.info("Auto-paired matching mask for campaign: %s", candidate.name)
            return candidate
        return None

    def _render_progress_bar(self, step: int, max_steps: int, node: str) -> None:
        """Render in-place terminal progress bar or structured log in headless/CI environments."""
        if max_steps <= 0:
            return
        percent = min(100.0, (step / max_steps) * 100.0)

        if not sys.stdout.isatty():
            # Headless CI environment: emit structured log at 25% checkpoints
            if step % max(1, max_steps // 4) == 0 or step == max_steps:
                LOGGER.info(
                    "[SAMPLER Node %s] Progress: %5.1f%% (%d/%d)",
                    node,
                    percent,
                    step,
                    max_steps,
                )
            return

        bar_length = 30
        filled = int((percent / 100.0) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        sys.stdout.write(
            f"\r{COLOR_CYAN}[SAMPLER Node {node}]{COLOR_RESET} |{bar}| {percent:5.1f}% ({step}/{max_steps})"
        )
        sys.stdout.flush()

    def process_single_model(
        self,
        model_file: Path,
        campaign_asset_name: str,
        mask_asset_name: str | None,
    ) -> JobResult:
        """Execute single model replacement job sequentially."""
        model_stem = model_file.stem
        output_prefix = f"{self._config.campaign_path.stem}_{model_stem}"
        job_start = time.time()
        client_id = str(uuid.uuid4())
        is_oom = False

        LOGGER.info(
            "%sStarting swap job for model:%s %s%s%s (Target: %s)",
            COLOR_BOLD,
            COLOR_RESET,
            COLOR_CYAN,
            model_file.name,
            COLOR_RESET,
            output_prefix,
        )

        model_asset_name: str | None = None
        try:
            # 1. Upload model reference asset
            self._housekeeper.record_pre_upload(model_file.name)
            model_asset_name = self._client.upload_image(model_file)
            self._housekeeper.record_uploaded_name(model_asset_name)

            # 2. Patch workflow graph
            prompt_payload = self._template.patch(
                campaign_image_name=campaign_asset_name,
                model_image_name=model_asset_name,
                output_prefix=output_prefix,
                mask_image_name=mask_asset_name,
            )

            # 3. Queue prompt
            prompt_id = self._client.queue_prompt(prompt_payload, client_id=client_id)
            LOGGER.info("Prompt accepted by ComfyUI. Assigned ID: %s", prompt_id)

            # 4. Stream execution events
            outputs = self._client.track_execution(
                prompt_id=prompt_id,
                client_id=client_id,
                timeout_seconds=self._config.timeout_seconds,
                on_progress=self._render_progress_bar,
            )
            # Clear line after progress
            if sys.stdout.isatty():
                sys.stdout.write("\r" + " " * 70 + "\r")
                sys.stdout.flush()

            if not outputs:
                raise ComfyAPIError("Job completed but SaveImage node produced zero output assets.")

            # 5. Download primary output
            primary_asset = outputs[0]
            final_target = self._config.output_dir / f"{output_prefix}.png"
            self._client.download_image(primary_asset, final_target)

            # 6. Housekeeping: Purge duplicate output asset from ComfyUI directory
            self._housekeeper.cleanup_output_asset(
                asset_filename=primary_asset.filename,
                asset_subfolder=primary_asset.subfolder,
                final_target=final_target,
            )

            elapsed = time.time() - job_start

            LOGGER.info(
                "%s[PASS]%s Successfully rendered %s (%.2fs)",
                COLOR_GREEN,
                COLOR_RESET,
                final_target.name,
                elapsed,
            )
            return JobResult(
                model_name=model_file.name,
                output_path=final_target,
                status=JobStatus.SUCCESS,
                duration_seconds=elapsed,
            )

        except (
            PipelineError,
            requests.RequestException,
            websocket.WebSocketException,
            OSError,
            ValueError,
            RuntimeError,
        ) as e:
            is_oom = "out of memory" in str(e).lower()
            elapsed = time.time() - job_start
            sys.stdout.write("\n")
            LOGGER.error(
                "%s[FAIL]%s Model %s failed: %s",
                COLOR_RED,
                COLOR_RESET,
                model_file.name,
                e,
            )
            self._client.interrupt_execution()
            return JobResult(
                model_name=model_file.name,
                output_path=None,
                status=JobStatus.FAILED,
                duration_seconds=elapsed,
                error_message=str(e),
            )
        finally:
            self._client.free_memory(unload_models=is_oom)
            if model_asset_name:
                self._housekeeper.cleanup_model_input(model_asset_name, model_file)

    def run_batch(self) -> list[JobResult]:
        """Execute model swap pipeline sequentially across all portraits."""
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        model_files = sorted(
            [
                p
                for p in self._config.models_dir.glob(self._config.model_filter)
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]
        )

        if not model_files:
            LOGGER.warning("No valid portrait images found in %s", self._config.models_dir)
            return []

        LOGGER.info(
            "Discovered %d target model portraits in %s",
            len(model_files),
            self._config.models_dir,
        )

        # Pre-flight ComfyUI server connectivity check
        if not self._client.check_connection():
            raise ComfyAPIError(
                f"Cannot connect to ComfyUI at {self._client._http_base}.\n"
                f"Troubleshooting tips:\n"
                f"  1. Verify ComfyUI is running.\n"
                f"  2. Check server port: ComfyUI Desktop typically runs on 8000, while portable/git uses 8188.\n"
                f"  3. Pass --server 127.0.0.1:<port> or export COMFYUI_SERVER env var."
            )

        # Ingest campaign base image
        LOGGER.info("Ingesting base campaign asset: %s", self._config.campaign_path)
        self._housekeeper.record_pre_upload(self._config.campaign_path.name)
        campaign_asset_name = self._client.upload_image(self._config.campaign_path)
        self._housekeeper.record_uploaded_name(campaign_asset_name)

        # Ingest inpainting mask if required or auto-detected
        mask_asset_name: str | None = None
        mask_file: Path | None = None
        if self._template.requires_mask():
            mask_file = self._resolve_mask_path()
            if mask_file is not None and mask_file.is_file():
                LOGGER.info("Ingesting inpainting mask: %s", mask_file)
                self._housekeeper.record_pre_upload(mask_file.name)
                mask_asset_name = self._client.upload_image(mask_file)
                self._housekeeper.record_uploaded_name(mask_asset_name)
            else:
                LOGGER.warning(
                    "Workflow expects mask placeholder, but no matching mask found for %s",
                    self._config.campaign_path.name,
                )
        else:
            LOGGER.info(
                "Workflow operates without external mask input (autonomous self-masking or maskless Qwen mode)."
            )

        results: list[JobResult] = []
        try:
            for index, model_path in enumerate(model_files, start=1):
                LOGGER.info(
                    "\n%s=== Processing Asset [%d/%d]: %s ===%s",
                    COLOR_BOLD,
                    index,
                    len(model_files),
                    model_path.name,
                    COLOR_RESET,
                )
                res = self.process_single_model(
                    model_file=model_path,
                    campaign_asset_name=campaign_asset_name,
                    mask_asset_name=mask_asset_name,
                )
                results.append(res)
        finally:
            self._housekeeper.cleanup_shared_inputs(
                [
                    (campaign_asset_name, self._config.campaign_path),
                    (mask_asset_name, mask_file),
                ]
            )

        self._print_batch_summary(results)
        return results

    def _print_batch_summary(self, results: list[JobResult]) -> None:
        """Print formatted execution summary table."""
        print(f"\n{COLOR_BOLD}{'=' * 75}{COLOR_RESET}")
        print(f"{COLOR_BOLD}COMFYUI BATCH MODEL SWAP - EXECUTION SUMMARY{COLOR_RESET}")
        print(f"{COLOR_BOLD}{'=' * 75}{COLOR_RESET}")
        print(f"{'MODEL':<28} | {'STATUS':<9} | {'TIME':<8} | {'EXPORT PATH / ERROR'}")
        print(f"{'-' * 28}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 24}")

        for r in results:
            status_color = COLOR_GREEN if r.status == JobStatus.SUCCESS else COLOR_RED
            status_str = f"{status_color}{r.status.value:<9}{COLOR_RESET}"
            detail = r.output_path.name if r.output_path else (r.error_message or "Unknown error")
            print(f"{r.model_name:<28} | {status_str} | {r.duration_seconds:6.2f}s | {detail}")

        passed = sum(1 for r in results if r.status == JobStatus.SUCCESS)
        total = len(results)
        summary_color = COLOR_GREEN if passed == total else COLOR_YELLOW
        print(f"{'-' * 75}")
        print(
            f"Total: {total} | {summary_color}Passed: {passed}{COLOR_RESET} | Failed: {total - passed}\n"
        )


def parse_cli_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Production Headless ComfyUI Batch Model Replacement Pipeline"
    )
    default_server = os.environ.get("COMFYUI_SERVER", "127.0.0.1:8188")
    parser.add_argument(
        "--server",
        default=default_server,
        help=f"ComfyUI server address (host:port). Default: $COMFYUI_SERVER or {default_server}",
    )
    parser.add_argument(
        "--campaign",
        required=True,
        type=Path,
        help="Path to the base campaign advertisement image",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Path to segmentation / inpainting mask (optional)",
    )
    parser.add_argument(
        "--models-dir",
        required=True,
        type=Path,
        help="Directory containing target reference portrait images",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data/output"),
        type=Path,
        help="Export target directory for localized campaign assets",
    )
    parser.add_argument(
        "--market-tag",
        default="global",
        help="Market localization tag (e.g. apac, latam, emea)",
    )
    parser.add_argument(
        "--workflow",
        default=Path("workflows/model_swap_qwen21_maskless_api.json"),
        type=Path,
        help="Path to ComfyUI workflow JSON template",
    )
    parser.add_argument(
        "--timeout",
        default=300.0,
        type=float,
        help="Maximum timeout in seconds per job (default: 300.0)",
    )
    parser.add_argument(
        "--filter",
        default="*",
        help="Glob pattern to filter models inside --models-dir (default: '*')",
    )
    parser.add_argument(
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        default=True,
        help="Disable automatic purging of intermediate ComfyUI assets",
    )
    parser.add_argument(
        "--comfy-input-dir",
        default=None,
        type=Path,
        help="Override ComfyUI input directory path for housekeeping",
    )
    parser.add_argument(
        "--comfy-output-dir",
        default=None,
        type=Path,
        help="Override ComfyUI output directory path for housekeeping",
    )
    return parser.parse_args(args)


def main() -> None:
    """CLI orchestrator entrypoint."""
    configure_logging()
    args = parse_cli_args()

    config = SwapperConfig(
        server_address=args.server,
        campaign_path=args.campaign,
        mask_path=args.mask,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        market_tag=args.market_tag,
        workflow_path=args.workflow,
        timeout_seconds=args.timeout,
        model_filter=args.filter,
        cleanup=args.cleanup,
        comfy_input_dir=args.comfy_input_dir,
        comfy_output_dir=args.comfy_output_dir,
    )

    swapper = BatchSwapper(config)
    results = swapper.run_batch()
    if not results:
        LOGGER.error("Zero models were processed. Verify --models-dir and --filter arguments.")
        sys.exit(2)
    failures = sum(1 for r in results if r.status != JobStatus.SUCCESS)
    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
