"""REST and WebSocket client for headless interaction with ComfyUI backend."""

from __future__ import annotations

import contextlib
import json
import logging
import mimetypes
import os
import time
import types
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import requests
import websocket

from batchpersona.models import (
    ComfyAPIError,
    ComfyExecutionError,
    ComfyTimeoutError,
    OutputAsset,
    normalize_server_address,
)

LOGGER = logging.getLogger("batch_swapper.client")


class ComfyUIClient:
    """REST and WebSocket client for headless interaction with ComfyUI."""

    def __init__(
        self,
        server_address: str,
        session: requests.Session | None = None,
        ws_factory: Callable[[str, float], Any] | None = None,
    ) -> None:
        self._server = normalize_server_address(server_address)
        self._http_base = f"http://{self._server}"
        self._ws_base = f"ws://{self._server}"
        self._session = session or requests.Session()
        self._ws_factory = ws_factory or self._default_ws_connect

    @staticmethod
    def _default_ws_connect(url: str, timeout: float) -> Any:
        return websocket.create_connection(url, timeout=timeout)

    @property
    def http_base(self) -> str:
        """Return base HTTP URL of ComfyUI server."""
        return self._http_base

    @property
    def server_address(self) -> str:
        """Return target host:port address string."""
        return self._server

    def check_connection(self, timeout: float = 3.0) -> bool:
        """Check if ComfyUI REST server is reachable and responsive."""
        try:
            resp = self._session.get(f"{self._http_base}/system_stats", timeout=timeout)
            return resp.status_code == 200
        except (requests.RequestException, OSError):
            return False

    def get_registered_node_types(self, timeout_seconds: float = 5.0) -> set[str] | None:
        """Retrieve set of registered node class names from ComfyUI /object_info endpoint."""
        url = f"{self._http_base}/object_info"
        try:
            response = self._session.get(url, timeout=timeout_seconds)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    return set(data.keys())
        except (requests.RequestException, ValueError, OSError) as e:
            LOGGER.debug("Could not query ComfyUI /object_info: %s", e)
            return None
        return None

    def upload_image(
        self,
        file_path: Path,
        image_type: str = "input",
        overwrite: bool = False,
    ) -> str:
        """Upload image to ComfyUI /upload/image endpoint. Returns stored filename."""
        if not file_path.is_file():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        mime_type, _ = mimetypes.guess_type(file_path.name)
        if not mime_type:
            mime_type = (
                "image/png" if file_path.suffix.lower() == ".png" else "application/octet-stream"
            )

        url = f"{self._http_base}/upload/image"
        with open(file_path, "rb") as f:
            files = {"image": (file_path.name, f, mime_type)}
            data = {"type": image_type, "overwrite": "true" if overwrite else "false"}
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
        temp_dest = destination.with_suffix(
            f"{destination.suffix}.{os.getpid()}_{time.time_ns()}.part"
        )
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

    @contextlib.contextmanager
    def open_websocket(self, client_id: str, timeout_seconds: float) -> Generator[Any, None, None]:
        """Establish and yield a WebSocket connection, ensuring deterministic cleanup."""
        ws_url = f"{self._ws_base}/ws?clientId={client_id}"
        ws = self._ws_factory(ws_url, timeout_seconds)
        try:
            yield ws
        finally:
            try:
                ws.close()
            except (websocket.WebSocketException, OSError) as ws_err:
                LOGGER.debug("Error while closing websocket: %s", ws_err)

    def get_prompt_history_outputs(self, prompt_id: str) -> list[OutputAsset]:
        """Query /history/{prompt_id} to retrieve output assets directly from execution history."""
        url = f"{self._http_base}/history/{prompt_id}"
        try:
            resp = self._session.get(url, timeout=10.0)
            if resp.status_code != 200:
                return []
            history_data = resp.json()
            if not isinstance(history_data, dict) or prompt_id not in history_data:
                return []
            prompt_entry = history_data[prompt_id]
            outputs = prompt_entry.get("outputs", {})
            assets: list[OutputAsset] = []
            for node_out in outputs.values():
                if isinstance(node_out, dict):
                    images = node_out.get("images", [])
                    for img in images:
                        if isinstance(img, dict):
                            assets.append(
                                OutputAsset(
                                    filename=img.get("filename", ""),
                                    subfolder=img.get("subfolder", ""),
                                    image_type=img.get("type", "output"),
                                )
                            )
            return assets
        except (requests.RequestException, ValueError, OSError) as e:
            LOGGER.debug("Could not query /history/%s: %s", prompt_id, e)
            return []

    def track_execution(
        self,
        prompt_id: str,
        client_id: str,
        timeout_seconds: float,
        on_progress: Callable[[int, int, str], None] | None = None,
        ws: Any | None = None,
    ) -> list[OutputAsset]:
        """Listen to WebSocket events until prompt completes, fails, or times out."""
        created_ws = False
        if ws is None:
            ws_url = f"{self._ws_base}/ws?clientId={client_id}"
            active_ws = self._ws_factory(ws_url, timeout_seconds)
            created_ws = True
        else:
            active_ws = ws

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
                settimeout_fn = getattr(active_ws, "settimeout", None)
                if callable(settimeout_fn):
                    settimeout_fn(min(2.0, remaining))

                try:
                    raw_frame = active_ws.recv()
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
            if created_ws:
                try:
                    active_ws.close()
                except (websocket.WebSocketException, OSError) as ws_err:
                    LOGGER.debug("Error while closing websocket: %s", ws_err)

        # Fallback to /history if execution completed but output assets were not intercepted via WS
        if not output_assets:
            history_assets = self.get_prompt_history_outputs(prompt_id)
            if history_assets:
                LOGGER.info(
                    "Retrieved %d output asset(s) via /history fallback for %s",
                    len(history_assets),
                    prompt_id,
                )
                output_assets.extend(history_assets)

        return output_assets
