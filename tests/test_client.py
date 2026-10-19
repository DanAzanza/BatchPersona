"""Unit and integration tests for ComfyUIClient network transport and events."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests
import websocket
from PIL import Image, PngImagePlugin

from batchpersona.client import ComfyUIClient
from batchpersona.models import (
    DEFAULT_SERVER_ADDRESS,
    ComfyAPIError,
    ComfyExecutionError,
    ComfyTimeoutError,
    OutputAsset,
    normalize_server_address,
)
from tests.conftest import MockWebSocket


def test_client_upload_image_success(tmp_path: Path) -> None:
    """Verify upload_image issues multipart request and returns ComfyUI asset name."""
    test_img = tmp_path / "sample.png"
    test_img.write_bytes(b"dummy_png_bytes")

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"name": "sample_uploaded.png"}
    mock_session.post.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    uploaded = client.upload_image(test_img)

    assert uploaded == "sample_uploaded.png"
    mock_session.post.assert_called_once()
    assert "upload/image" in mock_session.post.call_args[0][0]


def test_client_upload_image_error(tmp_path: Path) -> None:
    """Verify upload_image raises ComfyAPIError when server returns non-200."""
    test_img = tmp_path / "sample.png"
    test_img.write_bytes(b"dummy_png_bytes")

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = "Bad image format"
    mock_session.post.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    with pytest.raises(ComfyAPIError, match="Upload failed"):
        client.upload_image(test_img)


def test_client_upload_image_dynamic_mime_and_safe_overwrite(tmp_path: Path) -> None:
    """Verify upload_image detects JPEG/WebP MIME type and sets overwrite=false by default."""
    test_jpg = tmp_path / "model.jpg"
    test_jpg.write_bytes(b"dummy_jpg_bytes")

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"name": "model_001.jpg"}
    mock_session.post.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    uploaded = client.upload_image(test_jpg)

    assert uploaded == "model_001.jpg"
    _, kwargs = mock_session.post.call_args
    files = kwargs["files"]
    data = kwargs["data"]
    assert files["image"][2] == "image/jpeg"
    assert data["overwrite"] == "false"


def test_client_open_websocket() -> None:
    """Verify open_websocket context manager connects and closes websocket reliably."""
    mock_ws = MagicMock()
    ws_factory = MagicMock(return_value=mock_ws)
    client = ComfyUIClient(server_address="127.0.0.1:8188", ws_factory=ws_factory)

    with client.open_websocket("client-123", timeout_seconds=10.0) as ws:
        assert ws is mock_ws
        ws_factory.assert_called_once_with("ws://127.0.0.1:8188/ws?clientId=client-123", 10.0)

    mock_ws.close.assert_called_once()


def test_client_history_fallback() -> None:
    """Verify track_execution falls back to /history when websocket finishes with zero output frames."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "p-fast": {
            "outputs": {
                "9": {
                    "images": [{"filename": "cached_out.png", "subfolder": "sub", "type": "output"}]
                }
            }
        }
    }
    mock_session.get.return_value = mock_resp

    mock_ws = MockWebSocket(
        [json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "p-fast"}})]
    )
    client = ComfyUIClient(
        server_address="127.0.0.1:8188",
        session=mock_session,
        ws_factory=lambda url, timeout: mock_ws,
    )

    outputs = client.track_execution("p-fast", "c-1", timeout_seconds=5.0)
    assert len(outputs) == 1
    assert outputs[0].filename == "cached_out.png"
    assert outputs[0].subfolder == "sub"


def test_client_queue_prompt_success() -> None:
    """Verify queue_prompt posts payload and extracts prompt_id."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"prompt_id": "test-prompt-uuid-123"}
    mock_session.post.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    prompt_id = client.queue_prompt(prompt={"node": 1}, client_id="client-456")

    assert prompt_id == "test-prompt-uuid-123"
    mock_session.post.assert_called_once()


def test_client_download_image(tmp_path: Path) -> None:
    """Verify download_image writes content cleanly with atomic rename."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_content.return_value = [b"chunk1", b"chunk2"]
    mock_session.get.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    asset = OutputAsset(filename="out.png", subfolder="", image_type="output")
    target = tmp_path / "downloaded.png"

    client.download_image(asset, target)

    assert target.is_file()
    assert target.read_bytes() == b"chunk1chunk2"
    assert not any(tmp_path.glob("*.part"))


def test_client_download_image_strips_workflow_and_embeds_provenance(tmp_path: Path) -> None:
    """Verify download_image strips ComfyUI workflow/prompt and embeds EU AI Act provenance."""
    img = Image.new("RGB", (16, 16), color=(255, 0, 0))
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("prompt", '{"1": {"class_type": "KSampler"}}')
    pnginfo.add_text("workflow", '{"nodes": [{"id": 1, "type": "KSampler"}]}')
    pnginfo.add_text("custom_tag", "keep_me")

    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=pnginfo)
    png_bytes = buf.getvalue()

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_content.return_value = [png_bytes]
    mock_session.get.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    asset = OutputAsset(filename="result.png", subfolder="", image_type="output")
    target = tmp_path / "result_processed.png"

    client.download_image(asset, target, market_tag="latam", keep_workflow=False)

    assert target.is_file()
    with Image.open(target) as processed_im:
        meta = getattr(processed_im, "text", {})
        assert "prompt" not in meta
        assert "workflow" not in meta
        assert meta["Software"] == "BatchPersona"
        assert meta["AI-Generated"] == "true"
        assert meta["AI-System"] == "BatchPersona / ComfyUI"
        assert meta["AI-Compliance"] == "EU-AI-Act-Article-50-Synthetic-Media"
        assert meta["Market-Tag"] == "latam"
        assert meta["custom_tag"] == "keep_me"


def test_client_download_image_preserves_workflow_when_requested(tmp_path: Path) -> None:
    """Verify download_image retains workflow and prompt when keep_workflow is True."""
    img = Image.new("RGB", (16, 16), color=(0, 255, 0))
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("workflow", '{"nodes": []}')
    pnginfo.add_text("prompt", '{"1": {}}')

    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=pnginfo)
    png_bytes = buf.getvalue()

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_content.return_value = [png_bytes]
    mock_session.get.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    asset = OutputAsset(filename="raw.png", subfolder="", image_type="output")
    target = tmp_path / "raw_processed.png"

    client.download_image(asset, target, keep_workflow=True)

    with Image.open(target) as processed_im:
        meta = getattr(processed_im, "text", {})
        assert "workflow" in meta
        assert "prompt" in meta
        assert meta["Software"] == "BatchPersona"
        assert meta["AI-Generated"] == "true"
        assert meta["Market-Tag"] == "global"


def test_client_track_execution_success() -> None:
    """Verify WebSocket listener parses events, ignores binary frames, and tracks outputs."""
    prompt_id = "target-prompt-id"
    frames: list[str | bytes] = [
        # 1. Binary preview frame (should be ignored)
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00",
        # 2. Event for different prompt (should be ignored)
        json.dumps(
            {
                "type": "progress",
                "data": {"prompt_id": "other-id", "value": 1, "max": 10},
            }
        ),
        # 3. Matching progress event
        json.dumps(
            {
                "type": "progress",
                "data": {"prompt_id": prompt_id, "value": 5, "max": 25, "node": "12"},
            }
        ),
        # 4. Executed event with output images
        json.dumps(
            {
                "type": "executed",
                "data": {
                    "prompt_id": prompt_id,
                    "node": "14",
                    "output": {
                        "images": [
                            {
                                "filename": "campaign_latam_m01_00001_.png",
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    },
                },
            }
        ),
        # 5. Executing event with node=None (workflow completion signal)
        json.dumps({"type": "executing", "data": {"prompt_id": prompt_id, "node": None}}),
    ]

    mock_ws = MockWebSocket(frames)
    client = ComfyUIClient(
        server_address="127.0.0.1:8188",
        ws_factory=lambda url, timeout: mock_ws,
    )

    progress_records: list[tuple[int, int, str]] = []

    def on_progress(cur: int, total: int, node: str) -> None:
        progress_records.append((cur, total, node))

    outputs = client.track_execution(
        prompt_id=prompt_id,
        client_id="test-client",
        timeout_seconds=5.0,
        on_progress=on_progress,
    )

    assert len(outputs) == 1
    assert outputs[0].filename == "campaign_latam_m01_00001_.png"
    assert len(progress_records) == 1
    assert progress_records[0] == (5, 25, "12")
    assert mock_ws.closed


def test_client_track_execution_error() -> None:
    """Verify WebSocket listener raises ComfyExecutionError on execution_error event."""
    prompt_id = "failing-prompt-id"
    frames: list[str | bytes] = [
        json.dumps(
            {
                "type": "execution_error",
                "data": {
                    "prompt_id": prompt_id,
                    "node_id": "12",
                    "node_type": "KSampler",
                    "exception_message": "CUDA out of memory",
                    "exception_type": "torch.cuda.OutOfMemoryError",
                },
            }
        ),
    ]

    mock_ws = MockWebSocket(frames)
    client = ComfyUIClient(
        server_address="127.0.0.1:8188",
        ws_factory=lambda url, timeout: mock_ws,
    )

    with pytest.raises(ComfyExecutionError, match="CUDA out of memory"):
        client.track_execution(
            prompt_id=prompt_id,
            client_id="test-client",
            timeout_seconds=5.0,
        )


def test_client_track_execution_timeout() -> None:
    """Verify track_execution aborts and raises ComfyTimeoutError if duration exceeded."""
    prompt_id = "hanging-prompt-id"
    frames: list[str | bytes] = [
        json.dumps({"type": "status", "data": {"prompt_id": prompt_id}}),
    ]

    mock_ws = MockWebSocket(frames)
    client = ComfyUIClient(
        server_address="127.0.0.1:8188",
        ws_factory=lambda url, timeout: mock_ws,
    )

    # Use timeout_seconds = -1 to trigger immediate timeout
    with pytest.raises(ComfyTimeoutError, match="timeout limit"):
        client.track_execution(
            prompt_id=prompt_id,
            client_id="test-client",
            timeout_seconds=-0.1,
        )


def test_client_upload_image_not_found(tmp_path: Path) -> None:
    """Verify upload_image raises FileNotFoundError for missing input file."""
    client = ComfyUIClient(server_address="127.0.0.1:8188")
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        client.upload_image(tmp_path / "ghost.png")


def test_client_queue_prompt_errors() -> None:
    """Verify queue_prompt raises ComfyAPIError on HTTP failure or missing prompt_id."""
    mock_session = MagicMock()
    # 1. HTTP 500 error
    mock_resp_fail = MagicMock()
    mock_resp_fail.status_code = 500
    mock_resp_fail.text = "Internal Server Error"
    mock_session.post.return_value = mock_resp_fail

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    with pytest.raises(ComfyAPIError, match="Failed to queue prompt"):
        client.queue_prompt({"1": {}}, client_id="c1")

    # 2. Missing prompt_id in response
    mock_resp_missing = MagicMock()
    mock_resp_missing.status_code = 200
    mock_resp_missing.json.return_value = {"other_key": 123}
    mock_session.post.return_value = mock_resp_missing
    with pytest.raises(ComfyAPIError, match="Missing prompt_id"):
        client.queue_prompt({"1": {}}, client_id="c1")


def test_client_download_image_error(tmp_path: Path) -> None:
    """Verify download_image raises ComfyAPIError when server returns 404."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "Not Found"
    mock_session.get.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    asset = OutputAsset(filename="missing.png", subfolder="", image_type="output")
    with pytest.raises(ComfyAPIError, match="Failed to download asset"):
        client.download_image(asset, tmp_path / "dest.png")


def test_client_interrupt_and_free() -> None:
    """Verify interrupt and free memory requests execute cleanly."""
    mock_session = MagicMock()
    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)

    client.interrupt_execution()
    assert mock_session.post.call_count == 1

    client.free_memory()
    assert mock_session.post.call_count == 2


def test_client_context_manager_and_close() -> None:
    """Verify ComfyUIClient context manager closes the underlying HTTP session cleanly."""
    mock_session = MagicMock()
    with ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session) as client:
        assert client._session is mock_session
    mock_session.close.assert_called_once()
    client.close()
    assert mock_session.close.call_count == 2


def test_client_free_memory_with_unload_models() -> None:
    """Verify free_memory passes unload_models parameter to ComfyUI /free endpoint."""
    client = ComfyUIClient(server_address="127.0.0.1:8188")
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_session.post.return_value = mock_resp
    client._session = mock_session

    client.free_memory(unload_models=True)
    mock_session.post.assert_called_once_with(
        "http://127.0.0.1:8188/free",
        json={"unload_models": True, "free_memory": True},
        timeout=5.0,
    )


def test_client_check_connection() -> None:
    """Verify check_connection returns True on 200 and False on error."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_session.get.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    assert client.check_connection() is True

    mock_session.get.side_effect = requests.RequestException("Offline")
    assert client.check_connection() is False


def test_client_get_registered_node_types_success() -> None:
    """Verify get_registered_node_types parses /object_info dictionary keys."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "LoadImage": {"input": {}},
        "KSampler": {"input": {}},
        "VAEDecode": {"input": {}},
    }
    mock_session.get.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    registered = client.get_registered_node_types()
    assert registered == {"LoadImage", "KSampler", "VAEDecode"}
    mock_session.get.assert_called_once()
    assert "object_info" in mock_session.get.call_args[0][0]


def test_client_get_registered_node_types_failure() -> None:
    """Verify get_registered_node_types returns None gracefully on error or non-200."""
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_session.get.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    assert client.get_registered_node_types() is None

    # Exception during GET
    mock_session.get.side_effect = requests.RequestException("Network error")
    assert client.get_registered_node_types() is None


def test_client_open_websocket_close_exception() -> None:
    """Verify open_websocket cleanly suppresses and logs websocket close exceptions."""
    mock_ws = MagicMock()
    mock_ws.close.side_effect = websocket.WebSocketException("Already closed")
    client = ComfyUIClient(
        server_address="127.0.0.1:8188",
        ws_factory=lambda url, timeout: mock_ws,
    )
    with client.open_websocket("client-123", timeout_seconds=5.0) as ws:
        assert ws is mock_ws
    assert mock_ws.close.called


def test_client_get_prompt_history_outputs_negative_branches() -> None:
    """Verify get_prompt_history_outputs handles 404, missing data, and network errors gracefully."""
    mock_session = MagicMock()
    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)

    # 1. Status != 200
    mock_resp_404 = MagicMock(status_code=404)
    mock_session.get.return_value = mock_resp_404
    assert client.get_prompt_history_outputs("p1") == []

    # 2. JSON is not dict
    mock_resp_list = MagicMock(status_code=200)
    mock_resp_list.json.return_value = ["not", "dict"]
    mock_session.get.return_value = mock_resp_list
    assert client.get_prompt_history_outputs("p1") == []

    # 3. Prompt ID missing from JSON dict
    mock_resp_missing = MagicMock(status_code=200)
    mock_resp_missing.json.return_value = {"other_p": {}}
    mock_session.get.return_value = mock_resp_missing
    assert client.get_prompt_history_outputs("p1") == []

    # 4. RequestException raised
    mock_session.get.side_effect = requests.RequestException("Timeout")
    assert client.get_prompt_history_outputs("p1") == []


def test_client_upload_image_custom_mime_and_binary_fallback(tmp_path: Path) -> None:
    """Verify upload_image handles custom MIME type overrides and fallback for unknown extensions."""
    mock_session = MagicMock()
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"name": "test.unknown"}
    mock_session.post.return_value = mock_resp

    client = ComfyUIClient(server_address="127.0.0.1:8188", session=mock_session)
    dummy_file = tmp_path / "test.unknown_ext"
    dummy_file.write_bytes(b"\x00\x01\x02\x03")

    result = client.upload_image(dummy_file, overwrite=True)
    assert result == "test.unknown"
    call_args = mock_session.post.call_args
    files = call_args[1]["files"]
    file_tuple = files["image"]
    assert file_tuple[2] == "application/octet-stream"


def test_client_track_execution_with_external_ws() -> None:
    """Verify track_execution accepts externally provided WebSocket and does not close it."""
    prompt_id = "test-p1"
    frames: list[str | bytes] = [
        json.dumps({"type": "executing", "data": {"node": None, "prompt_id": prompt_id}}),
    ]
    mock_ws = MockWebSocket(frames)
    client = ComfyUIClient(server_address="127.0.0.1:8188")

    assets = client.track_execution(
        prompt_id=prompt_id,
        client_id="client-1",
        timeout_seconds=5.0,
        ws=mock_ws,
    )
    assert assets == []
    assert not mock_ws.closed


def test_normalize_server_address_default_fallback() -> None:
    """Verify normalize_server_address falls back to DEFAULT_SERVER_ADDRESS when empty."""
    assert normalize_server_address("") == DEFAULT_SERVER_ADDRESS
    assert normalize_server_address("   ") == DEFAULT_SERVER_ADDRESS
    assert normalize_server_address("http://127.0.0.1:8000/") == "127.0.0.1:8000"
    assert normalize_server_address("https://127.0.0.1:8188/prompt") == "127.0.0.1:8188"
    assert normalize_server_address("ws://localhost:8000") == "localhost:8000"
    assert normalize_server_address("wss://remote-host:9000/") == "remote-host:9000"
    assert DEFAULT_SERVER_ADDRESS == "127.0.0.1:8000"


def test_package_exports_default_server_address() -> None:
    """Verify batchpersona root package exports DEFAULT_SERVER_ADDRESS in __all__."""
    import batchpersona

    assert hasattr(batchpersona, "DEFAULT_SERVER_ADDRESS")
    assert "DEFAULT_SERVER_ADDRESS" in batchpersona.__all__
    assert batchpersona.DEFAULT_SERVER_ADDRESS == "127.0.0.1:8000"


def test_client_tls_and_wss_support() -> None:
    """Verify ComfyUIClient correctly configures HTTPS and WSS for secure servers."""
    # 1. Plain loopback / HTTP
    client_plain = ComfyUIClient("127.0.0.1:8000")
    assert client_plain.is_tls is False
    assert client_plain.http_base == "http://127.0.0.1:8000"
    assert client_plain.ws_base == "ws://127.0.0.1:8000"

    # 2. HTTPS scheme provided
    client_https = ComfyUIClient("https://remote-comfyui.cloud:8443")
    assert client_https.is_tls is True
    assert client_https.http_base == "https://remote-comfyui.cloud:8443"
    assert client_https.ws_base == "wss://remote-comfyui.cloud:8443"

    # 3. WSS scheme provided
    client_wss = ComfyUIClient("wss://secure-host.internal")
    assert client_wss.is_tls is True
    assert client_wss.http_base == "https://secure-host.internal"
    assert client_wss.ws_base == "wss://secure-host.internal"

    # 4. Standard port 443 provided
    client_443 = ComfyUIClient("comfy.mycorp.com:443")
    assert client_443.is_tls is True
    assert client_443.http_base == "https://comfy.mycorp.com:443"
    assert client_443.ws_base == "wss://comfy.mycorp.com:443"
