"""Shared test fixtures and mocks for BatchPersona test suites."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from batchpersona.client import ComfyUIClient


class MockWebSocket:
    """Mock WebSocket client simulating ComfyUI message framing and lifecycle."""

    def __init__(self, frames: list[str | bytes]) -> None:
        self._frames = list(frames)
        self.closed = False

    def recv(self) -> str | bytes:
        if not self._frames:
            raise StopIteration("No more mock frames")
        return self._frames.pop(0)

    def close(self) -> None:
        self.closed = True


def make_mock_client(**overrides: Any) -> MagicMock:
    """Create a mock conforming strictly to ComfyUIClient interface."""
    client = MagicMock(spec=ComfyUIClient)
    client.http_base = overrides.get("http_base", "http://127.0.0.1:8188")
    client.server_address = overrides.get("server_address", "127.0.0.1:8188")
    client.check_connection.return_value = overrides.get("check_connection", True)
    client.get_registered_node_types.return_value = overrides.get(
        "registered_nodes",
        {
            "LoadImage",
            "SaveImage",
            "KSampler",
            "VAEDecode",
            "EmptyLatentImage",
            "CLIPTextEncode",
            "ImageCompositeMasked",
            "SpecialCustomNode",
        },
    )
    for k, v in overrides.items():
        if hasattr(client, k):
            setattr(client, k, v)
    return client
