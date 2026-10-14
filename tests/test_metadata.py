"""Unit tests for EU AI Act Article 50 provenance metadata and workflow stripping."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image, PngImagePlugin

from batchpersona.metadata import (
    DEFAULT_AI_COMPLIANCE_TAG,
    DEFAULT_AI_GENERATED_TAG,
    DEFAULT_AI_SYSTEM_TAG,
    DEFAULT_SOFTWARE_TAG,
    process_png_provenance_metadata,
)


def _create_sample_png(path: Path, metadata: dict[str, str] | None = None) -> None:
    """Helper to create a small PNG image with optional text chunks."""
    img = Image.new("RGB", (8, 8), color=(100, 150, 200))
    pnginfo = PngImagePlugin.PngInfo()
    if metadata:
        for k, v in metadata.items():
            pnginfo.add_text(k, v)
    img.save(path, format="PNG", pnginfo=pnginfo)


def test_metadata_non_existent_file(tmp_path: Path) -> None:
    """Verify missing file returns False cleanly without error."""
    missing = tmp_path / "does_not_exist.png"
    assert process_png_provenance_metadata(missing) is False


def test_metadata_non_png_file(tmp_path: Path) -> None:
    """Verify non-PNG formats (e.g., text or JPEG) return False without modification."""
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Hello world", encoding="utf-8")
    assert process_png_provenance_metadata(text_file) is False
    assert text_file.read_text(encoding="utf-8") == "Hello world"

    jpg_file = tmp_path / "sample.jpg"
    img = Image.new("RGB", (8, 8), color=(255, 255, 255))
    img.save(jpg_file, format="JPEG")
    assert process_png_provenance_metadata(jpg_file) is False


def test_metadata_corrupted_png_bytes(tmp_path: Path) -> None:
    """Verify corrupted or truncated byte streams return False gracefully."""
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRbadbytes")
    assert process_png_provenance_metadata(corrupt_file) is False


def test_metadata_strips_workflow_and_injects_provenance(tmp_path: Path) -> None:
    """Verify ComfyUI workflow/prompt chunks are removed and Article 50 tags injected."""
    sample_png = tmp_path / "image.png"
    _create_sample_png(
        sample_png,
        metadata={
            "prompt": '{"3": {"inputs": {}}}',
            "workflow": '{"nodes": [{"id": 3}]}',
            "Author": "Studio Studio",
        },
    )

    success = process_png_provenance_metadata(sample_png, market_tag="apac", keep_workflow=False)
    assert success is True

    with Image.open(sample_png) as im:
        meta = getattr(im, "text", {})
        assert "prompt" not in meta
        assert "workflow" not in meta
        assert meta["Author"] == "Studio Studio"
        assert meta["Software"] == DEFAULT_SOFTWARE_TAG
        assert meta["AI-Generated"] == DEFAULT_AI_GENERATED_TAG
        assert meta["AI-System"] == DEFAULT_AI_SYSTEM_TAG
        assert meta["AI-Compliance"] == DEFAULT_AI_COMPLIANCE_TAG
        assert meta["Market-Tag"] == "apac"


def test_metadata_preserves_workflow_when_flag_enabled(tmp_path: Path) -> None:
    """Verify workflow metadata is retained when keep_workflow is True."""
    sample_png = tmp_path / "image_raw.png"
    _create_sample_png(
        sample_png,
        metadata={
            "prompt": '{"3": {"inputs": {}}}',
            "workflow": '{"nodes": [{"id": 3}]}',
        },
    )

    success = process_png_provenance_metadata(sample_png, market_tag="", keep_workflow=True)
    assert success is True

    with Image.open(sample_png) as im:
        meta = getattr(im, "text", {})
        assert "prompt" in meta
        assert "workflow" in meta
        assert meta["Market-Tag"] == "global"


def test_metadata_save_failure_cleans_up_tmp(tmp_path: Path) -> None:
    """Verify cleanup of temporary files if image saving encounters an OS error."""
    sample_png = tmp_path / "image_fail.png"
    _create_sample_png(sample_png)

    with patch.object(Image.Image, "save", side_effect=OSError("Disk write error")):
        success = process_png_provenance_metadata(sample_png)
        assert success is False

    assert not any(tmp_path.glob("*.meta.tmp"))
