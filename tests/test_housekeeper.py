"""Unit test suite for Smart Local Housekeeping in scripts/housekeeper.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.batch_swapper import BatchSwapper, JobStatus, OutputAsset, SwapperConfig
from scripts.housekeeper import (
    ComfyUIHousekeeper,
    create_housekeeper,
    is_loopback_host,
)


def test_is_loopback_host() -> None:
    """Verify loopback host detection correctly identifies local addresses."""
    assert is_loopback_host("127.0.0.1:8000") is True
    assert is_loopback_host("http://127.0.0.1:8188") is True
    assert is_loopback_host("localhost:8000") is True
    assert is_loopback_host("http://localhost:8188/ws") is True
    assert is_loopback_host("0.0.0.0:8000") is True
    assert is_loopback_host("::1") is True
    assert is_loopback_host("127.0.0.5:8000") is True

    # Remote hosts
    assert is_loopback_host("192.168.1.100:8188") is False
    assert is_loopback_host("https://comfyui.cloud.runpod.net") is False
    assert is_loopback_host("http://ai-server.lan:8000") is False


def test_housekeeper_disabled_operations(tmp_path: Path) -> None:
    """Verify disabled housekeeper performs no file mutations and returns cleanly."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    hk = ComfyUIHousekeeper(input_dir=input_dir, output_dir=output_dir, enabled=False)

    test_file = input_dir / "test.png"
    test_file.write_bytes(b"data")

    hk.record_pre_upload("test.png")
    hk.record_uploaded_name("test.png")
    assert hk.cleanup_model_input("test.png", tmp_path / "source.png") is False
    assert hk.cleanup_shared_inputs([("test.png", tmp_path / "source.png")]) == 0
    assert hk.cleanup_output_asset("test.png", "", tmp_path / "final.png") is False
    assert test_file.exists()


def test_housekeeper_pre_existing_input_protection(tmp_path: Path) -> None:
    """Verify pre-existing user files in input/ are never deleted."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # File pre-existed before pipeline ran
    pre_existing = input_dir / "user_portrait.png"
    pre_existing.write_bytes(b"user_data")

    source_path = tmp_path / "data_models" / "user_portrait.png"

    hk = ComfyUIHousekeeper(input_dir=input_dir, enabled=True)
    hk.record_pre_upload("user_portrait.png")
    hk.record_uploaded_name("user_portrait.png")

    # Attempt to clean up -> MUST REFUSE TO DELETE
    purged = hk.cleanup_model_input("user_portrait.png", source_path)
    assert purged is False
    assert pre_existing.exists()


def test_housekeeper_canonical_source_identity_protection(tmp_path: Path) -> None:
    """Verify source asset self-deletion is prevented if source is inside input_dir."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Source asset is in ComfyUI input directory
    source_path = input_dir / "campaign.png"
    source_path.write_bytes(b"campaign_data")

    hk = ComfyUIHousekeeper(input_dir=input_dir, enabled=True)
    hk._tracked_inputs.add(source_path.resolve())

    # Attempt to clean up -> MUST REFUSE TO DELETE
    purged = hk._purge_input_asset(source_path.resolve(), source_path)
    assert purged is False
    assert source_path.exists()


def test_housekeeper_canonical_export_identity_protection(tmp_path: Path) -> None:
    """Verify output asset self-deletion is prevented if export target matches ComfyUI output."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    intermediate = output_dir / "result.png"
    intermediate.write_bytes(b"rendered_image")

    hk = ComfyUIHousekeeper(output_dir=output_dir, enabled=True)

    # final_target is identical to intermediate
    purged = hk.cleanup_output_asset(
        asset_filename="result.png",
        asset_subfolder="",
        final_target=intermediate,
    )
    assert purged is False
    assert intermediate.exists()


def test_housekeeper_happy_path_input_and_output(tmp_path: Path) -> None:
    """Verify normal cleanup removes intermediate assets and preserves exports."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    export_dir = tmp_path / "data" / "output"
    export_dir.mkdir(parents=True)
    final_export = export_dir / "final_result.png"
    final_export.write_bytes(b"exported_content")

    # 1. Output cleanup
    intermediate_out = output_dir / "final_result.png"
    intermediate_out.write_bytes(b"intermediate_content")

    hk = ComfyUIHousekeeper(input_dir=input_dir, output_dir=output_dir, enabled=True)
    out_purged = hk.cleanup_output_asset(
        asset_filename="final_result.png",
        asset_subfolder="",
        final_target=final_export,
    )
    assert out_purged is True
    assert not intermediate_out.exists()
    assert final_export.exists()

    # 2. Output cleanup with subfolder
    sub_dir = output_dir / "batch_1"
    sub_dir.mkdir()
    sub_out = sub_dir / "sub_result.png"
    sub_out.write_bytes(b"sub_content")
    sub_purged = hk.cleanup_output_asset(
        asset_filename="sub_result.png",
        asset_subfolder="batch_1",
        final_target=final_export,
    )
    assert sub_purged is True
    assert not sub_out.exists()

    # 3. Model input cleanup
    source_model = tmp_path / "models" / "model_01.png"
    source_model.parent.mkdir()
    source_model.write_bytes(b"source_content")

    comfy_input_model = input_dir / "model_01.png"
    hk.record_pre_upload("model_01.png")
    comfy_input_model.write_bytes(b"uploaded_content")
    hk.record_uploaded_name("model_01.png")

    model_purged = hk.cleanup_model_input("model_01.png", source_model)
    assert model_purged is True
    assert not comfy_input_model.exists()
    assert source_model.exists()


def test_housekeeper_cleanup_shared_inputs(tmp_path: Path) -> None:
    """Verify batch completion shared inputs (campaign and mask) are purged."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    source_campaign = tmp_path / "campaign.png"
    source_campaign.write_bytes(b"src_camp")
    comfy_campaign = input_dir / "campaign.png"

    source_mask = tmp_path / "mask.png"
    source_mask.write_bytes(b"src_mask")
    comfy_mask = input_dir / "mask.png"

    hk = ComfyUIHousekeeper(input_dir=input_dir, enabled=True)

    hk.record_pre_upload("campaign.png")
    comfy_campaign.write_bytes(b"up_camp")
    hk.record_uploaded_name("campaign.png")

    hk.record_pre_upload("mask.png")
    comfy_mask.write_bytes(b"up_mask")
    hk.record_uploaded_name("mask.png")

    count = hk.cleanup_shared_inputs(
        [
            ("campaign.png", source_campaign),
            ("mask.png", source_mask),
            (None, None),  # Edge case: None entries
        ]
    )
    assert count == 2
    assert not comfy_campaign.exists()
    assert not comfy_mask.exists()
    assert source_campaign.exists()
    assert source_mask.exists()


def test_safe_delete_retry_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _safe_delete micro-retries on OSError and logs warning on exhaustion."""
    target_file = tmp_path / "locked.png"
    target_file.write_bytes(b"locked")

    hk = ComfyUIHousekeeper(
        input_dir=tmp_path, enabled=True, max_retries=2, retry_delay_seconds=0.01
    )

    # Simulate locked file (unlink raises PermissionError)
    def mock_unlink_locked() -> None:
        raise PermissionError("File is locked by another process")

    monkeypatch.setattr(Path, "unlink", lambda p: mock_unlink_locked())
    assert hk._safe_delete(target_file) is False
    assert target_file.exists()


def test_create_housekeeper_guards_and_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify create_housekeeper correctly resolves directories and honors guards."""
    # 1. Disabled flag
    hk1 = create_housekeeper("127.0.0.1:8000", enabled=False)
    assert hk1.enabled is False

    # 2. Remote server guard
    hk2 = create_housekeeper("192.168.1.200:8188", enabled=True)
    assert hk2.enabled is False

    # 3. Explicit directories
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()
    out.mkdir()
    hk3 = create_housekeeper("localhost:8000", enabled=True, input_dir=inp, output_dir=out)
    assert hk3.enabled is True
    assert hk3.input_dir == inp.resolve()
    assert hk3.output_dir == out.resolve()

    # 4. Environment variable fallback
    monkeypatch.setenv("COMFYUI_INPUT_DIR", str(inp))
    monkeypatch.setenv("COMFYUI_OUTPUT_DIR", str(out))
    hk4 = create_housekeeper("127.0.0.1:8000", enabled=True)
    assert hk4.input_dir == inp.resolve()
    assert hk4.output_dir == out.resolve()

    # 5. Desktop auto-detection mock
    monkeypatch.delenv("COMFYUI_INPUT_DIR", raising=False)
    monkeypatch.delenv("COMFYUI_OUTPUT_DIR", raising=False)

    mock_inst = MagicMock()
    mock_inst.input_dir = inp
    mock_inst.output_dir = out
    monkeypatch.setattr("scripts.launch_comfyui.detect_comfyui", lambda: mock_inst)

    hk5 = create_housekeeper("127.0.0.1:8000", enabled=True)
    assert hk5.input_dir == inp.resolve()
    assert hk5.output_dir == out.resolve()


def test_batch_swapper_with_housekeeper_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify BatchSwapper orchestrates housekeeper calls during process_single_model and run_batch."""
    campaign_file = tmp_path / "campaign.png"
    campaign_file.write_bytes(b"campaign")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_file = models_dir / "model_01.png"
    model_file.write_bytes(b"model")

    workflow_file = tmp_path / "wf.json"
    workflow_file.write_text(
        '{"1": {"class_type": "LoadImage", "inputs": {"image": "c"}}, "2": {"class_type": "LoadImage", "inputs": {"image": "m"}}, "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "p"}}}'
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    comfy_in = tmp_path / "comfy_in"
    comfy_out = tmp_path / "comfy_out"
    comfy_in.mkdir()
    comfy_out.mkdir()

    mock_client = MagicMock()
    mock_client.check_connection.return_value = True
    mock_client.upload_image.side_effect = lambda p: p.name
    mock_client.queue_prompt.return_value = "job_123"
    mock_client.track_execution.return_value = [
        OutputAsset(filename="campaign_model_01.png", subfolder="", image_type="output")
    ]

    def mock_download(asset: OutputAsset, target: Path) -> None:
        target.write_bytes(b"final_download")

    mock_client.download_image.side_effect = mock_download

    config = SwapperConfig(
        server_address="127.0.0.1:8000",
        campaign_path=campaign_file,
        models_dir=models_dir,
        output_dir=output_dir,
        workflow_path=workflow_file,
        cleanup=True,
        comfy_input_dir=comfy_in,
        comfy_output_dir=comfy_out,
    )

    housekeeper = ComfyUIHousekeeper(input_dir=comfy_in, output_dir=comfy_out, enabled=True)
    spy_cleanup_output = MagicMock(wraps=housekeeper.cleanup_output_asset)
    spy_cleanup_model = MagicMock(wraps=housekeeper.cleanup_model_input)
    spy_cleanup_shared = MagicMock(wraps=housekeeper.cleanup_shared_inputs)

    housekeeper.cleanup_output_asset = spy_cleanup_output  # type: ignore[method-assign]
    housekeeper.cleanup_model_input = spy_cleanup_model  # type: ignore[method-assign]
    housekeeper.cleanup_shared_inputs = spy_cleanup_shared  # type: ignore[method-assign]

    swapper = BatchSwapper(config, client=mock_client, housekeeper=housekeeper)
    results = swapper.run_batch()

    assert len(results) == 1
    assert results[0].status == JobStatus.SUCCESS
    assert spy_cleanup_output.call_count == 1
    assert spy_cleanup_model.call_count == 1
    assert spy_cleanup_shared.call_count == 1
