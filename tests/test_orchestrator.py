"""Unit and integration tests for BatchSwapper batch runner and orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from batchpersona.client import ComfyUIClient
from batchpersona.models import (
    ComfyAPIError,
    ComfyExecutionError,
    JobStatus,
    OutputAsset,
    SwapperConfig,
)
from batchpersona.orchestrator import BatchSwapper
from tests.conftest import make_mock_client


def test_batch_swapper_auto_pairs_mask(tmp_path: Path) -> None:
    """Verify BatchSwapper automatically detects and pairs <campaign>_mask.png if --mask is omitted."""
    campaign_file = tmp_path / "lookbook.png"
    campaign_file.write_bytes(b"campaign")
    mask_file = tmp_path / "lookbook_mask.png"
    mask_file.write_bytes(b"mask")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m.png").write_bytes(b"m")

    wf_file = tmp_path / "wf.json"
    wf_file.write_text(
        json.dumps(
            {
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": "{{CAMPAIGN_IMG}}"},
                },
                "2": {"class_type": "LoadImage", "inputs": {"image": "{{MASK_IMG}}"}},
                "3": {"class_type": "LoadImage", "inputs": {"image": "{{MODEL_IMG}}"}},
                "4": {
                    "class_type": "SaveImage",
                    "inputs": {"filename_prefix": "{{OUTPUT_PREFIX}}"},
                },
            }
        ),
        encoding="utf-8",
    )

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign_file,
        mask_path=None,
        models_dir=models_dir,
        output_dir=tmp_path / "out",
        market_tag="test",
        workflow_path=wf_file,
    )

    mock_client = make_mock_client()
    mock_client.upload_image.side_effect = lambda p: f"up_{p.name}"
    mock_client.queue_prompt.return_value = "p-1"
    mock_client.track_execution.return_value = [OutputAsset("out.png", "", "output")]

    swapper = BatchSwapper(config=config, client=mock_client)
    results = swapper.run_batch()

    assert len(results) == 1
    assert results[0].market_tag == "test"
    uploaded_files = [call[0][0].name for call in mock_client.upload_image.call_args_list]
    assert "lookbook.png" in uploaded_files
    assert "lookbook_mask.png" in uploaded_files


def test_batch_swapper_end_to_end_mock(tmp_path: Path) -> None:
    """Test full sequential batch execution across multiple models with mocked server."""
    models_dir = tmp_path / "input_models"
    models_dir.mkdir()
    (models_dir / "model_01.png").write_bytes(b"portrait_01")
    (models_dir / "model_02.png").write_bytes(b"portrait_02")

    campaign_file = tmp_path / "campaign.png"
    campaign_file.write_bytes(b"campaign_bytes")

    mask_file = tmp_path / "mask.png"
    mask_file.write_bytes(b"mask_bytes")

    output_dir = tmp_path / "output"

    workflow_file = tmp_path / "workflow.json"
    workflow_content = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "{{CAMPAIGN_IMG}}"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "{{MASK_IMG}}"}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "{{MODEL_IMG}}"}},
        "14": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "{{OUTPUT_PREFIX}}"},
        },
    }
    workflow_file.write_text(json.dumps(workflow_content), encoding="utf-8")

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign_file,
        mask_path=mask_file,
        models_dir=models_dir,
        output_dir=output_dir,
        market_tag="emea",
        workflow_path=workflow_file,
        timeout_seconds=10.0,
    )

    mock_client = make_mock_client()
    mock_client.upload_image.side_effect = lambda path: f"uploaded_{path.name}"
    mock_client.queue_prompt.side_effect = lambda prompt, client_id: f"prompt_{client_id[:8]}"
    mock_client.track_execution.return_value = [
        OutputAsset(filename="out_mock.png", subfolder="", image_type="output")
    ]

    def mock_download(asset: OutputAsset, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"rendered_model_result")

    mock_client.download_image.side_effect = mock_download

    swapper = BatchSwapper(config=config, client=mock_client)
    results = swapper.run_batch()

    assert len(results) == 2
    assert all(r.status == JobStatus.SUCCESS for r in results)
    assert (output_dir / "campaign_model_01.png").is_file()
    assert mock_client.free_memory.call_count == 3
    mock_client.free_memory.assert_called_with(unload_models=True)


def test_batch_swapper_handles_partial_failure(tmp_path: Path) -> None:
    """Verify that if one model fails, remaining models in batch continue processing."""
    models_dir = tmp_path / "input_models"
    models_dir.mkdir()
    (models_dir / "model_ok.png").write_bytes(b"portrait_ok")
    (models_dir / "model_fail.png").write_bytes(b"portrait_fail")

    campaign_file = tmp_path / "campaign.png"
    campaign_file.write_bytes(b"campaign_bytes")

    output_dir = tmp_path / "output"
    workflow_file = tmp_path / "workflow.json"
    workflow_file.write_text(
        json.dumps(
            {
                "1": {"class_type": "LoadImage", "inputs": {"image": "c"}},
                "3": {"class_type": "LoadImage", "inputs": {"image": "m"}},
                "14": {"class_type": "SaveImage", "inputs": {"filename_prefix": "o"}},
            }
        ),
        encoding="utf-8",
    )

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign_file,
        mask_path=None,
        models_dir=models_dir,
        output_dir=output_dir,
        market_tag="apac",
        workflow_path=workflow_file,
    )

    mock_client = make_mock_client()
    mock_client.upload_image.return_value = "img.png"
    mock_client.queue_prompt.return_value = "prompt-1"

    def mock_track(
        prompt_id: str,
        client_id: str,
        timeout_seconds: float,
        on_progress=None,
        **kwargs,
    ):
        if mock_client.track_execution.call_count == 1:
            raise ComfyExecutionError("OOM on model_fail")
        return [OutputAsset(filename="ok.png", subfolder="", image_type="output")]

    mock_client.track_execution.side_effect = mock_track

    def mock_download(asset: OutputAsset, destination: Path) -> None:
        destination.write_bytes(b"ok")

    mock_client.download_image.side_effect = mock_download

    swapper = BatchSwapper(config=config, client=mock_client)
    results = swapper.run_batch()

    assert len(results) == 2
    failed_results = [r for r in results if r.status == JobStatus.FAILED]
    success_results = [r for r in results if r.status == JobStatus.SUCCESS]

    assert len(failed_results) == 1
    assert "OOM on model_fail" in (failed_results[0].error_message or "")
    assert len(success_results) == 1


def test_batch_swapper_empty_models_dir(tmp_path: Path) -> None:
    """Verify run_batch returns empty list when models directory has no image files."""
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    workflow_file = tmp_path / "wf.json"
    workflow_file.write_text(
        json.dumps({"1": {"class_type": "LoadImage", "inputs": {}}}), encoding="utf-8"
    )

    campaign_file = tmp_path / "c.png"
    campaign_file.write_bytes(b"dummy_campaign")

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign_file,
        mask_path=None,
        models_dir=empty_dir,
        output_dir=tmp_path / "out",
        market_tag="apac",
        workflow_path=workflow_file,
    )
    swapper = BatchSwapper(config=config, client=MagicMock())
    results = swapper.run_batch()
    assert results == []


def test_batch_swapper_zero_outputs_error(tmp_path: Path) -> None:
    """Verify process_single_model fails if execution outputs zero images."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m1.png").write_bytes(b"model")

    campaign_file = tmp_path / "c.png"
    campaign_file.write_bytes(b"dummy_campaign")

    workflow_file = tmp_path / "wf.json"
    workflow_file.write_text(
        json.dumps(
            {
                "1": {"class_type": "LoadImage", "inputs": {}},
                "3": {"class_type": "LoadImage", "inputs": {}},
                "14": {"class_type": "SaveImage", "inputs": {}},
            }
        ),
        encoding="utf-8",
    )

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign_file,
        mask_path=None,
        models_dir=models_dir,
        output_dir=tmp_path / "out",
        market_tag="emea",
        workflow_path=workflow_file,
    )

    mock_client = MagicMock(spec=ComfyUIClient)
    mock_client.upload_image.return_value = "uploaded.png"
    mock_client.queue_prompt.return_value = "prompt-1"
    mock_client.track_execution.return_value = []

    swapper = BatchSwapper(config=config, client=mock_client)
    res = swapper.process_single_model(
        model_file=models_dir / "m1.png",
        campaign_asset_name="c.png",
        mask_asset_name=None,
    )
    assert res.status == JobStatus.FAILED
    assert "zero output assets" in (res.error_message or "")


def test_swapper_config_validation_failures(tmp_path: Path) -> None:
    """Verify SwapperConfig.__post_init__ catches invalid inputs fail-fast."""
    valid_file = tmp_path / "valid.png"
    valid_file.write_bytes(b"data")
    valid_wf = tmp_path / "wf.json"
    valid_wf.write_bytes(b"{}")
    valid_dir = tmp_path / "models"
    valid_dir.mkdir()

    # 1. Non-existent campaign
    with pytest.raises(FileNotFoundError, match="Campaign image does not exist"):
        SwapperConfig(
            server_address="127.0.0.1:8188",
            campaign_path=tmp_path / "nonexistent.png",
            models_dir=valid_dir,
            output_dir=tmp_path / "out",
            workflow_path=valid_wf,
        )

    # 2. Non-existent models directory
    with pytest.raises(NotADirectoryError, match="Models directory does not exist"):
        SwapperConfig(
            server_address="127.0.0.1:8188",
            campaign_path=valid_file,
            models_dir=tmp_path / "nonexistent_dir",
            output_dir=tmp_path / "out",
            workflow_path=valid_wf,
        )

    # 3. Non-existent workflow template
    with pytest.raises(FileNotFoundError, match="Workflow file does not exist"):
        SwapperConfig(
            server_address="127.0.0.1:8188",
            campaign_path=valid_file,
            models_dir=valid_dir,
            output_dir=tmp_path / "out",
            workflow_path=tmp_path / "nonexistent_wf.json",
        )

    # 4. Non-existent mask when explicitly specified
    with pytest.raises(FileNotFoundError, match="Mask file does not exist"):
        SwapperConfig(
            server_address="127.0.0.1:8188",
            campaign_path=valid_file,
            mask_path=tmp_path / "nonexistent_mask.png",
            models_dir=valid_dir,
            output_dir=tmp_path / "out",
            workflow_path=valid_wf,
        )

    # 5. Invalid non-positive timeout
    with pytest.raises(ValueError, match="Timeout must be positive"):
        SwapperConfig(
            server_address="127.0.0.1:8188",
            campaign_path=valid_file,
            models_dir=valid_dir,
            output_dir=tmp_path / "out",
            workflow_path=valid_wf,
            timeout_seconds=0.0,
        )


def test_render_progress_bar_non_tty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify progress bar logs structured 25% milestone updates in CI / non-TTY environments."""
    campaign_file = tmp_path / "c.png"
    campaign_file.write_bytes(b"data")
    wf_file = tmp_path / "wf.json"
    wf_file.write_text(
        json.dumps({"1": {"class_type": "LoadImage", "inputs": {}}}), encoding="utf-8"
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign_file,
        models_dir=models_dir,
        output_dir=tmp_path / "out",
        workflow_path=wf_file,
    )
    swapper = BatchSwapper(config=config, client=MagicMock())

    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    with caplog.at_level(logging.INFO):
        swapper._render_progress_bar(5, 20, "node_1")
        assert any("25.0%" in record.message for record in caplog.records)


def test_batch_swapper_preflight_connection_failure(tmp_path: Path) -> None:
    """Verify run_batch raises ComfyAPIError with troubleshooting tips if server is offline."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m.png").write_bytes(b"data")
    campaign = tmp_path / "c.png"
    campaign.write_bytes(b"data")
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"1": {"class_type": "LoadImage", "inputs": {}}}), encoding="utf-8")

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign,
        models_dir=models_dir,
        output_dir=tmp_path / "out",
        workflow_path=wf,
    )
    mock_client = MagicMock(spec=ComfyUIClient)
    mock_client.check_connection.return_value = False
    mock_client._http_base = "http://127.0.0.1:8188"

    swapper = BatchSwapper(config=config, client=mock_client)
    with pytest.raises(ComfyAPIError, match="Cannot connect to ComfyUI"):
        swapper.run_batch()


def test_batch_swapper_skip_existing_all_skipped(tmp_path: Path) -> None:
    """Verify that when --skip-existing is set and all files exist, no server calls occur."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m1.png").write_bytes(b"model1")
    (models_dir / "m2.png").write_bytes(b"model2")

    campaign_file = tmp_path / "camp.png"
    campaign_file.write_bytes(b"camp")

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "camp_m1.png").write_bytes(b"already_rendered_1")
    (output_dir / "camp_m2.png").write_bytes(b"already_rendered_2")

    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"1": {"class_type": "LoadImage", "inputs": {}}}), encoding="utf-8")

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign_file,
        models_dir=models_dir,
        output_dir=output_dir,
        workflow_path=wf,
        skip_existing=True,
    )
    mock_client = MagicMock(spec=ComfyUIClient)

    swapper = BatchSwapper(config=config, client=mock_client)
    results = swapper.run_batch()

    assert len(results) == 2
    assert all(r.status == JobStatus.SKIPPED for r in results)
    assert all(r.duration_seconds == 0.0 for r in results)
    assert mock_client.check_connection.call_count == 0
    assert mock_client.upload_image.call_count == 0
    assert mock_client.queue_prompt.call_count == 0


def test_batch_swapper_skip_existing_partial(tmp_path: Path) -> None:
    """Verify that only existing files are skipped while missing ones are rendered."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m_exist.png").write_bytes(b"model_exist")
    (models_dir / "m_new.png").write_bytes(b"model_new")

    campaign_file = tmp_path / "camp.png"
    campaign_file.write_bytes(b"camp")

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "camp_m_exist.png").write_bytes(b"existing_data")

    wf = tmp_path / "wf.json"
    wf_graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "{{CAMPAIGN_IMG}}"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "{{MODEL_IMG}}"}},
        "3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "{{OUTPUT_PREFIX}}"}},
    }
    wf.write_text(json.dumps(wf_graph), encoding="utf-8")

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign_file,
        models_dir=models_dir,
        output_dir=output_dir,
        workflow_path=wf,
        skip_existing=True,
    )
    mock_client = make_mock_client()
    mock_client.upload_image.side_effect = lambda path: f"uploaded_{path.name}"
    mock_client.queue_prompt.return_value = "prompt-123"
    mock_client.track_execution.return_value = [
        OutputAsset(filename="out.png", subfolder="", image_type="output")
    ]

    def mock_download(asset: OutputAsset, destination: Path) -> None:
        destination.write_bytes(b"newly_rendered")

    mock_client.download_image.side_effect = mock_download

    swapper = BatchSwapper(config=config, client=mock_client)
    results = swapper.run_batch()

    assert len(results) == 2
    status_by_model = {r.model_name: r.status for r in results}
    assert status_by_model["m_exist.png"] == JobStatus.SKIPPED
    assert status_by_model["m_new.png"] == JobStatus.SUCCESS

    uploaded_names = [call[0][0].name for call in mock_client.upload_image.call_args_list]
    assert "camp.png" in uploaded_names
    assert "m_new.png" in uploaded_names
    assert "m_exist.png" not in uploaded_names


def test_batch_swapper_force_overrides_skip_existing(tmp_path: Path) -> None:
    """Verify --force forces re-rendering even if output exists."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m.png").write_bytes(b"model")

    campaign_file = tmp_path / "camp.png"
    campaign_file.write_bytes(b"camp")

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "camp_m.png").write_bytes(b"old_data")

    wf = tmp_path / "wf.json"
    wf_graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "{{CAMPAIGN_IMG}}"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "{{MODEL_IMG}}"}},
        "3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "{{OUTPUT_PREFIX}}"}},
    }
    wf.write_text(json.dumps(wf_graph), encoding="utf-8")

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign_file,
        models_dir=models_dir,
        output_dir=output_dir,
        workflow_path=wf,
        skip_existing=True,
        force=True,
    )
    mock_client = make_mock_client()
    mock_client.upload_image.return_value = "up.png"
    mock_client.queue_prompt.return_value = "prompt-id"
    mock_client.track_execution.return_value = [
        OutputAsset(filename="out.png", subfolder="", image_type="output")
    ]

    swapper = BatchSwapper(config=config, client=mock_client)
    results = swapper.run_batch()

    assert len(results) == 1
    assert results[0].status == JobStatus.SUCCESS
    assert mock_client.queue_prompt.call_count == 1


def test_batch_swapper_skip_existing_corrupt_file_rerendered(tmp_path: Path) -> None:
    """Verify that a 0-byte corrupt file is NOT skipped when --skip-existing is set."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m.png").write_bytes(b"model")

    campaign_file = tmp_path / "camp.png"
    campaign_file.write_bytes(b"camp")

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "camp_m.png").touch()
    assert (output_dir / "camp_m.png").stat().st_size == 0

    wf = tmp_path / "wf.json"
    wf_graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "{{CAMPAIGN_IMG}}"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "{{MODEL_IMG}}"}},
        "3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "{{OUTPUT_PREFIX}}"}},
    }
    wf.write_text(json.dumps(wf_graph), encoding="utf-8")

    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign_file,
        models_dir=models_dir,
        output_dir=output_dir,
        workflow_path=wf,
        skip_existing=True,
    )
    mock_client = make_mock_client()
    mock_client.upload_image.return_value = "up.png"
    mock_client.queue_prompt.return_value = "prompt-id"
    mock_client.track_execution.return_value = [
        OutputAsset(filename="out.png", subfolder="", image_type="output")
    ]

    swapper = BatchSwapper(config=config, client=mock_client)
    results = swapper.run_batch()

    assert len(results) == 1
    assert results[0].status == JobStatus.SUCCESS
    assert mock_client.queue_prompt.call_count == 1


def test_batch_swapper_validate_workflow_nodes(tmp_path: Path) -> None:
    """Verify validate_workflow_nodes checks custom nodes against server registry."""
    wf_file = tmp_path / "wf.json"
    wf_file.write_text(
        json.dumps(
            {
                "1": {"class_type": "LoadImage", "inputs": {}},
                "2": {"class_type": "SpecialCustomNode", "inputs": {}},
            }
        ),
        encoding="utf-8",
    )
    camp = tmp_path / "camp.png"
    camp.touch()
    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=camp,
        mask_path=None,
        models_dir=tmp_path,
        output_dir=tmp_path,
        workflow_path=wf_file,
    )

    # 1. Success case: all nodes available
    mock_client_ok = MagicMock(spec=ComfyUIClient)
    mock_client_ok.http_base = "http://127.0.0.1:8188"
    mock_client_ok.get_registered_node_types.return_value = {
        "LoadImage",
        "SpecialCustomNode",
        "Other",
    }
    swapper_ok = BatchSwapper(config=config, client=mock_client_ok)
    swapper_ok.validate_workflow_nodes()

    # 2. Failure case: SpecialCustomNode missing
    mock_client_missing = MagicMock(spec=ComfyUIClient)
    mock_client_missing.http_base = "http://127.0.0.1:8188"
    mock_client_missing.get_registered_node_types.return_value = {"LoadImage", "KSampler"}
    swapper_missing = BatchSwapper(config=config, client=mock_client_missing)
    with pytest.raises(
        ComfyAPIError, match=r"(?s)missing required custom node type.*SpecialCustomNode"
    ):
        swapper_missing.validate_workflow_nodes()

    # 3. Fail-closed: endpoint unavailable (returns None) raises ComfyAPIError
    mock_client_none = MagicMock(spec=ComfyUIClient)
    mock_client_none.http_base = "http://127.0.0.1:8188"
    mock_client_none.get_registered_node_types.return_value = None
    swapper_none = BatchSwapper(config=config, client=mock_client_none)
    with pytest.raises(ComfyAPIError, match=r"(?s)Cannot query ComfyUI /object_info"):
        swapper_none.validate_workflow_nodes()


def test_batch_swapper_run_batch_preflight_node_validation_aborts(tmp_path: Path) -> None:
    """Verify run_batch aborts immediately before image ingestion if nodes are missing."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m.png").write_bytes(b"data")
    campaign = tmp_path / "c.png"
    campaign.write_bytes(b"data")
    wf_file = tmp_path / "wf.json"
    wf_file.write_text(
        json.dumps({"1": {"class_type": "MissingCustomNode", "inputs": {}}}),
        encoding="utf-8",
    )
    config = SwapperConfig(
        server_address="127.0.0.1:8188",
        campaign_path=campaign,
        mask_path=None,
        models_dir=models_dir,
        output_dir=tmp_path / "out",
        workflow_path=wf_file,
    )

    mock_client = MagicMock(spec=ComfyUIClient)
    mock_client.http_base = "http://127.0.0.1:8188"
    mock_client.check_connection.return_value = True
    mock_client.get_registered_node_types.return_value = {"LoadImage", "KSampler"}

    swapper = BatchSwapper(config=config, client=mock_client)
    with pytest.raises(ComfyAPIError, match="missing required custom node type"):
        swapper.run_batch()

    assert mock_client.upload_image.call_count == 0
