"""Unit and integration tests for BatchSwapper incremental caching and --skip-existing flows."""

from __future__ import annotations

import json
from pathlib import Path

from batchpersona.models import JobStatus, OutputAsset, SwapperConfig
from batchpersona.orchestrator import BatchSwapper
from tests.conftest import make_mock_client


def test_batch_swapper_skip_existing_all_skipped(tmp_path: Path) -> None:
    """Verify run_batch immediately skips rendering when all outputs exist and are valid."""
    campaign = tmp_path / "lookbook.png"
    campaign.touch()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m1.png").touch()
    (models_dir / "m2.png").touch()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"1": {"class_type": "LoadImage", "inputs": {}}}), encoding="utf-8")

    # Pre-populate both outputs
    (output_dir / "lookbook_m1.png").write_bytes(b"existing_valid_output_1")
    (output_dir / "lookbook_m2.png").write_bytes(b"existing_valid_output_2")

    config = SwapperConfig(
        server_address="127.0.0.1:8000",
        campaign_path=campaign,
        models_dir=models_dir,
        output_dir=output_dir,
        workflow_path=wf,
        skip_existing=True,
    )
    mock_client = make_mock_client()
    swapper = BatchSwapper(config=config, client=mock_client)
    results = swapper.run_batch()

    assert len(results) == 2
    assert all(r.status == JobStatus.SKIPPED for r in results)
    assert mock_client.upload_image.call_count == 0
    assert mock_client.queue_prompt.call_count == 0


def test_batch_swapper_skip_existing_partial(tmp_path: Path) -> None:
    """Verify run_batch skips existing models but renders missing ones in the same batch."""
    campaign = tmp_path / "lookbook.png"
    campaign.touch()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m1.png").touch()
    (models_dir / "m2.png").touch()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    wf = tmp_path / "wf.json"
    wf.write_text(
        json.dumps(
            {
                "1": {"class_type": "LoadImage", "inputs": {"image": "{{CAMPAIGN_IMG}}"}},
                "2": {"class_type": "LoadImage", "inputs": {"image": "{{MODEL_IMG}}"}},
            }
        ),
        encoding="utf-8",
    )

    # Pre-populate only m1
    (output_dir / "lookbook_m1.png").write_bytes(b"existing_valid_output_1")

    config = SwapperConfig(
        server_address="127.0.0.1:8000",
        campaign_path=campaign,
        models_dir=models_dir,
        output_dir=output_dir,
        workflow_path=wf,
        skip_existing=True,
    )
    mock_client = make_mock_client()
    mock_client.upload_image.side_effect = lambda p: f"up_{p.name}"
    mock_client.queue_prompt.return_value = "prompt_m2"
    mock_client.track_execution.return_value = [
        OutputAsset(filename="out_m2.png", subfolder="", image_type="output")
    ]
    mock_client.download_image.side_effect = lambda a, dst, *args, **kwargs: dst.write_bytes(
        b"rendered_m2"
    )

    swapper = BatchSwapper(config=config, client=mock_client)
    results = swapper.run_batch()

    assert len(results) == 2
    res_m1 = next(r for r in results if r.model_name == "m1.png")
    res_m2 = next(r for r in results if r.model_name == "m2.png")
    assert res_m1.status == JobStatus.SKIPPED
    assert res_m2.status == JobStatus.SUCCESS
    # Only campaign and m2 should be uploaded, m1 should not be processed
    assert mock_client.queue_prompt.call_count == 1
    assert (output_dir / "lookbook_m2.png").read_bytes() == b"rendered_m2"


def test_batch_swapper_force_overrides_skip_existing(tmp_path: Path) -> None:
    """Verify --force overrides --skip-existing and re-renders already existing outputs."""
    campaign = tmp_path / "lookbook.png"
    campaign.touch()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m1.png").touch()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    wf = tmp_path / "wf.json"
    wf.write_text(
        json.dumps(
            {
                "1": {"class_type": "LoadImage", "inputs": {"image": "{{CAMPAIGN_IMG}}"}},
                "2": {"class_type": "LoadImage", "inputs": {"image": "{{MODEL_IMG}}"}},
            }
        ),
        encoding="utf-8",
    )

    # Pre-populate m1
    (output_dir / "lookbook_m1.png").write_bytes(b"old_output")

    config = SwapperConfig(
        server_address="127.0.0.1:8000",
        campaign_path=campaign,
        models_dir=models_dir,
        output_dir=output_dir,
        workflow_path=wf,
        skip_existing=True,
        force=True,  # Force override
    )
    mock_client = make_mock_client()
    mock_client.upload_image.side_effect = lambda p: f"up_{p.name}"
    mock_client.queue_prompt.return_value = "prompt_m1"
    mock_client.track_execution.return_value = [
        OutputAsset(filename="out_m1.png", subfolder="", image_type="output")
    ]
    mock_client.download_image.side_effect = lambda a, dst, *args, **kwargs: dst.write_bytes(
        b"new_overwritten_output"
    )

    swapper = BatchSwapper(config=config, client=mock_client)
    results = swapper.run_batch()

    assert len(results) == 1
    assert results[0].status == JobStatus.SUCCESS
    assert (output_dir / "lookbook_m1.png").read_bytes() == b"new_overwritten_output"


def test_batch_swapper_skip_existing_corrupt_file_rerendered(tmp_path: Path) -> None:
    """Verify 0-byte corrupt output is treated as missing and re-rendered even with skip_existing."""
    campaign = tmp_path / "lookbook.png"
    campaign.touch()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m1.png").touch()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    wf = tmp_path / "wf.json"
    wf.write_text(
        json.dumps(
            {
                "1": {"class_type": "LoadImage", "inputs": {"image": "{{CAMPAIGN_IMG}}"}},
                "2": {"class_type": "LoadImage", "inputs": {"image": "{{MODEL_IMG}}"}},
            }
        ),
        encoding="utf-8",
    )

    # Pre-populate m1 as a 0-byte corrupted file
    (output_dir / "lookbook_m1.png").touch()
    assert (output_dir / "lookbook_m1.png").stat().st_size == 0

    config = SwapperConfig(
        server_address="127.0.0.1:8000",
        campaign_path=campaign,
        models_dir=models_dir,
        output_dir=output_dir,
        workflow_path=wf,
        skip_existing=True,
    )
    mock_client = make_mock_client()
    mock_client.upload_image.side_effect = lambda p: f"up_{p.name}"
    mock_client.queue_prompt.return_value = "prompt_m1"
    mock_client.track_execution.return_value = [
        OutputAsset(filename="out_m1.png", subfolder="", image_type="output")
    ]
    mock_client.download_image.side_effect = lambda a, dst, *args, **kwargs: dst.write_bytes(
        b"repaired_output"
    )

    swapper = BatchSwapper(config=config, client=mock_client)
    results = swapper.run_batch()

    assert len(results) == 1
    assert results[0].status == JobStatus.SUCCESS
    assert (output_dir / "lookbook_m1.png").read_bytes() == b"repaired_output"
