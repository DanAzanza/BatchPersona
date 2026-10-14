"""Unit and contract tests for batchpersona domain models and configuration schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from batchpersona.models import (
    DEFAULT_SERVER_ADDRESS,
    JobResult,
    JobStatus,
    OutputAsset,
    SwapperConfig,
    normalize_server_address,
)


def test_normalize_server_address() -> None:
    """Verify normalize_server_address strips protocol schemes and trailing slashes."""
    assert normalize_server_address("http://127.0.0.1:8188/") == "127.0.0.1:8188"
    assert normalize_server_address("https://remote.server:9000") == "remote.server:9000"
    assert normalize_server_address("ws://127.0.0.1:8000") == "127.0.0.1:8000"
    assert normalize_server_address("  127.0.0.1:8000/  ") == "127.0.0.1:8000"
    assert normalize_server_address("") == DEFAULT_SERVER_ADDRESS
    assert normalize_server_address("   ") == DEFAULT_SERVER_ADDRESS


def test_output_asset_and_job_models(tmp_path: Path) -> None:
    """Verify OutputAsset and JobResult dataclass integrity."""
    asset = OutputAsset(filename="result.png", subfolder="batch", image_type="output")
    assert asset.filename == "result.png"
    assert asset.subfolder == "batch"
    assert asset.image_type == "output"

    out_file = tmp_path / "result.png"
    result_ok = JobResult(
        model_name="model_01.png",
        output_path=out_file,
        status=JobStatus.SUCCESS,
        duration_seconds=4.2,
        market_tag="emea",
    )
    assert result_ok.status == JobStatus.SUCCESS
    assert result_ok.error_message is None
    assert result_ok.market_tag == "emea"

    result_fail = JobResult(
        model_name="model_02.png",
        output_path=None,
        status=JobStatus.FAILED,
        duration_seconds=1.1,
        error_message="OOM in KSampler",
    )
    assert result_fail.status == JobStatus.FAILED
    assert result_fail.error_message == "OOM in KSampler"


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


def test_swapper_config_market_tag_target_path_and_prefix(tmp_path: Path) -> None:
    """Verify SwapperConfig derives target path and output prefix according to market_tag."""
    campaign = tmp_path / "lookbook.png"
    campaign.touch()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"1": {"class_type": "LoadImage", "inputs": {}}}), encoding="utf-8")
    output_dir = tmp_path / "output"
    model_file = Path("model_east_asia.png")

    # 1. Default "global" -> backward-compatible un-tagged filename
    cfg_default = SwapperConfig(
        server_address="127.0.0.1:8000",
        campaign_path=campaign,
        models_dir=models_dir,
        output_dir=output_dir,
        workflow_path=wf,
    )
    assert cfg_default.clean_market_tag == ""
    assert cfg_default.get_target_path(model_file) == output_dir / "lookbook_model_east_asia.png"
    assert cfg_default.get_output_prefix(model_file) == "batchpersona_lookbook_model_east_asia"

    # 2. Case-insensitive "GLOBAL" or whitespace/empty -> un-tagged
    for tag in ("GLOBAL", "Global", "", "   "):
        cfg = SwapperConfig(
            server_address="127.0.0.1:8000",
            campaign_path=campaign,
            models_dir=models_dir,
            output_dir=output_dir,
            workflow_path=wf,
            market_tag=tag,
        )
        assert cfg.clean_market_tag == ""
        assert cfg.get_target_path(model_file) == output_dir / "lookbook_model_east_asia.png"
        assert cfg.get_output_prefix(model_file) == "batchpersona_lookbook_model_east_asia"

    # 3. Non-global market tags -> sanitized lowercase included in path and prefix
    cases = [
        (
            "apac",
            "apac",
            "lookbook_apac_model_east_asia.png",
            "batchpersona_lookbook_apac_model_east_asia",
        ),
        (
            "LATAM",
            "latam",
            "lookbook_latam_model_east_asia.png",
            "batchpersona_lookbook_latam_model_east_asia",
        ),
        (
            "US / East Coast",
            "us_east_coast",
            "lookbook_us_east_coast_model_east_asia.png",
            "batchpersona_lookbook_us_east_coast_model_east_asia",
        ),
        (
            "eu-north-1",
            "eu-north-1",
            "lookbook_eu-north-1_model_east_asia.png",
            "batchpersona_lookbook_eu-north-1_model_east_asia",
        ),
    ]
    for tag, expected_clean, expected_filename, expected_prefix in cases:
        cfg = SwapperConfig(
            server_address="127.0.0.1:8000",
            campaign_path=campaign,
            models_dir=models_dir,
            output_dir=output_dir,
            workflow_path=wf,
            market_tag=tag,
        )
        assert cfg.clean_market_tag == expected_clean
        assert cfg.get_target_path(model_file) == output_dir / expected_filename
        assert cfg.get_output_prefix(model_file) == expected_prefix
