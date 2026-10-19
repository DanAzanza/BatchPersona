"""Unit tests for command-line arguments and script entrypoints."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from batchpersona.client import ComfyUIClient
from batchpersona.models import (
    DEFAULT_SERVER_ADDRESS,
    JobResult,
    JobStatus,
    SwapperConfig,
)
from batchpersona.orchestrator import BatchSwapper, parse_cli_args
from scripts.batch_swapper import configure_logging
from scripts.batch_swapper import main as batch_main
from scripts.prepare_fullbody_dataset import main as prep_main
from scripts.prepare_fullbody_dataset import parse_cli_args as prep_parse_cli_args


def test_parse_cli_args() -> None:
    """Verify CLI argument parser defaults and flags."""
    args = parse_cli_args(
        [
            "--campaign",
            "data/campaign.png",
            "--models-dir",
            "data/models",
            "--market-tag",
            "latam",
            "--timeout",
            "120",
        ]
    )
    assert args.campaign == Path("data/campaign.png")
    assert args.models_dir == Path("data/models")
    assert args.market_tag == "latam"
    assert args.timeout == 120.0
    assert args.server == DEFAULT_SERVER_ADDRESS
    assert args.server == "127.0.0.1:8000"
    assert args.filter == "*"
    assert args.workflow == Path("workflows/model_swap_qwen21_maskless_api.json")

    args_custom = parse_cli_args(
        [
            "--campaign",
            "c.png",
            "--models-dir",
            "m",
            "--filter",
            "*_f*",
        ]
    )
    assert args_custom.filter == "*_f*"


def test_parse_cli_args_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify parse_cli_args respects COMFYUI_SERVER environment variable."""
    monkeypatch.setenv("COMFYUI_SERVER", "192.168.1.50:8000")
    args = parse_cli_args(["--campaign", "c.png", "--models-dir", "m"])
    assert args.server == "192.168.1.50:8000"


def test_parse_cli_args_new_overrides() -> None:
    """Verify parse_cli_args supports --skip-existing, --force, --seed, --resolution, --prompt."""
    cli_args = [
        "--campaign",
        "c.png",
        "--models-dir",
        "models",
        "--skip-existing",
        "--force",
        "--seed",
        "424242",
        "--resolution",
        "1024",
        "--prompt",
        "A high fashion lookbook model",
    ]
    parsed = parse_cli_args(cli_args)
    assert parsed.skip_existing is True
    assert parsed.force is True
    assert parsed.seed == 424242
    assert parsed.resolution == 1024
    assert parsed.prompt == "A high fashion lookbook model"
    assert parsed.keep_workflow_metadata is False

    parsed_keep = parse_cli_args(
        ["--campaign", "c.png", "--models-dir", "m", "--keep-workflow-metadata"]
    )
    assert parsed_keep.keep_workflow_metadata is True


def test_prepare_dataset_cli_args() -> None:
    """Verify parse_cli_args in prepare_fullbody_dataset parses paths cleanly."""
    args = prep_parse_cli_args(["--source-dir", "custom/src", "--output-dir", "custom/dst"])
    assert args.source_dir == Path("custom/src")
    assert args.output_dir == Path("custom/dst")
    assert args.campaign_src is None
    assert args.models_src is None

    args_custom = prep_parse_cli_args(
        [
            "--campaign-src",
            "campaigns/shoot.jpg",
            "--models-src",
            "portraits/",
            "--output-dir",
            "custom/data",
        ]
    )
    assert args_custom.campaign_src == Path("campaigns/shoot.jpg")
    assert args_custom.models_src == Path("portraits/")
    assert args_custom.output_dir == Path("custom/data")


def test_configure_logging() -> None:
    """Verify configure_logging initializes root logger without error."""
    configure_logging(level=logging.DEBUG)
    assert logging.getLogger().level == logging.DEBUG


def test_batch_swapper_main_zero_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify main() CLI exits with code 2 when zero models match."""
    campaign = tmp_path / "c.png"
    campaign.write_bytes(b"data")
    models = tmp_path / "models"
    models.mkdir()
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"1": {"class_type": "LoadImage", "inputs": {}}}), encoding="utf-8")

    test_args = [
        "batch_swapper.py",
        "--campaign",
        str(campaign),
        "--models-dir",
        str(models),
        "--output-dir",
        str(tmp_path / "out"),
        "--workflow",
        str(wf),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as exc_info:
        batch_main()
    assert exc_info.value.code == 2


def test_prepare_fullbody_dataset_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify scripts.prepare_fullbody_dataset main entrypoint handles exit codes."""
    empty_src = tmp_path / "empty_artifacts"
    empty_src.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_fullbody_dataset.py",
            "--source-dir",
            str(empty_src),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        prep_main()
    assert exc_info.value.code == 1


def test_prepare_fullbody_dataset_main_explicit_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify scripts.prepare_fullbody_dataset main entrypoint exits 0 upon successful processing."""
    from PIL import Image

    src_img = tmp_path / "custom_campaign.png"
    Image.new("RGB", (500, 700), color=(12, 34, 56)).save(src_img)
    out_dir = tmp_path / "out_cli"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_fullbody_dataset.py",
            "--campaign-src",
            str(src_img),
            "--output-dir",
            str(out_dir),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        prep_main()
    assert exc_info.value.code == 0
    assert (out_dir / "input_campaign" / "custom_campaign.png").is_file()


def test_batch_swapper_main_exit_code_zero_when_all_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify main returns exit code 0 when all outputs are skipped via --skip-existing."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "m.png").write_bytes(b"data")
    campaign = tmp_path / "c.png"
    campaign.write_bytes(b"data")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "c_m.png").write_bytes(b"already_done")
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"1": {"class_type": "LoadImage", "inputs": {}}}), encoding="utf-8")

    test_args = [
        "batch_swapper.py",
        "--campaign",
        str(campaign),
        "--models-dir",
        str(models_dir),
        "--output-dir",
        str(out_dir),
        "--workflow",
        str(wf),
        "--skip-existing",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as exc_info:
        batch_main()
    assert exc_info.value.code == 0


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

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    with caplog.at_level(logging.INFO):
        swapper._render_progress_bar(5, 20, "node_1")
        assert any("25.0%" in record.message for record in caplog.records)


def test_print_batch_summary_displays_market_tag(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Verify _print_batch_summary renders market localization in header and column."""
    campaign = tmp_path / "camp.png"
    campaign.touch()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"1": {"class_type": "LoadImage", "inputs": {}}}), encoding="utf-8")
    cfg = SwapperConfig(
        server_address="127.0.0.1:8000",
        campaign_path=campaign,
        models_dir=models_dir,
        output_dir=tmp_path / "out",
        workflow_path=wf,
        market_tag="apac",
    )
    swapper = BatchSwapper(config=cfg, client=MagicMock(spec=ComfyUIClient))
    results = [
        JobResult(
            model_name="model_01.png",
            output_path=tmp_path / "out" / "camp_apac_model_01.png",
            status=JobStatus.SUCCESS,
            duration_seconds=3.5,
            market_tag="apac",
        ),
        JobResult(
            model_name="model_02.png",
            output_path=None,
            status=JobStatus.FAILED,
            duration_seconds=1.2,
            error_message="Network glitch",
            market_tag="apac",
        ),
    ]
    swapper._print_batch_summary(results)
    captured = capsys.readouterr().out
    assert "[MARKET: APAC]" in captured
    assert "MARKET" in captured
    assert "apac" in captured
    assert "camp_apac_model_01.png" in captured
    assert "Network glitch" in captured
