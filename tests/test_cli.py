"""Unit tests for command-line arguments and script entrypoints."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

from batchpersona.orchestrator import parse_cli_args
from scripts.batch_swapper import configure_logging
from scripts.batch_swapper import main as batch_main
from scripts.generate_testdata import main as gen_main
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
    assert args.server == "127.0.0.1:8188"
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


def test_prepare_dataset_cli_args() -> None:
    """Verify parse_cli_args in prepare_fullbody_dataset parses paths cleanly."""
    args = prep_parse_cli_args(["--source-dir", "custom/src", "--output-dir", "custom/dst"])
    assert args.source_dir == Path("custom/src")
    assert args.output_dir == Path("custom/dst")


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


def test_generate_testdata_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify scripts.generate_testdata main entrypoint runs smoothly."""
    out_dir = tmp_path / "test_data"
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_testdata.py", "--output-dir", str(out_dir), "--size", "64"],
    )
    gen_main()
    assert (out_dir / "input_campaign" / "campaign_summer_lookbook.png").is_file()


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
