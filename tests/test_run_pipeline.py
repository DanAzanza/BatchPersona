"""Unit and integration tests for scripts/run_pipeline.py."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.launch_comfyui import ComfyInstance
from scripts.run_pipeline import (
    display_menu,
    ensure_server_ready,
    interactive_menu_loop,
    main,
    run_commercial_swap,
    run_quality_gate,
    run_quick_dry_run,
)


def test_ensure_server_ready_already_online() -> None:
    """Verify ensure_server_ready returns True immediately when online."""
    with patch("scripts.run_pipeline.is_server_online", return_value=True):
        assert ensure_server_ready("127.0.0.1:8000") is True


def test_ensure_server_ready_detect_none() -> None:
    """Verify ensure_server_ready returns False when offline and no instance found."""
    with (
        patch("scripts.run_pipeline.is_server_online", return_value=False),
        patch("scripts.run_pipeline.detect_comfyui", return_value=None),
    ):
        assert ensure_server_ready("127.0.0.1:8000") is False


def test_ensure_server_ready_launch_success() -> None:
    """Verify ensure_server_ready returns True when launched successfully."""
    mock_inst = ComfyInstance(name="Test", mode="headless")
    with (
        patch("scripts.run_pipeline.is_server_online", return_value=False),
        patch("scripts.run_pipeline.detect_comfyui", return_value=mock_inst),
        patch("scripts.run_pipeline.launch_server", return_value=True),
    ):
        assert ensure_server_ready("127.0.0.1:8000") is True


def test_ensure_server_ready_launch_failure() -> None:
    """Verify ensure_server_ready returns False when launch fails."""
    mock_inst = ComfyInstance(name="Test", mode="headless")
    with (
        patch("scripts.run_pipeline.is_server_online", return_value=False),
        patch("scripts.run_pipeline.detect_comfyui", return_value=mock_inst),
        patch("scripts.run_pipeline.launch_server", return_value=False),
    ):
        assert ensure_server_ready("127.0.0.1:8000") is False


def test_run_commercial_swap_server_offline() -> None:
    """Verify run_commercial_swap aborts if server cannot be made ready."""
    with patch("scripts.run_pipeline.ensure_server_ready", return_value=False):
        assert run_commercial_swap() == 1


def test_run_commercial_swap_nonexistent_dir(tmp_path: Path) -> None:
    """Verify run_commercial_swap returns 1 when campaign_dir is missing."""
    with patch("scripts.run_pipeline.ensure_server_ready", return_value=True):
        assert run_commercial_swap(campaign_dir=tmp_path / "nonexistent") == 1


def test_run_commercial_swap_empty_dir(tmp_path: Path) -> None:
    """Verify run_commercial_swap returns 0 when no supported images found."""
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    with patch("scripts.run_pipeline.ensure_server_ready", return_value=True):
        assert run_commercial_swap(campaign_dir=campaign_dir) == 0


def test_run_commercial_swap_success_flow(tmp_path: Path) -> None:
    """Verify run_commercial_swap iterates over all valid images."""
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / "lookbook_a.png").touch()
    (campaign_dir / "lookbook_b.jpg").touch()

    mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))

    with (
        patch("scripts.run_pipeline.ensure_server_ready", return_value=True),
        patch("subprocess.run", mock_run),
    ):
        code = run_commercial_swap(campaign_dir=campaign_dir)
        assert code == 0
        assert mock_run.call_count == 2


def test_run_commercial_swap_partial_failure(tmp_path: Path) -> None:
    """Verify run_commercial_swap records non-zero exit code if one item fails."""
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / "lookbook_1.png").touch()

    mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=42))

    with (
        patch("scripts.run_pipeline.ensure_server_ready", return_value=True),
        patch("subprocess.run", mock_run),
    ):
        code = run_commercial_swap(campaign_dir=campaign_dir)
        assert code == 42


def test_run_quick_dry_run_server_offline() -> None:
    """Verify run_quick_dry_run aborts when server is offline."""
    with patch("scripts.run_pipeline.ensure_server_ready", return_value=False):
        assert run_quick_dry_run() == 1


def test_run_quick_dry_run_with_missing_synthetic_data(tmp_path: Path) -> None:
    """Verify synthetic data generation is triggered if missing."""
    synthetic_dir = tmp_path / "synthetic"
    mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))

    with (
        patch("scripts.run_pipeline.ensure_server_ready", return_value=True),
        patch("subprocess.run", mock_run),
    ):
        code = run_quick_dry_run(synthetic_dir=synthetic_dir)
        assert code == 0
        # 1 call to generate_testdata, 1 call to batch_swapper
        assert mock_run.call_count == 2


def test_run_quick_dry_run_generation_failure(tmp_path: Path) -> None:
    """Verify failure to generate synthetic data stops execution."""
    synthetic_dir = tmp_path / "synthetic"
    mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=5))

    with (
        patch("scripts.run_pipeline.ensure_server_ready", return_value=True),
        patch("subprocess.run", mock_run),
    ):
        code = run_quick_dry_run(synthetic_dir=synthetic_dir)
        assert code == 5
        assert mock_run.call_count == 1


def test_run_quick_dry_run_existing_data(tmp_path: Path) -> None:
    """Verify synthetic data generation is skipped if already present."""
    synthetic_dir = tmp_path / "synthetic"
    camp_dir = synthetic_dir / "input_campaign"
    camp_dir.mkdir(parents=True)
    (camp_dir / "campaign_summer_lookbook.png").touch()

    mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))

    with (
        patch("scripts.run_pipeline.ensure_server_ready", return_value=True),
        patch("subprocess.run", mock_run),
    ):
        code = run_quick_dry_run(synthetic_dir=synthetic_dir)
        assert code == 0
        assert mock_run.call_count == 1


def test_run_quality_gate() -> None:
    """Verify run_quality_gate invokes run_ci_locally.py."""
    mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
    with patch("subprocess.run", mock_run):
        code = run_quality_gate()
        assert code == 0
        assert mock_run.call_count == 1


def test_display_menu(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify menu layout renders essential options."""
    display_menu("127.0.0.1:8000")
    captured = capsys.readouterr().out
    assert "BatchPersona" in captured
    assert "[1] Run Commercial Model Swap" in captured
    assert "[2] Run Quick Dry-Run" in captured
    assert "[3] Run Quality Gate and Tests" in captured
    assert "[0] Exit" in captured


def test_interactive_menu_loop_exit_immediately() -> None:
    """Verify menu exits cleanly on option 0."""
    with patch("builtins.input", side_effect=["0"]):
        assert interactive_menu_loop("127.0.0.1:8000") == 0


def test_interactive_menu_loop_options() -> None:
    """Verify menu loop executes choices and exits."""
    with (
        patch("builtins.input", side_effect=["1", "", "2", "", "3", "", "invalid", "0"]),
        patch("scripts.run_pipeline.run_commercial_swap", return_value=0) as mock_swap,
        patch("scripts.run_pipeline.run_quick_dry_run", return_value=0) as mock_dry,
        patch("scripts.run_pipeline.run_quality_gate", return_value=0) as mock_ci,
    ):
        assert interactive_menu_loop("127.0.0.1:8000") == 0
        assert mock_swap.call_count == 1
        assert mock_dry.call_count == 1
        assert mock_ci.call_count == 1


def test_interactive_menu_loop_keyboard_interrupt() -> None:
    """Verify graceful handling of KeyboardInterrupt."""
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        assert interactive_menu_loop("127.0.0.1:8000") == 0


def test_main_with_cli_args() -> None:
    """Verify main forwards direct CLI arguments to batch_swapper.py."""
    mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
    with patch("subprocess.run", mock_run):
        code = main(["--campaign", "lookbook.png", "--models-dir", "models"])
        assert code == 0
        assert mock_run.call_count == 1
        called_args = mock_run.call_args[0][0]
        assert "scripts/batch_swapper.py" in called_args[1]
        assert "--campaign" in called_args


def test_main_without_args() -> None:
    """Verify main triggers interactive menu loop when no arguments are provided."""
    with patch("scripts.run_pipeline.interactive_menu_loop", return_value=0) as mock_menu:
        assert main([]) == 0
        assert mock_menu.call_count == 1
