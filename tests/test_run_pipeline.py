"""Unit and integration tests for scripts/run_pipeline.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from batchpersona.launcher import ComfyInstance
from batchpersona.models import JobStatus
from batchpersona.pipeline import (
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
    with patch("batchpersona.pipeline.is_server_online", return_value=True):
        assert ensure_server_ready("127.0.0.1:8000") is True


def test_ensure_server_ready_detect_none() -> None:
    """Verify ensure_server_ready returns False when offline and no instance found."""
    with (
        patch("batchpersona.pipeline.is_server_online", return_value=False),
        patch("batchpersona.pipeline.detect_comfyui", return_value=None),
    ):
        assert ensure_server_ready("127.0.0.1:8000") is False


def test_ensure_server_ready_launch_success() -> None:
    """Verify ensure_server_ready returns True when launched successfully."""
    mock_inst = ComfyInstance(name="Test", mode="headless")
    with (
        patch("batchpersona.pipeline.is_server_online", return_value=False),
        patch("batchpersona.pipeline.detect_comfyui", return_value=mock_inst),
        patch("batchpersona.pipeline.launch_server", return_value=True),
    ):
        assert ensure_server_ready("127.0.0.1:8000") is True


def test_ensure_server_ready_launch_failure() -> None:
    """Verify ensure_server_ready returns False when launch fails."""
    mock_inst = ComfyInstance(name="Test", mode="headless")
    with (
        patch("batchpersona.pipeline.is_server_online", return_value=False),
        patch("batchpersona.pipeline.detect_comfyui", return_value=mock_inst),
        patch("batchpersona.pipeline.launch_server", return_value=False),
    ):
        assert ensure_server_ready("127.0.0.1:8000") is False


def test_run_commercial_swap_server_offline() -> None:
    """Verify run_commercial_swap aborts if server cannot be made ready."""
    with patch("batchpersona.pipeline.ensure_server_ready", return_value=False):
        assert run_commercial_swap() == 1


def test_run_commercial_swap_nonexistent_dir(tmp_path: Path) -> None:
    """Verify run_commercial_swap returns 1 when campaign_dir is missing."""
    with patch("batchpersona.pipeline.ensure_server_ready", return_value=True):
        assert run_commercial_swap(campaign_dir=tmp_path / "nonexistent") == 1


def test_run_commercial_swap_empty_dir(tmp_path: Path) -> None:
    """Verify run_commercial_swap returns 0 when no supported images found."""
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    with patch("batchpersona.pipeline.ensure_server_ready", return_value=True):
        assert run_commercial_swap(campaign_dir=campaign_dir) == 0


def test_run_commercial_swap_success_flow(tmp_path: Path) -> None:
    """Verify run_commercial_swap iterates over all valid images."""
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / "lookbook_a.png").touch()
    (campaign_dir / "lookbook_b.jpg").touch()

    mock_result = MagicMock()
    mock_result.status = "SUCCESS"

    with (
        patch("batchpersona.pipeline.ensure_server_ready", return_value=True),
        patch("batchpersona.pipeline.BatchSwapper") as mock_swapper_cls,
    ):
        mock_swapper = mock_swapper_cls.return_value
        mock_swapper.run_batch.return_value = [mock_result]

        code = run_commercial_swap(campaign_dir=campaign_dir)
        assert code == 0
        assert mock_swapper_cls.call_count == 2


def test_run_commercial_swap_partial_failure(tmp_path: Path) -> None:
    """Verify run_commercial_swap records non-zero exit code if one item fails."""
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / "lookbook_1.png").touch()

    mock_result = MagicMock()
    mock_result.status = JobStatus.FAILED

    with (
        patch("batchpersona.pipeline.ensure_server_ready", return_value=True),
        patch("batchpersona.pipeline.BatchSwapper") as mock_swapper_cls,
    ):
        mock_swapper = mock_swapper_cls.return_value
        mock_swapper.run_batch.return_value = [mock_result]

        code = run_commercial_swap(campaign_dir=campaign_dir)
        assert code == 1


def test_run_quick_dry_run_server_offline() -> None:
    """Verify run_quick_dry_run aborts when server is offline."""
    with patch("batchpersona.pipeline.ensure_server_ready", return_value=False):
        assert run_quick_dry_run() == 1


def test_run_quick_dry_run_with_missing_synthetic_data(tmp_path: Path) -> None:
    """Verify synthetic data generation is triggered if missing."""
    synthetic_dir = tmp_path / "synthetic"
    mock_result = MagicMock()
    mock_result.status = "SUCCESS"

    def fake_generate(**kwargs: object) -> list[Path]:
        (synthetic_dir / "input_campaign").mkdir(parents=True, exist_ok=True)
        (synthetic_dir / "input_models").mkdir(parents=True, exist_ok=True)
        f1 = synthetic_dir / "input_campaign" / "campaign_summer_lookbook.png"
        f2 = synthetic_dir / "input_campaign" / "campaign_summer_mask.png"
        f1.touch()
        f2.touch()
        return [f1, f2]

    with (
        patch("batchpersona.pipeline.ensure_server_ready", return_value=True),
        patch("batchpersona.pipeline.generate_all_testdata", side_effect=fake_generate) as mock_gen,
        patch("batchpersona.pipeline.BatchSwapper") as mock_swapper_cls,
    ):
        mock_swapper = mock_swapper_cls.return_value
        mock_swapper.run_batch.return_value = [mock_result]

        code = run_quick_dry_run(synthetic_dir=synthetic_dir)
        assert code == 0
        assert mock_gen.call_count == 1
        assert mock_swapper_cls.call_count == 1


def test_run_quick_dry_run_generation_failure(tmp_path: Path) -> None:
    """Verify failure to generate synthetic data stops execution."""
    synthetic_dir = tmp_path / "synthetic"

    with (
        patch("batchpersona.pipeline.ensure_server_ready", return_value=True),
        patch("batchpersona.pipeline.generate_all_testdata", return_value=[]) as mock_gen,
        patch("batchpersona.pipeline.BatchSwapper") as mock_swapper_cls,
    ):
        code = run_quick_dry_run(synthetic_dir=synthetic_dir)
        assert code == 1
        assert mock_gen.call_count == 1
        assert mock_swapper_cls.call_count == 0


def test_run_quick_dry_run_existing_data(tmp_path: Path) -> None:
    """Verify synthetic data generation is skipped if already present."""
    synthetic_dir = tmp_path / "synthetic"
    camp_dir = synthetic_dir / "input_campaign"
    camp_dir.mkdir(parents=True)
    (camp_dir / "campaign_summer_lookbook.png").touch()
    (camp_dir / "campaign_summer_mask.png").touch()
    (synthetic_dir / "input_models").mkdir(parents=True)

    mock_result = MagicMock()
    mock_result.status = "SUCCESS"

    with (
        patch("batchpersona.pipeline.ensure_server_ready", return_value=True),
        patch("batchpersona.pipeline.generate_all_testdata") as mock_gen,
        patch("batchpersona.pipeline.BatchSwapper") as mock_swapper_cls,
    ):
        mock_swapper = mock_swapper_cls.return_value
        mock_swapper.run_batch.return_value = [mock_result]

        code = run_quick_dry_run(synthetic_dir=synthetic_dir)
        assert code == 0
        assert mock_gen.call_count == 0
        assert mock_swapper_cls.call_count == 1


def test_run_quality_gate() -> None:
    """Verify run_quality_gate invokes run_all_ci_stages."""
    with patch("batchpersona.pipeline.run_all_ci_stages", return_value=0) as mock_ci:
        code = run_quality_gate()
        assert code == 0
        assert mock_ci.call_count == 1


def test_display_menu(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify menu layout renders essential options."""
    display_menu("127.0.0.1:8000")
    captured = capsys.readouterr().out
    assert "BatchPersona" in captured
    assert "[1] Run Commercial Model Swap" in captured
    assert "[2] Run Quick Dry-Run" in captured
    assert "[3] Run Quality Gate and Tests" in captured
    assert "[4] Close ComfyUI Console" in captured
    assert "[0] Exit" in captured


def test_interactive_menu_loop_exit_immediately() -> None:
    """Verify menu exits cleanly on option 0."""
    with patch("builtins.input", side_effect=["0"]):
        assert interactive_menu_loop("127.0.0.1:8000") == 0


def test_interactive_menu_loop_options() -> None:
    """Verify menu loop executes choices and exits."""
    with (
        patch(
            "builtins.input",
            side_effect=["1", "", "2", "", "3", "", "4", "", "4", "", "invalid", "0"],
        ),
        patch("batchpersona.pipeline.run_commercial_swap", return_value=0) as mock_swap,
        patch("batchpersona.pipeline.run_quick_dry_run", return_value=0) as mock_dry,
        patch("batchpersona.pipeline.run_quality_gate", return_value=0) as mock_ci,
        patch("batchpersona.pipeline.stop_server", side_effect=[True, False]) as mock_stop,
    ):
        assert interactive_menu_loop("127.0.0.1:8000") == 0
        assert mock_swap.call_count == 1
        assert mock_dry.call_count == 1
        assert mock_ci.call_count == 1
        assert mock_stop.call_count == 2


def test_interactive_menu_loop_auto_cleanup_on_exit() -> None:
    """Verify auto-started server is terminated on exit."""
    with (
        patch("batchpersona.pipeline._AUTO_STARTED_SERVER", "127.0.0.1:8000"),
        patch("batchpersona.pipeline.stop_server") as mock_stop,
        patch("builtins.input", side_effect=["0"]),
    ):
        assert interactive_menu_loop("127.0.0.1:8000") == 0
        mock_stop.assert_called_once_with("127.0.0.1:8000")


def test_interactive_menu_loop_keyboard_interrupt() -> None:
    """Verify graceful handling of KeyboardInterrupt and server cleanup."""
    with (
        patch("batchpersona.pipeline._AUTO_STARTED_SERVER", "127.0.0.1:8000"),
        patch("batchpersona.pipeline.stop_server") as mock_stop,
        patch("builtins.input", side_effect=KeyboardInterrupt),
    ):
        assert interactive_menu_loop("127.0.0.1:8000") == 0
        mock_stop.assert_called_once_with("127.0.0.1:8000")


def test_main_with_cli_args() -> None:
    """Verify main forwards direct CLI arguments to orchestrator_main."""
    with patch("batchpersona.pipeline.orchestrator_main", return_value=0) as mock_orch:
        code = main(["--campaign", "lookbook.png", "--models-dir", "models"])
        assert code == 0
        mock_orch.assert_called_once_with(["--campaign", "lookbook.png", "--models-dir", "models"])


def test_main_without_args() -> None:
    """Verify main triggers interactive menu loop when no arguments are provided."""
    with patch("batchpersona.pipeline.interactive_menu_loop", return_value=0) as mock_menu:
        assert main([]) == 0
        assert mock_menu.call_count == 1
