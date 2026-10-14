"""Unit and integration tests for ComfyUI server lifecycle, PID management, and termination."""

from __future__ import annotations

import subprocess
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

from batchpersona.launcher import (
    clear_server_pid_file,
    find_pid_by_port,
    get_pid_file_path,
    main,
    read_server_pid,
    record_server_pid,
    stop_server,
    unload_comfyui_vram,
)


def test_pid_file_lifecycle(tmp_path: Path) -> None:
    """Verify PID recording, reading, and clearing."""
    assert get_pid_file_path().name == ".comfyui_server.pid"
    fake_pid_file = tmp_path / ".comfyui_server.pid"
    with patch("batchpersona.launcher.get_pid_file_path", return_value=fake_pid_file):
        assert read_server_pid("127.0.0.1:8000") is None
        record_server_pid("127.0.0.1:8000", 9999)
        assert read_server_pid("127.0.0.1:8000") == 9999
        # Mismatched server
        assert read_server_pid("127.0.0.1:9000") is None
        # Corrupted PID file
        fake_pid_file.write_text("invalid json", encoding="utf-8")
        assert read_server_pid("127.0.0.1:8000") is None
        # Clear PID file
        fake_pid_file.write_text('{"pid": 1234}', encoding="utf-8")
        clear_server_pid_file()
        assert not fake_pid_file.exists()


def test_find_pid_by_port_win32() -> None:
    """Verify Windows netstat parsing logic for active port."""
    netstat_output = (
        "  TCP    127.0.0.1:8000         0.0.0.0:0              LISTENING       4321\n"
        "  TCP    127.0.0.1:8080         0.0.0.0:0              LISTENING       8765\n"
    )
    mock_run = MagicMock(
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=netstat_output)
    )
    with patch("sys.platform", "win32"), patch("subprocess.run", mock_run):
        assert find_pid_by_port(8000) == 4321
        assert find_pid_by_port(9999) is None

    # Error case: netstat raises OSError
    with patch("sys.platform", "win32"), patch("subprocess.run", side_effect=OSError("not found")):
        assert find_pid_by_port(8000) is None


def test_find_pid_by_port_posix() -> None:
    """Verify POSIX lsof parsing logic for active port."""
    mock_run = MagicMock(
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="5678\n")
    )
    with patch("sys.platform", "linux"), patch("subprocess.run", mock_run):
        assert find_pid_by_port(8000) == 5678

    # Error case: lsof raises OSError
    with patch("sys.platform", "linux"), patch("subprocess.run", side_effect=OSError("not found")):
        assert find_pid_by_port(8000) is None


def test_unload_comfyui_vram() -> None:
    """Verify unload_comfyui_vram issues /interrupt and /free requests."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_cm):
        assert unload_comfyui_vram("127.0.0.1:8000") is True

    # Error case: connection refused / exception
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        assert unload_comfyui_vram("127.0.0.1:8000") is False


def test_stop_server_already_offline() -> None:
    """Verify stop_server returns True immediately when server is already down."""
    with (
        patch("batchpersona.launcher.is_server_online", return_value=False),
        patch("batchpersona.launcher._ACTIVE_SERVER_PROCESS", None),
        patch("batchpersona.launcher.read_server_pid", return_value=None),
    ):
        assert stop_server("127.0.0.1:8000") is True


def test_stop_server_active_process_and_win32_kill() -> None:
    """Verify stop_server handles active process, taskkill, and VRAM unload."""
    mock_proc = MagicMock()
    mock_proc.pid = 1234

    with (
        patch("batchpersona.launcher.is_server_online", side_effect=[True, True, False]),
        patch("batchpersona.launcher._ACTIVE_SERVER_PROCESS", mock_proc),
        patch("batchpersona.launcher.unload_comfyui_vram", return_value=True) as mock_unload,
        patch("sys.platform", "win32"),
        patch("subprocess.run") as mock_run,
        patch("time.sleep"),
    ):
        assert stop_server("127.0.0.1:8000", timeout=2.0) is True
        mock_unload.assert_called_once_with("127.0.0.1:8000", timeout=3.0)
        mock_proc.terminate.assert_called_once()
        assert mock_run.call_count == 1
        assert "taskkill" in mock_run.call_args[0][0]


def test_stop_server_posix_signals() -> None:
    """Verify stop_server issues SIGTERM/SIGKILL on non-Windows platforms."""
    with (
        patch("batchpersona.launcher.is_server_online", side_effect=[True, False]),
        patch("batchpersona.launcher._ACTIVE_SERVER_PROCESS", None),
        patch("batchpersona.launcher.read_server_pid", return_value=7777),
        patch("batchpersona.launcher.is_comfyui_process", return_value=True),
        patch("batchpersona.launcher.unload_comfyui_vram", return_value=True),
        patch("sys.platform", "linux"),
        patch("os.kill") as mock_kill,
        patch("time.sleep"),
    ):
        assert stop_server("127.0.0.1:8000", timeout=2.0) is True
        assert mock_kill.call_count >= 1


def test_stop_server_port_lookup_fallback() -> None:
    """Verify stop_server uses port lookup when process and PID file are absent."""
    with (
        patch("batchpersona.launcher.is_server_online", side_effect=[True, False]),
        patch("batchpersona.launcher._ACTIVE_SERVER_PROCESS", None),
        patch("batchpersona.launcher.read_server_pid", return_value=None),
        patch("batchpersona.launcher.find_pid_by_port", return_value=8888),
        patch("batchpersona.launcher.is_comfyui_process", return_value=True),
        patch("batchpersona.launcher.unload_comfyui_vram", return_value=True),
        patch("sys.platform", "win32"),
        patch("subprocess.run") as mock_run,
        patch("time.sleep"),
    ):
        assert stop_server("127.0.0.1:8000", timeout=2.0) is True
        assert mock_run.call_count == 1


def test_stop_server_skips_unverified_pid() -> None:
    """Verify stop_server protects non-ComfyUI processes from accidental kill."""
    with (
        patch("batchpersona.launcher.is_server_online", return_value=True),
        patch("batchpersona.launcher._ACTIVE_SERVER_PROCESS", None),
        patch("batchpersona.launcher.read_server_pid", return_value=9999),
        patch("batchpersona.launcher.is_comfyui_process", return_value=False),
        patch("batchpersona.launcher.unload_comfyui_vram", return_value=True),
        patch("subprocess.run") as mock_run,
        patch("os.kill") as mock_kill,
        patch("time.sleep"),
    ):
        # Should not attempt kill on unverified process
        assert stop_server("127.0.0.1:8000", timeout=0.1) is False
        assert mock_run.call_count == 0
        assert mock_kill.call_count == 0


def test_stop_server_timeout_failure() -> None:
    """Verify stop_server returns False if server remains online after timeout."""
    with (
        patch("batchpersona.launcher.is_server_online", return_value=True),
        patch("batchpersona.launcher._ACTIVE_SERVER_PROCESS", None),
        patch("batchpersona.launcher.read_server_pid", return_value=None),
        patch("batchpersona.launcher.find_pid_by_port", return_value=None),
        patch("time.sleep"),
    ):
        assert stop_server("127.0.0.1:8000", timeout=0.1) is False


def test_cli_main_stop_flag() -> None:
    """Verify main() handles --stop flag."""
    with (
        patch("batchpersona.launcher.stop_server", return_value=True),
        patch.object(sys, "argv", ["launch_comfyui.py", "--stop"]),
    ):
        assert main() == 0

    with (
        patch("batchpersona.launcher.stop_server", return_value=False),
        patch.object(sys, "argv", ["launch_comfyui.py", "--stop"]),
    ):
        assert main() == 1
