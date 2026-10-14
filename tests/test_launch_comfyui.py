"""Unit and integration tests for scripts/launch_comfyui.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from batchpersona.launcher import (
    ComfyInstance,
    build_headless_command,
    detect_comfy_desktop_instance,
    detect_comfyui,
    detect_desktop_gui_exe,
    detect_standalone_instance,
    get_desktop_appdata_dir,
    is_server_online,
    launch_server,
    main,
)


def test_build_headless_command_all_options(tmp_path: Path) -> None:
    """Verify build_headless_command correctly assembles all CLI parameters."""
    python_exe = tmp_path / "python.exe"
    main_py = tmp_path / "main.py"
    base_dir = tmp_path / "base"
    user_dir = base_dir / "user"
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    yaml_path = tmp_path / "models.yaml"

    instance = ComfyInstance(
        name="TestInstance",
        mode="headless",
        python_exe=python_exe,
        main_py=main_py,
        base_dir=base_dir,
        user_dir=user_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        extra_model_paths=yaml_path,
        extra_args=("--enable-manager", "--highvram"),
    )

    cmd = build_headless_command(instance, host="0.0.0.0", port="8188")

    assert cmd[0] == str(python_exe)
    assert cmd[1] == str(main_py)
    assert "--listen" in cmd and cmd[cmd.index("--listen") + 1] == "0.0.0.0"
    assert "--port" in cmd and cmd[cmd.index("--port") + 1] == "8188"
    assert "--base-directory" in cmd and cmd[cmd.index("--base-directory") + 1] == str(base_dir)
    assert "--user-directory" in cmd and cmd[cmd.index("--user-directory") + 1] == str(user_dir)
    assert "--input-directory" in cmd and cmd[cmd.index("--input-directory") + 1] == str(input_dir)
    assert "--output-directory" in cmd and cmd[cmd.index("--output-directory") + 1] == str(
        output_dir
    )
    assert "--extra-model-paths-config" in cmd and cmd[
        cmd.index("--extra-model-paths-config") + 1
    ] == str(yaml_path)
    assert "--enable-manager" in cmd
    assert "--highvram" in cmd


def test_build_headless_command_missing_required() -> None:
    """Verify error raised when python_exe or main_py are missing."""
    instance = ComfyInstance(name="Incomplete", mode="headless")
    with pytest.raises(ValueError, match="Missing python_exe or main_py"):
        build_headless_command(instance, "127.0.0.1", "8000")


def test_is_server_online_success() -> None:
    """Verify is_server_online returns True on HTTP 200 with valid JSON body."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps({"system": {"os": "nt"}, "devices": []}).encode(
        "utf-8"
    )
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        assert is_server_online("127.0.0.1:8000") is True


def test_is_server_online_failure() -> None:
    """Verify is_server_online returns False on connection error."""
    with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
        assert is_server_online("127.0.0.1:8000") is False


def test_is_server_online_invalid_json() -> None:
    """Verify is_server_online returns False on unexpected payload format."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps({"unrelated": True}).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        assert is_server_online("127.0.0.1:8000") is False


def test_get_desktop_appdata_dir_platforms(tmp_path: Path) -> None:
    """Verify get_desktop_appdata_dir across simulated platforms."""
    with patch("sys.platform", "win32"), patch.dict("os.environ", {"APPDATA": str(tmp_path)}):
        assert get_desktop_appdata_dir() == tmp_path / "Comfy Desktop"

    with patch("sys.platform", "darwin"), patch("pathlib.Path.home", return_value=tmp_path):
        assert (
            get_desktop_appdata_dir()
            == tmp_path / "Library" / "Application Support" / "Comfy Desktop"
        )

    with (
        patch("sys.platform", "linux"),
        patch.dict("os.environ", {"XDG_CONFIG_HOME": str(tmp_path)}),
    ):
        assert get_desktop_appdata_dir() == tmp_path / "Comfy Desktop"

    with (
        patch("sys.platform", "linux"),
        patch.dict("os.environ", {}, clear=True),
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        assert get_desktop_appdata_dir() == tmp_path / ".config" / "Comfy Desktop"


def test_detect_comfy_desktop_instance_mocked(tmp_path: Path) -> None:
    """Verify parsing of Comfy Desktop installations.json with valid structure."""
    config_dir = tmp_path / "Comfy Desktop"
    config_dir.mkdir()
    models_dir = config_dir / "instance-model-paths"
    models_dir.mkdir()

    python_exe = tmp_path / "python.exe"
    python_exe.touch()
    install_dir = tmp_path / "ComfyUI"
    install_dir.mkdir()
    main_py = install_dir / "main.py"
    main_py.touch()

    yaml_file = models_dir / "inst-123.yaml"
    yaml_file.touch()

    installations_json = config_dir / "installations.json"
    installations_json.write_text(
        json.dumps(
            [
                {
                    "id": "inst-123",
                    "name": "MockComfy",
                    "status": "installed",
                    "adoptedPythonPath": str(python_exe),
                    "installPath": str(install_dir),
                    "adoptedBaseDir": str(install_dir),
                    "inputDir": str(install_dir / "input"),
                    "outputDir": str(install_dir / "output"),
                }
            ]
        ),
        encoding="utf-8",
    )

    with patch("batchpersona.launcher.get_desktop_appdata_dir", return_value=config_dir):
        inst = detect_comfy_desktop_instance()
        assert inst is not None
        assert inst.name == "MockComfy"
        assert inst.mode == "headless"
        assert inst.python_exe == python_exe
        assert inst.main_py == main_py
        assert inst.extra_model_paths == yaml_file


def test_detect_comfy_desktop_instance_edge_cases(tmp_path: Path) -> None:
    """Verify detect_comfy_desktop_instance gracefully handles malformed files."""
    # Config dir doesn't exist
    with patch(
        "batchpersona.launcher.get_desktop_appdata_dir", return_value=tmp_path / "nonexistent"
    ):
        assert detect_comfy_desktop_instance() is None

    # installations.json doesn't exist
    config_dir = tmp_path / "empty_conf"
    config_dir.mkdir()
    with patch("batchpersona.launcher.get_desktop_appdata_dir", return_value=config_dir):
        assert detect_comfy_desktop_instance() is None

    # Invalid JSON
    inst_file = config_dir / "installations.json"
    inst_file.write_text("invalid json", encoding="utf-8")
    with patch("batchpersona.launcher.get_desktop_appdata_dir", return_value=config_dir):
        assert detect_comfy_desktop_instance() is None

    # Non-list JSON
    inst_file.write_text("{}", encoding="utf-8")
    with patch("batchpersona.launcher.get_desktop_appdata_dir", return_value=config_dir):
        assert detect_comfy_desktop_instance() is None

    # Missing fields or missing paths
    inst_file.write_text(
        json.dumps([{"status": "not_installed"}, {"status": "installed"}]), encoding="utf-8"
    )
    with patch("batchpersona.launcher.get_desktop_appdata_dir", return_value=config_dir):
        assert detect_comfy_desktop_instance() is None


def test_detect_standalone_instance_env(tmp_path: Path) -> None:
    """Verify detection via COMFYUI_DIR environment variable."""
    fake_dir = tmp_path / "MyComfy"
    fake_dir.mkdir()
    main_py = fake_dir / "main.py"
    main_py.touch()
    fake_py = tmp_path / "python.exe"
    fake_py.touch()

    with patch.dict("os.environ", {"COMFYUI_DIR": str(fake_dir), "COMFYUI_PYTHON": str(fake_py)}):
        inst = detect_standalone_instance()
        assert inst is not None
        assert inst.mode == "headless"
        assert inst.python_exe == fake_py
        assert inst.main_py == main_py


def test_detect_standalone_instance_search_paths(tmp_path: Path) -> None:
    """Verify candidate search paths detection."""
    docs_dir = tmp_path / "Documents" / "ComfyUI"
    docs_dir.mkdir(parents=True)
    main_py = docs_dir / "main.py"
    main_py.touch()

    with (
        patch.dict("os.environ", {}, clear=True),
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        inst = detect_standalone_instance()
        assert inst is not None
        assert inst.mode == "headless"
        assert inst.main_py == main_py


def test_detect_desktop_gui_exe(tmp_path: Path) -> None:
    """Verify fallback detection of Desktop GUI executable."""
    prog_dir = tmp_path / "ProgramFiles"
    prog_dir.mkdir()
    app_dir = prog_dir / "Comfy Desktop"
    app_dir.mkdir()
    fake_exe = app_dir / "Comfy Desktop.exe"
    fake_exe.touch()

    empty_local = tmp_path / "LocalEmpty"
    empty_local.mkdir()

    with (
        patch("sys.platform", "win32"),
        patch.dict("os.environ", {"PROGRAMFILES": str(prog_dir), "LOCALAPPDATA": str(empty_local)}),
    ):
        inst = detect_desktop_gui_exe()
        assert inst is not None
        assert inst.mode == "gui"
        assert inst.gui_exe == fake_exe


def test_detect_comfyui_priority_chain() -> None:
    """Verify detect_comfyui adheres to priority fallback chain."""
    mock_desktop = ComfyInstance(name="DesktopBackend", mode="headless")
    mock_standalone = ComfyInstance(name="Standalone", mode="headless")
    mock_gui = ComfyInstance(name="GUI", mode="gui")

    # 1. Desktop takes priority
    with patch("batchpersona.launcher.detect_comfy_desktop_instance", return_value=mock_desktop):
        assert detect_comfyui() == mock_desktop

    # 2. Standalone second
    with (
        patch("batchpersona.launcher.detect_comfy_desktop_instance", return_value=None),
        patch("batchpersona.launcher.detect_standalone_instance", return_value=mock_standalone),
    ):
        assert detect_comfyui() == mock_standalone

    # 3. GUI third
    with (
        patch("batchpersona.launcher.detect_comfy_desktop_instance", return_value=None),
        patch("batchpersona.launcher.detect_standalone_instance", return_value=None),
        patch("batchpersona.launcher.detect_desktop_gui_exe", return_value=mock_gui),
    ):
        assert detect_comfyui() == mock_gui

    # 4. None if all fail
    with (
        patch("batchpersona.launcher.detect_comfy_desktop_instance", return_value=None),
        patch("batchpersona.launcher.detect_standalone_instance", return_value=None),
        patch("batchpersona.launcher.detect_desktop_gui_exe", return_value=None),
    ):
        assert detect_comfyui() is None


def test_launch_server_already_online() -> None:
    """Verify launch_server returns True immediately if server is already online."""
    with patch("batchpersona.launcher.is_server_online", return_value=True):
        instance = ComfyInstance(name="Running", mode="headless")
        assert launch_server(instance, "127.0.0.1:8000") is True


def test_launch_server_headless_win32_and_posix(tmp_path: Path) -> None:
    """Verify launch_server spawns correctly under win32 and posix."""
    py = tmp_path / "python.exe"
    main_script = tmp_path / "main.py"
    py.touch()
    main_script.touch()

    instance = ComfyInstance(name="Headless", mode="headless", python_exe=py, main_py=main_script)

    # Windows simulation
    with (
        patch("batchpersona.launcher.is_server_online", side_effect=[False, True]),
        patch("sys.platform", "win32"),
        patch("subprocess.Popen") as mock_popen,
    ):
        success = launch_server(instance, "127.0.0.1:8000", timeout=5.0)
        assert success is True
        assert mock_popen.called

    # Linux / Darwin simulation
    with (
        patch("batchpersona.launcher.is_server_online", side_effect=[False, True]),
        patch("sys.platform", "linux"),
        patch("subprocess.Popen") as mock_popen,
    ):
        success = launch_server(instance, "127.0.0.1:8000", timeout=5.0)
        assert success is True
        assert mock_popen.called


def test_launch_server_gui_mode(tmp_path: Path) -> None:
    """Verify launch_server spawns GUI executable and handles timeout."""
    fake_exe = tmp_path / "Comfy Desktop.exe"
    fake_exe.touch()
    instance = ComfyInstance(name="GUI", mode="gui", gui_exe=fake_exe)

    with (
        patch("batchpersona.launcher.is_server_online", return_value=False),
        patch("subprocess.Popen") as mock_popen,
    ):
        success = launch_server(instance, "127.0.0.1:8000", timeout=0.1)
        assert success is False
        assert mock_popen.called


def test_launch_server_unsupported_mode() -> None:
    """Verify error handled on unknown instance mode."""
    instance = ComfyInstance(name="Bad", mode="unknown")
    with patch("batchpersona.launcher.is_server_online", return_value=False):
        assert launch_server(instance, "127.0.0.1:8000") is False


def test_launch_server_headless_missing_paths() -> None:
    """Verify launch_server returns False if headless instance is missing python_exe or main_py."""
    with patch("batchpersona.launcher.is_server_online", return_value=False):
        instance = ComfyInstance(
            name="MissingPaths", mode="headless", python_exe=None, main_py=None
        )
        assert launch_server(instance, "127.0.0.1:8000") is False


def test_cli_main_check_and_detect() -> None:
    """Verify main() CLI flags --check and --detect."""
    # --check online
    with (
        patch("batchpersona.launcher.is_server_online", return_value=True),
        patch.object(sys, "argv", ["launch_comfyui.py", "--check"]),
    ):
        assert main() == 0

    # --check offline
    with (
        patch("batchpersona.launcher.is_server_online", return_value=False),
        patch.object(sys, "argv", ["launch_comfyui.py", "--check"]),
    ):
        assert main() == 1

    # --detect found
    mock_inst = ComfyInstance(
        name="Found", mode="headless", python_exe=Path("python"), main_py=Path("main.py")
    )
    with (
        patch("batchpersona.launcher.detect_comfyui", return_value=mock_inst),
        patch.object(sys, "argv", ["launch_comfyui.py", "--detect"]),
    ):
        assert main() == 0

    # --detect not found
    with (
        patch("batchpersona.launcher.detect_comfyui", return_value=None),
        patch.object(sys, "argv", ["launch_comfyui.py", "--detect"]),
    ):
        assert main() == 1


def test_cli_main_launch_flows() -> None:
    """Verify main() CLI --launch flows."""
    # Already online
    with (
        patch("batchpersona.launcher.is_server_online", return_value=True),
        patch.object(sys, "argv", ["launch_comfyui.py", "--launch"]),
    ):
        assert main() == 0

    # Offline, none found
    with (
        patch("batchpersona.launcher.is_server_online", return_value=False),
        patch("batchpersona.launcher.detect_comfyui", return_value=None),
        patch.object(sys, "argv", ["launch_comfyui.py", "--launch"]),
    ):
        assert main() == 1

    # Offline, launch successful
    mock_inst = ComfyInstance(name="Found", mode="headless")
    with (
        patch("batchpersona.launcher.is_server_online", return_value=False),
        patch("batchpersona.launcher.detect_comfyui", return_value=mock_inst),
        patch("batchpersona.launcher.launch_server", return_value=True),
        patch.object(sys, "argv", ["launch_comfyui.py", "--launch"]),
    ):
        assert main() == 0

    # Offline, launch failed
    with (
        patch("batchpersona.launcher.is_server_online", return_value=False),
        patch("batchpersona.launcher.detect_comfyui", return_value=mock_inst),
        patch("batchpersona.launcher.launch_server", return_value=False),
        patch.object(sys, "argv", ["launch_comfyui.py", "--launch"]),
    ):
        assert main() == 1


def test_detect_comfy_desktop_fallback_main_py(tmp_path: Path) -> None:
    """Verify fallback to installPath/main.py when ComfyUI/main.py is absent."""
    config_dir = tmp_path / "Comfy Desktop"
    config_dir.mkdir()
    python_exe = tmp_path / "python.exe"
    python_exe.touch()
    install_dir = tmp_path / "ComfyUI"
    install_dir.mkdir()
    main_py = install_dir / "main.py"
    main_py.touch()

    installations_json = config_dir / "installations.json"
    installations_json.write_text(
        json.dumps(
            [
                {
                    "id": "inst-1",
                    "status": "installed",
                    "adoptedPythonPath": str(python_exe),
                    "installPath": str(install_dir),
                }
            ]
        ),
        encoding="utf-8",
    )

    with patch("batchpersona.launcher.get_desktop_appdata_dir", return_value=config_dir):
        inst = detect_comfy_desktop_instance()
        assert inst is not None
        assert inst.main_py == main_py
        assert inst.working_dir == install_dir


def test_build_headless_command_default_server_split(tmp_path: Path) -> None:
    """Verify launch_server parses server with and without colon."""
    py = tmp_path / "python.exe"
    main_script = tmp_path / "main.py"
    py.touch()
    main_script.touch()
    instance = ComfyInstance(name="Headless", mode="headless", python_exe=py, main_py=main_script)

    with (
        patch("batchpersona.launcher.is_server_online", side_effect=[False, True]),
        patch("subprocess.Popen"),
    ):
        assert launch_server(instance, "127.0.0.1", timeout=2.0) is True


def test_detect_standalone_nested_and_embedded_python(tmp_path: Path) -> None:
    """Verify detection when main.py is nested and python is embedded."""
    docs_dir = tmp_path / "Documents" / "ComfyUI"
    nested_main = docs_dir / "ComfyUI" / "main.py"
    nested_main.parent.mkdir(parents=True)
    nested_main.touch()

    py_embedded = docs_dir.parent / "python_embeded" / "python.exe"
    py_embedded.parent.mkdir(parents=True)
    py_embedded.touch()

    with (
        patch.dict("os.environ", {}, clear=True),
        patch("pathlib.Path.home", return_value=tmp_path),
    ):
        inst = detect_standalone_instance()
        assert inst is not None
        assert inst.main_py == nested_main
        assert inst.python_exe == py_embedded


def test_detect_desktop_gui_exe_darwin(tmp_path: Path) -> None:
    """Verify detection of Desktop GUI executable on macOS."""
    fake_mac_exe = tmp_path / "Comfy Desktop"
    fake_mac_exe.touch()

    with patch("sys.platform", "darwin"), patch("pathlib.Path.exists", return_value=True):
        inst = detect_desktop_gui_exe()
        assert inst is not None
        assert inst.mode == "gui"
