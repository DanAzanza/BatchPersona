"""Automated ComfyUI Server Launcher & Healthcheck for BatchPersona."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple

from batchpersona.models import DEFAULT_SERVER_ADDRESS, normalize_server_address

LOGGER = logging.getLogger("batchpersona.launcher")

PID_FILE_NAME = ".comfyui_server.pid"
_ACTIVE_SERVER_PROCESS: subprocess.Popen[Any] | None = None
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def get_pid_file_path() -> Path:
    """Return path to persistent ComfyUI server PID tracking file."""
    try:
        if REPO_ROOT.is_dir() and os.access(REPO_ROOT, os.W_OK):
            return REPO_ROOT / PID_FILE_NAME
    except OSError:
        pass
    return Path.cwd() / PID_FILE_NAME


def record_server_pid(server: str, pid: Any) -> None:
    """Record running server PID to file for lifecycle management."""
    pid_file = get_pid_file_path()
    try:
        val: int | None = int(pid)
    except (TypeError, ValueError):
        val = None
    if val is None:
        return
    with contextlib.suppress(OSError):
        pid_file.write_text(json.dumps({"pid": val, "server": server}), encoding="utf-8")


def read_server_pid(server: str | None = None) -> int | None:
    """Read server PID from tracking file if valid."""
    pid_file = get_pid_file_path()
    if not pid_file.is_file():
        return None
    try:
        data = json.loads(pid_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        if server is not None and data.get("server") != server:
            return None
        pid = data.get("pid")
        return int(pid) if pid is not None else None
    except (OSError, ValueError, TypeError):
        return None


def clear_server_pid_file() -> None:
    """Remove server PID tracking file if it exists."""
    pid_file = get_pid_file_path()
    if pid_file.is_file():
        with contextlib.suppress(OSError):
            pid_file.unlink()


class ComfyInstance(NamedTuple):
    """Metadata describing a detected ComfyUI installation."""

    name: str
    mode: str  # 'headless' or 'gui'
    python_exe: Path | None = None
    main_py: Path | None = None
    working_dir: Path | None = None
    base_dir: Path | None = None
    user_dir: Path | None = None
    input_dir: Path | None = None
    output_dir: Path | None = None
    extra_model_paths: Path | None = None
    extra_args: tuple[str, ...] = ()
    gui_exe: Path | None = None


def get_desktop_appdata_dir() -> Path | None:
    """Return the platform-specific configuration directory for Comfy Desktop."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Comfy Desktop"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Comfy Desktop"
    else:
        config_home = os.environ.get("XDG_CONFIG_HOME")
        if config_home:
            return Path(config_home) / "Comfy Desktop"
        return Path.home() / ".config" / "Comfy Desktop"
    return None


def detect_comfy_desktop_instance() -> ComfyInstance | None:
    """Detect local ComfyUI backend configured via Comfy Desktop installations.json."""
    config_dir = get_desktop_appdata_dir()
    if not config_dir or not config_dir.exists():
        return None

    installations_file = config_dir / "installations.json"
    if not installations_file.exists():
        return None

    try:
        data = json.loads(installations_file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return None

        for inst in data:
            if not isinstance(inst, dict):
                continue
            if inst.get("status") != "installed":
                continue

            python_path_raw = inst.get("adoptedPythonPath")
            install_path_raw = inst.get("installPath")
            if not python_path_raw or not install_path_raw:
                continue

            python_exe = Path(python_path_raw)
            install_path = Path(install_path_raw)
            if not python_exe.exists() or not install_path.exists():
                continue

            # Look for main.py inside installPath/ComfyUI or installPath
            main_py = install_path / "ComfyUI" / "main.py"
            if not main_py.exists():
                main_py = install_path / "main.py"
            if not main_py.exists():
                continue

            working_dir = (
                install_path / "ComfyUI" if (install_path / "ComfyUI").exists() else install_path
            )

            base_dir_raw = inst.get("adoptedBaseDir")
            base_dir = Path(base_dir_raw) if base_dir_raw and Path(base_dir_raw).exists() else None
            user_dir = base_dir / "user" if base_dir else None

            input_dir_raw = inst.get("inputDir")
            input_dir = (
                Path(input_dir_raw) if input_dir_raw and Path(input_dir_raw).exists() else None
            )

            output_dir_raw = inst.get("outputDir")
            output_dir = (
                Path(output_dir_raw) if output_dir_raw and Path(output_dir_raw).exists() else None
            )

            inst_id = inst.get("id", "")
            extra_models_yaml = config_dir / "instance-model-paths" / f"{inst_id}.yaml"
            extra_model_paths = extra_models_yaml if extra_models_yaml.exists() else None

            extra_args = ["--enable-manager"]

            return ComfyInstance(
                name=inst.get("name", "ComfyUI Desktop Backend"),
                mode="headless",
                python_exe=python_exe,
                main_py=main_py,
                working_dir=working_dir,
                base_dir=base_dir,
                user_dir=user_dir,
                input_dir=input_dir,
                output_dir=output_dir,
                extra_model_paths=extra_model_paths,
                extra_args=tuple(extra_args),
            )
    except (json.JSONDecodeError, OSError, TypeError, KeyError):
        return None
    return None


def detect_standalone_instance() -> ComfyInstance | None:
    """Detect standalone git clones, portable distributions, or environment overrides."""
    # Check environment variable overrides first
    env_dir = os.environ.get("COMFYUI_DIR")
    env_py = os.environ.get("COMFYUI_PYTHON")
    if env_dir:
        cand_dir = Path(env_dir)
        cand_main = cand_dir / "main.py"
        if cand_main.exists():
            py_exe = Path(env_py) if env_py and Path(env_py).exists() else Path(sys.executable)
            return ComfyInstance(
                name="Environment ComfyUI",
                mode="headless",
                python_exe=py_exe,
                main_py=cand_main,
                working_dir=cand_dir,
                base_dir=cand_dir,
            )

    candidates = [
        Path.home() / "Documents" / "ComfyUI",
        Path.home() / "ComfyUI",
        Path("C:/ComfyUI_windows_portable/ComfyUI"),
        Path("D:/ComfyUI_windows_portable/ComfyUI"),
    ]

    for cand in candidates:
        if not cand.exists():
            continue

        cand_main = cand / "main.py"
        if not cand_main.exists():
            cand_main = cand / "ComfyUI" / "main.py"
        if not cand_main.exists():
            continue

        # Check virtual environments inside or adjacent
        venv_py = (
            cand
            / ".venv"
            / ("Scripts" if sys.platform == "win32" else "bin")
            / ("python.exe" if sys.platform == "win32" else "python")
        )
        if not venv_py.exists():
            venv_py = cand.parent / "python_embeded" / "python.exe"
        if not venv_py.exists():
            venv_py = Path(sys.executable)

        return ComfyInstance(
            name=f"Standalone ({cand.name})",
            mode="headless",
            python_exe=venv_py,
            main_py=cand_main,
            working_dir=cand_main.parent,
            base_dir=cand,
        )

    return None


def detect_desktop_gui_exe() -> ComfyInstance | None:
    """Detect Comfy Desktop Electron executable as GUI fallback."""
    gui_paths: list[Path] = []
    if sys.platform == "win32":
        prog_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
        local_app = os.environ.get("LOCALAPPDATA", "")
        gui_paths.extend(
            [
                Path(prog_files) / "Comfy Desktop" / "Comfy Desktop.exe",
                Path(local_app) / "Programs" / "Comfy Desktop" / "Comfy Desktop.exe",
                Path(prog_files) / "ComfyUI" / "ComfyUI.exe",
                Path(local_app) / "Programs" / "ComfyUI" / "ComfyUI.exe",
            ]
        )
    elif sys.platform == "darwin":
        gui_paths.append(Path("/Applications/Comfy Desktop.app/Contents/MacOS/Comfy Desktop"))

    for p in gui_paths:
        if p.exists():
            return ComfyInstance(
                name="Comfy Desktop GUI",
                mode="gui",
                gui_exe=p,
            )

    return None


def detect_comfyui() -> ComfyInstance | None:
    """Detect the best available ComfyUI instance (headless backend prioritized)."""
    # 1. Comfy Desktop headless backend (highest fidelity with existing models & settings)
    inst = detect_comfy_desktop_instance()
    if inst:
        return inst

    # 2. Standalone / portable ComfyUI directory
    inst = detect_standalone_instance()
    if inst:
        return inst

    # 3. Comfy Desktop Electron GUI application
    inst = detect_desktop_gui_exe()
    if inst:
        return inst

    return None


def is_server_online(server: str = DEFAULT_SERVER_ADDRESS, timeout: float = 1.5) -> bool:
    """Check if ComfyUI server is responding to /system_stats."""
    clean_server = normalize_server_address(server)
    url = f"http://{clean_server}/system_stats"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BatchPersona-Launcher"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                payload = json.loads(response.read().decode("utf-8"))
                return "devices" in payload or "system" in payload
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return False
    return False


def build_headless_command(instance: ComfyInstance, host: str, port: str) -> list[str]:
    """Build the command list to launch ComfyUI headless."""
    if not instance.python_exe or not instance.main_py:
        raise ValueError("Missing python_exe or main_py for headless launch.")

    cmd = [
        str(instance.python_exe),
        str(instance.main_py),
        "--listen",
        host,
        "--port",
        port,
    ]

    if instance.base_dir:
        cmd.extend(["--base-directory", str(instance.base_dir)])
    if instance.user_dir:
        cmd.extend(["--user-directory", str(instance.user_dir)])
    if instance.input_dir:
        cmd.extend(["--input-directory", str(instance.input_dir)])
    if instance.output_dir:
        cmd.extend(["--output-directory", str(instance.output_dir)])
    if instance.extra_model_paths:
        cmd.extend(["--extra-model-paths-config", str(instance.extra_model_paths)])
    if instance.extra_args:
        cmd.extend(instance.extra_args)

    return cmd


def launch_server(
    instance: ComfyInstance,
    server: str = DEFAULT_SERVER_ADDRESS,
    timeout: float = 60.0,
) -> bool:
    """Launch the ComfyUI server and wait until it is ready."""
    clean_server = normalize_server_address(server)
    if is_server_online(clean_server, timeout=1.0):
        print(f"[OK] ComfyUI server is already online and responding at {clean_server}.")
        return True

    # Parse host and port
    if ":" in clean_server:
        host, port = clean_server.split(":", 1)
    else:
        host = clean_server
        port = DEFAULT_SERVER_ADDRESS.split(":", 1)[1] if ":" in DEFAULT_SERVER_ADDRESS else "8000"

    if instance.mode == "headless":
        if not instance.python_exe or not instance.main_py:
            print(f"[ERROR] Headless instance '{instance.name}' is missing python_exe or main_py.")
            return False

        python_exe = instance.python_exe
        main_py = instance.main_py
        cmd = build_headless_command(instance, host, port)
        print(f"[LAUNCH] Starting headless ComfyUI backend ({instance.name})...")
        print(f"         Command: {python_exe} {main_py.name}")

        cwd = str(instance.working_dir) if instance.working_dir else str(main_py.parent)

        if sys.platform == "win32":
            # SW_MINIMIZE keeps the window out of the way while giving it a valid console
            startupinfo = None
            startupinfo_cls = getattr(subprocess, "STARTUPINFO", None)
            if startupinfo_cls is not None:
                startupinfo = startupinfo_cls()
                use_show_window = getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
                if use_show_window:
                    startupinfo.dwFlags |= use_show_window
                startupinfo.wShowWindow = 6  # SW_MINIMIZE

            creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                startupinfo=startupinfo,
                creationflags=creationflags,
            )
        else:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        global _ACTIVE_SERVER_PROCESS
        _ACTIVE_SERVER_PROCESS = proc
        record_server_pid(clean_server, proc.pid)

    elif instance.mode == "gui" and instance.gui_exe:
        print(f"[LAUNCH] Starting Comfy Desktop GUI ({instance.gui_exe})...")
        print(
            "[NOTICE] If the desktop window opens on the dashboard, please click 'Launch' or 'Open' on your instance."
        )
        if sys.platform == "win32":
            detached = getattr(subprocess, "DETACHED_PROCESS", 0)
            proc = subprocess.Popen([str(instance.gui_exe)], creationflags=detached)
        else:
            proc = subprocess.Popen([str(instance.gui_exe)], start_new_session=True)

        _ACTIVE_SERVER_PROCESS = proc
        record_server_pid(clean_server, proc.pid)
    else:
        print(f"[ERROR] Unsupported instance mode: {instance.mode}")
        return False

    # Wait for server initialization
    print(
        f"[WAITING] Waiting for ComfyUI server to become ready at {clean_server} (timeout: {int(timeout)}s)..."
    )
    start_time = time.monotonic()
    last_dot_time = start_time

    while time.monotonic() - start_time < timeout:
        if is_server_online(clean_server, timeout=1.5):
            print("\n[OK] ComfyUI server is online and ready for batch processing!")
            return True

        if time.monotonic() - last_dot_time >= 2.0:
            sys.stdout.write(".")
            sys.stdout.flush()
            last_dot_time = time.monotonic()

        time.sleep(1.0)

    print(
        f"\n[TIMEOUT] ComfyUI server at {clean_server} did not respond within {int(timeout)} seconds."
    )
    if instance.mode == "gui":
        print(
            "          Please ensure you have clicked 'Launch' or 'Open' on your instance in the Comfy Desktop window."
        )
    return False


def find_pid_by_port(port: int) -> int | None:
    """Find process ID listening on specified TCP port."""
    if sys.platform == "win32":
        try:
            res = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in res.stdout.splitlines():
                parts = line.strip().split()
                if (
                    len(parts) >= 5
                    and parts[0].upper() == "TCP"
                    and parts[3].upper() == "LISTENING"
                    and parts[1].endswith(f":{port}")
                ):
                    try:
                        return int(parts[4])
                    except ValueError:
                        continue
        except OSError:
            return None
    else:
        try:
            res = subprocess.run(
                ["lsof", "-i", f":{port}", "-sTCP:LISTEN", "-t"],
                capture_output=True,
                text=True,
                check=False,
            )
            stdout = res.stdout.strip()
            if stdout:
                first_line = stdout.splitlines()[0].strip()
                if first_line.isdigit():
                    return int(first_line)
        except OSError:
            return None
    return None


def is_comfyui_process(pid: int) -> bool:
    """Verify whether a PID belongs to a Python or ComfyUI process."""
    if pid <= 4:
        return False
    if sys.platform == "win32":
        try:
            res = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
            )
            out = res.stdout.strip().lower()
            if not out or "no tasks are running" in out:
                return False
            first_col = out.split(",")[0].strip('"').lower()
            return first_col in {
                "python.exe",
                "pythonw.exe",
                "comfyui.exe",
                "comfy.exe",
            }
        except OSError:
            return False
    else:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw_bytes = f.read()
                args = [
                    a.lower()
                    for a in raw_bytes.decode("utf-8", errors="replace").split("\x00")
                    if a
                ]
                if not args:
                    return False
                exe = os.path.basename(args[0])
                if "comfy" in exe:
                    return True
                if "python" in exe:
                    return any("comfy" in arg or "main.py" in arg for arg in args[1:])
                return False
        except (OSError, FileNotFoundError):
            return False


def unload_comfyui_vram(server: str = DEFAULT_SERVER_ADDRESS, timeout: float = 3.0) -> bool:
    """Send /interrupt and /free signals to ComfyUI to immediately release GPU VRAM."""
    clean_server = normalize_server_address(server)
    url_interrupt = f"http://{clean_server}/interrupt"
    url_free = f"http://{clean_server}/free"
    headers = {"User-Agent": "BatchPersona-Launcher", "Content-Type": "application/json"}

    # 1. Interrupt any active jobs
    try:
        req = urllib.request.Request(url_interrupt, data=b"", headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        pass

    # 2. Trigger memory/model unloading
    try:
        payload = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
        req = urllib.request.Request(url_free, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def stop_server(
    server: str = DEFAULT_SERVER_ADDRESS,
    timeout: float = 5.0,
    unload_first: bool = True,
) -> bool:
    """Stop ComfyUI server, release GPU VRAM, and close console window.

    Args:
        server: Host:port of ComfyUI server (default: DEFAULT_SERVER_ADDRESS).
        timeout: Maximum seconds to wait for server process to terminate.
        unload_first: Whether to dispatch /interrupt and /free to release GPU VRAM first.

    Returns:
        True if server is stopped or was already offline, False if termination failed.
    """
    global _ACTIVE_SERVER_PROCESS

    clean_server = normalize_server_address(server)
    was_online = is_server_online(clean_server, timeout=1.0)
    # If server is already offline and no process is tracked, exit early
    if not was_online and _ACTIVE_SERVER_PROCESS is None and read_server_pid(clean_server) is None:
        clear_server_pid_file()
        print(f"[OK] ComfyUI server at {clean_server} is already offline.")
        return True

    # 1. Release GPU VRAM before killing process
    if unload_first and was_online:
        print(f"[SHUTDOWN] Unloading ComfyUI models and releasing GPU VRAM ({clean_server})...")
        unload_comfyui_vram(clean_server, timeout=3.0)
        time.sleep(0.5)

    # 2. Resolve target PID
    pid: int | None = None
    is_self_spawned = False
    if _ACTIVE_SERVER_PROCESS is not None:
        pid = _ACTIVE_SERVER_PROCESS.pid
        is_self_spawned = True
    if pid is None:
        pid = read_server_pid(clean_server)
    if pid is None and was_online:
        try:
            default_port = (
                int(DEFAULT_SERVER_ADDRESS.split(":")[1]) if ":" in DEFAULT_SERVER_ADDRESS else 8000
            )
            port = int(clean_server.split(":")[1]) if ":" in clean_server else default_port
            candidate_pid = find_pid_by_port(port)
            if candidate_pid is not None and is_comfyui_process(candidate_pid):
                pid = candidate_pid
        except ValueError:
            pid = None

    # 3. Terminate process tree
    if pid is not None:
        if not is_self_spawned and not is_comfyui_process(pid):
            print(
                f"[WARNING] PID {pid} is not a verified ComfyUI/Python process; skipping termination to prevent collateral damage."
            )
        else:
            print(f"[SHUTDOWN] Terminating ComfyUI server process (PID: {pid})...")
            if sys.platform == "win32":
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True,
                        check=False,
                    )
                except OSError as e:
                    print(f"[WARNING] taskkill failed: {e}")
            else:
                try:
                    import signal

                    os.kill(pid, signal.SIGTERM)
                    time.sleep(0.5)
                    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
                    with contextlib.suppress(OSError):
                        os.kill(pid, sigkill)
                except OSError:
                    pass

    if _ACTIVE_SERVER_PROCESS is not None:
        with contextlib.suppress(OSError):
            _ACTIVE_SERVER_PROCESS.terminate()
        _ACTIVE_SERVER_PROCESS = None

    clear_server_pid_file()

    # 4. Confirm server shutdown
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        if not is_server_online(clean_server, timeout=1.0):
            print(f"[OK] ComfyUI server at {clean_server} has been successfully closed.")
            return True
        time.sleep(0.5)

    if is_server_online(clean_server, timeout=1.0):
        print(f"[WARNING] ComfyUI server at {clean_server} is still responding.")
        return False

    print(f"[OK] ComfyUI server at {clean_server} has stopped.")
    return True


def main() -> int:
    """CLI entrypoint for ComfyUI launcher."""
    parser = argparse.ArgumentParser(
        description="BatchPersona - ComfyUI Server Launcher & Healthcheck",
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER_ADDRESS,
        help=f"Target ComfyUI server host:port (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if server is currently online and exit with status code (0=online, 1=offline)",
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Detect available ComfyUI installations and print details",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch ComfyUI server headlessly if offline and wait until ready",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop ComfyUI server, release GPU VRAM, and close console window",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for server initialization or shutdown (default: 60.0)",
    )

    args = parser.parse_args()

    # Stop mode
    if args.stop:
        return 0 if stop_server(args.server, timeout=args.timeout) else 1

    # Fast check mode
    if args.check:
        if is_server_online(args.server, timeout=1.5):
            print(f"[ONLINE] ComfyUI server is running at {args.server}")
            return 0
        print(f"[OFFLINE] ComfyUI server is not responding at {args.server}")
        return 1

    # Detect mode
    if args.detect:
        inst = detect_comfyui()
        if not inst:
            print("[NOT FOUND] No ComfyUI installation detected on this system.")
            return 1
        print(f"[FOUND] Installation: {inst.name} (mode: {inst.mode})")
        if inst.mode == "headless":
            print(f"  Python:      {inst.python_exe}\n  Main Script: {inst.main_py}")
            print(f"  Base Dir:    {inst.base_dir}\n  Model YAML:  {inst.extra_model_paths}")
        elif inst.mode == "gui":
            print(f"  GUI Path:    {inst.gui_exe}")
        return 0

    # Launch mode (default when --launch or no action flag specified)
    if args.launch or not (args.check or args.detect):
        if is_server_online(args.server, timeout=1.5):
            print(f"[OK] ComfyUI server is already online at {args.server}.")
            return 0

        inst = detect_comfyui()
        if not inst:
            print(
                f"[ERROR] ComfyUI server is offline at {args.server} and no local installation was detected."
            )
            print("        Please start ComfyUI manually or check your installation.")
            return 1

        return 0 if launch_server(inst, server=args.server, timeout=args.timeout) else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
