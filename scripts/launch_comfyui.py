#!/usr/bin/env python3
"""
Automated ComfyUI Server Launcher & Healthcheck for BatchPersona.

Discovers local ComfyUI installations (ComfyUI Desktop, standalone, portable,
or custom paths), launches the backend headlessly in the background, and polls
the healthcheck endpoint until the server is ready to accept batch prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NamedTuple


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
    except Exception:
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


def is_server_online(server: str = "127.0.0.1:8000", timeout: float = 1.5) -> bool:
    """Check if ComfyUI server is responding to /system_stats."""
    url = f"http://{server}/system_stats"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BatchPersona-Launcher"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                payload = json.loads(response.read().decode("utf-8"))
                return "devices" in payload or "system" in payload
    except Exception:
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
    server: str = "127.0.0.1:8000",
    timeout: float = 60.0,
) -> bool:
    """Launch the ComfyUI server and wait until it is ready."""
    if is_server_online(server, timeout=1.0):
        print(f"[OK] ComfyUI server is already online and responding at {server}.")
        return True

    # Parse host and port
    if ":" in server:
        host, port = server.split(":", 1)
    else:
        host, port = server, "8000"

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
            subprocess.Popen(
                cmd,
                cwd=cwd,
                startupinfo=startupinfo,
                creationflags=creationflags,
            )
        else:
            subprocess.Popen(
                cmd,
                cwd=cwd,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    elif instance.mode == "gui" and instance.gui_exe:
        print(f"[LAUNCH] Starting Comfy Desktop GUI ({instance.gui_exe})...")
        print(
            "[NOTICE] If the desktop window opens on the dashboard, please click 'Launch' or 'Open' on your instance."
        )
        if sys.platform == "win32":
            detached = getattr(subprocess, "DETACHED_PROCESS", 0)
            subprocess.Popen([str(instance.gui_exe)], creationflags=detached)
        else:
            subprocess.Popen([str(instance.gui_exe)], start_new_session=True)
    else:
        print(f"[ERROR] Unsupported instance mode: {instance.mode}")
        return False

    # Wait for server initialization
    print(
        f"[WAITING] Waiting for ComfyUI server to become ready at {server} (timeout: {int(timeout)}s)..."
    )
    start_time = time.monotonic()
    last_dot_time = start_time

    while time.monotonic() - start_time < timeout:
        if is_server_online(server, timeout=1.5):
            print("\n[OK] ComfyUI server is online and ready for batch processing!")
            return True

        if time.monotonic() - last_dot_time >= 2.0:
            sys.stdout.write(".")
            sys.stdout.flush()
            last_dot_time = time.monotonic()

        time.sleep(1.0)

    print(f"\n[TIMEOUT] ComfyUI server at {server} did not respond within {int(timeout)} seconds.")
    if instance.mode == "gui":
        print(
            "          Please ensure you have clicked 'Launch' or 'Open' on your instance in the Comfy Desktop window."
        )
    return False


def main() -> int:
    """CLI entrypoint for ComfyUI launcher."""
    parser = argparse.ArgumentParser(
        description="BatchPersona - ComfyUI Server Launcher & Healthcheck",
    )
    parser.add_argument(
        "--server",
        default="127.0.0.1:8000",
        help="Target ComfyUI server host:port (default: 127.0.0.1:8000)",
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
        "--timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for server initialization (default: 60.0)",
    )

    args = parser.parse_args()

    # Fast check mode
    if args.check:
        online = is_server_online(args.server, timeout=1.5)
        if online:
            print(f"[ONLINE] ComfyUI server is running at {args.server}")
            return 0
        else:
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
            print(f"  Python:      {inst.python_exe}")
            print(f"  Main Script: {inst.main_py}")
            print(f"  Base Dir:    {inst.base_dir}")
            print(f"  Model YAML:  {inst.extra_model_paths}")
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

        success = launch_server(inst, server=args.server, timeout=args.timeout)
        return 0 if success else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
