"""Backward compatibility shim forwarding to batchpersona.launcher."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for p in [str(REPO_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from batchpersona.launcher import (
    ComfyInstance,
    build_headless_command,
    detect_comfy_desktop_instance,
    detect_comfyui,
    detect_desktop_gui_exe,
    detect_standalone_instance,
    find_pid_by_port,
    get_desktop_appdata_dir,
    is_server_online,
    launch_server,
    main,
    stop_server,
    unload_comfyui_vram,
)

__all__ = [
    "ComfyInstance",
    "build_headless_command",
    "detect_comfy_desktop_instance",
    "detect_comfyui",
    "detect_desktop_gui_exe",
    "detect_standalone_instance",
    "find_pid_by_port",
    "get_desktop_appdata_dir",
    "is_server_online",
    "launch_server",
    "main",
    "stop_server",
    "unload_comfyui_vram",
]

if __name__ == "__main__":
    sys.exit(main())
