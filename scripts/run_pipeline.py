"""Backward compatibility shim forwarding to batchpersona.pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for p in [str(REPO_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from batchpersona.launcher import (
    detect_comfyui,
    is_server_online,
    launch_server,
)
from batchpersona.pipeline import (
    VALID_IMAGE_EXTENSIONS,
    display_menu,
    ensure_server_ready,
    interactive_menu_loop,
    main,
    run_commercial_swap,
    run_quality_gate,
    run_quick_dry_run,
)

__all__ = [
    "VALID_IMAGE_EXTENSIONS",
    "detect_comfyui",
    "display_menu",
    "ensure_server_ready",
    "interactive_menu_loop",
    "is_server_online",
    "launch_server",
    "main",
    "run_commercial_swap",
    "run_quality_gate",
    "run_quick_dry_run",
]

if __name__ == "__main__":
    sys.exit(main())
