"""Backward compatibility shim forwarding to batchpersona.housekeeper."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for p in [str(REPO_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from batchpersona.housekeeper import (
    ComfyUIHousekeeper,
    create_housekeeper,
    is_loopback_host,
)

__all__ = [
    "ComfyUIHousekeeper",
    "create_housekeeper",
    "is_loopback_host",
]
