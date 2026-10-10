"""Backward compatibility shim forwarding to batchpersona.preprocess."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for p in [str(REPO_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from batchpersona.preprocess import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    main,
    parse_cli_args,
    process_dataset,
    resize_and_save,
    setup_logging,
)

__all__ = [
    "TARGET_HEIGHT",
    "TARGET_WIDTH",
    "main",
    "parse_cli_args",
    "process_dataset",
    "resize_and_save",
    "setup_logging",
]

if __name__ == "__main__":
    main()
