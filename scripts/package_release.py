"""Backward compatibility shim forwarding to batchpersona.packaging."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for p in [str(REPO_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import batchpersona.packaging as _pkg

build_release_package = _pkg.build_release_package
compute_sha256 = _pkg.compute_sha256
validate_package = _pkg.validate_package
collect_distribution_files = _pkg.collect_distribution_files
main = _pkg.main

__all__ = [
    "build_release_package",
    "collect_distribution_files",
    "compute_sha256",
    "main",
    "validate_package",
]

if __name__ == "__main__":
    sys.exit(main())
