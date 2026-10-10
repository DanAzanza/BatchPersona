"""Backward compatibility shim forwarding to batchpersona.ci."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for p in [str(REPO_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import batchpersona.ci as _ci

CI_TEMP_DIR = _ci.CI_TEMP_DIR
setup_logging = _ci.setup_logging
parse_cli_args = _ci.parse_cli_args


def main(args: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint for running CI checks locally."""
    parsed = parse_cli_args(args)
    setup_logging(verbose=parsed.verbose)
    current_module = sys.modules[__name__]
    runner = getattr(current_module, "run_all_ci_stages", _ci.run_all_ci_stages)
    return runner(
        skip_lint=parsed.skip_lint,
        skip_tests=parsed.skip_tests,
    )


def run_stage(stage_name: str, cmd: Sequence[str], cwd: Path = REPO_ROOT) -> bool:
    """Execute a single CI pipeline stage."""
    return _ci.run_stage(stage_name, cmd, cwd=cwd)


def run_all_ci_stages(
    python_executable: str = sys.executable,
    skip_lint: bool = False,
    skip_tests: bool = False,
    **kwargs: Any,
) -> int:
    """Run all CI pipeline stages in sequence."""
    current_module = sys.modules[__name__]
    stage_runner = kwargs.get("run_stage_fn") or getattr(current_module, "run_stage", _ci.run_stage)
    staging_dir = kwargs.get("staging_dir") or getattr(
        current_module, "CI_TEMP_DIR", _ci.CI_TEMP_DIR
    )
    return _ci.run_all_ci_stages(
        python_executable=python_executable,
        skip_lint=skip_lint,
        skip_tests=skip_tests,
        run_stage_fn=stage_runner,
        staging_dir=staging_dir,
    )


__all__ = [
    "CI_TEMP_DIR",
    "main",
    "parse_cli_args",
    "run_all_ci_stages",
    "run_stage",
    "setup_logging",
]

if __name__ == "__main__":
    sys.exit(main())
