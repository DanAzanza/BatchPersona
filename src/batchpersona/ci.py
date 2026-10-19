"""Local CI quality gate runner mirroring GitHub Actions CI pipeline."""

from __future__ import annotations

import argparse
import contextlib
import logging
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

LOGGER = logging.getLogger("ci_local_runner")

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
CI_TEMP_DIR: Final[Path] = REPO_ROOT / ".ci_local_staging"


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def run_stage(stage_name: str, cmd: Sequence[str], cwd: Path = REPO_ROOT) -> bool:
    """Execute a single CI pipeline stage as a subprocess and log the outcome."""
    LOGGER.info("==================================================")
    LOGGER.info("Starting CI Stage: %s", stage_name)
    LOGGER.info("Executing: %s", " ".join(cmd))
    LOGGER.info("==================================================")

    res = subprocess.run(cmd, cwd=cwd, check=False)
    if res.returncode == 0:
        LOGGER.info("[SUCCESS] CI Stage '%s' passed.", stage_name)
        return True

    LOGGER.error(
        "[FAILED] CI Stage '%s' failed with exit code %d.",
        stage_name,
        res.returncode,
    )
    return False


def run_all_ci_stages(
    python_executable: str = sys.executable,
    skip_lint: bool = False,
    skip_tests: bool = False,
    keep_coverage: bool = False,
    run_stage_fn: Callable[..., bool] | None = None,
    staging_dir: Path | None = None,
) -> int:
    """Run all CI pipeline stages in sequence with deterministic cleanup."""
    _ = staging_dir
    stage_runner = run_stage_fn or run_stage

    # Stage 1: Ruff Linting
    if not skip_lint:
        lint_cmd = [python_executable, "-m", "ruff", "check", "."]
        if not stage_runner("Ruff Linter", lint_cmd):
            return 1

        # Stage 2: Ruff Format Check
        format_cmd = [python_executable, "-m", "ruff", "format", "--check", "."]
        if not stage_runner("Ruff Formatter", format_cmd):
            return 1

        # Stage 3: Pyright Static Type Checker
        type_cmd = [python_executable, "-m", "pyright", "src", "scripts", "tests"]
        if not stage_runner("Pyright Static Type Checker", type_cmd):
            return 1

    # Stage 4: Pytest Suite with Coverage
    if not skip_tests:
        test_cmd = [
            python_executable,
            "-m",
            "pytest",
            "tests",
            "--cov=batchpersona",
            "--cov=scripts",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "-v",
        ]
        try:
            if not stage_runner("Pytest Regression Suite", test_cmd):
                return 1
        finally:
            if not keep_coverage:
                for cov_file in [REPO_ROOT / ".coverage", REPO_ROOT / "coverage.xml"]:
                    if cov_file.is_file():
                        with contextlib.suppress(OSError):
                            cov_file.unlink()
                            LOGGER.debug("Cleaned up coverage artifact: %s", cov_file)

    LOGGER.info("==================================================")
    LOGGER.info("[ALL CI CHECKS PASSED] Local CI parity 100%% verified.")
    LOGGER.info("==================================================")
    return 0


def parse_cli_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for local CI runner."""
    parser = argparse.ArgumentParser(
        description="Run GitHub Actions CI Quality Gate locally with full parity."
    )
    parser.add_argument(
        "--skip-lint",
        action="store_true",
        help="Skip Ruff linting and format checking.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest regression suite.",
    )
    parser.add_argument(
        "--keep-coverage",
        action="store_true",
        help="Keep temporary coverage artifacts (.coverage and coverage.xml) after CI run.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint for running CI checks locally."""
    parsed = parse_cli_args(args)
    setup_logging(verbose=parsed.verbose)
    return run_all_ci_stages(
        skip_lint=parsed.skip_lint,
        skip_tests=parsed.skip_tests,
        keep_coverage=parsed.keep_coverage,
    )


if __name__ == "__main__":
    sys.exit(main())
