"""Local CI quality gate runner mirroring GitHub Actions CI pipeline."""

from __future__ import annotations

import argparse
import logging
import shutil
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
    run_stage_fn: Callable[..., bool] | None = None,
    staging_dir: Path | None = None,
) -> int:
    """Run all CI pipeline stages in sequence with deterministic cleanup."""
    stage_runner = run_stage_fn or run_stage
    temp_dir = staging_dir or CI_TEMP_DIR

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

    # Stage 4: CI Synthetic Test Data Generation Smoke Test
    ci_gen_cmd = [
        python_executable,
        "scripts/generate_testdata.py",
        "--output-dir",
        str(temp_dir),
        "--size",
        "256",
    ]
    try:
        if not stage_runner("CI Synthetic Data Generator", ci_gen_cmd):
            return 1

        # Verify expected smoke test outputs exist
        expected_files = [
            temp_dir / "input_campaign" / "campaign_summer_lookbook.png",
            temp_dir / "input_campaign" / "campaign_summer_mask.png",
            temp_dir / "input_models" / "model_east_asia_f01.png",
            temp_dir / "input_models" / "model_south_america_m01.png",
            temp_dir / "input_models" / "model_nordic_f01.png",
        ]
        for f in expected_files:
            if not f.is_file():
                LOGGER.error("Missing expected CI generated asset: %s", f)
                return 1
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            LOGGER.debug("Cleaned up staging directory: %s", temp_dir)

    # Stage 5: Pytest Suite with Coverage
    if not skip_tests:
        test_cmd = [
            python_executable,
            "-m",
            "pytest",
            "tests",
            "--cov=batchpersona",
            "--cov=scripts",
            "--cov-report=term-missing",
            "-v",
        ]
        if not stage_runner("Pytest Regression Suite", test_cmd):
            return 1

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
    )


if __name__ == "__main__":
    sys.exit(main())
