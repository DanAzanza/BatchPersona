#!/usr/bin/env python3
"""
Interactive Pipeline Runner & CLI Orchestrator for BatchPersona.

Coordinates batch model swaps across lookbooks, instant zero-GPU dry-runs,
local CI verification, and headless ComfyUI auto-launching with a unified,
cross-platform interface.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

# Ensure Windows terminal standard streams handle UTF-8 properly
if sys.platform == "win32":
    reconfig_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfig_stdout):
        reconfig_stdout(encoding="utf-8")
    reconfig_stderr = getattr(sys.stderr, "reconfigure", None)
    if callable(reconfig_stderr):
        reconfig_stderr(encoding="utf-8")

from scripts.launch_comfyui import (
    detect_comfyui,
    is_server_online,
    launch_server,
)

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def ensure_server_ready(server: str = "127.0.0.1:8000", timeout: float = 90.0) -> bool:
    """Ensure ComfyUI server is online, launching it headlessly if needed."""
    if is_server_online(server, timeout=1.5):
        return True

    print(f"\n[NOTICE] ComfyUI server is offline at {server}.")
    print("[AUTO-START] Automatically starting ComfyUI backend in background...")

    instance = detect_comfyui()
    if not instance:
        print(f"[ERROR] No ComfyUI installation detected to launch for {server}.")
        print("        Please start ComfyUI manually or check your installation.")
        return False

    success = launch_server(instance, server=server, timeout=timeout)
    if not success:
        print(f"[ERROR] ComfyUI server failed to become ready at {server} within {int(timeout)}s.")
        return False

    return True


def run_commercial_swap(
    server: str = "127.0.0.1:8000",
    campaign_dir: Path = Path("data/input_campaign"),
    models_dir: Path = Path("data/input_models"),
    output_dir: Path = Path("data/output"),
    workflow_path: Path = Path("workflows/model_swap_qwen21_maskless_api.json"),
    timeout: float = 300.0,
) -> int:
    """Run batch model swap across all campaign lookbook images."""
    if not ensure_server_ready(server):
        return 1

    print("\n[RUNNING] Executing Commercial Batch Model Swap across Campaign Lookbooks...")
    if not campaign_dir.exists():
        print(f"[WARNING] Campaign directory does not exist: {campaign_dir}")
        return 1

    campaign_files = sorted(
        p
        for p in campaign_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )

    if not campaign_files:
        print(
            f"[WARNING] No campaign images found in {campaign_dir} (supported: .png, .jpg, .jpeg, .webp)"
        )
        return 0

    overall_exit_code = 0
    for idx, campaign_file in enumerate(campaign_files, start=1):
        print("\n" + "=" * 75)
        print(f"[CAMPAIGN {idx}/{len(campaign_files)}] Processing Lookbook: {campaign_file.name}")
        print("=" * 75)

        cmd = [
            sys.executable,
            "scripts/batch_swapper.py",
            "--server",
            server,
            "--campaign",
            str(campaign_file),
            "--models-dir",
            str(models_dir),
            "--output-dir",
            str(output_dir),
            "--workflow",
            str(workflow_path),
            "--timeout",
            str(timeout),
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            overall_exit_code = result.returncode

    return overall_exit_code


def run_quick_dry_run(
    server: str = "127.0.0.1:8000",
    synthetic_dir: Path = Path("data_synthetic"),
    timeout: float = 30.0,
) -> int:
    """Execute instant zero-GPU dry-run using synthetic assets and composite workflow."""
    if not ensure_server_ready(server):
        return 1

    print("\n[RUNNING] Executing Instant Composite Test Dry-Run (No GPU Required)...")
    test_lookbook = synthetic_dir / "input_campaign" / "campaign_summer_lookbook.png"
    test_mask = synthetic_dir / "input_campaign" / "campaign_summer_mask.png"
    test_models = synthetic_dir / "input_models"
    test_output = synthetic_dir / "output"

    if not test_lookbook.exists():
        print("[INFO] Generating isolated synthetic test dataset for dry-run...")
        cmd_gen = [
            sys.executable,
            "scripts/generate_testdata.py",
            "--output-dir",
            str(synthetic_dir),
            "--size",
            "896",
        ]
        gen_result = subprocess.run(cmd_gen)
        if gen_result.returncode != 0:
            print("[ERROR] Failed to generate synthetic test dataset.")
            return gen_result.returncode

    cmd_swap = [
        sys.executable,
        "scripts/batch_swapper.py",
        "--server",
        server,
        "--campaign",
        str(test_lookbook),
        "--mask",
        str(test_mask),
        "--models-dir",
        str(test_models),
        "--output-dir",
        str(test_output),
        "--workflow",
        "workflows/model_swap_composite_api.json",
        "--market-tag",
        "test",
        "--timeout",
        str(timeout),
    ]

    result = subprocess.run(cmd_swap)
    return result.returncode


def run_quality_gate() -> int:
    """Execute complete local CI test suite and verification."""
    print("\n[RUNNING] Running Complete Quality Gate and CI Verification locally...")
    cmd = [sys.executable, "scripts/run_ci_locally.py"]
    result = subprocess.run(cmd)
    return result.returncode


def display_menu(server: str) -> None:
    """Render the interactive CLI dashboard menu."""
    header = "=" * 75
    print("\n" + header)
    print("      BatchPersona - Headless ComfyUI Model Replacement Pipeline")
    print(header)
    print(f"  Python runtime: {sys.executable}")
    print(f"  Server address: {server}")
    print(header)
    print()
    print("  [1] Run Commercial Model Swap  (Campaign Lookbooks - Diverse Personas)")
    print("  [2] Run Quick Dry-Run          (Instant Zero-GPU Verification, < 1s)")
    print("  [3] Run Quality Gate and Tests (Pytest + Coverage + Ruff)")
    print("  [0] Exit")
    print()
    print(header)


def interactive_menu_loop(server: str) -> int:
    """Run the interactive console menu loop until exited."""
    while True:
        display_menu(server)
        try:
            choice = input("Select an option [1-3, 0]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting BatchPersona. Goodbye!")
            return 0

        exit_code = 0
        if choice == "1":
            exit_code = run_commercial_swap(server)
        elif choice == "2":
            exit_code = run_quick_dry_run(server)
        elif choice == "3":
            exit_code = run_quality_gate()
        elif choice == "0":
            print("Exiting BatchPersona. Goodbye!")
            return 0
        else:
            print(f"[WARNING] Invalid selection: '{choice}'. Please enter 1, 2, 3, or 0.")
            continue

        print()
        if exit_code == 0:
            print("[SUCCESS] Operation completed successfully.")
        else:
            print(f"[FAILED] Operation exited with error code: {exit_code}")

        try:
            input("\nPress Enter to return to menu...")
        except (KeyboardInterrupt, EOFError):
            return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: forwards direct arguments or starts the interactive menu."""
    if argv is None:
        argv = sys.argv[1:]

    # Direct pass-through if batch_swapper arguments are passed
    if argv:
        cmd = [sys.executable, "scripts/batch_swapper.py", *argv]
        return subprocess.run(cmd).returncode

    default_server = os.environ.get("COMFYUI_SERVER", "127.0.0.1:8000")
    return interactive_menu_loop(default_server)


if __name__ == "__main__":
    sys.exit(main())
