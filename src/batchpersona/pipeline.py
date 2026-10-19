#!/usr/bin/env python3
"""Interactive Pipeline Runner & CLI Orchestrator for BatchPersona."""

from __future__ import annotations

import os
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

from batchpersona.ci import run_all_ci_stages
from batchpersona.launcher import (
    detect_comfyui,
    is_server_online,
    launch_server,
    stop_server,
)
from batchpersona.models import DEFAULT_SERVER_ADDRESS, JobStatus, SwapperConfig
from batchpersona.orchestrator import BatchSwapper
from batchpersona.orchestrator import main as orchestrator_main

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_AUTO_STARTED_SERVER: str | None = None


def cleanup_auto_started_server() -> None:
    """Terminate and release resources for auto-started ComfyUI backend."""
    global _AUTO_STARTED_SERVER
    if _AUTO_STARTED_SERVER:
        print(
            f"\n[AUTO-SHUTDOWN] Closing auto-started ComfyUI console at {_AUTO_STARTED_SERVER}..."
        )
        stop_server(_AUTO_STARTED_SERVER)
        _AUTO_STARTED_SERVER = None


def ensure_server_ready(server: str = DEFAULT_SERVER_ADDRESS, timeout: float = 90.0) -> bool:
    """Ensure ComfyUI server is online, launching it headlessly if needed."""
    global _AUTO_STARTED_SERVER
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

    _AUTO_STARTED_SERVER = server
    return True


def run_commercial_swap(
    server: str = DEFAULT_SERVER_ADDRESS,
    campaign_dir: Path = Path("data/input_campaign"),
    models_dir: Path = Path("data/input_models"),
    output_dir: Path = Path("data/output"),
    workflow_path: Path = Path("workflows/model_swap_qwen21_maskless_api.json"),
    timeout: float = 300.0,
    skip_existing: bool = False,
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

        config = SwapperConfig(
            server_address=server,
            campaign_path=campaign_file,
            models_dir=models_dir,
            output_dir=output_dir,
            workflow_path=workflow_path,
            timeout_seconds=timeout,
            skip_existing=skip_existing,
        )
        swapper = BatchSwapper(config)
        results = swapper.run_batch()
        if not results:
            overall_exit_code = 2
        elif any(r.status == JobStatus.FAILED for r in results):
            overall_exit_code = 1

    return overall_exit_code


def run_quick_dry_run(
    server: str = DEFAULT_SERVER_ADDRESS,
    synthetic_dir: Path = Path("tests/data_synthetic"),
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

    if not test_lookbook.is_file() or not test_models.is_dir():
        print(f"[ERROR] Required synthetic test assets not found in {synthetic_dir}.")
        return 1

    config = SwapperConfig(
        server_address=server,
        campaign_path=test_lookbook,
        mask_path=test_mask,
        models_dir=test_models,
        output_dir=test_output,
        workflow_path=Path("workflows/model_swap_composite_api.json"),
        market_tag="test",
        timeout_seconds=timeout,
    )
    swapper = BatchSwapper(config)
    results = swapper.run_batch()
    if not results:
        return 2
    return 1 if any(r.status == JobStatus.FAILED for r in results) else 0


def run_quality_gate() -> int:
    """Execute complete local CI test suite and verification."""
    print("\n[RUNNING] Running Complete Quality Gate and CI Verification locally...")
    return run_all_ci_stages()


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
    print("  [4] Close ComfyUI Console      (Free GPU VRAM & Terminate Server)")
    print("  [0] Exit")
    print()
    print(header)


def interactive_menu_loop(server: str) -> int:
    """Run the interactive console menu loop until exited."""
    try:
        while True:
            display_menu(server)
            try:
                choice = input("Select an option [1-4, 0]: ").strip()
            except (KeyboardInterrupt, EOFError):
                cleanup_auto_started_server()
                print("\nExiting BatchPersona. Goodbye!")
                return 0

            exit_code = 0
            if choice == "1":
                exit_code = run_commercial_swap(server)
            elif choice == "2":
                exit_code = run_quick_dry_run(server)
            elif choice == "3":
                exit_code = run_quality_gate()
            elif choice == "4":
                print(f"\n[SHUTDOWN] Terminating ComfyUI server at {server}...")
                stopped = stop_server(server)
                if stopped:
                    print("[OK] ComfyUI server and console closed. GPU VRAM released.")
                else:
                    print("[WARNING] Could not stop ComfyUI server or server is still responding.")
                exit_code = 0
            elif choice == "0":
                cleanup_auto_started_server()
                print("Exiting BatchPersona. Goodbye!")
                return 0
            else:
                print(f"[WARNING] Invalid selection: '{choice}'. Please enter 1, 2, 3, 4, or 0.")
                continue

            print()
            if exit_code == 0:
                print("[SUCCESS] Operation completed successfully.")
            else:
                print(f"[FAILED] Operation exited with error code: {exit_code}")

            try:
                input("\nPress Enter to return to menu...")
            except (KeyboardInterrupt, EOFError):
                cleanup_auto_started_server()
                return 0
    finally:
        cleanup_auto_started_server()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: forwards direct arguments or starts the interactive menu."""
    if argv is None:
        argv = sys.argv[1:]

    # Direct pass-through if batch_swapper arguments are passed
    if argv:
        try:
            orchestrator_main(argv)
            return 0
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)

    default_server = os.environ.get("COMFYUI_SERVER", DEFAULT_SERVER_ADDRESS)
    return interactive_menu_loop(default_server)


if __name__ == "__main__":
    sys.exit(main())
