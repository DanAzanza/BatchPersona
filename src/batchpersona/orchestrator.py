"""Production-grade headless batch-processing orchestrator for ComfyUI model replacement."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
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

import requests
import websocket

from batchpersona.client import ComfyUIClient
from batchpersona.housekeeper import ComfyUIHousekeeper, create_housekeeper
from batchpersona.models import (
    COLOR_BOLD,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_RESET,
    COLOR_YELLOW,
    ComfyAPIError,
    JobResult,
    JobStatus,
    PipelineError,
    SwapperConfig,
)
from batchpersona.template import WorkflowTemplate

LOGGER = logging.getLogger("batch_swapper")


def configure_logging(level: int = logging.INFO) -> None:
    """Initialize structured console logging format."""
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


class BatchSwapper:
    """High-level batch pipeline orchestrator managing localized model replacement."""

    def __init__(
        self,
        config: SwapperConfig,
        client: ComfyUIClient | None = None,
        template: WorkflowTemplate | None = None,
        housekeeper: ComfyUIHousekeeper | None = None,
    ) -> None:
        self._config = config
        self._client = client or ComfyUIClient(config.server_address)
        self._template = template or WorkflowTemplate.load_from_file(config.workflow_path)
        self._housekeeper = housekeeper or create_housekeeper(
            server_address=config.server_address,
            enabled=config.cleanup,
            input_dir=config.comfy_input_dir,
            output_dir=config.comfy_output_dir,
        )

    def _resolve_mask_path(self) -> Path | None:
        """Resolve mask path explicitly or via automatic matching pattern."""
        if self._config.mask_path is not None and self._config.mask_path.is_file():
            return self._config.mask_path

        # Auto-pairing pattern: look for <campaign_stem>_mask.png in the campaign folder
        candidate = self._config.campaign_path.with_name(
            f"{self._config.campaign_path.stem}_mask.png"
        )
        if candidate.is_file():
            LOGGER.info("Auto-paired matching mask for campaign: %s", candidate.name)
            return candidate
        return None

    def _render_progress_bar(self, step: int, max_steps: int, node: str) -> None:
        """Render in-place terminal progress bar or structured log in headless/CI environments."""
        if max_steps <= 0:
            return
        percent = min(100.0, (step / max_steps) * 100.0)

        if not sys.stdout.isatty():
            # Headless CI environment: emit structured log at 25% checkpoints
            if step % max(1, max_steps // 4) == 0 or step == max_steps:
                LOGGER.info(
                    "[SAMPLER Node %s] Progress: %5.1f%% (%d/%d)",
                    node,
                    percent,
                    step,
                    max_steps,
                )
            return

        bar_length = 30
        filled = int((percent / 100.0) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        sys.stdout.write(
            f"\r{COLOR_CYAN}[SAMPLER Node {node}]{COLOR_RESET} |{bar}| {percent:5.1f}% ({step}/{max_steps})"
        )
        sys.stdout.flush()

    def process_single_model(
        self,
        model_file: Path,
        campaign_asset_name: str,
        mask_asset_name: str | None,
    ) -> JobResult:
        """Execute single model replacement job sequentially."""
        model_stem = model_file.stem
        output_prefix = f"{self._config.campaign_path.stem}_{model_stem}"
        final_target = self._config.output_dir / f"{output_prefix}.png"

        # Check skip-existing guard before starting job
        if (
            self._config.skip_existing
            and not self._config.force
            and final_target.is_file()
            and final_target.stat().st_size > 0
        ):
            LOGGER.info(
                "%s[SKIP]%s Existing output found: %s",
                COLOR_YELLOW,
                COLOR_RESET,
                final_target.name,
            )
            return JobResult(
                model_name=model_file.name,
                output_path=final_target,
                status=JobStatus.SKIPPED,
                duration_seconds=0.0,
            )

        job_start = time.time()
        client_id = str(uuid.uuid4())
        is_oom = False

        LOGGER.info(
            "%sStarting swap job for model:%s %s%s%s (Target: %s)",
            COLOR_BOLD,
            COLOR_RESET,
            COLOR_CYAN,
            model_file.name,
            COLOR_RESET,
            output_prefix,
        )

        model_asset_name: str | None = None
        try:
            # 1. Upload model reference asset
            self._housekeeper.record_pre_upload(model_file.name)
            model_asset_name = self._client.upload_image(model_file)
            self._housekeeper.record_uploaded_name(model_asset_name)

            # 2. Patch workflow graph with assets, overrides (seed, resolution, prompt)
            prompt_payload = self._template.patch(
                campaign_image_name=campaign_asset_name,
                model_image_name=model_asset_name,
                output_prefix=output_prefix,
                mask_image_name=mask_asset_name,
                seed=self._config.seed,
                resolution=self._config.resolution,
                prompt=self._config.prompt,
            )

            # 3. Queue prompt
            prompt_id = self._client.queue_prompt(prompt_payload, client_id=client_id)
            LOGGER.info("Prompt accepted by ComfyUI. Assigned ID: %s", prompt_id)

            # 4. Stream execution events
            outputs = self._client.track_execution(
                prompt_id=prompt_id,
                client_id=client_id,
                timeout_seconds=self._config.timeout_seconds,
                on_progress=self._render_progress_bar,
            )
            # Clear line after progress
            if sys.stdout.isatty():
                sys.stdout.write("\r" + " " * 70 + "\r")
                sys.stdout.flush()

            if not outputs:
                raise ComfyAPIError("Job completed but SaveImage node produced zero output assets.")

            # 5. Download primary output
            primary_asset = outputs[0]
            self._client.download_image(primary_asset, final_target)

            # 6. Housekeeping: Purge duplicate output asset from ComfyUI directory
            self._housekeeper.cleanup_output_asset(
                asset_filename=primary_asset.filename,
                asset_subfolder=primary_asset.subfolder,
                final_target=final_target,
            )

            elapsed = time.time() - job_start

            LOGGER.info(
                "%s[PASS]%s Successfully rendered %s (%.2fs)",
                COLOR_GREEN,
                COLOR_RESET,
                final_target.name,
                elapsed,
            )
            return JobResult(
                model_name=model_file.name,
                output_path=final_target,
                status=JobStatus.SUCCESS,
                duration_seconds=elapsed,
            )

        except (
            PipelineError,
            requests.RequestException,
            websocket.WebSocketException,
            OSError,
            ValueError,
            RuntimeError,
        ) as e:
            is_oom = "out of memory" in str(e).lower()
            elapsed = time.time() - job_start
            sys.stdout.write("\n")
            LOGGER.error(
                "%s[FAIL]%s Model %s failed: %s",
                COLOR_RED,
                COLOR_RESET,
                model_file.name,
                e,
            )
            self._client.interrupt_execution()
            return JobResult(
                model_name=model_file.name,
                output_path=None,
                status=JobStatus.FAILED,
                duration_seconds=elapsed,
                error_message=str(e),
            )
        finally:
            self._client.free_memory(unload_models=is_oom)
            if model_asset_name:
                self._housekeeper.cleanup_model_input(model_asset_name, model_file)

    def run_batch(self) -> list[JobResult]:
        """Execute model swap pipeline sequentially across all portraits."""
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        model_files = sorted(
            [
                p
                for p in self._config.models_dir.glob(self._config.model_filter)
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]
        )

        if not model_files:
            LOGGER.warning("No valid portrait images found in %s", self._config.models_dir)
            return []

        LOGGER.info(
            "Discovered %d target model portraits in %s",
            len(model_files),
            self._config.models_dir,
        )

        # Early check: if skip-existing is enabled, determine if all models can be skipped upfront
        if self._config.skip_existing and not self._config.force:
            pending_models = [
                m
                for m in model_files
                if not (
                    (
                        self._config.output_dir / f"{self._config.campaign_path.stem}_{m.stem}.png"
                    ).is_file()
                    and (
                        self._config.output_dir / f"{self._config.campaign_path.stem}_{m.stem}.png"
                    )
                    .stat()
                    .st_size
                    > 0
                )
            ]
            if not pending_models:
                LOGGER.info(
                    "%s[SKIP ALL]%s All %d outputs already exist in %s. Skipping batch.",
                    COLOR_YELLOW,
                    COLOR_RESET,
                    len(model_files),
                    self._config.output_dir,
                )
                results = [
                    JobResult(
                        model_name=m.name,
                        output_path=self._config.output_dir
                        / f"{self._config.campaign_path.stem}_{m.stem}.png",
                        status=JobStatus.SKIPPED,
                        duration_seconds=0.0,
                    )
                    for m in model_files
                ]
                self._print_batch_summary(results)
                return results

        # Pre-flight ComfyUI server connectivity check
        if not self._client.check_connection():
            raise ComfyAPIError(
                f"Cannot connect to ComfyUI at {self._client._http_base}.\n"
                f"Troubleshooting tips:\n"
                f"  1. Verify ComfyUI is running.\n"
                f"  2. Check server port: ComfyUI Desktop typically runs on 8000, while portable/git uses 8188.\n"
                f"  3. Pass --server 127.0.0.1:<port> or export COMFYUI_SERVER env var."
            )

        # Ingest campaign base image
        LOGGER.info("Ingesting base campaign asset: %s", self._config.campaign_path)
        self._housekeeper.record_pre_upload(self._config.campaign_path.name)
        campaign_asset_name = self._client.upload_image(self._config.campaign_path)
        self._housekeeper.record_uploaded_name(campaign_asset_name)

        # Ingest inpainting mask if required or auto-detected
        mask_asset_name: str | None = None
        mask_file: Path | None = None
        if self._template.requires_mask():
            mask_file = self._resolve_mask_path()
            if mask_file is not None and mask_file.is_file():
                LOGGER.info("Ingesting inpainting mask: %s", mask_file)
                self._housekeeper.record_pre_upload(mask_file.name)
                mask_asset_name = self._client.upload_image(mask_file)
                self._housekeeper.record_uploaded_name(mask_asset_name)
            else:
                LOGGER.warning(
                    "Workflow expects mask placeholder, but no matching mask found for %s",
                    self._config.campaign_path.name,
                )
        else:
            LOGGER.info(
                "Workflow operates without external mask input (autonomous self-masking or maskless Qwen mode)."
            )

        results: list[JobResult] = []
        try:
            for index, model_path in enumerate(model_files, start=1):
                LOGGER.info(
                    "\n%s=== Processing Asset [%d/%d]: %s ===%s",
                    COLOR_BOLD,
                    index,
                    len(model_files),
                    model_path.name,
                    COLOR_RESET,
                )
                res = self.process_single_model(
                    model_file=model_path,
                    campaign_asset_name=campaign_asset_name,
                    mask_asset_name=mask_asset_name,
                )
                results.append(res)
        finally:
            self._housekeeper.cleanup_shared_inputs(
                [
                    (campaign_asset_name, self._config.campaign_path),
                    (mask_asset_name, mask_file),
                ]
            )

        self._print_batch_summary(results)
        return results

    def _print_batch_summary(self, results: list[JobResult]) -> None:
        """Print formatted execution summary table."""
        print(f"\n{COLOR_BOLD}{'=' * 75}{COLOR_RESET}")
        print(f"{COLOR_BOLD}COMFYUI BATCH MODEL SWAP - EXECUTION SUMMARY{COLOR_RESET}")
        print(f"{COLOR_BOLD}{'=' * 75}{COLOR_RESET}")
        print(f"{'MODEL':<28} | {'STATUS':<9} | {'TIME':<8} | {'EXPORT PATH / ERROR'}")
        print(f"{'-' * 28}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 24}")

        for r in results:
            if r.status == JobStatus.SUCCESS:
                status_color = COLOR_GREEN
            elif r.status == JobStatus.SKIPPED:
                status_color = COLOR_YELLOW
            else:
                status_color = COLOR_RED

            status_str = f"{status_color}{r.status.value:<9}{COLOR_RESET}"
            detail = r.output_path.name if r.output_path else (r.error_message or "Unknown error")
            print(f"{r.model_name:<28} | {status_str} | {r.duration_seconds:6.2f}s | {detail}")

        passed = sum(1 for r in results if r.status == JobStatus.SUCCESS)
        skipped = sum(1 for r in results if r.status == JobStatus.SKIPPED)
        failed = sum(1 for r in results if r.status == JobStatus.FAILED)
        total = len(results)

        summary_color = COLOR_GREEN if failed == 0 else COLOR_RED
        print(f"{'-' * 75}")
        print(
            f"Total: {total} | {COLOR_GREEN}Passed: {passed}{COLOR_RESET} | "
            f"{COLOR_YELLOW}Skipped: {skipped}{COLOR_RESET} | "
            f"{summary_color}Failed: {failed}{COLOR_RESET}\n"
        )


def parse_cli_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Production Headless ComfyUI Batch Model Replacement Pipeline"
    )
    default_server = os.environ.get("COMFYUI_SERVER", "127.0.0.1:8188")
    parser.add_argument(
        "--server",
        default=default_server,
        help=f"ComfyUI server address (host:port). Default: $COMFYUI_SERVER or {default_server}",
    )
    parser.add_argument(
        "--campaign",
        required=True,
        type=Path,
        help="Path to the base campaign advertisement image",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Path to segmentation / inpainting mask (optional)",
    )
    parser.add_argument(
        "--models-dir",
        required=True,
        type=Path,
        help="Directory containing target reference portrait images",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data/output"),
        type=Path,
        help="Export target directory for localized campaign assets",
    )
    parser.add_argument(
        "--market-tag",
        default="global",
        help="Market localization tag (e.g. apac, latam, emea)",
    )
    parser.add_argument(
        "--workflow",
        default=Path("workflows/model_swap_qwen21_maskless_api.json"),
        type=Path,
        help="Path to ComfyUI workflow JSON template",
    )
    parser.add_argument(
        "--timeout",
        default=300.0,
        type=float,
        help="Maximum timeout in seconds per job (default: 300.0)",
    )
    parser.add_argument(
        "--filter",
        default="*",
        help="Glob pattern to filter models inside --models-dir (default: '*')",
    )
    parser.add_argument(
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        default=True,
        help="Disable automatic purging of intermediate ComfyUI assets",
    )
    parser.add_argument(
        "--comfy-input-dir",
        default=None,
        type=Path,
        help="Override ComfyUI input directory path for housekeeping",
    )
    parser.add_argument(
        "--comfy-output-dir",
        default=None,
        type=Path,
        help="Override ComfyUI output directory path for housekeeping",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip models whose output asset already exists in --output-dir",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-rendering even if output asset already exists",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override diffusion sampler seed (0 to 2^64 - 1)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override model resolution (e.g. 1024, 1280)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override positive instruction prompt",
    )
    return parser.parse_args(args)


def main() -> None:
    """CLI orchestrator entrypoint."""
    configure_logging()
    args = parse_cli_args()

    config = SwapperConfig(
        server_address=args.server,
        campaign_path=args.campaign,
        mask_path=args.mask,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        market_tag=args.market_tag,
        workflow_path=args.workflow,
        timeout_seconds=args.timeout,
        model_filter=args.filter,
        cleanup=args.cleanup,
        comfy_input_dir=args.comfy_input_dir,
        comfy_output_dir=args.comfy_output_dir,
        seed=args.seed,
        resolution=args.resolution,
        prompt=args.prompt,
        skip_existing=args.skip_existing,
        force=args.force,
    )

    swapper = BatchSwapper(config)
    results = swapper.run_batch()
    if not results:
        LOGGER.error("Zero models were processed. Verify --models-dir and --filter arguments.")
        sys.exit(2)
    failures = sum(1 for r in results if r.status == JobStatus.FAILED)
    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
