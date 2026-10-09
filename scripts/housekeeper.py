"""Smart Local Housekeeping for ComfyUI intermediate assets.

Safely purges duplicate input uploads and generated output files in local ComfyUI
directories while protecting original user assets, preventing path collision wipeouts,
and gracefully disabling on remote servers.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

# Ensure repository root is on sys.path when executed directly as a script
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("batch_swapper.housekeeper")

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})


def is_loopback_host(server: str) -> bool:
    """Determine whether server address points to the local machine loopback."""
    raw = server.strip().lower()
    if raw in LOOPBACK_HOSTS or raw.startswith("::1"):
        return True
    if not raw.startswith(("http://", "https://", "ws://", "wss://")):
        raw = f"http://{raw}"

    try:
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        return host in LOOPBACK_HOSTS or host.startswith("127.") or host.startswith("::1")
    except Exception:
        return False


class ComfyUIHousekeeper:
    """Manages safe purging of ComfyUI intermediate assets on local filesystems."""

    def __init__(
        self,
        input_dir: Path | None = None,
        output_dir: Path | None = None,
        enabled: bool = True,
        max_retries: int = 3,
        retry_delay_seconds: float = 0.05,
    ) -> None:
        self.input_dir = input_dir.resolve() if input_dir and input_dir.exists() else None
        self.output_dir = output_dir.resolve() if output_dir and output_dir.exists() else None
        self.enabled = enabled
        self.max_retries = max(1, max_retries)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)

        self._pre_existing_inputs: set[Path] = set()
        self._tracked_inputs: set[Path] = set()

    def record_pre_upload(self, asset_name: str) -> None:
        """Inspect if the asset name already exists in ComfyUI input directory prior to upload."""
        if not self.enabled or self.input_dir is None:
            return

        candidate = (self.input_dir / asset_name).resolve()
        if candidate.exists():
            self._pre_existing_inputs.add(candidate)
            LOGGER.debug("Recorded pre-existing asset in ComfyUI input: %s", candidate.name)
        else:
            self._tracked_inputs.add(candidate)

    def record_uploaded_name(self, server_asset_name: str) -> None:
        """Register the actual server filename assigned by ComfyUI if different from original."""
        if not self.enabled or self.input_dir is None:
            return

        candidate = (self.input_dir / server_asset_name).resolve()
        if candidate not in self._pre_existing_inputs:
            self._tracked_inputs.add(candidate)

    def cleanup_model_input(
        self,
        model_asset_name: str | None,
        source_path: Path,
    ) -> bool:
        """Purge model portrait uploaded to ComfyUI input directory."""
        if not self.enabled or self.input_dir is None or not model_asset_name:
            return False

        target = (self.input_dir / model_asset_name).resolve()
        return self._purge_input_asset(target, source_path)

    def cleanup_shared_inputs(
        self,
        assets: Sequence[tuple[str | None, Path | None]],
    ) -> int:
        """Purge shared session inputs (campaign lookbook, masks) upon batch completion."""
        if not self.enabled or self.input_dir is None:
            return 0

        purged_count = 0
        for asset_name, source_path in assets:
            if not asset_name or not source_path:
                continue
            target = (self.input_dir / asset_name).resolve()
            if self._purge_input_asset(target, source_path):
                purged_count += 1

        return purged_count

    def cleanup_output_asset(
        self,
        asset_filename: str,
        asset_subfolder: str,
        final_target: Path,
    ) -> bool:
        """Purge duplicate generated image from ComfyUI output directory."""
        if not self.enabled or self.output_dir is None or not asset_filename:
            return False

        intermediate = (
            (self.output_dir / asset_subfolder / asset_filename).resolve()
            if asset_subfolder
            else (self.output_dir / asset_filename).resolve()
        )

        if not intermediate.exists():
            return False

        # Guard: Never delete if final export target is identical to intermediate path
        if intermediate == final_target.resolve():
            LOGGER.debug(
                "Skipping output cleanup: export target is identical to ComfyUI output (%s)",
                intermediate,
            )
            return False

        success = self._safe_delete(intermediate)
        if success:
            LOGGER.info("Cleaned up intermediate ComfyUI output asset: %s", intermediate.name)
        return success

    def _purge_input_asset(self, target: Path, source_path: Path) -> bool:
        """Internal helper validating safety guards before deleting an input asset."""
        if not target.exists():
            return False

        # Guard 1: Never delete original user source asset
        if target == source_path.resolve():
            LOGGER.debug(
                "Skipping input cleanup: target is identical to source asset (%s)",
                target,
            )
            return False

        # Guard 2: Never delete files that pre-existed before this session
        if target in self._pre_existing_inputs:
            LOGGER.debug(
                "Skipping input cleanup: asset pre-existed in ComfyUI input (%s)",
                target.name,
            )
            return False

        # Guard 3: Only delete tracked assets uploaded during this session
        if target not in self._tracked_inputs:
            LOGGER.debug(
                "Skipping input cleanup: asset was not tracked during this session (%s)",
                target.name,
            )
            return False

        success = self._safe_delete(target)
        if success:
            self._tracked_inputs.discard(target)
            LOGGER.info("Cleaned up intermediate ComfyUI input asset: %s", target.name)
        return success

    def _safe_delete(self, path: Path) -> bool:
        """Safely delete file with bounded micro-retries for Windows file lock resilience."""
        for attempt in range(self.max_retries):
            try:
                path.unlink()
                return True
            except FileNotFoundError:
                return True
            except OSError as err:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay_seconds)
                else:
                    LOGGER.warning(
                        "Could not clean up intermediate ComfyUI asset %s: %s",
                        path.name,
                        err,
                    )
        return False


def create_housekeeper(
    server_address: str,
    enabled: bool = True,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> ComfyUIHousekeeper:
    """Instantiate a configured housekeeper, guarding against remote servers."""
    if not enabled:
        return ComfyUIHousekeeper(enabled=False)

    if not is_loopback_host(server_address):
        LOGGER.info(
            "ComfyUI server is remote (%s); local filesystem housekeeping disabled.",
            server_address,
        )
        return ComfyUIHousekeeper(enabled=False)

    resolved_input: Path | None = input_dir
    resolved_output: Path | None = output_dir

    # 1. Environment variable fallbacks
    if resolved_input is None and "COMFYUI_INPUT_DIR" in os.environ:
        p = Path(os.environ["COMFYUI_INPUT_DIR"])
        if p.exists():
            resolved_input = p

    if resolved_output is None and "COMFYUI_OUTPUT_DIR" in os.environ:
        p = Path(os.environ["COMFYUI_OUTPUT_DIR"])
        if p.exists():
            resolved_output = p

    # 2. ComfyUI desktop auto-detection fallback
    if resolved_input is None or resolved_output is None:
        try:
            from scripts.launch_comfyui import detect_comfyui

            instance = detect_comfyui()
            if instance:
                if resolved_input is None and instance.input_dir and instance.input_dir.exists():
                    resolved_input = instance.input_dir
                if resolved_output is None and instance.output_dir and instance.output_dir.exists():
                    resolved_output = instance.output_dir
        except Exception as e:
            LOGGER.debug("Auto-detecting ComfyUI directories encountered error: %s", e)

    if resolved_input or resolved_output:
        LOGGER.info(
            "Local ComfyUI housekeeping active (input: %s, output: %s)",
            resolved_input.name if resolved_input else "none",
            resolved_output.name if resolved_output else "none",
        )

    return ComfyUIHousekeeper(
        input_dir=resolved_input,
        output_dir=resolved_output,
        enabled=True,
    )
