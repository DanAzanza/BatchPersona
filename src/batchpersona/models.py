"""Core domain models, enumerations, exceptions, and configuration schemas for BatchPersona."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

# ANSI Color codes for high-visibility terminal output
COLOR_RESET: Final[str] = "\033[0m"
COLOR_CYAN: Final[str] = "\033[36m"
COLOR_GREEN: Final[str] = "\033[32m"
COLOR_YELLOW: Final[str] = "\033[33m"
COLOR_RED: Final[str] = "\033[31m"
COLOR_BOLD: Final[str] = "\033[1m"


class JobStatus(str, Enum):
    """Execution status of an individual model swap job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class PipelineError(Exception):
    """Base domain exception for ComfyUI pipeline failures."""


class ComfyAPIError(PipelineError):
    """Exception raised when ComfyUI REST API returns a non-200 status code."""


class ComfyExecutionError(PipelineError):
    """Exception raised when ComfyUI returns an execution_error WebSocket event."""


class ComfyTimeoutError(PipelineError):
    """Exception raised when an execution exceeds the maximum allocated duration."""


@dataclass(frozen=True)
class SwapperConfig:
    """Immutable configuration parameters for the batch swapper."""

    server_address: str
    campaign_path: Path
    models_dir: Path
    output_dir: Path
    workflow_path: Path
    mask_path: Path | None = None
    market_tag: str = "global"
    timeout_seconds: float = 300.0
    model_filter: str = "*"
    validate_paths: bool = True
    cleanup: bool = True
    comfy_input_dir: Path | None = None
    comfy_output_dir: Path | None = None
    seed: int | None = None
    resolution: int | None = None
    prompt: str | None = None
    skip_existing: bool = False
    force: bool = False

    def __post_init__(self) -> None:
        """Validate configuration paths and parameters fail-fast."""
        if self.validate_paths:
            if not self.campaign_path.is_file():
                raise FileNotFoundError(f"Campaign image does not exist: {self.campaign_path}")
            if not self.models_dir.is_dir():
                raise NotADirectoryError(f"Models directory does not exist: {self.models_dir}")
            if not self.workflow_path.is_file():
                raise FileNotFoundError(f"Workflow file does not exist: {self.workflow_path}")
            if self.mask_path is not None and not self.mask_path.is_file():
                raise FileNotFoundError(f"Mask file does not exist: {self.mask_path}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"Timeout must be positive, got {self.timeout_seconds}")
        if self.seed is not None and not (0 <= self.seed <= 18446744073709551615):
            raise ValueError(
                f"Seed must be an unsigned 64-bit integer (0 <= seed <= 2^64-1), got {self.seed}"
            )
        if self.resolution is not None and self.resolution <= 0:
            raise ValueError(f"Resolution must be a positive integer, got {self.resolution}")


@dataclass(frozen=True)
class OutputAsset:
    """Descriptor for an output image asset returned by ComfyUI."""

    filename: str
    subfolder: str
    image_type: str


@dataclass(frozen=True)
class JobResult:
    """Outcome and diagnostics of a model swap execution."""

    model_name: str
    output_path: Path | None
    status: JobStatus
    duration_seconds: float
    error_message: str | None = None
