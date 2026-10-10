"""BatchPersona - Headless batch-processing pipeline for automated model replacement in advertising campaigns."""

from __future__ import annotations

from batchpersona.client import ComfyUIClient
from batchpersona.housekeeper import ComfyUIHousekeeper, create_housekeeper, is_loopback_host
from batchpersona.launcher import (
    ComfyInstance,
    build_headless_command,
    detect_comfyui,
    is_server_online,
    launch_server,
)
from batchpersona.models import (
    COLOR_BOLD,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_RESET,
    COLOR_YELLOW,
    ComfyAPIError,
    ComfyExecutionError,
    ComfyTimeoutError,
    JobResult,
    JobStatus,
    OutputAsset,
    PipelineError,
    SwapperConfig,
)
from batchpersona.orchestrator import BatchSwapper, configure_logging
from batchpersona.template import WorkflowTemplate

__version__ = "1.0.0"

__all__ = [
    "COLOR_BOLD",
    "COLOR_CYAN",
    "COLOR_GREEN",
    "COLOR_RED",
    "COLOR_RESET",
    "COLOR_YELLOW",
    "BatchSwapper",
    "ComfyAPIError",
    "ComfyExecutionError",
    "ComfyInstance",
    "ComfyTimeoutError",
    "ComfyUIClient",
    "ComfyUIHousekeeper",
    "JobResult",
    "JobStatus",
    "OutputAsset",
    "PipelineError",
    "SwapperConfig",
    "WorkflowTemplate",
    "__version__",
    "build_headless_command",
    "configure_logging",
    "create_housekeeper",
    "detect_comfyui",
    "is_loopback_host",
    "is_server_online",
    "launch_server",
]
