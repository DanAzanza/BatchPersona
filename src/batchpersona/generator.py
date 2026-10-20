"""Synthetic workflow environment and prompt configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

DEFAULT_QWEN_EDIT_PROMPT: Final[str] = (
    "Edit <image1>: Replace only the person's face, hair, and skin tone across all visible skin "
    "with the identity from <image2>. Keep the exact body pose, clothing, shoes, and background "
    "of <image1> unchanged."
)
QWEN_EDIT_PROMPT: Final[str] = DEFAULT_QWEN_EDIT_PROMPT


def generate_synthetic_environment(
    workflow_path: Path | str | None = None,
) -> None:
    """Ensure synthetic workflow files are populated with the refined Qwen 2.1 prompt."""
    target_path = (
        Path(workflow_path)
        if workflow_path
        else Path("workflows/model_swap_qwen21_maskless_api.json")
    )
    if target_path.is_file():
        data = json.loads(target_path.read_text(encoding="utf-8"))
        if "6" in data and "inputs" in data["6"]:
            data["6"]["inputs"]["prompt"] = DEFAULT_QWEN_EDIT_PROMPT
            target_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
