"""Backward compatibility shim forwarding to batchpersona.generator."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for p in [str(REPO_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from batchpersona.generator import (
    DEFAULT_IMAGE_SIZE,
    EAST_ASIA_PALETTE,
    FEATHER_BLUR_RADIUS,
    NORDIC_PALETTE,
    SOUTH_AMERICA_PALETTE,
    ColorPalette,
    generate_all_testdata,
    main,
    parse_arguments,
    synthesize_campaign_lookbook,
    synthesize_campaign_mask,
    synthesize_model_portrait,
)

__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "EAST_ASIA_PALETTE",
    "FEATHER_BLUR_RADIUS",
    "NORDIC_PALETTE",
    "SOUTH_AMERICA_PALETTE",
    "ColorPalette",
    "generate_all_testdata",
    "main",
    "parse_arguments",
    "synthesize_campaign_lookbook",
    "synthesize_campaign_mask",
    "synthesize_model_portrait",
]

if __name__ == "__main__":
    main()
