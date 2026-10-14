"""EU AI Act Article 50 provenance metadata injection and ComfyUI workflow stripping utility."""

from __future__ import annotations

import contextlib
import logging
import uuid
from pathlib import Path
from typing import Final

from PIL import Image, PngImagePlugin, UnidentifiedImageError

LOGGER = logging.getLogger("batchpersona.metadata")

DEFAULT_SOFTWARE_TAG: Final[str] = "BatchPersona"
DEFAULT_AI_GENERATED_TAG: Final[str] = "true"
DEFAULT_AI_SYSTEM_TAG: Final[str] = "BatchPersona / ComfyUI"
DEFAULT_AI_COMPLIANCE_TAG: Final[str] = "EU-AI-Act-Article-50-Synthetic-Media"
DEFAULT_MARKET_TAG: Final[str] = "global"

PROPRIETARY_CHUNKS_TO_STRIP: Final[frozenset[str]] = frozenset({"prompt", "workflow"})


def process_png_provenance_metadata(
    image_path: Path,
    market_tag: str | None = None,
    keep_workflow: bool = False,
) -> bool:
    """Inspect and update PNG metadata for EU AI Act compliance and IP protection.

    1. Checks if the file is a valid PNG. If not (e.g. mock byte stream in tests or
       non-PNG format), skips processing without error.
    2. Strips proprietary ComfyUI workflow/prompt chunks unless `keep_workflow` is True.
    3. Embeds machine-readable EU AI Act Article 50 provenance metadata tags.

    Returns True if the image was processed and rewritten, False otherwise.
    """
    if not image_path.is_file():
        return False

    try:
        with Image.open(image_path) as im:
            if im.format != "PNG":
                LOGGER.debug("Skipping metadata processing on non-PNG format: %s", im.format)
                return False

            im_copy = im.copy()
            raw_text = getattr(im, "text", None)
            existing_text: dict[str, str] = (
                {str(k): str(v) for k, v in raw_text.items()} if isinstance(raw_text, dict) else {}
            )

        # Build clean PngInfo
        pnginfo = PngImagePlugin.PngInfo()
        for k, v in existing_text.items():
            if not keep_workflow and k.lower() in PROPRIETARY_CHUNKS_TO_STRIP:
                continue
            pnginfo.add_text(k, str(v))

        # Embed EU AI Act Article 50 Provenance Metadata
        clean_tag = market_tag.strip() if market_tag and market_tag.strip() else DEFAULT_MARKET_TAG
        pnginfo.add_text("Software", DEFAULT_SOFTWARE_TAG)
        pnginfo.add_text("AI-Generated", DEFAULT_AI_GENERATED_TAG)
        pnginfo.add_text("AI-System", DEFAULT_AI_SYSTEM_TAG)
        pnginfo.add_text("AI-Compliance", DEFAULT_AI_COMPLIANCE_TAG)
        pnginfo.add_text("Market-Tag", clean_tag)

        # Atomic replacement avoiding file locking race conditions
        tmp_target = image_path.with_name(f"{image_path.stem}_{uuid.uuid4().hex[:8]}.meta.tmp")
        try:
            im_copy.save(tmp_target, format="PNG", pnginfo=pnginfo, optimize=True)
            tmp_target.replace(image_path)
            LOGGER.debug("Embedded EU AI Act provenance metadata into %s", image_path.name)
            return True
        finally:
            if tmp_target.exists():
                with contextlib.suppress(OSError):
                    tmp_target.unlink()

    except (UnidentifiedImageError, OSError, ValueError) as err:
        LOGGER.debug("Skipping PNG metadata processing on %s: %s", image_path.name, err)
        return False
