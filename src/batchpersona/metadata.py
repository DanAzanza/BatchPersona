"""EU AI Act Article 50 provenance metadata injection and ComfyUI workflow stripping utility."""

from __future__ import annotations

import contextlib
import logging
import struct
import uuid
import zlib
from pathlib import Path
from typing import Final

LOGGER = logging.getLogger("batchpersona.metadata")

DEFAULT_SOFTWARE_TAG: Final[str] = "BatchPersona"
DEFAULT_AI_GENERATED_TAG: Final[str] = "true"
DEFAULT_AI_SYSTEM_TAG: Final[str] = "BatchPersona / ComfyUI"
DEFAULT_AI_COMPLIANCE_TAG: Final[str] = "EU-AI-Act-Article-50-Synthetic-Media"
DEFAULT_MARKET_TAG: Final[str] = "global"

PROPRIETARY_CHUNKS_TO_STRIP: Final[frozenset[str]] = frozenset({"prompt", "workflow"})
PROVENANCE_TAG_KEYS: Final[frozenset[str]] = frozenset(
    {"software", "ai-generated", "ai-system", "ai-compliance", "market-tag"}
)

PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"


def _build_text_chunk(keyword: str, text: str) -> bytes:
    """Build a standard PNG tEXt chunk with valid CRC32."""
    kw_bytes = keyword.encode("latin-1", errors="replace")
    txt_bytes = text.encode("latin-1", errors="replace")
    chunk_data = kw_bytes + b"\x00" + txt_bytes
    chunk_type = b"tEXt"
    length = struct.pack(">I", len(chunk_data))
    crc = struct.pack(">I", zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF)
    return length + chunk_type + chunk_data + crc


def process_png_provenance_metadata(
    image_path: Path,
    market_tag: str | None = None,
    keep_workflow: bool = False,
) -> bool:
    """Inspect and update PNG metadata for EU AI Act compliance and IP protection.

    Fast zero-recompression implementation: directly parses and rewrites PNG chunks
    without decompressing or re-encoding pixel data (IDAT).

    1. Checks if the file is a valid PNG. If not (e.g. non-PNG format or corrupt
       byte streams), skips processing without error.
    2. Strips proprietary ComfyUI workflow/prompt chunks unless `keep_workflow` is True.
    3. Embeds machine-readable EU AI Act Article 50 provenance metadata tags.

    Returns True if the image was processed and rewritten, False otherwise.
    """
    if not image_path.is_file():
        return False

    tmp_target: Path | None = None
    try:
        raw_bytes = image_path.read_bytes()
        if not raw_bytes.startswith(PNG_SIGNATURE):
            LOGGER.debug("Skipping metadata processing on non-PNG file: %s", image_path.name)
            return False

        clean_tag = market_tag.strip() if market_tag and market_tag.strip() else DEFAULT_MARKET_TAG
        tags_to_inject = [
            ("Software", DEFAULT_SOFTWARE_TAG),
            ("AI-Generated", DEFAULT_AI_GENERATED_TAG),
            ("AI-System", DEFAULT_AI_SYSTEM_TAG),
            ("AI-Compliance", DEFAULT_AI_COMPLIANCE_TAG),
            ("Market-Tag", clean_tag),
        ]

        output_chunks: list[bytes] = [PNG_SIGNATURE]
        offset = len(PNG_SIGNATURE)
        total_len = len(raw_bytes)
        has_ihdr = False
        has_iend = False

        while offset + 8 <= total_len:
            length = struct.unpack(">I", raw_bytes[offset : offset + 4])[0]
            chunk_type = raw_bytes[offset + 4 : offset + 8]
            chunk_total_len = 12 + length

            if offset + chunk_total_len > total_len:
                # Truncated chunk
                LOGGER.debug("Truncated PNG chunk detected in %s", image_path.name)
                return False

            chunk_data = raw_bytes[offset + 8 : offset + 8 + length]
            chunk_crc = struct.unpack(
                ">I", raw_bytes[offset + 8 + length : offset + chunk_total_len]
            )[0]

            # Verify chunk CRC
            calc_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
            if chunk_crc != calc_crc:
                LOGGER.debug("CRC mismatch in PNG chunk %s in %s", chunk_type, image_path.name)
                return False

            if chunk_type == b"IHDR":
                has_ihdr = True

            # Filter metadata text chunks
            if chunk_type in (b"tEXt", b"iTXt", b"zTXt"):
                parts = chunk_data.split(b"\x00", 1)
                keyword = parts[0].decode("latin-1", errors="ignore").lower() if parts else ""

                if not keep_workflow and keyword in PROPRIETARY_CHUNKS_TO_STRIP:
                    offset += chunk_total_len
                    continue

                if keyword in PROVENANCE_TAG_KEYS:
                    offset += chunk_total_len
                    continue

            if chunk_type == b"IEND":
                has_iend = True
                # Inject provenance tags right before IEND
                for kw, val in tags_to_inject:
                    output_chunks.append(_build_text_chunk(kw, val))

            output_chunks.append(raw_bytes[offset : offset + chunk_total_len])
            offset += chunk_total_len

        if not has_ihdr or not has_iend:
            LOGGER.debug("Malformed PNG missing IHDR or IEND in %s", image_path.name)
            return False

        tmp_target = image_path.with_name(f"{image_path.stem}_{uuid.uuid4().hex[:8]}.meta.tmp")
        tmp_target.write_bytes(b"".join(output_chunks))
        tmp_target.replace(image_path)
        LOGGER.debug("Embedded EU AI Act provenance metadata into %s", image_path.name)
        return True

    except (OSError, ValueError, struct.error, zlib.error) as err:
        LOGGER.debug("Skipping PNG metadata processing on %s: %s", image_path.name, err)
        return False
    finally:
        if tmp_target is not None and tmp_target.exists():
            with contextlib.suppress(OSError):
                tmp_target.unlink()
