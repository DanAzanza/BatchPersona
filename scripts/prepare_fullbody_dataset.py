"""Dataset preparation utility for full-body photorealistic models and campaign advertisements.

Processes full-body commercial fashion lookbooks and multi-ethnic model reference photos,
standardizing resolutions to 896x1152 (canonical 3:4 diffusion aspect ratio) using
centering-aware proportional cropping (ImageOps.fit) to avoid anatomy squashing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Final

from PIL import Image, ImageOps

LOGGER = logging.getLogger("dataset_preparer")

TARGET_WIDTH: Final[int] = 896
TARGET_HEIGHT: Final[int] = 1152


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured console logging."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def resize_and_save(src_path: Path, dst_path: Path) -> None:
    """Standardize source image to 896x1152 PNG using aspect-ratio preserving fit.

    Uses ImageOps.fit with vertical centering biased towards upper torso (0.35)
    to prevent head cutoff and eliminate stretching/squashing distortion.
    """
    if not src_path.is_file():
        raise FileNotFoundError(f"Source image not found: {src_path}")

    with Image.open(src_path) as im:
        rgb_img = im.convert("RGB")
        # Proportional fit preserving aspect ratio (centering: 0.5 x, 0.35 y for portrait lookbooks)
        fitted = ImageOps.fit(
            rgb_img,
            (TARGET_WIDTH, TARGET_HEIGHT),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.35),
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        fitted.save(dst_path, format="PNG", optimize=True)
        LOGGER.info(
            "Exported asset: %s (%dx%d, %d bytes)",
            dst_path.name,
            TARGET_WIDTH,
            TARGET_HEIGHT,
            dst_path.stat().st_size,
        )


def process_dataset(artifacts_dir: Path, workspace_data_dir: Path) -> int:
    """Ingest artifact images and build the standardized full-body dataset.

    Returns the count of successfully processed assets.
    """
    if not artifacts_dir.is_dir():
        LOGGER.warning("Artifacts directory does not exist: %s", artifacts_dir)
        return 0

    campaign_dir = workspace_data_dir / "input_campaign"
    models_dir = workspace_data_dir / "input_models"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover and map artifact files to canonical names
    artifact_files = list(artifacts_dir.glob("*.jpg")) + list(artifacts_dir.glob("*.png"))
    name_map: dict[str, Path] = {}
    for f in artifact_files:
        fname = f.name.lower()
        if "campaign_editorial_female" in fname:
            name_map["campaign_female"] = f
        elif "campaign_editorial_male" in fname:
            name_map["campaign_male"] = f
        elif "model_east_asia_f" in fname:
            name_map["model_east_asia_f01"] = f
        elif "model_west_africa_m" in fname:
            name_map["model_west_africa_m01"] = f
        elif "model_nordic_f" in fname:
            name_map["model_nordic_f01"] = f
        elif "model_south_america_m" in fname:
            name_map["model_south_america_m01"] = f

    LOGGER.info("Matched %d source images in %s", len(name_map), artifacts_dir)

    # 2. Process Campaign Images
    campaign_mapping = [
        ("campaign_female", campaign_dir / "campaign_fashion_female.png"),
        ("campaign_male", campaign_dir / "campaign_fashion_male.png"),
    ]
    processed_count = 0
    for key, img_dst in campaign_mapping:
        if key in name_map:
            resize_and_save(name_map[key], img_dst)
            processed_count += 1

    # 3. Process Diverse Full-Body Target Models
    model_mapping = [
        ("model_east_asia_f01", models_dir / "model_east_asia_f01.png"),
        ("model_west_africa_m01", models_dir / "model_west_africa_m01.png"),
        ("model_nordic_f01", models_dir / "model_nordic_f01.png"),
        ("model_south_america_m01", models_dir / "model_south_america_m01.png"),
    ]
    for key, dst in model_mapping:
        if key in name_map:
            resize_and_save(name_map[key], dst)
            processed_count += 1

    LOGGER.info(
        "Processed %d standardized full-body assets into %s",
        processed_count,
        workspace_data_dir,
    )
    return processed_count


def parse_cli_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Standardize full-body campaign and model assets to 896x1152."
    )
    default_src = Path(os.environ.get("SOURCE_ARTIFACTS_DIR", "."))
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_src,
        help="Directory containing source candidate images (default: $SOURCE_ARTIFACTS_DIR or '.')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Target workspace data directory (default: data)",
    )
    return parser.parse_args(args)


def main() -> None:
    """CLI entry point."""
    setup_logging()
    cli_args = parse_cli_args(sys.argv[1:])
    count = process_dataset(cli_args.source_dir, cli_args.output_dir)
    sys.exit(0 if count > 0 else 1)


if __name__ == "__main__":
    main()
