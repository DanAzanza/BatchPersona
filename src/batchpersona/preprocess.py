"""Dataset preparation utility for full-body photorealistic models and campaign advertisements."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Final

from PIL import Image, ImageOps

LOGGER = logging.getLogger("dataset_preparer")

TARGET_WIDTH: Final[int] = 896
TARGET_HEIGHT: Final[int] = 1152
VALID_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured console logging."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def resize_and_save(src_path: Path, dst_path: Path) -> None:
    """Standardize source image to 896x1152 PNG using aspect-ratio preserving fit.

    Performs an atomic write via a temporary file to prevent corruption on interruption.
    """
    if not src_path.is_file():
        raise FileNotFoundError(f"Source image not found: {src_path}")

    # If source and destination resolve to the same path and already meet specifications, skip
    if src_path.resolve() == dst_path.resolve():
        try:
            with Image.open(src_path) as current:
                if current.size == (TARGET_WIDTH, TARGET_HEIGHT) and current.format == "PNG":
                    LOGGER.info("Asset already standardized at %s", dst_path.name)
                    return
        except (OSError, ValueError, SyntaxError):
            pass

    with Image.open(src_path) as im:
        rgb_img = im.convert("RGB")
        fitted = ImageOps.fit(
            rgb_img,
            (TARGET_WIDTH, TARGET_HEIGHT),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.35),
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dst = dst_path.with_name(f"{dst_path.stem}_{uuid.uuid4().hex[:8]}.tmp.png")
        try:
            fitted.save(tmp_dst, format="PNG", optimize=True)
            tmp_dst.replace(dst_path)
        finally:
            if tmp_dst.is_file():
                with contextlib.suppress(OSError):
                    tmp_dst.unlink(missing_ok=True)

        LOGGER.info(
            "Exported asset: %s (%dx%d, %d bytes)",
            dst_path.name,
            TARGET_WIDTH,
            TARGET_HEIGHT,
            dst_path.stat().st_size,
        )


def process_dataset(
    artifacts_dir: Path,
    workspace_data_dir: Path,
    campaign_src: Path | None = None,
    models_src: Path | None = None,
) -> int:
    """Ingest artifact images and build the standardized full-body dataset.

    Supports:
    1. Explicit sources via `campaign_src` (file or dir) and/or `models_src` (dir or file).
    2. Legacy artifact filename matching if specific legacy substrings exist in `artifacts_dir`.
    3. Generic mode on `artifacts_dir` (or subdirectories `input_campaign` / `input_models`):
       - Files starting with `campaign_` -> `workspace_data_dir / input_campaign / {stem}.png`
       - All other valid image files -> `workspace_data_dir / input_models / {stem}.png`
    """
    campaign_dir = workspace_data_dir / "input_campaign"
    models_dir = workspace_data_dir / "input_models"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0

    # 1. Explicit campaign_src or models_src mode
    if campaign_src is not None or models_src is not None:
        if campaign_src is not None:
            for f in _get_valid_image_files(campaign_src, label="campaign"):
                resize_and_save(f, campaign_dir / f"{f.stem}.png")
                processed_count += 1

        if models_src is not None:
            for f in _get_valid_image_files(models_src, label="models"):
                resize_and_save(f, models_dir / f"{f.stem}.png")
                processed_count += 1

        LOGGER.info(
            "Processed %d standardized full-body assets into %s",
            processed_count,
            workspace_data_dir,
        )
        return processed_count

    # 2. Directory scan mode
    if not artifacts_dir.is_dir():
        LOGGER.warning("Artifacts directory does not exist: %s", artifacts_dir)
        return 0

    artifact_files = sorted(
        [
            f
            for f in artifacts_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
    )

    name_map: dict[str, Path] = {}
    for f in artifact_files:
        fname = f.name.lower()
        if "campaign_editorial_female" in fname:
            name_map["campaign_female"] = f
        elif "campaign_editorial_male" in fname:
            name_map["campaign_male"] = f
        elif "campaign_urban_female" in fname:
            name_map["campaign_urban_female"] = f
        elif "campaign_arch_male" in fname:
            name_map["campaign_arch_male"] = f
        elif "campaign_resort_female" in fname:
            name_map["campaign_resort_female"] = f
        elif "campaign_street_male" in fname:
            name_map["campaign_street_male"] = f
        elif "model_east_asia_f" in fname:
            name_map["model_east_asia_f01"] = f
        elif "model_west_africa_m" in fname:
            name_map["model_west_africa_m01"] = f
        elif "model_nordic_f" in fname:
            name_map["model_nordic_f01"] = f
        elif "model_south_america_m" in fname:
            name_map["model_south_america_m01"] = f

    LOGGER.info("Matched %d source images in %s", len(name_map), artifacts_dir)

    handled_files: set[Path] = set()

    campaign_mapping = [
        ("campaign_female", campaign_dir / "campaign_fashion_female.png"),
        ("campaign_male", campaign_dir / "campaign_fashion_male.png"),
        ("campaign_urban_female", campaign_dir / "campaign_fashion_urban_female.png"),
        ("campaign_arch_male", campaign_dir / "campaign_fashion_arch_male.png"),
        ("campaign_resort_female", campaign_dir / "campaign_fashion_resort_female.png"),
        ("campaign_street_male", campaign_dir / "campaign_fashion_street_male.png"),
    ]
    for key, img_dst in campaign_mapping:
        if key in name_map:
            src_f = name_map[key]
            resize_and_save(src_f, img_dst)
            handled_files.add(src_f)
            processed_count += 1

    model_mapping = [
        ("model_east_asia_f01", models_dir / "model_east_asia_f01.png"),
        ("model_west_africa_m01", models_dir / "model_west_africa_m01.png"),
        ("model_nordic_f01", models_dir / "model_nordic_f01.png"),
        ("model_south_america_m01", models_dir / "model_south_america_m01.png"),
    ]
    for key, dst in model_mapping:
        if key in name_map:
            src_f = name_map[key]
            resize_and_save(src_f, dst)
            handled_files.add(src_f)
            processed_count += 1

    # Generic Directory Mode for files in artifacts_dir not matched by legacy mapping
    for f in artifact_files:
        if f in handled_files:
            continue
        if f.name.lower().startswith("campaign_"):
            resize_and_save(f, campaign_dir / f"{f.stem}.png")
        else:
            resize_and_save(f, models_dir / f"{f.stem}.png")
        handled_files.add(f)
        processed_count += 1

    # Generic Directory Mode for subdirectories input_campaign and input_models
    sub_campaign = artifacts_dir / "input_campaign"
    if sub_campaign.is_dir() and sub_campaign.resolve() != campaign_dir.resolve():
        for f in sorted(sub_campaign.iterdir()):
            if (
                f.is_file()
                and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
                and f not in handled_files
            ):
                resize_and_save(f, campaign_dir / f"{f.stem}.png")
                handled_files.add(f)
                processed_count += 1

    sub_models = artifacts_dir / "input_models"
    if sub_models.is_dir() and sub_models.resolve() != models_dir.resolve():
        for f in sorted(sub_models.iterdir()):
            if (
                f.is_file()
                and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
                and f not in handled_files
            ):
                resize_and_save(f, models_dir / f"{f.stem}.png")
                handled_files.add(f)
                processed_count += 1

    LOGGER.info(
        "Processed %d standardized full-body assets into %s",
        processed_count,
        workspace_data_dir,
    )
    return processed_count


def _get_valid_image_files(path: Path, label: str = "source") -> list[Path]:
    """Retrieve list of valid image files from file or directory with diagnostic warnings."""
    if path.is_file():
        if path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
            return [path]
        LOGGER.warning("%s source file has unsupported extension: %s", label.capitalize(), path)
        return []
    if path.is_dir():
        return [
            f
            for f in sorted(path.iterdir())
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
    not_found_msg = (
        f"{label.capitalize()} source directory not found: %s"
        if label == "models"
        else f"{label.capitalize()} source path not found: %s"
    )
    LOGGER.warning(not_found_msg, path)
    return []


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
    parser.add_argument(
        "--campaign-src",
        type=Path,
        default=None,
        help="Optional source file or directory for campaign images.",
    )
    parser.add_argument(
        "--models-src",
        type=Path,
        default=None,
        help="Optional source directory (or file) for model images.",
    )
    return parser.parse_args(args)


def main() -> None:
    """CLI entry point."""
    setup_logging()
    cli_args = parse_cli_args(sys.argv[1:])
    count = process_dataset(
        cli_args.source_dir,
        cli_args.output_dir,
        campaign_src=cli_args.campaign_src,
        models_src=cli_args.models_src,
    )
    sys.exit(0 if count > 0 else 1)


if __name__ == "__main__":
    main()
