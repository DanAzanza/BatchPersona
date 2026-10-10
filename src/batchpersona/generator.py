"""Synthetic test data generator for advertising campaign model replacement."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

LOGGER = logging.getLogger("testdata_generator")

DEFAULT_IMAGE_SIZE: Final[int] = 1024
FEATHER_BLUR_RADIUS: Final[float] = 3.5


@dataclass(frozen=True)
class ColorPalette:
    """Color palette definition for synthetic portrait generation."""

    skin_base: tuple[int, int, int]
    skin_shadow: tuple[int, int, int]
    skin_highlight: tuple[int, int, int]
    hair: tuple[int, int, int]
    eyes: tuple[int, int, int]
    lips: tuple[int, int, int]
    backdrop: tuple[int, int, int]
    accent: tuple[int, int, int]


EAST_ASIA_PALETTE: Final[ColorPalette] = ColorPalette(
    skin_base=(242, 218, 188),
    skin_shadow=(214, 183, 149),
    skin_highlight=(255, 238, 216),
    hair=(24, 24, 28),
    eyes=(42, 28, 20),
    lips=(198, 92, 92),
    backdrop=(235, 238, 242),
    accent=(45, 85, 125),
)

SOUTH_AMERICA_PALETTE: Final[ColorPalette] = ColorPalette(
    skin_base=(175, 122, 78),
    skin_shadow=(138, 92, 54),
    skin_highlight=(205, 149, 102),
    hair=(35, 26, 22),
    eyes=(58, 38, 25),
    lips=(150, 80, 68),
    backdrop=(228, 232, 224),
    accent=(160, 95, 45),
)

NORDIC_PALETTE: Final[ColorPalette] = ColorPalette(
    skin_base=(252, 228, 218),
    skin_shadow=(228, 196, 184),
    skin_highlight=(255, 245, 240),
    hair=(238, 218, 154),
    eyes=(65, 115, 155),
    lips=(215, 120, 128),
    backdrop=(240, 240, 245),
    accent=(120, 140, 160),
)


def _setup_logging() -> None:
    """Configure structured console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _create_studio_gradient(
    size: int, color_top: tuple[int, int, int], color_bottom: tuple[int, int, int]
) -> Image.Image:
    """Generate a smooth vertical studio background gradient with subtle micro-noise."""
    y_coords = np.linspace(0.0, 1.0, size).reshape(size, 1)
    top_arr = np.array(color_top, dtype=np.float32)
    bottom_arr = np.array(color_bottom, dtype=np.float32)

    gradient = (1.0 - y_coords) * top_arr + y_coords * bottom_arr
    base_img = np.tile(gradient[:, np.newaxis, :], (1, size, 1))

    # Add high-frequency subtle texture to prevent zero-variance CLIP embeddings
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(loc=0.0, scale=2.5, size=(size, size, 3))
    composited = np.clip(base_img + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(composited, mode="RGB")


def _draw_textured_ellipse(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    width: int = 1,
) -> None:
    """Draw a filled ellipse with optional outline."""
    draw.ellipse(bbox, fill=fill, outline=outline, width=width)


def synthesize_campaign_lookbook(size: int = DEFAULT_IMAGE_SIZE) -> Image.Image:
    """Synthesize high-end commercial fashion advertisement with blue blazer and subject."""
    img = _create_studio_gradient(size, (245, 247, 250), (210, 218, 228))
    draw = ImageDraw.ImageDraw(img)

    center_x = size // 2
    scale = size / 1024.0

    # Studio decorative geometric lighting arches (commercial ad backdrop)
    arch_box = (
        int(center_x - 360 * scale),
        int(80 * scale),
        int(center_x + 360 * scale),
        int(920 * scale),
    )
    draw.arc(arch_box, start=180, end=360, fill=(190, 202, 218), width=int(6 * scale))
    draw.line(
        [(arch_box[0], int(500 * scale)), (arch_box[0], int(960 * scale))],
        fill=(190, 202, 218),
        width=int(6 * scale),
    )
    draw.line(
        [(arch_box[2], int(500 * scale)), (arch_box[2], int(960 * scale))],
        fill=(190, 202, 218),
        width=int(6 * scale),
    )

    # Torso & Shoulders: Tailored Cobalt Blue Jacket
    jacket_points = [
        (int(center_x - 310 * scale), size),
        (int(center_x - 240 * scale), int(640 * scale)),
        (int(center_x - 140 * scale), int(560 * scale)),
        (int(center_x + 140 * scale), int(560 * scale)),
        (int(center_x + 240 * scale), int(640 * scale)),
        (int(center_x + 310 * scale), size),
    ]
    draw.polygon(jacket_points, fill=(24, 68, 142))

    # Jacket lapels (darker navy shadow lines)
    left_lapel = [
        (int(center_x - 140 * scale), int(560 * scale)),
        (int(center_x - 30 * scale), int(760 * scale)),
        (int(center_x - 90 * scale), size),
        (int(center_x - 240 * scale), int(640 * scale)),
    ]
    draw.polygon(left_lapel, fill=(18, 48, 108))

    right_lapel = [
        (int(center_x + 140 * scale), int(560 * scale)),
        (int(center_x + 30 * scale), int(760 * scale)),
        (int(center_x + 90 * scale), size),
        (int(center_x + 240 * scale), int(640 * scale)),
    ]
    draw.polygon(right_lapel, fill=(18, 48, 108))

    # Inner White Shirt & Tie/V-Neck opening
    shirt_points = [
        (int(center_x - 65 * scale), int(540 * scale)),
        (center_x, int(680 * scale)),
        (int(center_x + 65 * scale), int(540 * scale)),
    ]
    draw.polygon(shirt_points, fill=(248, 250, 252))

    # Neck
    neck_box = (
        int(center_x - 55 * scale),
        int(460 * scale),
        int(center_x + 55 * scale),
        int(570 * scale),
    )
    draw.rectangle(neck_box, fill=(225, 192, 168))

    # Base Head Silhouette & Placeholder Facial Region
    head_box = (
        int(center_x - 130 * scale),
        int(230 * scale),
        int(center_x + 130 * scale),
        int(510 * scale),
    )
    draw.ellipse(head_box, fill=(235, 202, 178), outline=(200, 165, 140), width=int(2 * scale))

    # Hair silhouette (Dark Neutral Base)
    hair_dome = (
        int(center_x - 145 * scale),
        int(190 * scale),
        int(center_x + 145 * scale),
        int(400 * scale),
    )
    draw.arc(hair_dome, start=180, end=360, fill=(35, 30, 30), width=int(45 * scale))

    # Brand typography overlay placeholder
    draw.text(
        (int(60 * scale), int(70 * scale)),
        "MAISON VIRTUAL // SUMMER LOOKBOOK",
        fill=(100, 115, 135),
    )
    draw.text(
        (int(60 * scale), int(100 * scale)),
        "COLLECTION 2026 - AUTONOMOUS APPAREL",
        fill=(140, 155, 175),
    )

    return img


def synthesize_campaign_mask(size: int = DEFAULT_IMAGE_SIZE) -> Image.Image:
    """Synthesize high-fidelity inpainting mask isolating subject face, neck, and hair."""
    mask_canvas = Image.new("L", (size, size), color=0)
    draw = ImageDraw.ImageDraw(mask_canvas)

    center_x = size // 2
    scale = size / 1024.0

    # Target region includes head, hair contour, ears, and neck down to shirt collar
    head_box = (
        int(center_x - 165 * scale),
        int(180 * scale),
        int(center_x + 165 * scale),
        int(535 * scale),
    )
    draw.ellipse(head_box, fill=255)

    # Neck region down to collar
    neck_polygon = [
        (int(center_x - 75 * scale), int(480 * scale)),
        (int(center_x + 75 * scale), int(480 * scale)),
        (int(center_x + 65 * scale), int(565 * scale)),
        (int(center_x - 65 * scale), int(565 * scale)),
    ]
    draw.polygon(neck_polygon, fill=255)

    # Upper hair volume padding
    hair_box = (
        int(center_x - 170 * scale),
        int(170 * scale),
        int(center_x + 170 * scale),
        int(380 * scale),
    )
    draw.ellipse(hair_box, fill=255)

    # Feather borders using Gaussian blur to eliminate 8x VAE latent seam artifacts
    feathered_mask = mask_canvas.filter(ImageFilter.GaussianBlur(radius=FEATHER_BLUR_RADIUS))
    return feathered_mask


def synthesize_model_portrait(
    palette: ColorPalette,
    archetype_title: str,
    size: int = DEFAULT_IMAGE_SIZE,
    is_masculine_features: bool = False,
) -> Image.Image:
    """Synthesize photorealistic archetype portrait with studio lighting and high-contrast features."""
    img = _create_studio_gradient(
        size,
        palette.backdrop,
        (palette.backdrop[0] - 25, palette.backdrop[1] - 25, palette.backdrop[2] - 25),
    )
    draw = ImageDraw.ImageDraw(img)

    center_x = size // 2
    scale = size / 1024.0

    # Decorative portrait studio framing / contrast markers
    marker_box = (
        int(60 * scale),
        int(60 * scale),
        int((1024 - 60) * scale),
        int((1024 - 60) * scale),
    )
    draw.rectangle(marker_box, outline=palette.accent, width=int(2 * scale))

    # Shoulders & Base Garment (Neutral studio drape)
    shoulder_y = int(680 * scale)
    shoulder_points = [
        (int(center_x - 340 * scale), size),
        (int(center_x - 260 * scale), shoulder_y),
        (int(center_x + 260 * scale), shoulder_y),
        (int(center_x + 340 * scale), size),
    ]
    draw.polygon(shoulder_points, fill=palette.accent)

    # Neck
    neck_width = 75 if is_masculine_features else 60
    neck_box = (
        int(center_x - neck_width * scale),
        int(530 * scale),
        int(center_x + neck_width * scale),
        int(700 * scale),
    )
    draw.rectangle(neck_box, fill=palette.skin_shadow)

    # Head Structure (Base Oval)
    jaw_spread = 155 if is_masculine_features else 140
    head_box = (
        int(center_x - jaw_spread * scale),
        int(240 * scale),
        int(center_x + jaw_spread * scale),
        int(610 * scale),
    )
    draw.ellipse(head_box, fill=palette.skin_base)

    # Skin Highlight: Forehead & Cheekbones
    highlight_box = (
        int(center_x - 80 * scale),
        int(280 * scale),
        int(center_x + 80 * scale),
        int(420 * scale),
    )
    draw.ellipse(highlight_box, fill=palette.skin_highlight)

    # Eyebrows
    brow_y = int(370 * scale)
    brow_thickness = int(7 * scale if is_masculine_features else 4 * scale)
    draw.line(
        [
            (int(center_x - 85 * scale), brow_y),
            (int(center_x - 30 * scale), brow_y - int(10 * scale)),
        ],
        fill=palette.hair,
        width=brow_thickness,
    )
    draw.line(
        [
            (int(center_x + 30 * scale), brow_y - int(10 * scale)),
            (int(center_x + 85 * scale), brow_y),
        ],
        fill=palette.hair,
        width=brow_thickness,
    )

    # Eyes & Catchlight Reflections
    eye_y = int(410 * scale)
    eye_spacing = int(58 * scale)
    eye_width = int(26 * scale)
    eye_height = int(14 * scale)

    # Left eye
    left_eye_box = (
        center_x - eye_spacing - eye_width,
        eye_y - eye_height,
        center_x - eye_spacing + eye_width,
        eye_y + eye_height,
    )
    draw.ellipse(left_eye_box, fill=(250, 250, 252), outline=palette.skin_shadow, width=1)
    draw.ellipse(
        (
            center_x - eye_spacing - int(10 * scale),
            eye_y - int(10 * scale),
            center_x - eye_spacing + int(10 * scale),
            eye_y + int(10 * scale),
        ),
        fill=palette.eyes,
    )
    draw.ellipse(
        (
            center_x - eye_spacing - int(4 * scale),
            eye_y - int(5 * scale),
            center_x - eye_spacing,
            eye_y - int(1 * scale),
        ),
        fill=(255, 255, 255),
    )

    # Right eye
    right_eye_box = (
        center_x + eye_spacing - eye_width,
        eye_y - eye_height,
        center_x + eye_spacing + eye_width,
        eye_y + eye_height,
    )
    draw.ellipse(right_eye_box, fill=(250, 250, 252), outline=palette.skin_shadow, width=1)
    draw.ellipse(
        (
            center_x + eye_spacing - int(10 * scale),
            eye_y - int(10 * scale),
            center_x + eye_spacing + int(10 * scale),
            eye_y + int(10 * scale),
        ),
        fill=palette.eyes,
    )
    draw.ellipse(
        (
            center_x + eye_spacing + int(2 * scale),
            eye_y - int(5 * scale),
            center_x + eye_spacing + int(6 * scale),
            eye_y - int(1 * scale),
        ),
        fill=(255, 255, 255),
    )

    # Nose Bridge & Base
    nose_tip_y = int(475 * scale)
    draw.line(
        [(center_x, int(420 * scale)), (center_x, nose_tip_y)],
        fill=palette.skin_shadow,
        width=int(3 * scale),
    )
    draw.ellipse(
        (
            int(center_x - 14 * scale),
            nose_tip_y - int(6 * scale),
            int(center_x + 14 * scale),
            nose_tip_y + int(6 * scale),
        ),
        fill=palette.skin_shadow,
    )

    # Lips
    lip_y = int(530 * scale)
    lip_width = int(35 * scale)
    lip_height = int(12 * scale)
    draw.ellipse(
        (
            center_x - lip_width,
            lip_y - lip_height,
            center_x + lip_width,
            lip_y + lip_height,
        ),
        fill=palette.lips,
    )

    # Facial hair contour if masculine archetype
    if is_masculine_features:
        beard_box = (
            int(center_x - 120 * scale),
            int(480 * scale),
            int(center_x + 120 * scale),
            int(620 * scale),
        )
        draw.arc(beard_box, start=30, end=150, fill=palette.hair, width=int(18 * scale))

    # Hair Sculpt / Coiffure
    hair_top = (
        int(center_x - 170 * scale),
        int(180 * scale),
        int(center_x + 170 * scale),
        int(420 * scale),
    )
    draw.arc(hair_top, start=175, end=365, fill=palette.hair, width=int(42 * scale))

    # Archetype Studio Metadata Watermark
    draw.text(
        (int(80 * scale), int(930 * scale)),
        f"ARCHETYPE: {archetype_title.upper()}",
        fill=palette.accent,
    )
    draw.text(
        (int(80 * scale), int(955 * scale)),
        "STUDIO REFERENCE 1024x1024 // 2026 STANDARDS",
        fill=(130, 140, 150),
    )

    return img


def generate_all_testdata(output_root: Path, size: int = DEFAULT_IMAGE_SIZE) -> list[Path]:
    """Generate the complete set of synthetic test campaign assets and model references."""
    campaign_dir = output_root / "input_campaign"
    models_dir = output_root / "input_models"
    output_dir = output_root / "output"

    campaign_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[Path] = []

    # 1. Base Campaign Lookbook
    lookbook_path = campaign_dir / "campaign_summer_lookbook.png"
    lookbook_img = synthesize_campaign_lookbook(size=size)
    lookbook_img.save(lookbook_path, format="PNG")
    generated_files.append(lookbook_path)
    LOGGER.info(
        "Generated Lookbook Asset: %s (%dx%d, %d bytes)",
        lookbook_path.name,
        size,
        size,
        lookbook_path.stat().st_size,
    )

    # 2. Campaign Segmentation Mask
    mask_path = campaign_dir / "campaign_summer_mask.png"
    mask_img = synthesize_campaign_mask(size=size)
    mask_img.save(mask_path, format="PNG")
    generated_files.append(mask_path)
    LOGGER.info(
        "Generated Inpaint Mask: %s (%dx%d, %d bytes)",
        mask_path.name,
        size,
        size,
        mask_path.stat().st_size,
    )

    # 3. Diverse Model Portraits
    models_to_create: Sequence[tuple[str, ColorPalette, str, bool]] = [
        (
            "model_east_asia_f01.png",
            EAST_ASIA_PALETTE,
            "East Asia Female (Archetype 01)",
            False,
        ),
        (
            "model_south_america_m01.png",
            SOUTH_AMERICA_PALETTE,
            "South America Male (Archetype 01)",
            True,
        ),
        ("model_nordic_f01.png", NORDIC_PALETTE, "Nordic Female (Archetype 01)", False),
    ]

    for filename, palette, title, is_masculine in models_to_create:
        model_path = models_dir / filename
        portrait = synthesize_model_portrait(
            palette=palette,
            archetype_title=title,
            size=size,
            is_masculine_features=is_masculine,
        )
        portrait.save(model_path, format="PNG")
        generated_files.append(model_path)
        LOGGER.info(
            "Generated Target Model: %s (%dx%d, %d bytes)",
            model_path.name,
            size,
            size,
            model_path.stat().st_size,
        )

    return generated_files


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Synthesize high-fidelity test datasets for ComfyUI model replacement pipeline."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Root directory for generated datasets (default: data)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Output image square resolution (default: 1024)",
    )
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    _setup_logging()
    args = parse_arguments()
    LOGGER.info("Initiating synthetic test data generation in %s...", args.output_dir)
    generated = generate_all_testdata(output_root=args.output_dir, size=args.size)
    LOGGER.info(
        "Synthetic generation completed successfully. Total assets created: %d",
        len(generated),
    )


if __name__ == "__main__":
    main()
