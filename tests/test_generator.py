"""Unit tests for synthetic dataset generation and fullbody dataset preparation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from batchpersona.generator import (
    EAST_ASIA_PALETTE,
    NORDIC_PALETTE,
    SOUTH_AMERICA_PALETTE,
    generate_all_testdata,
    synthesize_campaign_lookbook,
    synthesize_campaign_mask,
    synthesize_model_portrait,
)


def test_synthesize_campaign_lookbook() -> None:
    """Verify generated campaign lookbook has expected dimensions, mode, and content."""
    img = synthesize_campaign_lookbook(size=256)
    assert img.size == (256, 256)
    assert img.mode == "RGB"

    # Verify lookbook is not blank or flat
    extrema = img.getextrema()
    assert isinstance(extrema, tuple)
    for band_range in extrema:
        if isinstance(band_range, tuple):
            ch_min, ch_max = band_range
            assert ch_max - ch_min > 50


def test_synthesize_campaign_mask() -> None:
    """Verify generated mask is grayscale, contains binary regions and feathered edges."""
    mask = synthesize_campaign_mask(size=256)
    assert mask.size == (256, 256)
    assert mask.mode == "L"

    # Mask should have both 0 (unmasked) and 255 (masked target) regions
    extrema = mask.getextrema()
    assert extrema[0] == 0
    assert extrema[1] == 255


def test_synthesize_model_portraits() -> None:
    """Verify distinct archetype generation works across all palettes."""
    palettes = [
        (EAST_ASIA_PALETTE, "East Asia Female", False),
        (SOUTH_AMERICA_PALETTE, "South America Male", True),
        (NORDIC_PALETTE, "Nordic Female", False),
    ]
    for pal, title, is_masculine in palettes:
        img = synthesize_model_portrait(pal, title, size=256, is_masculine_features=is_masculine)
        assert img.size == (256, 256)
        assert img.mode == "RGB"


def test_generate_all_testdata(tmp_path: Path) -> None:
    """Verify end-to-end file generation writes all required assets with valid formats."""
    files = generate_all_testdata(output_root=tmp_path, size=128)
    assert len(files) == 5

    expected_names = {
        "campaign_summer_lookbook.png",
        "campaign_summer_mask.png",
        "model_east_asia_f01.png",
        "model_south_america_m01.png",
        "model_nordic_f01.png",
    }
    actual_names = {f.name for f in files}
    assert expected_names == actual_names

    for file_path in files:
        assert file_path.is_file()
        assert file_path.stat().st_size > 0
        with Image.open(file_path) as im:
            assert im.size == (128, 128)


def test_fullbody_campaign_assets() -> None:
    """Verify full-body campaign lookbooks have valid dimensions and formats without external mask requirements."""
    campaign_dir = Path("data/input_campaign")
    campaign_images = [
        "campaign_fashion_female.png",
        "campaign_fashion_male.png",
    ]
    for img_name in campaign_images:
        img_path = campaign_dir / img_name
        assert img_path.is_file()

        with Image.open(img_path) as im:
            assert im.size == (896, 1152)
            assert im.mode == "RGB"


def test_fullbody_model_assets() -> None:
    """Verify all diverse multi-ethnic target model assets are full-body 896x1152 images."""
    models_dir = Path("data/input_models")
    expected_models = [
        "model_east_asia_f01.png",
        "model_west_africa_m01.png",
        "model_nordic_f01.png",
        "model_south_america_m01.png",
    ]
    for m_name in expected_models:
        m_path = models_dir / m_name
        assert m_path.is_file()
        with Image.open(m_path) as im:
            assert im.size == (896, 1152)
            assert im.mode == "RGB"


def test_resize_and_save_utility(tmp_path: Path) -> None:
    """Verify resize_and_save standardizes any resolution cleanly to 896x1152."""
    from scripts.prepare_fullbody_dataset import resize_and_save

    src = tmp_path / "arbitrary.png"
    Image.new("RGB", (500, 700), color=(100, 150, 200)).save(src)
    dst = tmp_path / "standardized.png"

    resize_and_save(src, dst)
    assert dst.is_file()
    with Image.open(dst) as im:
        assert im.size == (896, 1152)


def test_prepare_dataset_full_flow(tmp_path: Path) -> None:
    """Verify process_dataset in scripts.prepare_fullbody_dataset correctly ingests source assets."""
    from scripts.prepare_fullbody_dataset import process_dataset

    assert process_dataset(tmp_path / "nonexistent", tmp_path / "out_data") == 0

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    out_data = tmp_path / "out_data"

    dummy_files = [
        "campaign_editorial_female_1.jpg",
        "campaign_editorial_male_2.png",
        "model_east_asia_f_3.jpg",
        "model_west_africa_m_4.png",
        "model_nordic_f_5.jpg",
        "model_south_america_m_6.png",
    ]
    for name in dummy_files:
        f = artifacts_dir / name
        Image.new("RGB", (400, 600), color=(128, 128, 128)).save(f)

    count = process_dataset(artifacts_dir, out_data)
    assert count == 6

    assert (out_data / "input_campaign" / "campaign_fashion_female.png").is_file()
    assert (out_data / "input_campaign" / "campaign_fashion_male.png").is_file()
    assert (out_data / "input_models" / "model_east_asia_f01.png").is_file()
    assert (out_data / "input_models" / "model_west_africa_m01.png").is_file()
    assert (out_data / "input_models" / "model_nordic_f01.png").is_file()
    assert (out_data / "input_models" / "model_south_america_m01.png").is_file()
