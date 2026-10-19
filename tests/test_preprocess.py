"""Unit tests for dataset preprocessing and standardization utility."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from batchpersona.preprocess import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    VALID_IMAGE_EXTENSIONS,
    main,
    parse_cli_args,
    process_dataset,
    resize_and_save,
    setup_logging,
)


def test_constants() -> None:
    """Verify target dimensions and valid image extension set."""
    assert TARGET_WIDTH == 896
    assert TARGET_HEIGHT == 1152
    assert ".png" in VALID_IMAGE_EXTENSIONS
    assert ".jpg" in VALID_IMAGE_EXTENSIONS
    assert ".jpeg" in VALID_IMAGE_EXTENSIONS
    assert ".webp" in VALID_IMAGE_EXTENSIONS


def test_resize_and_save_utility(tmp_path: Path) -> None:
    """Verify resize_and_save standardizes any resolution cleanly to 896x1152."""
    src = tmp_path / "arbitrary.png"
    Image.new("RGB", (500, 700), color=(100, 150, 200)).save(src)
    dst = tmp_path / "standardized.png"

    resize_and_save(src, dst)
    assert dst.is_file()
    with Image.open(dst) as im:
        assert im.size == (TARGET_WIDTH, TARGET_HEIGHT)
        assert im.mode == "RGB"


def test_resize_and_save_colocation_skip(tmp_path: Path) -> None:
    """Verify resize_and_save is safe and skips re-encoding when src and dst are identical."""
    target = tmp_path / "already_ready.png"
    Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), color=(99, 99, 99)).save(target, format="PNG")
    mtime_before = target.stat().st_mtime_ns

    # Call with identical src and dst
    resize_and_save(target, target)
    assert target.is_file()
    assert target.stat().st_mtime_ns == mtime_before
    with Image.open(target) as im:
        assert im.size == (TARGET_WIDTH, TARGET_HEIGHT)


def test_resize_and_save_not_found(tmp_path: Path) -> None:
    """Verify resize_and_save raises FileNotFoundError for missing images."""
    with pytest.raises(FileNotFoundError):
        resize_and_save(tmp_path / "non_existent.png", tmp_path / "dst.png")


def test_prepare_dataset_full_flow(tmp_path: Path) -> None:
    """Verify process_dataset correctly ingests source assets generically without hardcoded name mappings."""
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

    # Names are preserved directly from stems
    assert (out_data / "input_campaign" / "campaign_editorial_female_1.png").is_file()
    assert (out_data / "input_campaign" / "campaign_editorial_male_2.png").is_file()
    assert (out_data / "input_models" / "model_east_asia_f_3.png").is_file()
    assert (out_data / "input_models" / "model_west_africa_m_4.png").is_file()
    assert (out_data / "input_models" / "model_nordic_f_5.png").is_file()
    assert (out_data / "input_models" / "model_south_america_m_6.png").is_file()


def test_process_dataset_arbitrary_filenames(tmp_path: Path) -> None:
    """Verify arbitrary user filenames are standardized to 896x1152 and routed to correct directories."""
    source_dir = tmp_path / "user_photos"
    source_dir.mkdir()
    out_dir = tmp_path / "standardized_data"

    files = [
        ("campaign_winter.jpg", (600, 900)),
        ("custom_model_01.jpg", (450, 750)),
        ("lookbook_summer_resort.PNG", (700, 1000)),
        ("persona_fashion_latam.webp", (500, 600)),
    ]
    for fname, size in files:
        fpath = source_dir / fname
        Image.new("RGB", size, color=(120, 140, 160)).save(fpath)

    count = process_dataset(source_dir, out_dir)
    assert count == 4

    expected_campaign = [
        out_dir / "input_campaign" / "campaign_winter.png",
        out_dir / "input_campaign" / "lookbook_summer_resort.png",
    ]
    expected_models = [
        out_dir / "input_models" / "custom_model_01.png",
        out_dir / "input_models" / "persona_fashion_latam.png",
    ]

    for c_path in expected_campaign:
        assert c_path.is_file()
        with Image.open(c_path) as im:
            assert im.size == (TARGET_WIDTH, TARGET_HEIGHT)
            assert im.mode == "RGB"

    for m_path in expected_models:
        assert m_path.is_file()
        with Image.open(m_path) as im:
            assert im.size == (TARGET_WIDTH, TARGET_HEIGHT)
            assert im.mode == "RGB"


def test_process_dataset_nested_subdirectories(tmp_path: Path) -> None:
    """Verify images inside input_campaign/ and input_models/ subdirectories are ingested."""
    source_dir = tmp_path / "structured_dataset"
    sub_campaign = source_dir / "input_campaign"
    sub_models = source_dir / "input_models"
    sub_campaign.mkdir(parents=True)
    sub_models.mkdir(parents=True)

    Image.new("RGB", (400, 500), color=(20, 30, 40)).save(sub_campaign / "lookbook_haute.jpg")
    Image.new("RGB", (300, 400), color=(50, 60, 70)).save(sub_models / "model_runway_01.png")

    out_dir = tmp_path / "out_structured"
    count = process_dataset(source_dir, out_dir)
    assert count == 2

    assert (out_dir / "input_campaign" / "lookbook_haute.png").is_file()
    assert (out_dir / "input_models" / "model_runway_01.png").is_file()
    with Image.open(out_dir / "input_campaign" / "lookbook_haute.png") as im:
        assert im.size == (TARGET_WIDTH, TARGET_HEIGHT)


def test_process_dataset_explicit_sources(tmp_path: Path) -> None:
    """Verify explicit campaign_src and models_src parameters function for both single files and directories."""
    c_file = tmp_path / "single_campaign.jpg"
    Image.new("RGB", (800, 1000), color=(10, 20, 30)).save(c_file)

    m_dir = tmp_path / "models_folder"
    m_dir.mkdir()
    Image.new("RGB", (500, 600), color=(40, 50, 60)).save(m_dir / "custom_model_alpha.jpg")
    Image.new("RGB", (500, 600), color=(70, 80, 90)).save(m_dir / "custom_model_beta.png")

    out_dir = tmp_path / "out_explicit"
    count = process_dataset(
        tmp_path / "unused_artifacts",
        out_dir,
        campaign_src=c_file,
        models_src=m_dir,
    )
    assert count == 3

    assert (out_dir / "input_campaign" / "single_campaign.png").is_file()
    assert (out_dir / "input_models" / "custom_model_alpha.png").is_file()
    assert (out_dir / "input_models" / "custom_model_beta.png").is_file()

    # Verify single model file and directory campaign source
    c_dir = tmp_path / "campaign_folder"
    c_dir.mkdir()
    Image.new("RGB", (600, 800), color=(1, 2, 3)).save(c_dir / "campaign_a.jpg")
    single_model = tmp_path / "single_model.png"
    Image.new("RGB", (400, 600), color=(4, 5, 6)).save(single_model)

    out_dir_2 = tmp_path / "out_explicit_2"
    count2 = process_dataset(
        tmp_path,
        out_dir_2,
        campaign_src=c_dir,
        models_src=single_model,
    )
    assert count2 == 2
    assert (out_dir_2 / "input_campaign" / "campaign_a.png").is_file()
    assert (out_dir_2 / "input_models" / "single_model.png").is_file()


def test_process_dataset_non_image_files_ignored(tmp_path: Path) -> None:
    """Verify non-image files such as txt, json, ds_store are ignored cleanly."""
    source_dir = tmp_path / "mixed_files"
    source_dir.mkdir()
    (source_dir / "notes.txt").write_text("Hello world", encoding="utf-8")
    (source_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (source_dir / ".DS_Store").write_bytes(b"\x00\x01\x02")

    out_dir = tmp_path / "out_mixed"
    count = process_dataset(source_dir, out_dir)
    assert count == 0
    if (out_dir / "input_campaign").exists():
        assert list((out_dir / "input_campaign").iterdir()) == []
    if (out_dir / "input_models").exists():
        assert list((out_dir / "input_models").iterdir()) == []


def test_process_dataset_source_validation(tmp_path: Path) -> None:
    """Verify process_dataset warns and handles unsupported extensions or missing paths."""
    out_dir = tmp_path / "out_val"

    # Unsupported extensions
    bad_c = tmp_path / "campaign.txt"
    bad_c.write_text("not an image", encoding="utf-8")
    bad_m = tmp_path / "model.pdf"
    bad_m.write_bytes(b"%PDF-1.4")

    count = process_dataset(tmp_path, out_dir, campaign_src=bad_c, models_src=bad_m)
    assert count == 0

    # Non-existent paths
    count2 = process_dataset(
        tmp_path,
        out_dir,
        campaign_src=tmp_path / "missing_c",
        models_src=tmp_path / "missing_m",
    )
    assert count2 == 0


def test_setup_logging() -> None:
    """Verify setup_logging configures root logger without error."""
    setup_logging()


def test_preprocess_cli_and_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify parse_cli_args and main entrypoint function cleanly."""
    args = parse_cli_args(["--source-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")])
    assert args.source_dir == tmp_path
    assert args.output_dir == tmp_path / "out"

    # Test main exit code 1 when no files found
    monkeypatch.setattr(
        "sys.argv",
        [
            "batchpersona-preprocess",
            "--source-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1

    # Now create a file and test exit code 0
    Image.new("RGB", (200, 200), color=(1, 2, 3)).save(tmp_path / "campaign_test.png")
    with pytest.raises(SystemExit) as exc_info2:
        main()
    assert exc_info2.value.code == 0
