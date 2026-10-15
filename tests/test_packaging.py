"""Comprehensive test suite for release packaging module."""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from batchpersona.packaging import (
    REPO_ROOT,
    build_release_package,
    collect_distribution_files,
    compute_sha256,
    parse_args,
    resolve_default_version,
    should_ignore,
    validate_package,
)
from batchpersona.packaging import (
    main as pkg_main,
)


def test_compute_sha256(tmp_path: Path) -> None:
    """Verify SHA-256 hash calculation on known content."""
    test_file = tmp_path / "hello.txt"
    test_file.write_bytes(b"hello world\n")
    # sha256 of "hello world\n"
    expected = "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
    assert compute_sha256(test_file) == expected


def test_should_ignore() -> None:
    """Verify blacklist pattern matching."""
    assert should_ignore(Path("src/__pycache__/module.pyc")) is True
    assert should_ignore(Path(".pytest_cache/v/cache/nodeids")) is True
    assert should_ignore(Path(".ruff_cache/0.1/abc")) is True
    assert should_ignore(Path(".venv/lib/python.exe")) is True
    assert should_ignore(Path(".DS_Store")) is True
    assert should_ignore(Path("Thumbs.db")) is True
    assert should_ignore(Path("coverage.xml")) is True
    assert should_ignore(Path("temp.tmp")) is True
    assert should_ignore(Path("backup.bak")) is True

    # Legitimate paths should not be ignored
    assert should_ignore(Path("src/batchpersona/packaging.py")) is False
    assert should_ignore(Path("workflows/model_swap_api.json")) is False
    assert should_ignore(Path("install.bat")) is False
    assert should_ignore(Path("install.sh")) is False


def test_collect_distribution_files_from_real_repo() -> None:
    """Verify file collection from actual repository includes critical files."""
    collected = collect_distribution_files(REPO_ROOT)
    arcnames = {arc for _, arc in collected}

    assert "pyproject.toml" in arcnames
    assert "requirements.txt" in arcnames
    assert "install.bat" in arcnames
    assert "install.sh" in arcnames
    assert "run_pipeline.bat" in arcnames
    assert "run_pipeline.sh" in arcnames
    assert "src/batchpersona/__init__.py" in arcnames
    assert "src/batchpersona/packaging.py" in arcnames
    assert "scripts/package_release.py" in arcnames
    assert "workflows/model_swap_qwen21_maskless_api.json" in arcnames


def test_build_release_package_mock_repo(tmp_path: Path) -> None:
    """Verify build_release_package creates valid zip and checksum file."""
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()

    # Create mandatory root files
    for fname in [
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "LICENSE",
        ".env.example",
        "install.bat",
        "install.sh",
        "run_pipeline.bat",
        "run_pipeline.sh",
    ]:
        (fake_repo / fname).write_text(f"content of {fname}", encoding="utf-8")

    # Create directories
    (fake_repo / "src" / "batchpersona").mkdir(parents=True)
    (fake_repo / "src" / "batchpersona" / "__init__.py").write_text(
        "__version__ = '1.2.3'\n", encoding="utf-8"
    )
    (fake_repo / "scripts").mkdir()
    (fake_repo / "scripts" / "run_pipeline.py").write_text("print('run')", encoding="utf-8")
    (fake_repo / "workflows").mkdir()
    (fake_repo / "workflows" / "model_swap_qwen21_maskless_api.json").write_text(
        "{}", encoding="utf-8"
    )
    (fake_repo / "docs").mkdir()
    (fake_repo / "docs" / "ARCHITECTURE.md").write_text("# Docs", encoding="utf-8")
    (fake_repo / "tests").mkdir()
    (fake_repo / "tests" / "test_sample.py").write_text("def test_ok(): pass", encoding="utf-8")
    (fake_repo / "data" / "output").mkdir(parents=True)
    (fake_repo / "data" / "output" / ".gitkeep").write_text("", encoding="utf-8")

    out_dir = tmp_path / "dist"
    zip_path, sha_path = build_release_package(
        version="1.2.3",
        repo_root=fake_repo,
        output_dir=out_dir,
    )

    assert zip_path.is_file()
    assert sha_path.is_file()
    assert zip_path.name == "BatchPersona-v1.2.3.zip"

    # Check sha file format
    sha_text = sha_path.read_text(encoding="utf-8")
    assert "BatchPersona-v1.2.3.zip" in sha_text
    assert compute_sha256(zip_path) in sha_text

    # Verify zip content and permissions
    with zipfile.ZipFile(zip_path, mode="r") as zf:
        namelist = zf.namelist()
        assert "BatchPersona-v1.2.3/install.sh" in namelist
        assert "BatchPersona-v1.2.3/run_pipeline.sh" in namelist
        assert "BatchPersona-v1.2.3/src/batchpersona/__init__.py" in namelist
        assert "BatchPersona-v1.2.3/data/input_campaign/" in namelist
        assert "BatchPersona-v1.2.3/data/input_models/" in namelist
        assert "BatchPersona-v1.2.3/data/output/" in namelist

        # Verify executable permissions for shell scripts
        sh_info = zf.getinfo("BatchPersona-v1.2.3/install.sh")
        mode = (sh_info.external_attr >> 16) & 0o777
        assert mode == 0o755, f"Expected 0o755 permissions for install.sh, got {oct(mode)}"

        # Verify regular permissions for non-scripts
        py_info = zf.getinfo("BatchPersona-v1.2.3/src/batchpersona/__init__.py")
        py_mode = (py_info.external_attr >> 16) & 0o777
        assert py_mode == 0o644, f"Expected 0o644 permissions for __init__.py, got {oct(py_mode)}"


def test_validate_package_errors(tmp_path: Path) -> None:
    """Verify validate_package flags non-existent and corrupt files."""
    nonexistent = tmp_path / "does_not_exist.zip"
    errs = validate_package(nonexistent, "prefix")
    assert any("does not exist" in e for e in errs)

    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a valid zip file")
    errs_corrupt = validate_package(corrupt, "prefix")
    assert any("Failed to inspect zip file" in e for e in errs_corrupt)


def test_validate_package_missing_mandatory_file(tmp_path: Path) -> None:
    """Verify validate_package catches missing required entries."""
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("prefix/dummy.txt", "content")

    errs = validate_package(bad_zip, "prefix")
    assert any("Mandatory entry missing" in e for e in errs)


def test_validate_package_catches_blacklisted_entries(tmp_path: Path) -> None:
    """Verify validate_package flags blacklisted files."""
    bad_zip = tmp_path / "bad_cache.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("prefix/install.bat", "content")
        zf.writestr("prefix/install.sh", "content")
        zf.writestr("prefix/pyproject.toml", "content")
        zf.writestr("prefix/requirements.txt", "content")
        zf.writestr("prefix/run_pipeline.bat", "content")
        zf.writestr("prefix/run_pipeline.sh", "content")
        zf.writestr("prefix/src/batchpersona/__init__.py", "content")
        zf.writestr("prefix/workflows/model_swap_qwen21_maskless_api.json", "content")
        zf.writestr("prefix/src/batchpersona/__pycache__/bad.pyc", "bad")

    errs = validate_package(bad_zip, "prefix")
    assert any("Blacklisted transient file found" in e for e in errs)


def test_build_release_package_raises_on_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify build_release_package raises RuntimeError if validation fails."""
    monkeypatch.setattr(
        "batchpersona.packaging.validate_package",
        lambda *args, **kwargs: ["Mock validation failure"],
    )
    with pytest.raises(RuntimeError, match="Package validation failed"):
        build_release_package(
            version="1.0.0",
            repo_root=REPO_ROOT,
            output_dir=tmp_path / "dist",
        )


def test_parse_args() -> None:
    """Verify CLI argument parsing."""
    args = parse_args(["--version", "v2.0.0", "--verbose", "--output-dir", "custom_dist"])
    assert args.version == "v2.0.0"
    assert args.verbose is True
    assert args.output_dir == Path("custom_dist")


def test_resolve_default_version(tmp_path: Path) -> None:
    """Verify version resolution from pyproject.toml."""
    fake_root = tmp_path / "proj"
    fake_root.mkdir()
    pyproject = fake_root / "pyproject.toml"
    pyproject.write_text('version = "3.4.5"\n', encoding="utf-8")
    assert resolve_default_version(fake_root) == "3.4.5"

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    assert resolve_default_version(empty_root) == "1.0.0"


def test_pkg_main_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify pkg_main CLI runner on success."""
    mock_build = MagicMock(return_value=(tmp_path / "test.zip", tmp_path / "SHA256SUMS.txt"))
    monkeypatch.setattr("batchpersona.packaging.build_release_package", mock_build)

    code = pkg_main(["--version", "1.0.0"])
    assert code == 0
    assert mock_build.called


def test_pkg_main_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify pkg_main CLI runner handles exceptions gracefully."""
    mock_build = MagicMock(side_effect=RuntimeError("Build error"))
    monkeypatch.setattr("batchpersona.packaging.build_release_package", mock_build)

    code = pkg_main(["--version", "1.0.0"])
    assert code == 1
