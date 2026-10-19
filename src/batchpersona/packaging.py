"""Deterministic release packaging tool for BatchPersona distribution archives."""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

LOGGER = logging.getLogger("batchpersona.packaging")

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent

# Core root files strictly required for installation and execution
MANDATORY_ROOT_FILES: Final[tuple[str, ...]] = (
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    "LICENSE",
    ".env.example",
    "install.bat",
    "install.sh",
    "run_pipeline.bat",
    "run_pipeline.sh",
)

# Directories included recursively
INCLUDED_DIRECTORIES: Final[tuple[str, ...]] = (
    "src",
    "scripts",
    "workflows",
    "docs",
    "tests",
)

# Standard ignore patterns to prevent dev cache and transient leaks
IGNORED_PATTERNS: Final[tuple[str, ...]] = (
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".coverage",
    "coverage.xml",
    ".DS_Store",
    "Thumbs.db",
    "*.pyc",
    "*.pyo",
    "*.tmp",
    "*.bak",
)


def compute_sha256(file_path: Path) -> str:
    """Compute cryptographic SHA-256 hash of a file with streaming chunks."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def should_ignore(path: Path) -> bool:
    """Check if a path matches any blacklist pattern."""
    for part in path.parts:
        if part in ("__pycache__", ".pytest_cache", ".ruff_cache", ".venv", ".ci_local_staging"):
            return True
        if part.endswith((".egg-info", ".tmp", ".bak")):
            return True

    name = path.name
    if name in (".DS_Store", "Thumbs.db", ".coverage", "coverage.xml"):
        return True
    return name.endswith((".pyc", ".pyo", ".tmp", ".bak"))


def collect_distribution_files(repo_root: Path) -> list[tuple[Path, str]]:
    """Collect all files to include in the release distribution archive.

    Returns a list of (source_absolute_path, archive_relative_path) tuples.
    """
    collected: list[tuple[Path, str]] = []

    # 1. Root files
    for filename in MANDATORY_ROOT_FILES:
        file_path = repo_root / filename
        if file_path.is_file():
            collected.append((file_path, filename))
        else:
            LOGGER.warning("Mandatory root file not found: %s", filename)

    # 2. Whitelisted directories
    for directory_name in INCLUDED_DIRECTORIES:
        dir_path = repo_root / directory_name
        if not dir_path.is_dir():
            LOGGER.warning("Included directory not found: %s", directory_name)
            continue

        for path in sorted(dir_path.rglob("*")):
            if not path.is_file() or should_ignore(path):
                continue
            rel_path = path.relative_to(repo_root).as_posix()
            collected.append((path, rel_path))

    # 3. Clean runtime dataset scaffold and sample assets
    data_dir = repo_root / "data"
    if data_dir.is_dir():
        for path in sorted(data_dir.rglob("*")):
            if not path.is_file() or should_ignore(path):
                continue
            rel_path = path.relative_to(repo_root).as_posix()
            # In data/output, only include .gitkeep so batch run outputs are not packaged
            if rel_path.startswith("data/output/") and path.name != ".gitkeep":
                continue
            collected.append((path, rel_path))

    return collected


def build_release_package(
    version: str,
    repo_root: Path = REPO_ROOT,
    output_dir: Path | None = None,
    archive_prefix: str | None = None,
) -> tuple[Path, Path]:
    """Build a standalone zip distribution archive and companion SHA-256 checksum file.

    Returns (zip_path, sha256_path).
    """
    clean_version = version.lstrip("v")
    target_dir = output_dir or (repo_root / "dist")
    target_dir.mkdir(parents=True, exist_ok=True)

    prefix = archive_prefix or f"BatchPersona-v{clean_version}"
    zip_filename = f"{prefix}.zip"
    zip_path = target_dir / zip_filename
    sha256_path = target_dir / "SHA256SUMS.txt"

    LOGGER.info("Building release archive: %s", zip_path)
    files_to_pack = collect_distribution_files(repo_root)

    # POSIX permissions flags for external_attr:
    # 0o100755: regular file with executable bits (rwxr-xr-x)
    # 0o100644: regular file with read/write bits (rw-r--r--)
    executable_mode = 0o100755 << 16
    regular_mode = 0o100644 << 16

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Guarantee scaffold directories exist even if empty
        for scaffold_dir in (
            "data/input_campaign",
            "data/input_models",
            "data/output",
        ):
            dir_arcname = f"{prefix}/{scaffold_dir}/"
            dir_info = zipfile.ZipInfo(dir_arcname)
            dir_info.external_attr = 0o40755 << 16  # directory mode
            zf.writestr(dir_info, "")

        for src_path, arc_subpath in files_to_pack:
            arcname = f"{prefix}/{arc_subpath}"
            zinfo = zipfile.ZipInfo(arcname)
            mtime = time.gmtime(src_path.stat().st_mtime)[:6]
            zinfo.date_time = mtime

            # Set POSIX execution permission bits for shell scripts
            if src_path.suffix == ".sh" or src_path.name in ("install.sh", "run_pipeline.sh"):
                zinfo.external_attr = executable_mode
            else:
                zinfo.external_attr = regular_mode

            content = src_path.read_bytes()
            zf.writestr(zinfo, content)

    LOGGER.info("Verifying archive integrity for: %s", zip_path.name)
    validation_errors = validate_package(zip_path, prefix)
    if validation_errors:
        raise RuntimeError(
            f"Package validation failed for {zip_path.name}: {'; '.join(validation_errors)}"
        )

    # Compute and persist SHA-256 checksum
    checksum = compute_sha256(zip_path)
    sha256_content = f"{checksum}  {zip_filename}\n"
    sha256_path.write_text(sha256_content, encoding="utf-8")
    LOGGER.info("SHA-256: %s (%s)", checksum, zip_filename)

    return zip_path, sha256_path


def validate_package(zip_path: Path, prefix: str) -> list[str]:
    """Test archive integrity and assert presence of mandatory components."""
    errors: list[str] = []

    if not zip_path.is_file():
        return [f"Archive does not exist: {zip_path}"]

    try:
        with zipfile.ZipFile(zip_path, mode="r") as zf:
            corrupt = zf.testzip()
            if corrupt:
                errors.append(f"Archive testzip failed on entry: {corrupt}")

            entries = set(zf.namelist())

            # Verify mandatory root launchers and files
            for expected_file in (
                "pyproject.toml",
                "requirements.txt",
                "install.bat",
                "install.sh",
                "run_pipeline.bat",
                "run_pipeline.sh",
                "src/batchpersona/__init__.py",
                "workflows/model_swap_qwen21_maskless_api.json",
            ):
                expected_arcname = f"{prefix}/{expected_file}"
                if expected_arcname not in entries:
                    errors.append(f"Mandatory entry missing: {expected_arcname}")

            # Verify that dangerous dev artifacts were not packaged
            for entry in entries:
                if any(bad in entry for bad in ("__pycache__", ".venv", ".pytest_cache")):
                    errors.append(f"Blacklisted transient file found in package: {entry}")

    except Exception as exc:
        errors.append(f"Failed to inspect zip file: {exc}")

    return errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build deterministic BatchPersona distribution package."
    )
    parser.add_argument(
        "--version",
        type=str,
        default="",
        help="Release version string (e.g. 1.0.0 or v1.0.0). Defaults to pyproject.toml version.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Target output directory for the archive. Defaults to 'dist/'.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root directory. Defaults to detected repo root.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser.parse_args(argv)


def resolve_default_version(repo_root: Path) -> str:
    """Read version from pyproject.toml as fallback."""
    pyproject_path = repo_root / "pyproject.toml"
    if pyproject_path.is_file():
        content = pyproject_path.read_text(encoding="utf-8")
        match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
    return "1.0.0"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for batchpersona packaging."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    version = args.version.strip() if args.version else resolve_default_version(args.repo_root)

    try:
        zip_path, sha_path = build_release_package(
            version=version,
            repo_root=args.repo_root,
            output_dir=args.output_dir,
        )
        print(f"[SUCCESS] Release package created: {zip_path}")
        print(f"[SUCCESS] Checksum manifest created: {sha_path}")
        return 0
    except Exception as exc:
        LOGGER.exception("Failed to build release package: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
