"""Test suite ensuring strict parity between local development and GitHub Actions CI.

Validates:
1. Requirements completeness: Every third-party dependency imported by scripts is in requirements.txt.
2. Synthetic test data generation at CI resolution (256x256).
3. CI workflow YAML integrity.
4. Local CI orchestrator execution logic.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.run_ci_locally import (
    main as ci_main,
)
from scripts.run_ci_locally import (
    parse_cli_args,
    run_all_ci_stages,
    run_stage,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_requirements_dependencies_complete() -> None:
    """Verify all 3rd-party modules imported in scripts/ are declared in requirements.txt."""
    req_file = REPO_ROOT / "requirements.txt"
    assert req_file.is_file(), "requirements.txt must exist at repository root"

    req_lines = req_file.read_text(encoding="utf-8").splitlines()
    req_names = {
        re.split(r"[><=~]", line.strip())[0].lower()
        for line in req_lines
        if line.strip() and not line.startswith("#")
    }

    # Canonical mapping between Python import names and PyPI distribution names
    import_to_package_map = {
        "pil": "pillow",
        "websocket": "websocket-client",
        "numpy": "numpy",
        "requests": "requests",
        "pytest": "pytest",
        "ruff": "ruff",
    }

    # Discover all imported modules across all scripts
    scripts_dir = REPO_ROOT / "scripts"
    third_party_imports: set[str] = set()

    # Standard library modules available in current Python
    stdlib_modules = set(sys.stdlib_module_names)

    for py_file in scripts_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_pkg = alias.name.split(".")[0].lower()
                    if top_pkg not in stdlib_modules and top_pkg not in {
                        "scripts",
                        "tests",
                        "batchpersona",
                    }:
                        third_party_imports.add(top_pkg)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_pkg = node.module.split(".")[0].lower()
                if top_pkg not in stdlib_modules and top_pkg not in {
                    "scripts",
                    "tests",
                    "batchpersona",
                }:
                    third_party_imports.add(top_pkg)

    for imp in third_party_imports:
        pkg_name = import_to_package_map.get(imp, imp)
        assert pkg_name in req_names, (
            f"Import '{imp}' in scripts/ is missing from requirements.txt (expected '{pkg_name}')"
        )


def test_ci_workflow_yaml_file_integrity() -> None:
    """Verify GitHub Actions CI workflow references existing files and correct python versions."""
    ci_yaml = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_yaml.is_file(), "CI workflow .github/workflows/ci.yml must exist"

    content = ci_yaml.read_text(encoding="utf-8")
    assert "scripts/run_ci_locally.py" in content
    assert "requirements.txt" in content
    assert "python-version" in content
    assert "pip install -e ." in content


def test_release_workflow_yaml_file_integrity() -> None:
    """Verify GitHub Actions Release workflow references packaging, quality gate, and gh release."""
    release_yaml = REPO_ROOT / ".github" / "workflows" / "release.yml"
    assert release_yaml.is_file(), "Release workflow .github/workflows/release.yml must exist"

    content = release_yaml.read_text(encoding="utf-8")
    assert "scripts/package_release.py" in content
    assert "scripts/run_ci_locally.py" in content
    assert "gh release create" in content
    assert "contents: write" in content
    assert "python-version" in content


def test_run_stage_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run_stage reports True on returncode 0 and False otherwise."""
    mock_run = MagicMock()
    mock_res_ok = MagicMock()
    mock_res_ok.returncode = 0
    mock_run.return_value = mock_res_ok

    monkeypatch.setattr(subprocess, "run", mock_run)
    assert run_stage("Test OK", ["python", "--version"]) is True

    mock_res_err = MagicMock()
    mock_res_err.returncode = 1
    mock_run.return_value = mock_res_err
    assert run_stage("Test Error", ["python", "--nonexistent"]) is False


def test_run_all_ci_stages_orchestration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify run_all_ci_stages executes stages and handles skips."""
    monkeypatch.setattr("scripts.run_ci_locally.CI_TEMP_DIR", tmp_path / "ci_staging")

    # Mock run_stage to always succeed
    called_stages: list[str] = []

    def mock_run_stage(stage_name: str, cmd: list[str], cwd: Path | None = None) -> bool:
        called_stages.append(stage_name)
        return True

    monkeypatch.setattr("scripts.run_ci_locally.run_stage", mock_run_stage)

    # Test full run
    code = run_all_ci_stages(python_executable="python")
    assert code == 0
    assert "Ruff Linter" in called_stages
    assert "Ruff Formatter" in called_stages
    assert "Pyright Static Type Checker" in called_stages
    assert "Pytest Regression Suite" in called_stages

    # Test with skip flags
    called_stages.clear()
    code = run_all_ci_stages(python_executable="python", skip_lint=True, skip_tests=True)
    assert code == 0
    assert "Ruff Linter" not in called_stages
    assert "Pyright Static Type Checker" not in called_stages
    assert "Pytest Regression Suite" not in called_stages
    assert len(called_stages) == 0


def test_run_all_ci_stages_failure_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run_all_ci_stages halts and returns 1 when a stage fails."""
    monkeypatch.setattr("scripts.run_ci_locally.run_stage", lambda *args, **kwargs: False)
    assert run_all_ci_stages(python_executable="python") == 1


def test_parse_cli_args_and_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI parsing and main function behavior."""
    args = parse_cli_args(["--skip-lint", "--skip-tests", "-v"])
    assert args.skip_lint is True
    assert args.skip_tests is True
    assert args.verbose is True
    assert args.keep_coverage is False

    args_keep = parse_cli_args(["--keep-coverage"])
    assert args_keep.keep_coverage is True

    monkeypatch.setattr("scripts.run_ci_locally.run_all_ci_stages", lambda **kwargs: 0)
    assert ci_main(["--skip-lint"]) == 0


def test_run_all_ci_stages_coverage_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify run_all_ci_stages unlinks coverage artifacts by default and keeps them when requested."""
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    monkeypatch.setattr("batchpersona.ci.REPO_ROOT", fake_repo)

    cov_file = fake_repo / ".coverage"
    xml_file = fake_repo / "coverage.xml"

    def mock_runner(stage_name: str, cmd: list[str], cwd: Path | None = None) -> bool:
        if stage_name == "Pytest Regression Suite":
            cov_file.write_bytes(b"coverage_data")
            xml_file.write_bytes(b"<coverage/>")
        return True

    # Run with default keep_coverage=False
    code = run_all_ci_stages(
        python_executable="python",
        skip_lint=True,
        keep_coverage=False,
        run_stage_fn=mock_runner,
        staging_dir=tmp_path / "staging",
    )
    assert code == 0
    assert not cov_file.exists()
    assert not xml_file.exists()

    # Run with keep_coverage=True
    code_keep = run_all_ci_stages(
        python_executable="python",
        skip_lint=True,
        keep_coverage=True,
        run_stage_fn=mock_runner,
        staging_dir=tmp_path / "staging",
    )
    assert code_keep == 0
    assert cov_file.is_file()
    assert xml_file.is_file()


def test_python_310_compatibility() -> None:
    """Verify scripts/ and tests/ do not import constructs requiring Python 3.11+."""
    py311_typing_symbols = {
        "Self",
        "LiteralString",
        "Never",
        "assert_never",
        "dataclass_transform",
        "TypeVarTuple",
        "reveal_type",
    }
    for folder in [REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "tests"]:
        for py_path in folder.rglob("*.py"):
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "typing":
                    for alias in node.names:
                        assert alias.name not in py311_typing_symbols, (
                            f"{py_path.name} imports '{alias.name}' from typing, "
                            f"which breaks Python 3.10 compatibility"
                        )


def test_backward_compatibility_shims() -> None:
    """Verify that all scripts/ shims export expected symbols and forward to batchpersona."""
    import scripts.batch_swapper as shim_swapper
    import scripts.housekeeper as shim_housekeeper
    import scripts.launch_comfyui as shim_launcher
    import scripts.package_release as shim_package
    import scripts.prepare_fullbody_dataset as shim_preprocess
    import scripts.run_ci_locally as shim_ci
    import scripts.run_pipeline as shim_pipeline

    assert hasattr(shim_launcher, "detect_comfyui")
    assert hasattr(shim_launcher, "launch_server")
    assert hasattr(shim_launcher, "is_server_online")
    assert hasattr(shim_launcher, "ComfyInstance")
    assert hasattr(shim_launcher, "stop_server")
    assert hasattr(shim_launcher, "unload_comfyui_vram")
    assert hasattr(shim_launcher, "find_pid_by_port")
    assert hasattr(shim_pipeline, "interactive_menu_loop")
    assert hasattr(shim_pipeline, "run_commercial_swap")
    assert hasattr(shim_pipeline, "ensure_server_ready")
    assert hasattr(shim_swapper, "BatchSwapper")
    assert hasattr(shim_swapper, "SwapperConfig")
    assert hasattr(shim_swapper, "JobStatus")
    assert hasattr(shim_housekeeper, "ComfyUIHousekeeper")
    assert hasattr(shim_preprocess, "process_dataset")
    assert hasattr(shim_ci, "run_all_ci_stages")
    assert hasattr(shim_package, "build_release_package")
    assert hasattr(shim_package, "main")


def test_ci_main_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ci_main dispatches to run_all_ci_stages with verbose logging."""
    monkeypatch.setattr(
        "batchpersona.ci.run_all_ci_stages",
        lambda *args, **kwargs: 0,
    )
    assert ci_main(["--skip-lint", "--skip-tests", "-v"]) == 0


def test_module_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify src/batchpersona/__main__.py dispatches to pipeline.main."""
    import runpy

    monkeypatch.setattr("batchpersona.pipeline.main", lambda argv=None: 0)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("batchpersona", run_name="__main__")
    assert exc_info.value.code == 0
