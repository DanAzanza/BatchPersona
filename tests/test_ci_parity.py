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
from PIL import Image

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
                    if top_pkg not in stdlib_modules and top_pkg not in {"scripts", "tests"}:
                        third_party_imports.add(top_pkg)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_pkg = node.module.split(".")[0].lower()
                if top_pkg not in stdlib_modules and top_pkg not in {"scripts", "tests"}:
                    third_party_imports.add(top_pkg)

    for imp in third_party_imports:
        pkg_name = import_to_package_map.get(imp, imp)
        assert pkg_name in req_names, (
            f"Import '{imp}' in scripts/ is missing from requirements.txt (expected '{pkg_name}')"
        )


def test_ci_synthetic_data_generation_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify generate_testdata.py executes the exact CI step (--size 256) and produces valid assets."""
    from scripts.generate_testdata import main as gen_main

    ci_staging = tmp_path / "data_ci_test"
    test_args = [
        "generate_testdata.py",
        "--output-dir",
        str(ci_staging),
        "--size",
        "256",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    gen_main()

    expected_assets = [
        ci_staging / "input_campaign" / "campaign_summer_lookbook.png",
        ci_staging / "input_campaign" / "campaign_summer_mask.png",
        ci_staging / "input_models" / "model_east_asia_f01.png",
        ci_staging / "input_models" / "model_south_america_m01.png",
        ci_staging / "input_models" / "model_nordic_f01.png",
    ]
    for asset in expected_assets:
        assert asset.is_file(), f"Expected CI asset missing: {asset}"
        assert asset.stat().st_size > 0
        with Image.open(asset) as img:
            assert img.size == (256, 256), (
                f"Asset {asset.name} has size {img.size}, expected 256x256"
            )


def test_ci_workflow_yaml_file_integrity() -> None:
    """Verify GitHub Actions CI workflow references existing files and correct python versions."""
    ci_yaml = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_yaml.is_file(), "CI workflow .github/workflows/ci.yml must exist"

    content = ci_yaml.read_text(encoding="utf-8")
    assert "scripts/generate_testdata.py" in content
    assert "requirements.txt" in content
    assert "python-version" in content
    assert "ruff check ." in content
    assert "ruff format --check ." in content


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
        # Create dummy CI temp files when generator stage runs
        if stage_name == "CI Synthetic Data Generator":
            staging = tmp_path / "ci_staging"
            (staging / "input_campaign").mkdir(parents=True, exist_ok=True)
            (staging / "input_models").mkdir(parents=True, exist_ok=True)
            (staging / "input_campaign" / "campaign_summer_lookbook.png").write_bytes(b"data")
            (staging / "input_campaign" / "campaign_summer_mask.png").write_bytes(b"data")
            (staging / "input_models" / "model_east_asia_f01.png").write_bytes(b"data")
            (staging / "input_models" / "model_south_america_m01.png").write_bytes(b"data")
            (staging / "input_models" / "model_nordic_f01.png").write_bytes(b"data")
        return True

    monkeypatch.setattr("scripts.run_ci_locally.run_stage", mock_run_stage)

    # Test full run
    code = run_all_ci_stages(python_executable="python")
    assert code == 0
    assert "Ruff Linter" in called_stages
    assert "Ruff Formatter" in called_stages
    assert "Pyright Static Type Checker" in called_stages
    assert "CI Synthetic Data Generator" in called_stages
    assert "Pytest Regression Suite" in called_stages

    # Test with skip flags
    called_stages.clear()
    code = run_all_ci_stages(python_executable="python", skip_lint=True, skip_tests=True)
    assert code == 0
    assert "Ruff Linter" not in called_stages
    assert "Pyright Static Type Checker" not in called_stages
    assert "Pytest Regression Suite" not in called_stages
    assert "CI Synthetic Data Generator" in called_stages


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

    monkeypatch.setattr("scripts.run_ci_locally.run_all_ci_stages", lambda **kwargs: 0)
    assert ci_main(["--skip-lint"]) == 0


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
    for folder in [REPO_ROOT / "scripts", REPO_ROOT / "tests"]:
        for py_path in folder.glob("*.py"):
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "typing":
                    for alias in node.names:
                        assert alias.name not in py311_typing_symbols, (
                            f"{py_path.name} imports '{alias.name}' from typing, "
                            f"which breaks Python 3.10 compatibility"
                        )
