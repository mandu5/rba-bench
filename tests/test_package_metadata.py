from __future__ import annotations

import tomllib
from pathlib import Path


def test_public_rba_bench_import_alias() -> None:
    import rba_bench

    assert rba_bench.__version__ == "0.1.0"


def test_package_metadata_exposes_rba_bench_cli() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "rba-bench"
    assert pyproject["project"]["description"].startswith("RBA-Bench")
    scripts = pyproject["project"]["scripts"]
    assert scripts["rba-bench"] == "rba_bench.cli:main"
    assert scripts["rl-research"] == "rl_research.cli:main"
