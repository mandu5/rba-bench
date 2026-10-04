"""Phase G result-intake helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from rl_research.experiments.merge import merge_seed_sweep_inputs
from rl_research.utils.logging import write_json


@dataclass(frozen=True)
class PhaseGEnvironment:
    key: str
    label: str
    slug: str
    base_seeds: tuple[int, ...]
    add_seeds: tuple[int, ...]
    base_glob: str


PHASE_G_ENVIRONMENTS = {
    "hopper": PhaseGEnvironment(
        key="hopper",
        label="Hopper-v5",
        slug="hopper_v5",
        base_seeds=(21, 22, 23, 24, 25),
        add_seeds=(26, 27, 28, 29, 30),
        base_glob="*hopper*v5*phase_a*_merged/*_rows.csv",
    ),
    "walker2d": PhaseGEnvironment(
        key="walker2d",
        label="Walker2d-v5",
        slug="walker2d_v5",
        base_seeds=(31, 32, 33, 34, 35),
        add_seeds=(36, 37, 38, 39, 40),
        base_glob="*walker2d*v5*phase_b*_merged/*_rows.csv",
    ),
    "halfcheetah": PhaseGEnvironment(
        key="halfcheetah",
        label="HalfCheetah-v5",
        slug="halfcheetah_v5",
        base_seeds=(41, 42, 43, 44, 45),
        add_seeds=(46, 47, 48, 49, 50),
        base_glob="*halfcheetah*v5*phase_c*_merged/*_rows.csv",
    ),
    "ant": PhaseGEnvironment(
        key="ant",
        label="Ant-v5",
        slug="ant_v5",
        base_seeds=(51, 52, 53, 54, 55),
        add_seeds=(56, 57, 58, 59, 60),
        base_glob="*ant*v5*phase_c*_merged/*_rows.csv",
    ),
}

ENV_ALIASES = {
    "hopper-v5": "hopper",
    "walker": "walker2d",
    "walker2d-v5": "walker2d",
    "half": "halfcheetah",
    "halfcheetah-v5": "halfcheetah",
    "ant-v5": "ant",
}


def intake_phase_g_result(
    *,
    env_key: str,
    add_input: str | Path,
    base_input: str | Path | None = None,
    output_dir: str | Path = "results/merged",
    progress_path: str | Path = "docs/phase_g_progress.json",
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Merge existing 5-seed base evidence with a Phase G add-5 result bundle."""
    env = resolve_phase_g_environment(env_key)
    add_path = Path(add_input)
    if not add_path.exists():
        raise FileNotFoundError(add_path)
    base_path = resolve_base_input(base_input) if base_input is not None else find_default_base_rows(env)
    if not base_path.exists():
        raise FileNotFoundError(base_path)

    merge_name = f"{env.slug}_phase_g_10seed_merged"
    merged = merge_seed_sweep_inputs(
        [base_path, add_path],
        output_dir=output_dir,
        name=merge_name,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )

    status = "merged_10seed" if len(merged["seeds"]) == 10 else "merged_partial"
    row = {
        "env_key": env.key,
        "env_label": env.label,
        "status": status,
        "base_input": str(base_path),
        "add_input": str(add_path),
        "light_bundle": str(add_path),
        "merge_dir": merged["merge_dir"],
        "json_path": merged["json_path"],
        "n_rows": int(merged["n_rows"]),
        "n_seeds": int(len(merged["seeds"])),
        "seeds": merged["seeds"],
        "base_seeds": list(env.base_seeds),
        "add_seeds": list(env.add_seeds),
        "expected_base_seeds": list(env.base_seeds),
        "expected_add_seeds": list(env.add_seeds),
    }
    update_phase_g_progress(progress_path, row)
    return {**row, "progress_path": str(progress_path)}


def resolve_phase_g_environment(env_key: str) -> PhaseGEnvironment:
    normalized = env_key.strip().lower().replace("_", "-")
    key = ENV_ALIASES.get(normalized, normalized)
    if key not in PHASE_G_ENVIRONMENTS:
        allowed = ", ".join(sorted(PHASE_G_ENVIRONMENTS))
        raise ValueError(f"Unknown Phase G environment '{env_key}'. Allowed: {allowed}")
    return PHASE_G_ENVIRONMENTS[key]


def find_default_base_rows(env: PhaseGEnvironment, *, merged_root: str | Path = "results/merged") -> Path:
    root = Path(merged_root)
    matches = sorted(root.glob(env.base_glob))
    matches = [path for path in matches if "phase_g" not in str(path).lower()]
    if not matches:
        raise FileNotFoundError(f"No default 5-seed base rows found for {env.label} under {root}")
    return matches[-1]


def resolve_base_input(base_input: str | Path) -> Path:
    """Resolve a base input path, accepting either a rows CSV, zip, or merged dir."""
    path = Path(base_input)
    if path.is_dir():
        matches = sorted(path.glob("*_rows.csv"))
        if not matches:
            raise FileNotFoundError(f"No merged rows CSV found under base directory: {path}")
        return matches[-1]
    return path


def update_phase_g_progress(progress_path: str | Path, row: dict[str, Any]) -> None:
    path = Path(progress_path)
    existing = _load_progress_rows(path)
    kept = [item for item in existing if item.get("env_key") != row["env_key"]]
    kept.append(row)
    ordering = {key: index for index, key in enumerate(PHASE_G_ENVIRONMENTS)}
    kept = sorted(kept, key=lambda item: ordering.get(str(item.get("env_key")), 999))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, kept)


def _load_progress_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        rows = payload.get("rows", [])
        if isinstance(rows, list):
            return [item for item in rows if isinstance(item, dict)]
    raise ValueError(f"Unsupported progress JSON shape: {path}")
