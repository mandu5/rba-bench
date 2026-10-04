"""Export declared training settings and audit them against tracked run counts.

Reads only the five paper configurations and their merged per-seed CSVs. It
does not assert bitwise reproduction of training or replace raw-run receipts.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path

from rl_research.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "hopper": "hopper_v5_v21_main.yaml", "walker2d": "walker2d_v5_v21_main.yaml",
    "ant": "ant_v5_v21_main.yaml", "halfcheetah": "halfcheetah_v5_v21_main.yaml",
    "pcgym": "pcgym_case_study.yaml",
}


def build_training_setup() -> dict:
    envs = {}
    for env, filename in CONFIGS.items():
        path = ROOT / "configs" / "train" / filename
        config = load_experiment_config(path)
        pattern = "*pcgym*/*_rows.csv" if env == "pcgym" else f"*_{env}_v5_phase_g_10seed*/*_rows.csv"
        sources = list((ROOT / "results" / "merged").glob(pattern))
        if len(sources) != 1:
            raise ValueError(f"Expected one merged dataset for {env}, found {sources}")
        with sources[0].open() as handle:
            rows = list(csv.DictReader(handle))
        seeds = sorted({int(r["seed"]) for r in rows})
        policies = sorted({r["adaptation_policy"] for r in rows})
        if len(rows) != len(seeds) * len(policies):
            raise ValueError(f"Incomplete policy/seed grid for {env}")
        if len({(r["adaptation_policy"], int(r["seed"])) for r in rows}) != len(rows):
            raise ValueError(f"Duplicate policy/seed observation for {env}")
        adaptive = [r for r in rows if r["adaptation_policy"] != "Frozen"]
        updates = {int(r["update_cost"]) for r in adaptive}
        samples = {int(r["replay_samples"]) for r in adaptive}
        deploy_steps = config.train.total_steps - config.train.pretrain_steps
        expected_updates = deploy_steps // config.budget.window_steps * config.budget.max_updates_per_window
        if updates != {expected_updates} or samples != {expected_updates * config.model.batch_size}:
            raise ValueError(f"Observed budget disagrees with declared config for {env}")
        envs[env] = {
            "config_path": str(path.relative_to(ROOT)),
            "csv_path": str(sources[0].relative_to(ROOT)), "row_count": len(rows),
            "seeds": seeds, "policies": policies, "deploy_steps": deploy_steps,
            "observed_updates": next(iter(updates)), "observed_replay_samples": next(iter(samples)),
            **{key: asdict(getattr(config, key)) for key in ("train", "model", "budget", "eval", "drift")},
        }
    return {"scope": "Declared source configs and observed merged-run seed/budget audit; no new training.",
            "envs": envs}


def main() -> None:
    output = ROOT / "results" / "training_setup.json"
    output.write_text(json.dumps(build_training_setup(), indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
