from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from rl_research.experiments.merge import merge_seed_sweep_inputs


def _write_seed_zip(
    path: Path,
    seeds: list[int],
    *,
    novelty_name: str = "Budgeted-Novelty",
) -> None:
    rows = []
    for seed in seeds:
        frozen_return = 10.0 + seed
        for policy, offset, updates in [
            ("Frozen", 0.0, 0),
            ("RecentOnly", 5.0, 100),
            ("PER", 2.0, 100),
            (novelty_name, 3.0, 100),
        ]:
            run_dir = f"results/sweeps/test_seed_sweep/seed{seed}_{policy.lower()}"
            rows.append(
                {
                    "run_name": f"seed{seed}_{policy}",
                    "adaptation_policy": policy,
                    "env_id": "MockContinuousControl-v0",
                    "seed": seed,
                    "return_mean": frozen_return + offset,
                    "return_auc": frozen_return + offset,
                    "recovery_auc": 0.5 + offset / 100.0,
                    "steps_to_recovery": 1,
                    "update_cost": updates,
                    "replay_samples": updates * 4,
                    "wallclock_update_time": float(updates) / 10.0,
                    "forgetting": 10.0 - offset,
                    "reward_per_update": None if updates == 0 else (frozen_return + offset) / updates,
                    "improvement_over_frozen": offset,
                    "improvement_per_1k_updates": None if updates == 0 else offset / updates * 1000.0,
                    "normalized_budgeted_score": 0.0,
                    "normalized_budgeted_score_v2": None,
                    "reward_ctrl_mean": None,
                    "run_dir": run_dir,
                }
            )
    csv = pd.DataFrame(rows).to_csv(index=False)
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "results/sweeps/test_seed_sweep/test_seed_sweep_rows.csv",
            csv,
        )
        for row in rows:
            archive.writestr(
                f"{row['run_dir']}/config.snapshot.yaml",
                "train:\n  total_steps: 1000\n  pretrain_steps: 400\n",
            )


def test_merge_seed_sweeps_accepts_split_zips(tmp_path: Path) -> None:
    first = tmp_path / "seeds1_3.zip"
    second = tmp_path / "seeds4_5.zip"
    _write_seed_zip(first, [1, 2, 3])
    _write_seed_zip(second, [4, 5])

    output = merge_seed_sweep_inputs(
        [first, second],
        output_dir=tmp_path / "merged",
        name="unit_merge",
        bootstrap_resamples=50,
    )

    assert output["n_rows"] == 20
    assert output["seeds"] == [1, 2, 3, 4, 5]
    assert output["policies"] == ["Budgeted-Novelty", "Frozen", "PER", "RecentOnly"]
    assert Path(output["aggregate_path"]).exists()
    assert Path(output["budget_audit_path"]).exists()
    assert Path(output["figure1_path"]).exists()

    aggregate = pd.read_csv(output["aggregate_path"])
    recent = aggregate[aggregate["adaptation_policy"] == "RecentOnly"].iloc[0]
    assert recent["n_seeds"] == 5
    assert recent["return_mean_iqm"] > aggregate[aggregate["adaptation_policy"] == "PER"].iloc[0]["return_mean_iqm"]

    budget = pd.read_csv(output["budget_audit_path"])
    non_frozen = budget[budget["adaptation_policy"] != "Frozen"]
    assert (non_frozen["budget_used_ratio_mean"].round(6) == 1.0).all()


def test_merge_seed_sweeps_normalizes_legacy_budgeted_pred_alias(tmp_path: Path) -> None:
    legacy = tmp_path / "seeds1_2_legacy.zip"
    current = tmp_path / "seeds3_4_current.zip"
    _write_seed_zip(legacy, [1, 2], novelty_name="Budgeted-Pred")
    _write_seed_zip(current, [3, 4], novelty_name="Budgeted-Novelty")

    output = merge_seed_sweep_inputs(
        [legacy, current],
        output_dir=tmp_path / "merged",
        name="alias_merge",
        bootstrap_resamples=50,
    )

    assert output["n_rows"] == 16
    assert output["seeds"] == [1, 2, 3, 4]
    assert "Budgeted-Pred" not in output["policies"]
    assert "Budgeted-Novelty" in output["policies"]

    rows = pd.read_csv(output["rows_path"])
    assert rows["adaptation_policy"].nunique() == 4
    assert not rows.duplicated(subset=["seed", "adaptation_policy"]).any()
