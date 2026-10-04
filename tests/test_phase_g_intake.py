from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from rl_research.experiments.phase_g import intake_phase_g_result
from rl_research.experiments.phase_g import resolve_base_input


def _rows_for_seeds(seeds: list[int], env_id: str = "Hopper-v5") -> pd.DataFrame:
    rows = []
    for seed in seeds:
        frozen_return = 100.0 + seed
        for policy, offset, updates in [
            ("Frozen", 0.0, 0),
            ("Uniform", 8.0, 100),
            ("PER", 4.0, 100),
        ]:
            rows.append(
                {
                    "run_name": f"{env_id}_seed{seed}_{policy}",
                    "adaptation_policy": policy,
                    "env_id": env_id,
                    "seed": seed,
                    "return_mean": frozen_return + offset,
                    "recovery_auc": 0.7 + offset / 100.0,
                    "forgetting": 10.0 - offset / 2.0,
                    "update_cost": updates,
                    "replay_samples": updates * 4,
                    "wallclock_update_time": updates / 10.0,
                    "improvement_over_frozen": offset,
                    "improvement_per_1k_updates": None if updates == 0 else offset / updates * 1000.0,
                    "normalized_budgeted_score": 0.0,
                    "reward_ctrl_mean": None,
                    "run_dir": f"results/sweeps/{env_id}_seed{seed}_{policy}",
                }
            )
    return pd.DataFrame(rows)


def test_phase_g_intake_merges_base_rows_and_add5_zip(tmp_path: Path) -> None:
    base_rows = tmp_path / "hopper_base_rows.csv"
    _rows_for_seeds([21, 22, 23, 24, 25]).to_csv(base_rows, index=False)

    add_zip = tmp_path / "phaseg_hopper_add5_light_bundle.zip"
    add_rows = _rows_for_seeds([26, 27, 28, 29, 30]).to_csv(index=False)
    with ZipFile(add_zip, "w") as archive:
        archive.writestr(
            "hopper_v5_v21_main_seed_sweep_rows.csv",
            add_rows,
        )

    output = intake_phase_g_result(
        env_key="hopper",
        add_input=add_zip,
        base_input=base_rows,
        output_dir=tmp_path / "merged",
        progress_path=tmp_path / "phase_g_progress.json",
        bootstrap_resamples=50,
    )

    assert output["env_key"] == "hopper"
    assert output["n_rows"] == 30
    assert output["seeds"] == [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    assert output["status"] == "merged_10seed"
    assert Path(output["merge_dir"]).exists()
    assert Path(output["progress_path"]).exists()

    progress = pd.read_json(output["progress_path"])
    hopper = progress[progress["env_key"] == "hopper"].iloc[0]
    assert hopper["status"] == "merged_10seed"
    assert hopper["n_seeds"] == 10


def test_resolve_base_input_accepts_merged_directory(tmp_path: Path) -> None:
    merged_dir = tmp_path / "20260509T000000Z_hopper_v5_phase_a_merged"
    merged_dir.mkdir()
    rows_csv = merged_dir / "hopper_v5_phase_a_merged_rows.csv"
    rows_csv.write_text("seed,adaptation_policy\n21,Frozen\n", encoding="utf-8")

    assert resolve_base_input(merged_dir) == rows_csv
