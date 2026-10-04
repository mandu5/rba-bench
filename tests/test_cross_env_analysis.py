from __future__ import annotations

from pathlib import Path

import pandas as pd

from rl_research.experiments.cross_env import compare_environments


def _write_merged_env(root: Path, name: str, rows: list[dict[str, object]]) -> None:
    root.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(root / f"{name}_robust_aggregate.csv", index=False)
    budget_rows = []
    for row in rows:
        policy = row["adaptation_policy"]
        updates = 0.0 if policy == "Frozen" else 100.0
        budget_rows.append(
            {
                "adaptation_policy": policy,
                "n_seeds": 5,
                "env_steps_deploy_iqm": 600.0,
                "update_cost_iqm": updates,
                "replay_samples_iqm": updates * 4,
                "wallclock_update_time_iqm": updates / 10.0,
                "budget_used_ratio_iqm": 0.0 if policy == "Frozen" else 1.0,
            }
        )
    pd.DataFrame(budget_rows).to_csv(root / f"{name}_budget_audit.csv", index=False)


def _aggregate_row(
    policy: str,
    *,
    ret: float,
    recovery: float,
    forgetting: float,
    improvement: float,
) -> dict[str, object]:
    return {
        "adaptation_policy": policy,
        "n_seeds": 5,
        "return_mean_iqm": ret,
        "return_mean_iqm_ci_low": ret - 1,
        "return_mean_iqm_ci_high": ret + 1,
        "recovery_auc_iqm": recovery,
        "recovery_auc_iqm_ci_low": recovery - 0.01,
        "recovery_auc_iqm_ci_high": recovery + 0.01,
        "forgetting_iqm": forgetting,
        "forgetting_iqm_ci_low": forgetting - 1,
        "forgetting_iqm_ci_high": forgetting + 1,
        "improvement_over_frozen_iqm": improvement,
        "improvement_over_frozen_iqm_ci_low": improvement - 1,
        "improvement_over_frozen_iqm_ci_high": improvement + 1,
    }


def test_compare_envs_normalizes_legacy_budgeted_pred_alias(tmp_path: Path) -> None:
    env_a = tmp_path / "env_a"
    env_b = tmp_path / "env_b"
    _write_merged_env(
        env_a,
        "env_a",
        [
            _aggregate_row("Budgeted-Pred", ret=50.0, recovery=0.9, forgetting=10.0, improvement=40.0),
            _aggregate_row("RecentOnly", ret=40.0, recovery=0.85, forgetting=12.0, improvement=30.0),
            _aggregate_row("Uniform", ret=30.0, recovery=0.82, forgetting=8.0, improvement=20.0),
            _aggregate_row("PER", ret=20.0, recovery=0.8, forgetting=5.0, improvement=10.0),
            _aggregate_row("Frozen", ret=10.0, recovery=0.7, forgetting=1.0, improvement=0.0),
        ],
    )
    _write_merged_env(
        env_b,
        "env_b",
        [
            _aggregate_row("PER", ret=50.0, recovery=0.9, forgetting=8.0, improvement=40.0),
            _aggregate_row("Budgeted-Novelty", ret=40.0, recovery=0.8, forgetting=4.0, improvement=30.0),
            _aggregate_row("RecentOnly", ret=30.0, recovery=0.78, forgetting=10.0, improvement=20.0),
            _aggregate_row("Uniform", ret=20.0, recovery=0.7, forgetting=6.0, improvement=10.0),
            _aggregate_row("Frozen", ret=5.0, recovery=0.6, forgetting=1.0, improvement=0.0),
        ],
    )

    output = compare_environments(
        [env_a, env_b],
        env_labels=["EnvA", "EnvB"],
        output_dir=tmp_path / "cross",
        name="unit_cross",
    )

    combined = pd.read_csv(output["combined_summary_path"])
    assert len(combined) == 10
    assert "Budgeted-Pred" not in set(combined["adaptation_policy"])
    assert "Budgeted-Novelty" in set(combined["adaptation_policy"])

    rank_matrix = pd.read_csv(output["policy_return_rank_matrix_path"])
    novelty = rank_matrix[rank_matrix["adaptation_policy"] == "Budgeted-Novelty"].iloc[0]
    assert novelty["EnvA"] == 1
    assert novelty["EnvB"] == 2
    assert novelty["return_bottom3_count"] == 0
    frozen = rank_matrix[rank_matrix["adaptation_policy"] == "Frozen"].iloc[0]
    assert frozen["return_bottom3_count"] == 2

    disagreement = pd.read_csv(output["env_ranking_disagreement_path"])
    assert set(disagreement["metric"]) == {
        "return_mean",
        "recovery_auc",
        "forgetting",
        "improvement_over_frozen",
    }
    assert Path(output["figure1_path"]).exists()
    assert Path(output["return_rank_heatmap_path"]).exists()

    budget = pd.read_csv(output["same_budget_audit_path"])
    non_frozen = budget[budget["adaptation_policy"] != "Frozen"]
    assert (non_frozen["budget_used_ratio_iqm"] == 1.0).all()
