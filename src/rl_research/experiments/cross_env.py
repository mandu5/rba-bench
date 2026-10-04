"""Cross-environment analysis for merged seed-sweep summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rl_research.evaluation.robust_stats import kendall_tau, rank_names
from rl_research.experiments.merge import RANKING_METRICS
from rl_research.utils.logging import prepare_run_dir, write_csv, write_json


POLICY_ALIASES = {
    "Budgeted-Pred": "Budgeted-Novelty",
    "budgeted-pred": "Budgeted-Novelty",
}

RANK_COLUMN_NAMES = {
    "return_mean": "return_rank",
    "recovery_auc": "recovery_rank",
    "forgetting": "forgetting_rank",
    "improvement_over_frozen": "improvement_rank",
}


@dataclass
class MergedEnvInput:
    path: Path
    env_label: str
    aggregate_path: Path
    budget_audit_path: Path | None


def compare_environments(
    input_paths: list[str | Path],
    *,
    env_labels: list[str] | None = None,
    output_dir: str | Path = "results/cross_env",
    name: str = "cross_env_comparison",
) -> dict[str, Any]:
    """Build cross-environment ranking, budget, and disagreement summaries."""
    env_inputs = _resolve_env_inputs(input_paths, env_labels)
    combined = _load_combined_aggregates(env_inputs)
    combined = _attach_metric_ranks(combined)
    budget = _load_combined_budget_audits(env_inputs)
    disagreement_rows, disagreement_payload = _build_cross_env_disagreement(combined)
    rank_matrix = _build_return_rank_matrix(combined, [env.env_label for env in env_inputs])
    pattern_rows = _build_policy_pattern_rows(combined)

    run_dir = prepare_run_dir(output_dir, name)
    combined_path = run_dir / f"{name}_combined_summary.csv"
    rank_matrix_path = run_dir / f"{name}_policy_return_rank_matrix.csv"
    disagreement_path = run_dir / f"{name}_env_ranking_disagreement.csv"
    budget_path = run_dir / f"{name}_same_budget_audit.csv"
    patterns_path = run_dir / f"{name}_consistent_patterns.csv"
    figure1_path = run_dir / f"{name}_figure1_cross_env_ranking_disagreement.html"
    rank_heatmap_path = run_dir / f"{name}_return_rank_heatmap.html"
    json_path = run_dir / f"{name}_summary.json"

    combined.to_csv(combined_path, index=False)
    write_csv(rank_matrix_path, rank_matrix)
    write_csv(disagreement_path, disagreement_rows)
    write_csv(budget_path, budget.to_dict("records"))
    write_csv(patterns_path, pattern_rows)
    _write_cross_env_figure1(disagreement_rows, combined, figure1_path)
    _write_return_rank_heatmap(rank_matrix, [env.env_label for env in env_inputs], rank_heatmap_path)

    payload = {
        "name": name,
        "inputs": [str(env.path) for env in env_inputs],
        "env_labels": [env.env_label for env in env_inputs],
        "policies": sorted(combined["adaptation_policy"].unique().tolist()),
        "combined_summary_path": str(combined_path),
        "policy_return_rank_matrix_path": str(rank_matrix_path),
        "env_ranking_disagreement_path": str(disagreement_path),
        "same_budget_audit_path": str(budget_path),
        "consistent_patterns_path": str(patterns_path),
        "figure1_path": str(figure1_path),
        "return_rank_heatmap_path": str(rank_heatmap_path),
        "story_b_diagnostics": _build_story_b_diagnostics(disagreement_payload, pattern_rows),
    }
    write_json(json_path, payload)
    payload["json_path"] = str(json_path)
    return payload


def _resolve_env_inputs(
    input_paths: list[str | Path],
    env_labels: list[str] | None,
) -> list[MergedEnvInput]:
    if not input_paths:
        raise ValueError("At least one merged result directory or summary JSON is required.")
    if env_labels is not None and len(env_labels) != len(input_paths):
        raise ValueError("--env-labels must have the same length as --inputs.")

    resolved: list[MergedEnvInput] = []
    for index, raw_path in enumerate(input_paths):
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        root = path.parent if path.is_file() else path
        aggregate = _find_one(root, "*_robust_aggregate.csv")
        budget = _find_optional(root, "*_budget_audit.csv")
        label = env_labels[index] if env_labels is not None else _infer_env_label(root, aggregate)
        resolved.append(
            MergedEnvInput(
                path=path,
                env_label=label,
                aggregate_path=aggregate,
                budget_audit_path=budget,
            )
        )
    return resolved


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} found under {root}")
    if len(matches) > 1:
        raise ValueError(f"Multiple {pattern} files found under {root}: {matches}")
    return matches[0]


def _find_optional(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple {pattern} files found under {root}: {matches}")
    return matches[0]


def _infer_env_label(root: Path, aggregate_path: Path) -> str:
    summary_path = _find_optional(root, "*_summary.json")
    if summary_path is not None:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("aggregate_rows") or []
        if rows and rows[0].get("env_id"):
            return str(rows[0]["env_id"])
    return aggregate_path.stem.replace("_robust_aggregate", "")


def _normalize_policy_name(name: object) -> str:
    raw = str(name)
    return POLICY_ALIASES.get(raw, raw)


def _load_combined_aggregates(env_inputs: list[MergedEnvInput]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for env in env_inputs:
        frame = pd.read_csv(env.aggregate_path)
        if "adaptation_policy" not in frame.columns:
            raise ValueError(f"Missing adaptation_policy in {env.aggregate_path}")
        frame = frame.copy()
        frame["env_label"] = env.env_label
        frame["adaptation_policy"] = frame["adaptation_policy"].map(_normalize_policy_name)
        duplicate = frame.duplicated(subset=["env_label", "adaptation_policy"])
        if duplicate.any():
            dupes = frame.loc[duplicate, ["env_label", "adaptation_policy"]].to_dict("records")
            raise ValueError(f"Policy aliasing created duplicate rows: {dupes}")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["env_label", "adaptation_policy"]).reset_index(drop=True)


def _attach_metric_ranks(combined: pd.DataFrame) -> pd.DataFrame:
    combined = combined.copy()
    for rank_column in RANK_COLUMN_NAMES.values():
        combined[rank_column] = np.nan
    for env_label, group in combined.groupby("env_label", sort=False):
        rows = group.to_dict("records")
        for metric, higher_is_better in RANKING_METRICS.items():
            ranking = rank_names(rows, metric, higher_is_better=higher_is_better)
            rank_map = {policy: rank + 1 for rank, policy in enumerate(ranking)}
            column = RANK_COLUMN_NAMES[metric]
            mask = combined["env_label"] == env_label
            combined.loc[mask, column] = combined.loc[mask, "adaptation_policy"].map(rank_map)
    for rank_column in RANK_COLUMN_NAMES.values():
        combined[rank_column] = combined[rank_column].astype(int)
    return combined


def _load_combined_budget_audits(env_inputs: list[MergedEnvInput]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for env in env_inputs:
        if env.budget_audit_path is None:
            continue
        frame = pd.read_csv(env.budget_audit_path)
        if "adaptation_policy" not in frame.columns:
            raise ValueError(f"Missing adaptation_policy in {env.budget_audit_path}")
        frame = frame.copy()
        frame["env_label"] = env.env_label
        frame["adaptation_policy"] = frame["adaptation_policy"].map(_normalize_policy_name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["env_label", "adaptation_policy"])
    budget = pd.concat(frames, ignore_index=True)
    duplicate = budget.duplicated(subset=["env_label", "adaptation_policy"])
    if duplicate.any():
        dupes = budget.loc[duplicate, ["env_label", "adaptation_policy"]].to_dict("records")
        raise ValueError(f"Duplicate budget audit rows after aliasing: {dupes}")
    return budget.sort_values(["env_label", "adaptation_policy"]).reset_index(drop=True)


def _build_cross_env_disagreement(combined: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    for env_label, group in combined.groupby("env_label", sort=False):
        records = group.to_dict("records")
        reward_ranking = rank_names(records, "return_mean", higher_is_better=True)
        env_payload: dict[str, Any] = {"reward_only_iqm_ranking": reward_ranking, "comparisons": {}}
        for metric, higher_is_better in RANKING_METRICS.items():
            ranking = rank_names(records, metric, higher_is_better=higher_is_better)
            tau, concordant, discordant = kendall_tau(reward_ranking, ranking)
            record = {
                "env_label": env_label,
                "metric": metric,
                "higher_is_better": higher_is_better,
                "kendall_tau_vs_return_iqm": tau,
                "concordant_pairs": concordant,
                "discordant_pairs": discordant,
                "reward_only_ranking": " > ".join(reward_ranking),
                "metric_ranking": " > ".join(ranking),
            }
            rows_out.append(record)
            env_payload["comparisons"][metric] = record
        payload[env_label] = env_payload
    return rows_out, payload


def _build_return_rank_matrix(combined: pd.DataFrame, env_labels: list[str]) -> list[dict[str, Any]]:
    pivot = combined.pivot(index="adaptation_policy", columns="env_label", values="return_rank")
    n_policies = int(combined["adaptation_policy"].nunique())
    for env_label in env_labels:
        if env_label not in pivot.columns:
            pivot[env_label] = np.nan
    pivot = pivot[env_labels]
    records: list[dict[str, Any]] = []
    for policy, row in pivot.iterrows():
        values = pd.to_numeric(row, errors="coerce")
        record: dict[str, Any] = {"adaptation_policy": policy}
        for env_label in env_labels:
            value = values[env_label]
            record[env_label] = None if pd.isna(value) else int(value)
        record["mean_return_rank"] = float(values.mean())
        record["std_return_rank"] = float(values.std(ddof=0))
        record["best_return_rank"] = int(values.min())
        record["worst_return_rank"] = int(values.max())
        record["return_top3_count"] = int((values <= 3).sum())
        record["return_bottom3_count"] = int((values >= max(1, n_policies - 2)).sum())
        records.append(record)
    return sorted(records, key=lambda item: (item["mean_return_rank"], item["adaptation_policy"]))


def _build_policy_pattern_rows(combined: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    n_policies = int(combined["adaptation_policy"].nunique())
    for policy, group in combined.groupby("adaptation_policy", sort=True):
        return_ranks = group["return_rank"].astype(float)
        recovery_ranks = group["recovery_rank"].astype(float)
        forgetting_ranks = group["forgetting_rank"].astype(float)
        best_idx = return_ranks.idxmin()
        worst_idx = return_ranks.idxmax()
        output.append(
            {
                "adaptation_policy": str(policy),
                "n_envs": int(group["env_label"].nunique()),
                "mean_return_rank": float(return_ranks.mean()),
                "std_return_rank": float(return_ranks.std(ddof=0)),
                "best_return_rank": int(return_ranks.min()),
                "worst_return_rank": int(return_ranks.max()),
                "return_rank_range": int(return_ranks.max() - return_ranks.min()),
                "return_top3_count": int((return_ranks <= 3).sum()),
                "return_bottom3_count": int((return_ranks >= max(1, n_policies - 2)).sum()),
                "mean_recovery_rank": float(recovery_ranks.mean()),
                "mean_forgetting_rank": float(forgetting_ranks.mean()),
                "best_return_env": str(combined.loc[best_idx, "env_label"]),
                "worst_return_env": str(combined.loc[worst_idx, "env_label"]),
            }
        )
    return sorted(output, key=lambda item: (item["mean_return_rank"], item["adaptation_policy"]))


def _build_story_b_diagnostics(
    disagreement_payload: dict[str, Any],
    pattern_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tau_by_env = {
        env_label: payload["comparisons"]["forgetting"]["kendall_tau_vs_return_iqm"]
        for env_label, payload in disagreement_payload.items()
        if "forgetting" in payload.get("comparisons", {})
    }
    pattern_by_policy = {row["adaptation_policy"]: row for row in pattern_rows}
    return {
        "return_forgetting_tau_by_env": tau_by_env,
        "per_return_rank_summary": pattern_by_policy.get("PER"),
        "recencyweighted_return_rank_summary": pattern_by_policy.get("RecencyWeighted"),
        "recentonly_return_rank_summary": pattern_by_policy.get("RecentOnly"),
        "staticmix_return_rank_summary": pattern_by_policy.get("StaticMix"),
    }


def _write_cross_env_figure1(
    disagreement_rows: list[dict[str, Any]],
    combined: pd.DataFrame,
    output_path: Path,
) -> None:
    disagreement = pd.DataFrame(disagreement_rows)
    heatmap = disagreement[disagreement["metric"] != "return_mean"].copy()
    heatmap["kendall_tau_vs_return_iqm"] = pd.to_numeric(
        heatmap["kendall_tau_vs_return_iqm"], errors="coerce",
    )
    metrics = ["recovery_auc", "forgetting", "improvement_over_frozen"]
    envs = list(dict.fromkeys(heatmap["env_label"].tolist()))
    z = []
    text = []
    for env_label in envs:
        row = []
        text_row = []
        for metric in metrics:
            match = heatmap[(heatmap["env_label"] == env_label) & (heatmap["metric"] == metric)]
            value = float(match["kendall_tau_vs_return_iqm"].iloc[0]) if not match.empty else np.nan
            row.append(value)
            text_row.append("" if np.isnan(value) else f"{value:.3f}")
        z.append(row)
        text.append(text_row)

    shift_rows = []
    for _, row in combined.iterrows():
        shift_rows.append(
            {
                "env_label": row["env_label"],
                "adaptation_policy": row["adaptation_policy"],
                "abs_return_forgetting_rank_shift": abs(int(row["return_rank"]) - int(row["forgetting_rank"])),
            }
        )
    shift_df = pd.DataFrame(shift_rows)
    shift_summary = (
        shift_df.groupby("env_label", sort=False)["abs_return_forgetting_rank_shift"]
        .mean()
        .reset_index()
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Kendall tau vs return-only ranking",
            "Mean absolute return-forgetting rank shift",
        ),
        column_widths=[0.62, 0.38],
    )
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=metrics,
            y=envs,
            text=text,
            texttemplate="%{text}",
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="tau"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=shift_summary["env_label"],
            y=shift_summary["abs_return_forgetting_rank_shift"],
            name="rank shift",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="Figure 1 — Cross-Environment Reward-only vs Multi-axis Ranking Disagreement",
        xaxis_title="Alternative metric",
        yaxis_title="Environment",
        yaxis2_title="Mean absolute rank shift",
        showlegend=False,
    )
    fig.write_html(output_path)


def _write_return_rank_heatmap(
    rank_matrix: list[dict[str, Any]],
    env_labels: list[str],
    output_path: Path,
) -> None:
    policies = [row["adaptation_policy"] for row in rank_matrix]
    z = [[row.get(env_label) for env_label in env_labels] for row in rank_matrix]
    text = [[str(row.get(env_label, "")) for env_label in env_labels] for row in rank_matrix]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=env_labels,
            y=policies,
            text=text,
            texttemplate="%{text}",
            colorscale="Viridis_r",
            colorbar=dict(title="Return rank"),
        )
    )
    fig.update_layout(
        title="Return IQM Rank Matrix Across Environments (1 = best)",
        xaxis_title="Environment",
        yaxis_title="Policy",
    )
    fig.write_html(output_path)
