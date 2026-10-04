"""Merge split seed-sweep outputs and build paper-grade summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile
import io
import json

import pandas as pd
import plotly.graph_objects as go
import yaml

from rl_research.evaluation.robust_stats import (
    bootstrap_ci,
    iqm,
    kendall_tau,
    mean,
    rank_names,
    sample_std,
)
from rl_research.utils.logging import prepare_run_dir, write_csv, write_json


PRIMARY_METRICS = [
    "return_mean",
    "recovery_auc",
    "forgetting",
    "update_cost",
    "replay_samples",
    "wallclock_update_time",
    "improvement_over_frozen",
    "improvement_per_1k_updates",
    "normalized_budgeted_score",
    "normalized_budgeted_score_v2",
    "reward_ctrl_mean",
]

RANKING_METRICS = {
    "return_mean": True,
    "recovery_auc": True,
    "forgetting": False,
    "improvement_over_frozen": True,
}

POLICY_ALIASES = {
    "Budgeted-Pred": "Budgeted-Novelty",
    "budgeted-pred": "Budgeted-Novelty",
}


@dataclass
class SourceBundle:
    path: Path
    kind: str  # "zip" | "dir" | "csv"


def merge_seed_sweep_inputs(
    input_paths: list[str | Path],
    *,
    output_dir: str | Path = "results/merged",
    name: str = "merged_seed_sweep",
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Merge split seed-sweep outputs and write robust summaries."""
    sources = [_classify_source(Path(p)) for p in input_paths]
    rows = _load_all_rows(sources)
    if rows.empty:
        raise ValueError("No seed_sweep_rows.csv files found in inputs.")
    rows = _normalize_rows(rows)
    rows = _attach_missing_normalized_score_v2(rows)
    _validate_complete_grid(rows)

    aggregate_rows = build_robust_aggregate(
        rows,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    ranking_rows, ranking_payload = build_ranking_disagreement(aggregate_rows)
    budget_rows = build_budget_audit(rows, sources)

    run_dir = prepare_run_dir(output_dir, name)
    rows_path = run_dir / f"{name}_rows.csv"
    aggregate_path = run_dir / f"{name}_robust_aggregate.csv"
    ranking_path = run_dir / f"{name}_ranking_disagreement.csv"
    budget_path = run_dir / f"{name}_budget_audit.csv"
    json_path = run_dir / f"{name}_summary.json"
    figure1_path = run_dir / f"{name}_figure1_ranking_disagreement.html"

    rows.to_csv(rows_path, index=False)
    write_csv(aggregate_path, aggregate_rows)
    write_csv(ranking_path, ranking_rows)
    write_csv(budget_path, budget_rows)
    _write_figure1(aggregate_rows, ranking_payload, figure1_path)
    write_json(
        json_path,
        {
            "name": name,
            "inputs": [str(source.path) for source in sources],
            "n_rows": int(len(rows)),
            "seeds": sorted(int(seed) for seed in rows["seed"].unique()),
            "policies": sorted(str(policy) for policy in rows["adaptation_policy"].unique()),
            "rows_path": str(rows_path),
            "aggregate_path": str(aggregate_path),
            "ranking_disagreement_path": str(ranking_path),
            "budget_audit_path": str(budget_path),
            "figure1_path": str(figure1_path),
            "ranking_disagreement": ranking_payload,
            "aggregate_rows": aggregate_rows,
            "budget_audit_rows": budget_rows,
        },
    )
    return {
        "merge_dir": str(run_dir),
        "rows_path": str(rows_path),
        "aggregate_path": str(aggregate_path),
        "ranking_disagreement_path": str(ranking_path),
        "budget_audit_path": str(budget_path),
        "figure1_path": str(figure1_path),
        "json_path": str(json_path),
        "n_rows": int(len(rows)),
        "seeds": sorted(int(seed) for seed in rows["seed"].unique()),
        "policies": sorted(str(policy) for policy in rows["adaptation_policy"].unique()),
        "aggregate_rows": aggregate_rows,
        "ranking_disagreement": ranking_payload,
    }


def build_robust_aggregate(
    rows: pd.DataFrame,
    *,
    bootstrap_resamples: int = 5000,
    bootstrap_seed: int = 0,
) -> list[dict[str, Any]]:
    """Aggregate seed rows with mean/std/IQM/bootstrap-CI per policy."""
    output: list[dict[str, Any]] = []
    for policy, group in rows.groupby("adaptation_policy", sort=True):
        row: dict[str, Any] = {
            "adaptation_policy": str(policy),
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in PRIMARY_METRICS:
            if metric not in group.columns:
                continue
            values = group[metric].dropna().tolist()
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = sample_std(values)
            row[f"{metric}_iqm"] = iqm(values)
            low, high = bootstrap_ci(
                values,
                statistic=iqm,
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            )
            row[f"{metric}_iqm_ci_low"] = low
            row[f"{metric}_iqm_ci_high"] = high
        output.append(row)
    return output


def build_ranking_disagreement(
    aggregate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build reward-only vs alternative metric ranking disagreement summary."""
    reward_ranking = rank_names(aggregate_rows, "return_mean", higher_is_better=True)
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"reward_only_iqm_ranking": reward_ranking, "comparisons": {}}
    for metric, higher_is_better in RANKING_METRICS.items():
        ranking = rank_names(aggregate_rows, metric, higher_is_better=higher_is_better)
        tau, concordant, discordant = kendall_tau(reward_ranking, ranking)
        record = {
            "metric": metric,
            "higher_is_better": higher_is_better,
            "kendall_tau_vs_return_iqm": tau,
            "concordant_pairs": concordant,
            "discordant_pairs": discordant,
            "ranking": " > ".join(ranking),
        }
        rows.append(record)
        payload["comparisons"][metric] = record
    return rows, payload


def build_budget_audit(rows: pd.DataFrame, sources: list[SourceBundle]) -> list[dict[str, Any]]:
    """Build same-budget audit rows from merged seed data and config snapshots."""
    enriched = rows.copy()
    deploy_steps: list[int | None] = []
    for _, row in enriched.iterrows():
        cfg = _load_config_snapshot(sources, str(row.get("run_dir", "")))
        if not cfg:
            cfg = _load_local_config_for_env(str(row.get("env_id", "")))
        if cfg:
            train = cfg.get("train", {})
            total = int(train.get("total_steps", 0) or 0)
            pretrain = int(train.get("pretrain_steps", 0) or 0)
            deploy_steps.append(max(total - pretrain, 0))
        else:
            deploy_steps.append(None)
    enriched["env_steps_deploy"] = deploy_steps
    max_budget = float(enriched["update_cost"].fillna(0.0).max())
    if max_budget <= 0:
        max_budget = 1.0
    enriched["budget_used_ratio"] = enriched["update_cost"].fillna(0.0).astype(float) / max_budget

    metrics = [
        "env_steps_deploy",
        "update_cost",
        "replay_samples",
        "wallclock_update_time",
        "budget_used_ratio",
    ]
    output: list[dict[str, Any]] = []
    for policy, group in enriched.groupby("adaptation_policy", sort=True):
        record: dict[str, Any] = {
            "adaptation_policy": str(policy),
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            values = group[metric].dropna().tolist()
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_std"] = sample_std(values)
            record[f"{metric}_iqm"] = iqm(values)
        output.append(record)
    return output


def _classify_source(path: Path) -> SourceBundle:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        return SourceBundle(path=path, kind="zip")
    if path.is_file() and path.suffix.lower() == ".csv":
        return SourceBundle(path=path, kind="csv")
    if path.is_dir():
        return SourceBundle(path=path, kind="dir")
    raise ValueError(f"Unsupported merge input: {path}")


def _load_all_rows(sources: list[SourceBundle]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source in sources:
        frames.extend(_load_rows_from_source(source))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_rows_from_source(source: SourceBundle) -> list[pd.DataFrame]:
    if source.kind == "csv":
        return [pd.read_csv(source.path)]
    if source.kind == "dir":
        paths = sorted(source.path.rglob("*_seed_sweep_rows.csv"))
        return [pd.read_csv(path) for path in paths]
    frames: list[pd.DataFrame] = []
    with ZipFile(source.path) as archive:
        members = sorted(
            name for name in archive.namelist()
            if name.endswith("_seed_sweep_rows.csv")
        )
        for member in members:
            frames.append(pd.read_csv(io.BytesIO(archive.read(member))))
    return frames


def _normalize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    if "seed" not in rows.columns or "adaptation_policy" not in rows.columns:
        raise ValueError("Merged rows must include seed and adaptation_policy columns.")
    rows["seed"] = rows["seed"].astype(int)
    rows["adaptation_policy"] = rows["adaptation_policy"].map(_normalize_policy_name)
    for column in rows.columns:
        if column in {"run_name", "adaptation_policy", "env_id", "run_dir"}:
            continue
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.sort_values(["seed", "adaptation_policy"]).reset_index(drop=True)
    return rows


def _normalize_policy_name(name: object) -> str:
    text = str(name)
    return POLICY_ALIASES.get(text, text)


def _validate_complete_grid(rows: pd.DataFrame) -> None:
    duplicates = rows.duplicated(subset=["seed", "adaptation_policy"])
    if duplicates.any():
        dupes = rows.loc[duplicates, ["seed", "adaptation_policy"]].to_dict("records")
        raise ValueError(f"Duplicate seed/policy rows found: {dupes[:5]}")
    seeds = set(rows["seed"].unique())
    policies = set(rows["adaptation_policy"].unique())
    missing: list[tuple[int, str]] = []
    for seed in sorted(seeds):
        seed_policies = set(rows.loc[rows["seed"] == seed, "adaptation_policy"])
        for policy in policies - seed_policies:
            missing.append((int(seed), str(policy)))
    if missing:
        raise ValueError(f"Incomplete seed/policy grid; first missing cells: {missing[:10]}")


def _attach_missing_normalized_score_v2(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    if "normalized_budgeted_score_v2" not in rows.columns:
        rows["normalized_budgeted_score_v2"] = pd.NA
    if rows["normalized_budgeted_score_v2"].notna().all():
        return rows
    for seed, idx in rows.groupby("seed").groups.items():
        sub = rows.loc[idx]
        max_cost = float(sub["update_cost"].fillna(0.0).max())
        non_frozen = sub[sub["adaptation_policy"].astype(str).str.lower() != "frozen"]
        return_scale = float(non_frozen["return_mean"].abs().max()) if not non_frozen.empty else 0.0
        if max_cost <= 0 or return_scale <= 1e-6:
            continue
        for row_index, row in sub.iterrows():
            if pd.notna(row.get("normalized_budgeted_score_v2")):
                continue
            rows.at[row_index, "normalized_budgeted_score_v2"] = (
                float(row["return_mean"]) / return_scale
                - 0.5 * float(row["update_cost"]) / max_cost
            )
    rows["normalized_budgeted_score_v2"] = pd.to_numeric(
        rows["normalized_budgeted_score_v2"], errors="coerce",
    )
    return rows


def _load_config_snapshot(sources: list[SourceBundle], run_dir: str) -> dict[str, Any] | None:
    if not run_dir:
        return None
    member = f"{run_dir.strip('/')}/config.snapshot.yaml"
    for source in sources:
        if source.kind == "zip":
            with ZipFile(source.path) as archive:
                try:
                    with archive.open(member) as handle:
                        return yaml.safe_load(handle) or {}
                except KeyError:
                    continue
        elif source.kind == "dir":
            candidates = [
                source.path / "config.snapshot.yaml",
                Path(member),
                source.path / member,
            ]
            for path in candidates:
                if path.exists():
                    with path.open("r", encoding="utf-8") as handle:
                        return yaml.safe_load(handle) or {}
    return None


def _load_local_config_for_env(env_id: str) -> dict[str, Any] | None:
    """Fallback for light bundles that omit per-run config snapshots."""
    if not env_id:
        return None
    config_by_env = {
        "Hopper-v5": "configs/train/hopper_v5_v21_main.yaml",
        "Walker2d-v5": "configs/train/walker2d_v5_v21_main.yaml",
        "HalfCheetah-v5": "configs/train/halfcheetah_v5_v21_main.yaml",
        "Ant-v5": "configs/train/ant_v5_v21_main.yaml",
    }
    config_path = config_by_env.get(env_id)
    if not config_path:
        return None
    path = Path(config_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_figure1(
    aggregate_rows: list[dict[str, Any]],
    ranking_payload: dict[str, Any],
    output_path: Path,
) -> None:
    axes = ["return_mean", "recovery_auc", "forgetting", "improvement_over_frozen"]
    higher = {
        "return_mean": True,
        "recovery_auc": True,
        "forgetting": False,
        "improvement_over_frozen": True,
    }
    rankings = {
        axis: rank_names(aggregate_rows, axis, higher_is_better=higher[axis])
        for axis in axes
    }
    policies = rankings["return_mean"]
    fig = go.Figure()
    for policy in policies:
        y_values = [rankings[axis].index(policy) + 1 for axis in axes]
        fig.add_trace(
            go.Scatter(
                x=axes,
                y=y_values,
                mode="lines+markers",
                name=policy,
            )
        )
    annotations = []
    for axis in axes[1:]:
        comparison = ranking_payload["comparisons"].get(axis, {})
        tau = comparison.get("kendall_tau_vs_return_iqm")
        if tau is not None:
            annotations.append(f"τ(return,{axis})={float(tau):.3f}")
    fig.update_layout(
        title="Figure 1 Prototype — Reward-only vs Multi-axis Ranking Disagreement (IQM)",
        xaxis_title="Ranking axis",
        yaxis_title="Rank position (1 = best)",
        yaxis=dict(autorange="reversed", dtick=1),
        legend_title="Policy",
        annotations=[
            dict(
                text="<br>".join(annotations),
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.14,
                showarrow=False,
                align="right",
            )
        ],
    )
    fig.write_html(output_path)
