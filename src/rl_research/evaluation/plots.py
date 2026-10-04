"""Plotting utilities for sweep summaries (v2.1).

v2.0 plots preserved (`reward_cost`, `recovery_cost`, `ranking_disagreement`,
`budgeted_score`).

v2.1 additions:
  - `per_vs_budgeted_main`: bar comparing PER vs. Budgeted-{TD,Grad,Novelty}
  - `plasticity_over_time`: line plot of dormant_neuron_ratio across windows
  - `retention_pareto`: scatter of retention rate vs. update cost
  - `kendall_tau_seed_stability`: heatmap of seed-level ranking stability
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kendall_tau(a: list[str], b: list[str]) -> float:
    """Kendall's tau on two ranked lists of policy names (no ties)."""
    common = [x for x in a if x in b]
    if len(common) < 2:
        return float("nan")
    pos_b = {name: idx for idx, name in enumerate(b)}
    n = len(common)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = i, j
            bi, bj = pos_b[common[i]], pos_b[common[j]]
            if (ai - aj) * (bi - bj) > 0:
                concordant += 1
            elif (ai - aj) * (bi - bj) < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return float("nan")
    return float(concordant - discordant) / total


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def plot_sweep_summary(
    summary_path: str | Path,
    output_dir: str | Path | None = None,
    plasticity_csv_glob: str | None = None,
) -> dict[str, str]:
    """Generate v2.0 + v2.1 plots from a sweep JSON summary.

    `plasticity_csv_glob` (optional): glob pattern under sweep_dir that points
    to per-policy plasticity_snapshots.csv files. If omitted, plasticity plot
    is skipped.
    """
    summary_file = Path(summary_path)
    with summary_file.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)

    rows = payload.get("rows", [])
    if not rows:
        raise ValueError(f"No rows found in sweep summary: {summary_file}")

    df = pd.DataFrame(rows)
    if "reward_per_update" in df:
        df["reward_per_update"] = pd.to_numeric(df["reward_per_update"], errors="coerce")
    plot_dir = Path(output_dir) if output_dir else summary_file.parent / f"{summary_file.stem}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    paths["reward_cost"] = str(plot_dir / "reward_cost.html")
    paths["recovery_cost"] = str(plot_dir / "recovery_cost.html")
    paths["ranking_disagreement"] = str(plot_dir / "ranking_disagreement.html")
    paths["budgeted_score"] = str(plot_dir / "budgeted_score.html")
    paths["per_vs_budgeted_main"] = str(plot_dir / "per_vs_budgeted_main.html")
    paths["retention_pareto"] = str(plot_dir / "retention_pareto.html")
    paths["kendall_tau_seed_stability"] = str(plot_dir / "kendall_tau_seed_stability.html")

    # ---- v2.0 plots --------------------------------------------------------

    reward_fig = px.scatter(
        df,
        x="update_cost",
        y="return_mean",
        color="adaptation_policy",
        hover_data=[c for c in ["reward_per_update", "normalized_budgeted_score", "forgetting", "run_dir"] if c in df.columns],
        title="Reward-Cost Tradeoff",
        labels={"update_cost": "Update count", "return_mean": "Mean deployment return"},
    )
    reward_fig.write_html(paths["reward_cost"])

    recovery_fig = px.scatter(
        df,
        x="update_cost",
        y="recovery_auc",
        color="adaptation_policy",
        hover_data=[c for c in ["return_mean", "forgetting", "run_dir"] if c in df.columns],
        title="Recovery-Cost Tradeoff",
        labels={"update_cost": "Update count", "recovery_auc": "Recovery AUC"},
    )
    recovery_fig.write_html(paths["recovery_cost"])

    disagreement = payload.get("ranking_disagreement_vs_reward_only", {})
    ranking_fig = go.Figure(
        data=[go.Bar(x=list(disagreement.keys()), y=list(disagreement.values()))]
    )
    ranking_fig.update_layout(
        title="Pairwise Ranking Disagreement vs Reward-Only Ranking",
        xaxis_title="Alternative metric",
        yaxis_title="Pairwise disagreements",
    )
    ranking_fig.write_html(paths["ranking_disagreement"])

    if "normalized_budgeted_score" in df.columns:
        score_fig = px.bar(
            df.sort_values("normalized_budgeted_score", ascending=False),
            x="adaptation_policy",
            y="normalized_budgeted_score",
            color="adaptation_policy",
            hover_data=[c for c in ["return_mean", "update_cost", "improvement_over_frozen"] if c in df.columns],
            title="Normalized Budgeted Score",
            labels={"normalized_budgeted_score": "Score"},
        )
        score_fig.write_html(paths["budgeted_score"])

    # ---- v2.1 plot: PER vs Budgeted-{TD,Grad,Novelty} hero figure ----------

    hero_targets = {"PER", "Budgeted-TD", "Budgeted-Grad", "Budgeted-Novelty", "Budgeted-Pred"}
    hero_df = df[df["adaptation_policy"].isin(hero_targets)].copy()
    if not hero_df.empty:
        hero_df = hero_df.sort_values("adaptation_policy")
        hero_long = hero_df.melt(
            id_vars=["adaptation_policy"],
            value_vars=[
                c for c in ["return_mean", "recovery_auc", "normalized_budgeted_score", "forgetting"]
                if c in hero_df.columns
            ],
            var_name="metric",
            value_name="value",
        )
        hero_fig = px.bar(
            hero_long,
            x="adaptation_policy",
            y="value",
            color="metric",
            barmode="group",
            title="PER vs. Budgeted-{TD, Grad, Novelty} — main comparison",
        )
        hero_fig.write_html(paths["per_vs_budgeted_main"])
    else:
        # Generate a placeholder so downstream tooling does not break.
        placeholder = go.Figure()
        placeholder.update_layout(title="PER vs. Budgeted (no v2.1 baselines in sweep)")
        placeholder.write_html(paths["per_vs_budgeted_main"])

    # ---- v2.1 plot: retention pareto (uses retention_rate if present) -------

    retention_cols = [c for c in df.columns if c.startswith("retention_rate")]
    if retention_cols:
        primary = retention_cols[0]
        retention_fig = px.scatter(
            df,
            x="update_cost",
            y=primary,
            color="adaptation_policy",
            hover_data=[c for c in ["return_mean", "recovery_auc", "run_dir"] if c in df.columns],
            title=f"Retention vs. Cost ({primary})",
        )
        retention_fig.write_html(paths["retention_pareto"])
    else:
        placeholder = go.Figure()
        placeholder.update_layout(title="Retention vs. Cost (no retention_rate column found)")
        placeholder.write_html(paths["retention_pareto"])

    # ---- v2.1 plot: Kendall's tau across seeds (if seed column exists) ----

    if "seed" in df.columns and df["seed"].notna().any():
        rankings_per_seed: dict[Any, list[str]] = {}
        for seed_value, seed_rows in df.groupby("seed"):
            sorted_seed = seed_rows.sort_values("return_mean", ascending=False)
            rankings_per_seed[seed_value] = sorted_seed["adaptation_policy"].tolist()
        seeds = list(rankings_per_seed.keys())
        n = len(seeds)
        if n >= 2:
            tau_matrix = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    tau = _kendall_tau(rankings_per_seed[seeds[i]], rankings_per_seed[seeds[j]])
                    tau_matrix[i][j] = 0.0 if math.isnan(tau) else tau
            heatmap = go.Figure(data=go.Heatmap(
                z=tau_matrix,
                x=[f"seed={s}" for s in seeds],
                y=[f"seed={s}" for s in seeds],
                colorscale="RdBu",
                zmin=-1.0,
                zmax=1.0,
            ))
            heatmap.update_layout(title="Kendall's τ across seed-level rankings (reward-only)")
            heatmap.write_html(paths["kendall_tau_seed_stability"])
        else:
            placeholder = go.Figure()
            placeholder.update_layout(title="Kendall's τ (need ≥ 2 seeds)")
            placeholder.write_html(paths["kendall_tau_seed_stability"])
    else:
        placeholder = go.Figure()
        placeholder.update_layout(title="Kendall's τ (no seed column in sweep summary)")
        placeholder.write_html(paths["kendall_tau_seed_stability"])

    # ---- v2.1 plot: plasticity over time (optional) ------------------------

    if plasticity_csv_glob:
        sweep_dir = summary_file.parent
        plasticity_files = sorted(sweep_dir.glob(plasticity_csv_glob))
        if plasticity_files:
            frames = []
            for f in plasticity_files:
                try:
                    p_df = pd.read_csv(f)
                    p_df["source_file"] = str(f.name)
                    frames.append(p_df)
                except Exception:
                    continue
            if frames:
                p_all = pd.concat(frames, ignore_index=True)
                if "dormant_neuron_ratio" in p_all.columns and "step" in p_all.columns:
                    plast_fig = px.line(
                        p_all,
                        x="step",
                        y="dormant_neuron_ratio",
                        color="source_file",
                        title="Dormant Neuron Ratio over Steps (per-policy snapshots)",
                    )
                    paths["plasticity_over_time"] = str(plot_dir / "plasticity_over_time.html")
                    plast_fig.write_html(paths["plasticity_over_time"])

    return paths
