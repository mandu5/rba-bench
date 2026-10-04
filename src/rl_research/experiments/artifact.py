"""Project-page artifact generation for RBA-Bench outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import html
import json
import os
import shutil

from rl_research.utils.logging import prepare_run_dir, write_json


EXPECTED_ARTIFACTS = {
    "figure1": "*_figure1_cross_env_ranking_disagreement.html",
    "return_rank_heatmap": "*_return_rank_heatmap.html",
    "same_budget_audit": "*_same_budget_audit.csv",
    "combined_summary": "*_combined_summary.csv",
    "ranking_disagreement": "*_env_ranking_disagreement.csv",
    "return_rank_matrix": "*_policy_return_rank_matrix.csv",
    "consistent_patterns": "*_consistent_patterns.csv",
    "summary_json": "*_summary.json",
}

PUBLIC_ASSET_NAMES = {
    "figure1": "figure1.html",
    "return_rank_heatmap": "return_rank_heatmap.html",
    "same_budget_audit": "same_budget_audit.csv",
    "combined_summary": "combined_summary.csv",
    "ranking_disagreement": "ranking_disagreement.csv",
    "return_rank_matrix": "return_rank_matrix.csv",
    "consistent_patterns": "consistent_patterns.csv",
    "summary_json": "summary.json",
}


def build_artifact_page(
    cross_env_dir: str | Path,
    *,
    output_dir: str | Path = "results/artifacts",
    name: str = "rba_bench_artifact",
    phase_status: str = "preliminary",
    copy_assets: bool = False,
) -> dict[str, Any]:
    """Create a lightweight local HTML gallery for cross-environment outputs.

    The page is intentionally static and dependency-free so it can be copied into
    a GitHub Pages, Notion, or faculty-outreach artifact without requiring a web
    server.
    """
    source_dir = Path(cross_env_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Cross-environment directory not found: {source_dir}")

    artifacts = _collect_artifacts(source_dir)
    summary = _load_summary(artifacts.get("summary_json"))

    run_dir = prepare_run_dir(output_dir, name)
    public_artifacts = _copy_public_assets(artifacts, run_dir) if copy_assets else artifacts
    asset_dir = run_dir / "assets" if copy_assets else None
    manifest = {
        "name": name,
        "phase_status": phase_status,
        "source_dir": str(source_dir),
        "artifacts": {key: str(value) if value is not None else None for key, value in artifacts.items()},
        "public_artifacts": {
            key: str(value) if value is not None else None for key, value in public_artifacts.items()
        },
        "copy_assets": copy_assets,
        "story_b_diagnostics": summary.get("story_b_diagnostics", {}),
    }
    index_path = run_dir / "index.html"
    manifest_path = run_dir / "manifest.json"
    index_path.write_text(
        _render_html(
            name=name,
            phase_status=phase_status,
            source_dir=source_dir,
            run_dir=run_dir,
            artifacts=public_artifacts,
            summary=summary,
        ),
        encoding="utf-8",
    )
    write_json(manifest_path, manifest)
    return {
        "artifact_dir": str(run_dir),
        "index_path": str(index_path),
        "manifest_path": str(manifest_path),
        "source_dir": str(source_dir),
        "asset_dir": str(asset_dir) if asset_dir is not None else None,
        "missing_artifacts": [key for key, value in artifacts.items() if value is None],
    }


def _collect_artifacts(source_dir: Path) -> dict[str, Path | None]:
    artifacts: dict[str, Path | None] = {}
    for key, pattern in EXPECTED_ARTIFACTS.items():
        matches = sorted(source_dir.glob(pattern))
        artifacts[key] = matches[-1] if matches else None
    return artifacts


def _copy_public_assets(artifacts: dict[str, Path | None], run_dir: Path) -> dict[str, Path | None]:
    asset_dir = run_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path | None] = {}
    for key, source in artifacts.items():
        if source is None:
            copied[key] = None
            continue
        target_name = PUBLIC_ASSET_NAMES.get(key, source.name)
        target = asset_dir / target_name
        shutil.copy2(source, target)
        copied[key] = target
    return copied


def _load_summary(summary_path: Path | None) -> dict[str, Any]:
    if summary_path is None:
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _relative_link(target: Path | None, base_dir: Path) -> str | None:
    if target is None:
        return None
    return os.path.relpath(target, start=base_dir)


def _render_html(
    *,
    name: str,
    phase_status: str,
    source_dir: Path,
    run_dir: Path,
    artifacts: dict[str, Path | None],
    summary: dict[str, Any],
) -> str:
    title = "RBA-Bench: Fixed-Budget Continual RL Evaluation"
    links = []
    for label, key in [
        ("Figure 1: ranking disagreement", "figure1"),
        ("Return rank heatmap", "return_rank_heatmap"),
        ("Same-budget audit CSV", "same_budget_audit"),
        ("Robust combined summary CSV", "combined_summary"),
        ("Ranking disagreement CSV", "ranking_disagreement"),
        ("Return rank matrix CSV", "return_rank_matrix"),
        ("Consistent patterns CSV", "consistent_patterns"),
        ("Summary JSON", "summary_json"),
    ]:
        rel = _relative_link(artifacts.get(key), run_dir)
        if rel is None:
            links.append(f"<li><strong>{html.escape(label)}</strong>: missing</li>")
        else:
            links.append(
                f'<li><a href="{html.escape(rel)}">{html.escape(label)}</a></li>'
            )

    diagnostics = summary.get("story_b_diagnostics", {})
    tau_by_env = diagnostics.get("return_forgetting_tau_by_env", {})
    tau_rows = []
    if isinstance(tau_by_env, dict):
        for env_label, tau in tau_by_env.items():
            tau_rows.append(
                "<tr>"
                f"<td>{html.escape(str(env_label))}</td>"
                f"<td>{html.escape(_format_number(tau))}</td>"
                "</tr>"
            )

    policies = summary.get("policies", [])
    env_labels = summary.get("env_labels", [])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --ink: #18212f;
      --muted: #5a6678;
      --line: #d8dee8;
      --panel: #f7f4ec;
      --accent: #0f766e;
      --accent-2: #b45309;
    }}
    body {{
      margin: 0;
      font-family: Avenir Next, Charter, Georgia, serif;
      color: var(--ink);
      background: linear-gradient(135deg, #fbfaf6 0%, #edf5f4 48%, #f8efe2 100%);
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 48px 24px 72px;
    }}
    .hero {{
      border: 1px solid var(--line);
      border-radius: 24px;
      background: rgba(255, 255, 255, 0.78);
      padding: 32px;
      box-shadow: 0 16px 40px rgba(24, 33, 47, 0.08);
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: clamp(32px, 5vw, 56px);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    h2 {{
      margin-top: 34px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 8px;
    }}
    p, li {{
      font-size: 16px;
      line-height: 1.65;
    }}
    .badge {{
      display: inline-block;
      background: var(--accent);
      color: white;
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    .warning {{
      border-left: 4px solid var(--accent-2);
      background: #fff8ed;
      padding: 12px 16px;
      color: #4f3422;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: rgba(255,255,255,0.72);
    }}
    td, th {{
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
    }}
    code {{
      background: #edf2f7;
      padding: 2px 5px;
      border-radius: 6px;
    }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <span class="badge">{html.escape(phase_status)}</span>
      <h1>{html.escape(title)}</h1>
      <p>A reproducible fixed-budget evaluation toolkit for continual RL under non-stationary deployment drift.</p>
      <p class="warning">Claim guardrail: this page is an artifact gallery, not a final paper claim. Phase A-C outputs are preliminary unless Phase G 10-seed outputs are explicitly loaded.</p>
    </section>

    <h2>Core Artifact</h2>
    <div class="grid">
      <div class="card"><strong>Baselines</strong><br>{html.escape(str(len(policies)))} replay/adaptation policies</div>
      <div class="card"><strong>Environments</strong><br>{html.escape(', '.join(str(v) for v in env_labels) or 'not found')}</div>
      <div class="card"><strong>Source</strong><br><code>{html.escape(str(source_dir))}</code></div>
    </div>

    <h2>Result Links</h2>
    <ul>
      {''.join(links)}
    </ul>

    <h2>Story B Diagnostic: Return vs Forgetting</h2>
    <table>
      <thead><tr><th>Environment</th><th>Kendall tau(return, forgetting)</th></tr></thead>
      <tbody>{''.join(tau_rows) if tau_rows else '<tr><td colspan="2">No diagnostics found.</td></tr>'}</tbody>
    </table>

    <h2>Reproduction</h2>
    <p>Use <code>phase-g-intake</code> to merge light bundles, then <code>compare-envs</code> to regenerate this source directory. Use <code>build-artifact-page</code> to rebuild this page.</p>
  </main>
</body>
</html>
"""


def _format_number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
