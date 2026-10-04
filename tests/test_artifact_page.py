from __future__ import annotations

from pathlib import Path

import json

from rl_research.experiments.artifact import build_artifact_page


def test_build_artifact_page_from_cross_env_outputs(tmp_path: Path) -> None:
    source = tmp_path / "cross_env"
    source.mkdir()
    (source / "demo_figure1_cross_env_ranking_disagreement.html").write_text("<html>fig1</html>", encoding="utf-8")
    (source / "demo_return_rank_heatmap.html").write_text("<html>heatmap</html>", encoding="utf-8")
    (source / "demo_same_budget_audit.csv").write_text("adaptation_policy,budget_used_ratio_mean\nUniform,1.0\n", encoding="utf-8")
    (source / "demo_combined_summary.csv").write_text("adaptation_policy,return_mean_iqm\nUniform,1.0\n", encoding="utf-8")
    (source / "demo_env_ranking_disagreement.csv").write_text("env_label,metric,kendall_tau_vs_return_iqm\nMock,forgetting,-0.1\n", encoding="utf-8")
    (source / "demo_policy_return_rank_matrix.csv").write_text("adaptation_policy,Mock\nUniform,1\n", encoding="utf-8")
    (source / "demo_consistent_patterns.csv").write_text("adaptation_policy,mean_return_rank\nUniform,1\n", encoding="utf-8")
    (source / "demo_summary.json").write_text(
        json.dumps(
            {
                "policies": ["Uniform"],
                "env_labels": ["Mock-v0"],
                "story_b_diagnostics": {
                    "return_forgetting_tau_by_env": {
                        "Mock-v0": -0.1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    output = build_artifact_page(
        source,
        output_dir=tmp_path / "artifacts",
        name="unit_artifact",
        phase_status="unit-test",
    )

    index = Path(output["index_path"])
    manifest = Path(output["manifest_path"])
    assert index.exists()
    assert manifest.exists()
    assert output["missing_artifacts"] == []
    html = index.read_text(encoding="utf-8")
    assert "RBA-Bench" in html
    assert "unit-test" in html
    assert "Kendall tau(return, forgetting)" in html
    assert "-0.100" in html


def test_build_artifact_page_can_copy_assets_for_static_site(tmp_path: Path) -> None:
    source = tmp_path / "cross_env"
    source.mkdir()
    (source / "demo_figure1_cross_env_ranking_disagreement.html").write_text("<html>fig1</html>", encoding="utf-8")
    (source / "demo_return_rank_heatmap.html").write_text("<html>heatmap</html>", encoding="utf-8")
    (source / "demo_same_budget_audit.csv").write_text("adaptation_policy,budget_used_ratio_mean\nUniform,1.0\n", encoding="utf-8")
    (source / "demo_combined_summary.csv").write_text("adaptation_policy,return_mean_iqm\nUniform,1.0\n", encoding="utf-8")
    (source / "demo_env_ranking_disagreement.csv").write_text("env_label,metric,kendall_tau_vs_return_iqm\nMock,forgetting,-0.1\n", encoding="utf-8")
    (source / "demo_policy_return_rank_matrix.csv").write_text("adaptation_policy,Mock\nUniform,1\n", encoding="utf-8")
    (source / "demo_consistent_patterns.csv").write_text("adaptation_policy,mean_return_rank\nUniform,1\n", encoding="utf-8")
    (source / "demo_summary.json").write_text(
        json.dumps({"policies": ["Uniform"], "env_labels": ["Mock-v0"]}),
        encoding="utf-8",
    )

    output = build_artifact_page(
        source,
        output_dir=tmp_path / "site",
        name="unit_static_site",
        phase_status="unit-test",
        copy_assets=True,
    )

    site_dir = Path(output["artifact_dir"])
    assert (site_dir / "assets" / "figure1.html").exists()
    assert (site_dir / "assets" / "same_budget_audit.csv").exists()
    assert output["asset_dir"] == str(site_dir / "assets")
    html = Path(output["index_path"]).read_text(encoding="utf-8")
    assert 'href="assets/figure1.html"' in html
    assert "../../cross_env" not in html
