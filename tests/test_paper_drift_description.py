"""The manuscript's drift paragraph must match the configs it describes.

Section 2 of the paper states the drift schedule in words. Numbers derived from
the analysis artifacts are macro-generated and cannot drift; the drift schedule
is a *configuration* constant, so it has no such guard. It had already drifted
once -- the manuscript described a single dynamics change per environment and
called the reactor's drift a setpoint change, when the configs say four
escalating friction events and an actuator-gain fault. This test is the guard.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MAIN_TEX = ROOT / "paper" / "main.tex"

# What the paper's Setup paragraph asserts, keyed by config.
EXPECTED = {
    "walker2d_v5_v21_main.yaml": ("friction", 8000, 0.15, 8000, 4),
    "ant_v5_v21_main.yaml": ("friction", 8000, 0.15, 8000, 4),
    "halfcheetah_v5_v21_main.yaml": ("friction", 8000, 0.15, 8000, 4),
    "hopper_v5_v21_main.yaml": ("body_mass", 5000, 0.25, 0, 1),
    "pcgym_case_study.yaml": ("actuator_gain", 8000, 0.15, 8000, 2),
}


@pytest.mark.parametrize("name,expected", sorted(EXPECTED.items()))
def test_config_matches_the_described_schedule(name: str, expected: tuple) -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "train" / name).read_text())
    d = cfg["drift"]
    got = (d["shift_type"], d["shift_step"], d["severity"],
           d["repeat_interval"], d["repeat_count"])
    assert got == expected, (
        f"{name} drift block changed; paper/main.tex Setup describes {expected}"
    )


def test_escalation_factors_quoted_in_the_paper_are_the_ones_applied() -> None:
    """drift.py escalates by 1 + severity*(i+1), or 1 - severity*(i+1) for gain."""
    tex = MAIN_TEX.read_text()

    # Friction: severity 0.15 over 4 events -> 1.15 .. 1.60.
    first = 1.0 + 0.15 * 1
    last = 1.0 + 0.15 * 4
    assert rf"\times {first:.2f}" in tex
    assert rf"\times {last:.2f}" in tex

    # Hopper body mass: severity 0.25, one event -> 1.25.
    assert rf"\times {1.0 + 0.25:.2f}" in tex

    # Reactor actuator gain: severity 0.15 over 2 events -> 0.85, 0.70.
    assert rf"\times {1.0 - 0.15 * 1:.2f}" in tex
    assert rf"\times {1.0 - 0.15 * 2:.2f}" in tex


def test_every_drift_event_lands_in_the_first_half_of_deployment() -> None:
    """The paper claims this outright, so it has to hold for all five configs."""
    for name in EXPECTED:
        cfg = yaml.safe_load((ROOT / "configs" / "train" / name).read_text())
        d, t = cfg["drift"], cfg["train"]
        deploy = t["total_steps"] - t["pretrain_steps"]
        last_event = d["shift_step"] + max(d["repeat_count"] - 1, 0) * d["repeat_interval"]
        assert last_event < deploy / 2, (
            f"{name}: last drift event at {last_event} of {deploy} deployment steps"
        )


def test_the_reactor_drift_is_not_described_as_a_setpoint_change() -> None:
    """The specific false claim this test exists to prevent coming back."""
    tex = MAIN_TEX.read_text().lower()
    assert "setpoint" not in tex


def test_figures_fit_the_text_block_at_natural_size() -> None:
    """main.tex includes both figures unscaled, so they must fit unscaled.

    They are authored at the same font sizes; scaling them to a common
    \\linewidth fraction printed those fonts at two different sizes, one of
    them below the style file's own 7pt floor.
    """
    fitz = pytest.importorskip("fitz")
    textwidth_pt = 5.5 * 72  # neurips_2026.sty: textwidth=5.5in
    for name in ("fig_mechanism.pdf", "fig_power.pdf"):
        page = fitz.open(ROOT / "paper" / "figures" / name)[0]
        assert page.rect.width <= textwidth_pt, (
            f"{name} is {page.rect.width:.0f}pt wide, over the {textwidth_pt:.0f}pt "
            "text block; it would overflow the margin at natural size"
        )
