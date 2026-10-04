"""Section 7's wall-clock mechanism must match the policies' sampling paths.

The paper says RecentOnly and StaticMix draw their batch by indexing into the
buffer while the other six adaptive policies score every stored transition
first. That is a statement about which `sampling_mode` each policy declares,
and an earlier draft got it wrong -- it put Uniform and RecencyWeighted on the
cheap path, when both go through `sample_weighted` and score the whole buffer.
This pins the prose to the code.
"""
from __future__ import annotations

from pathlib import Path

from rl_research.adaptation import policies as P
from rl_research.config import AdaptationBudget

ROOT = Path(__file__).resolve().parents[1]
MAIN_TEX = ROOT / "paper" / "main.tex"

# The eight adaptive policies the paper compares, by class.
ADAPTIVE = {
    "RecentOnly": P.RecentOnlyPolicy,
    "StaticMix": P.StaticMixPolicy,
    "RecencyWeighted": P.RecencyWeightedPolicy,
    "Uniform": P.UniformPolicy,
    "PER": P.PERPolicy,
    "Budgeted-TD": P.BudgetedTDPolicy,
    "Budgeted-Grad": P.BudgetedGradPolicy,
    "Budgeted-Novelty": P.BudgetedNoveltyPolicy,
}
INDEXERS = {"RecentOnly", "StaticMix"}


def _sampling_mode(cls) -> str:
    """Instantiate with defaults and read the mode off one decision.

    `decide` accepts `stats=None` (the first window has no stats yet), so the
    only thing needed is a budget; every field of it is irrelevant to the mode.
    """
    budget = AdaptationBudget(window_steps=1, max_updates_per_window=1, recent_replay_ratio=0.5)
    return cls().decide(None, budget).sampling_mode


def test_exactly_recentonly_and_staticmix_index_into_the_buffer() -> None:
    modes = {name: _sampling_mode(cls) for name, cls in ADAPTIVE.items()}
    indexers = {n for n, m in modes.items() if m == "mixed"}
    scorers = {n for n, m in modes.items() if m == "weighted"}
    assert indexers == INDEXERS, modes
    assert scorers == set(ADAPTIVE) - INDEXERS, modes


def test_the_paper_names_the_right_two_indexers() -> None:
    tex = MAIN_TEX.read_text()
    i = tex.index("index into the buffer")
    window = tex[max(0, i - 200): i]
    assert r"\textsf{RecentOnly}" in window and r"\textsf{StaticMix}" in window
    # The sentence that was wrong before must not come back in any spelling.
    assert "recency and uniform policies index" not in tex
    assert "uniform policies index into" not in tex
