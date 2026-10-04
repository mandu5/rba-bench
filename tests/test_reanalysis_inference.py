"""Regression cases for the two selection/estimator issues in official review."""
import numpy as np
import pytest

from scripts import metric_disagreement_null as metric
from scripts import triage_rank_resolvability as rank
from scripts import validate_inference as validation


def test_metric_floor_inverts_each_pair_before_averaging() -> None:
    # Across the six balanced splits, the A self-flips are (0, 1/3, 1/3)
    # and B's are (1/3, 0, 1/3). Pair-specific floors average to
    # 4/9 - 1/(3 sqrt(3)) = 0.2519943547; pooled inversion gives 2/9.
    a = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [-1, -1, 2, 2]])
    b = a[[0, 2, 1]]
    rows = []
    for p, name in enumerate(["A", "B", "C"]):
        for s in range(4):
            row = {key: 0.0 for key in metric.METRICS}
            row.update(seed=s, adaptation_policy=name,
                       return_mean=a[p, s], recovery_auc=b[p, s])
            rows.append(row)
    grid = metric.Grid(rows)
    cross, floor = grid.excess("return_mean", "recovery_auc", np.arange(4),
                               4096, np.random.default_rng(42))
    assert cross == pytest.approx(1 / 3, abs=1e-12)
    assert floor == pytest.approx(4 / 9 - 1 / (3 * np.sqrt(3)), abs=1e-12)


def test_selected_adjacencies_are_protected_by_all_pair_holm() -> None:
    names = list("ABCDEFGH")
    x = np.tile(np.arange(8, 0, -1)[:, None], (1, 10))
    audit = rank.adjacent_pair_audit(x, names, names)
    # Seven selected pairs all have p=2/1024 and would all pass selected-only
    # BH. Holm across the 28 possible pairs cannot reject the first null.
    assert audit["exploratory_bh_rejections"] == 7
    assert audit["holm_rejections"] == 0
    assert audit["n_all_pairs"] == 28
    assert audit["cannot_reject_at_seed_count"] is True


def test_all_pair_holm_can_resolve_adjacent_steps_with_more_seeds() -> None:
    names = list("ABCDEFGH")
    x = np.tile(np.arange(8, 0, -1)[:, None], (1, 11))
    audit = rank.adjacent_pair_audit(x, names, names)
    assert audit["holm_rejections"] == 7
    assert audit["cannot_reject_at_seed_count"] is False


def test_calibration_uses_the_pair_specific_excess_estimand() -> None:
    a = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [-1, -1, 2, 2]])
    b = a[[0, 2, 1]]
    ii, jj = np.triu_indices(3, 1)
    got = validation.excess(a, b, np.arange(4), 4096, ii, jj,
                            np.random.default_rng(42))
    assert got == pytest.approx(1 / (3 * np.sqrt(3)) - 1 / 9, abs=1e-12)


def test_null_calibration_checks_zero_not_the_biased_sampling_mean() -> None:
    summary = validation.sampling_summary(np.array([0.1, 0.1]),
                                           np.array([0.02, 0.02]), 0.0)
    assert summary["interval_coverage"] == 0.0
    assert summary["one_sided_rejection_rate"] == 1.0
    assert summary["bias"] == pytest.approx(0.1)


def test_population_target_counts_discordant_pairs_when_noise_is_zero() -> None:
    truth_a = np.arange(8, 0, -1, dtype=float)
    truth_b = truth_a.copy()
    truth_b[:4] = truth_b[:4][::-1]
    target = validation.population_target(truth_a, truth_b, np.zeros(8),
                                          np.zeros(8), 5, 100,
                                          np.random.default_rng(42))
    assert target["excess"] == pytest.approx(6 / 28)


def test_training_setup_matches_observed_runs_and_budget() -> None:
    from scripts.export_training_setup import build_training_setup

    setup = build_training_setup()
    assert setup["envs"]["hopper"]["observed_updates"] == 35000
    assert setup["envs"]["pcgym"]["observed_updates"] == 20000
    assert setup["envs"]["hopper"]["seeds"] == list(range(21, 31))
    assert setup["envs"]["pcgym"]["seeds"] == list(range(61, 66))
    assert sum(e["row_count"] for e in setup["envs"].values()) == 405
