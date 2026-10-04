"""Calibrate the revised pair-specific estimator without recentering its bias.

Independent Gaussian policy/seed observations generate the same disjoint-half
IQM statistic used on the benchmark. Concordant latent orderings have excess
exactly zero. For a partial reversal, a separate large simulation approximates
the population half-sample misordering rates and hence the excess target.
Coverage is assessed against that population target, never against the mean of
the finite-sample estimator. Cases vary noise by policy and across metrics.

These are model checks, not a universal guarantee for benchmark observations.
The JSON records failures as findings; the script does not tune cases or label
the inferential procedure valid solely because its standard errors are large.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .metric_disagreement_null import pairwise_split_estimate
except ImportError:
    from metric_disagreement_null import pairwise_split_estimate

ROOT = Path(__file__).resolve().parents[1]


def excess(A: np.ndarray, B: np.ndarray, cols: np.ndarray, reps: int,
           ii: np.ndarray, jj: np.ndarray, rng) -> float:
    """Compatibility wrapper; benchmark and calibration share one estimator."""
    estimate = pairwise_split_estimate(A, B, cols, reps, rng)
    return estimate["cross"] - estimate["floor"]


def wilson_interval(count: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = count / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(centre - radius), float(centre + radius)]


def sampling_summary(points: np.ndarray, ses: np.ndarray, target: float) -> dict:
    """Evaluate nominal Wald intervals and one-sided tests against a fixed target."""
    covered = int(np.count_nonzero(np.abs(points - target) <= 1.96 * ses))
    rejected = int(np.count_nonzero(points > 1.6448536269514722 * ses))
    n = len(points)
    sd = float(points.std(ddof=1))
    return {
        "datasets": n, "population_excess": target,
        "mean_estimate": float(points.mean()), "bias": float(points.mean() - target),
        "sampling_sd": sd, "mean_jackknife_se": float(ses.mean()),
        "se_over_sampling_sd": float(ses.mean() / sd) if sd > 0 else None,
        "interval_coverage": covered / n,
        "coverage_mc_interval": wilson_interval(covered, n),
        "one_sided_rejection_rate": rejected / n,
        "rejection_mc_interval": wilson_interval(rejected, n),
        "covered_count": covered, "rejected_count": rejected,
    }


def population_target(truth_a: np.ndarray, truth_b: np.ndarray,
                      sigma_a: np.ndarray, sigma_b: np.ndarray, half: int,
                      draws: int, rng: np.random.Generator) -> dict:
    """Target for independent halves under symmetric continuous metric noise.

    A concordant pair's cross misordering probability is pa+pb-2pa*pb, so
    subtracting its pair-specific floor gives zero. A discordant latent pair
    contributes 1-2*(pa+pb-2pa*pb). Population pa/pb are estimated independently
    of calibration datasets; the null target remains exactly zero.
    """
    ii, jj = np.triu_indices(len(truth_a), 1)
    direction_a = truth_a[ii] > truth_a[jj]
    direction_b = truth_b[ii] > truth_b[jj]
    discordant = direction_a != direction_b
    if not np.any(discordant):
        return {"excess": 0.0, "method": "analytic_concordant_null", "draws": 0,
                "monte_carlo_se": 0.0}
    if draws < 20 or draws % 20:
        raise ValueError("Population draws must be a positive multiple of 20.")
    lo, hi = int(np.floor(half * .25)), int(np.ceil(half * .75))
    scores = []
    for truth, sigma in ((truth_a, sigma_a), (truth_b, sigma_b)):
        values = truth[None, :, None] + rng.normal(size=(draws, len(truth), half)) * sigma[None, :, None]
        scores.append(np.sort(values, axis=-1)[:, :, lo:hi].mean(axis=-1))
    mis_a = (scores[0][:, ii] > scores[0][:, jj]) != direction_a
    mis_b = (scores[1][:, ii] > scores[1][:, jj]) != direction_b

    def target_from_rates(a: np.ndarray, b: np.ndarray) -> float:
        floor = a + b - 2 * a * b
        return float(np.mean(discordant * (1 - 2 * floor)))

    target = target_from_rates(mis_a.mean(axis=0), mis_b.mean(axis=0))
    batches = [target_from_rates(a.mean(axis=0), b.mean(axis=0))
               for a, b in zip(np.array_split(mis_a, 20), np.array_split(mis_b, 20))]
    return {"excess": target, "method": "independent_population_half_simulation",
            "draws": draws, "monte_carlo_se": float(np.std(batches, ddof=1) / np.sqrt(20))}


def calibration_cases(policies: int, noise: float, unequal_ratio: float) -> list[dict]:
    truth = np.linspace(1.0, 0.0, policies)
    reverse = truth.copy()
    reverse[: policies // 2] = reverse[: policies // 2][::-1]
    equal = np.full(policies, noise)
    varying = noise * np.geomspace(.25, 2.5, policies)
    return [
        dict(name="homoscedastic_concordant", null=True, truth_a=truth, truth_b=truth,
             sigma_a=equal, sigma_b=equal),
        dict(name="heteroscedastic_concordant", null=True, truth_a=truth, truth_b=truth,
             sigma_a=varying, sigma_b=varying[::-1]),
        dict(name="heteroscedastic_unequal_concordant", null=True, truth_a=truth, truth_b=truth,
             sigma_a=varying, sigma_b=unequal_ratio * varying[::-1]),
        dict(name="heteroscedastic_partial_reversal", null=False, truth_a=truth, truth_b=reverse,
             sigma_a=varying, sigma_b=varying[::-1]),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policies", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--noise", type=float, default=.35)
    ap.add_argument("--noise-ratio", type=float, default=3.4,
                    help="B/A multiplier in the unequal-noise case")
    ap.add_argument("--reps", type=int, default=600, help="fallback splits above ten seeds")
    ap.add_argument("--datasets", type=int, default=400)
    ap.add_argument("--population-draws", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "inference_calibration.json")
    args = ap.parse_args()
    if args.datasets < 2 or args.seeds < 4:
        raise ValueError("Use at least two datasets and four seeds.")
    cols = np.arange(args.seeds)
    cases = []
    for case_id, case in enumerate(calibration_cases(args.policies, args.noise, args.noise_ratio)):
        rng = np.random.default_rng(args.seed + 1009 * case_id)
        population_rng = np.random.default_rng(args.seed + 1000003 + 1009 * case_id)
        target = population_target(case["truth_a"], case["truth_b"], case["sigma_a"],
                                   case["sigma_b"], args.seeds // 2,
                                   args.population_draws, population_rng)
        points, ses, clipped = [], [], []
        print(f"{case['name']}: {args.datasets} datasets, target {target['excess']:+.5f}", flush=True)
        for draw in range(args.datasets):
            A = case["truth_a"][:, None] + rng.normal(size=(args.policies, args.seeds)) * case["sigma_a"][:, None]
            B = case["truth_b"][:, None] + rng.normal(size=(args.policies, args.seeds)) * case["sigma_b"][:, None]
            estimate = pairwise_split_estimate(A, B, cols, args.reps, rng)
            points.append(estimate["cross"] - estimate["floor"])
            clipped.append(estimate["clamped_pair_count"] > 0)
            jk = []
            for i in cols:
                leave_out = pairwise_split_estimate(A, B, np.delete(cols, i), args.reps, rng)
                jk.append(leave_out["cross"] - leave_out["floor"])
            jk = np.asarray(jk)
            ses.append(float(np.sqrt((args.seeds - 1) / args.seeds * np.sum((jk - jk.mean()) ** 2))))
            if (draw + 1) % 100 == 0:
                print(f"  {draw + 1}/{args.datasets}", flush=True)
        summary = sampling_summary(np.asarray(points), np.asarray(ses), target["excess"])
        cases.append({"name": case["name"], "is_concordant_null": case["null"],
                      "truth_a": case["truth_a"].tolist(), "truth_b": case["truth_b"].tolist(),
                      "sigma_a": case["sigma_a"].tolist(), "sigma_b": case["sigma_b"].tolist(),
                      "data_seed": args.seed + 1009 * case_id,
                      "population_seed": args.seed + 1000003 + 1009 * case_id,
                      "population_target": target,
                      "clipped_dataset_fraction": float(np.mean(clipped)), **summary})
        print(f"  bias {summary['bias']:+.4f}; coverage {summary['interval_coverage']:.3f}; "
              f"one-sided rejection {summary['one_sided_rejection_rate']:.3f}", flush=True)
    payload = {
        "estimator": "mean_pair_specific_concordant_floor_v2",
        "split_scheme": "exhaustive_ordered_disjoint_halves" if args.seeds <= 10 else "random_disjoint_halves",
        "policies": args.policies, "seeds": args.seeds, "half_size": args.seeds // 2,
        "nominal_coverage": .95, "nominal_one_sided_alpha": .05,
        "model": "Independent Gaussian observations across policy, metric and seed; policy-specific SDs; IQM half scores.",
        "coverage_target": "Analytic zero for concordant nulls; separately simulated population excess for reversal.",
        "scope": "Model calibration only; not a distribution-free validity guarantee or a benchmark-wide exclusion bound.",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
