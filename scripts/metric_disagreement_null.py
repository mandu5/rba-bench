"""Is cross-metric ranking disagreement more than sampling noise?

"Metric A and metric B rank the methods differently on X% of pairs" has become a
standard argument for multi-axis evaluation. The number is normally reported
bare. It needs a reference, and the obvious one is not enough.

REFERENCE 1 -- independence null.
Permute one metric's values across methods within an evaluation group. This
destroys any association while preserving both marginals, and gives the flip
rate expected when the metrics share no information about method quality: 0.5.
A reported rate below 0.5 therefore means the metrics agree more than chance.
It does not establish that they disagree, because two metrics that are both pure
sampling noise also flip at 0.5.

REFERENCE 2 -- split-half self-disagreement.
Rank the methods by metric A on one half of the seeds and by A again on the
other half. Those flips are sampling noise alone, since the metric is identical.
The test then compares like with like,

    cross = flip( A(half 1), B(half 2) )

against the rate expected if the two metrics shared a true ordering and differed
only in noise. Getting that reference right matters more than it looks. Write
p_M for metric M's per-pair mis-ordering probability. With disjoint halves the
self-disagreement is f_M = 2 p_M (1 - p_M), and the concordant cross rate is
p_A + p_B - 2 p_A p_B. The arithmetic mean of the two self-rates is NOT that
quantity -- it is smaller by exactly (p_A - p_B)^2 -- so using it as the floor
makes any pair of metrics that share an ordering but differ in noise look like
a real difference. The inversion must be applied separately to each policy
pair before averaging its floor; inversion of the pooled self-rate is a
different nonlinear estimator. Finite-sample bias and nonsmooth inference
remain model-dependent; scripts/validate_inference.py records calibration.

INFERENCE.
Enumerating balanced splits estimates excess *conditional on the seeds
observed*; the spread across splits is procedure noise, not sampling error, and
must not be reported as a confidence interval. Seed-level uncertainty comes from
a leave-one-seed-out jackknife. Across a whole table of metric pairs the tests
are given nominal Benjamini-Hochberg flags. These are exploratory diagnostics,
not distribution-free significance guarantees.

Three diagnostics accompany the test:
  --power     what true rank separation could be detected at this seed count,
              so that "not detected" is distinguishable from "not there";
  --floor     descriptive self-disagreement by half size and an unvalidated
              log-log extrapolation, not an inferential bound;
  the independence null is reported alongside for context.

Usage:
    python3 scripts/metric_disagreement_null.py
    python3 scripts/metric_disagreement_null.py --power --floor --reps 800
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
from functools import lru_cache

import numpy as np

METRICS = {
    "return_mean": 1,
    "recovery_auc": 1,
    "forgetting": -1,
    "improvement_per_1k_updates": 1,
}
CONTROL = "Frozen"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Null models for cross-metric ranking disagreement.")
    p.add_argument("--results", default="results/merged")
    p.add_argument("--pattern", nargs="*", default=["phase_g_10seed"])
    p.add_argument("--reps", type=int, default=600, help="fallback splits above ten seeds")
    p.add_argument("--jack-reps", type=int, default=300, help="fallback jackknife splits above ten seeds")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", action="store_true", help="run the power simulation")
    p.add_argument("--floor", action="store_true", help="run the noise-floor size sweep")
    p.add_argument("--drop-metric", default=None,
                   help="rerun the table without this metric, to price its redundancy")
    p.add_argument("--json-out", default="results/metric_disagreement_null.json")
    return p.parse_args()


def iqm(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def _flips(sa: np.ndarray, sb: np.ndarray, ii: np.ndarray, jj: np.ndarray) -> float:
    return float(np.mean((sa[ii] > sa[jj]) != (sb[ii] > sb[jj])))


def _mis_order_rate(self_flip):
    """Invert f = 2p(1-p) for p, the per-pair mis-ordering probability.

    Project finite-sample self-rates above 0.5 into the inversion domain.
    Clipping gives p=0.5; it does not establish a latent chance ordering.
    """
    f = np.minimum(np.asarray(self_flip, dtype=float), 0.5)
    value = (1.0 - np.sqrt(np.maximum(0.0, 1.0 - 2.0 * f))) / 2.0
    return float(value) if value.ndim == 0 else value


def concordant_floor(self_a, self_b):
    """Cross-metric flip rate expected when the two metrics share a true order."""
    pa, pb = _mis_order_rate(self_a), _mis_order_rate(self_b)
    return pa + pb - 2.0 * pa * pb


@lru_cache(maxsize=12)
def _balanced_partitions(n: int) -> tuple[np.ndarray, np.ndarray]:
    """All ordered disjoint half-samples; odd n leaves one seed out."""
    h = n // 2
    first, second = [], []
    for a in itertools.combinations(range(n), h):
        remaining = [i for i in range(n) if i not in a]
        for b in itertools.combinations(remaining, h):
            first.append(a)
            second.append(b)
    return np.asarray(first), np.asarray(second)


def _half_scores(values: np.ndarray, halves: np.ndarray) -> np.ndarray:
    """Vectorized policy IQMs, one row per half-sample."""
    h = halves.shape[1]
    lo, hi = int(np.floor(h * 0.25)), int(np.ceil(h * 0.75))
    selected = np.sort(values[:, halves], axis=-1)
    return selected[:, :, lo:hi].mean(axis=-1).T


def pairwise_split_estimate(A: np.ndarray, B: np.ndarray, cols: np.ndarray,
                            reps: int, rng) -> dict:
    """Estimate each policy-pair floor before averaging across pairs.

    Exhaustive splitting at n<=10 removes split Monte Carlo error from both
    point estimates and delete-one replicates. For larger samples, use reps
    random ordered disjoint halves. The inversion assumes independent half
    samples from a continuous latent order, with error probability <=1/2.
    """
    n, h = len(cols), len(cols) // 2
    if n <= 10:
        h1, h2 = _balanced_partitions(n)
        scheme = "exhaustive_ordered_disjoint_halves"
    else:
        perms = np.asarray([rng.permutation(n) for _ in range(reps)])
        h1, h2 = perms[:, :h], perms[:, h:2 * h]
        scheme = "random_ordered_disjoint_halves"
    a1, a2 = _half_scores(A[:, cols], h1), _half_scores(A[:, cols], h2)
    b1, b2 = _half_scores(B[:, cols], h1), _half_scores(B[:, cols], h2)
    ii, jj = np.triu_indices(A.shape[0], 1)
    oa1, oa2 = a1[:, ii] > a1[:, jj], a2[:, ii] > a2[:, jj]
    ob1, ob2 = b1[:, ii] > b1[:, jj], b2[:, ii] > b2[:, jj]
    self_a, self_b = (oa1 != oa2).mean(axis=0), (ob1 != ob2).mean(axis=0)
    floors = concordant_floor(self_a, self_b)
    return {"cross": float((oa1 != ob2).mean()), "floor": float(floors.mean()),
            "pair_self_a": self_a.tolist(), "pair_self_b": self_b.tolist(),
            "pair_floors": floors.tolist(),
            "clamped_pair_count": int(((self_a > 0.5) | (self_b > 0.5)).sum()),
            "split_count": len(h1), "half_size": h, "split_scheme": scheme}


class Grid:
    """Per-seed metric values for one environment, as arrays."""

    def __init__(self, rows: list[dict]):
        self.seeds = sorted({r["seed"] for r in rows})
        self.pol = sorted({r["adaptation_policy"] for r in rows} - {CONTROL})
        idx = {(r["adaptation_policy"], r["seed"]): r for r in rows}
        # values[metric][policy, seed], already sign-oriented so higher is better
        self.values = {
            c: np.array([[METRICS[c] * float(idx[(p, s)][c]) for s in self.seeds] for p in self.pol])
            for c in METRICS
        }
        self.ii, self.jj = np.triu_indices(len(self.pol), k=1)

    def score(self, metric: str, cols: np.ndarray) -> np.ndarray:
        v = self.values[metric][:, cols]
        return np.array([iqm(row) for row in v])

    def excess(self, a: str, b: str, cols: np.ndarray, reps: int, rng) -> tuple[float, float]:
        """Mean cross flip rate and the concordant noise floor, over disjoint halves.

        Self-disagreement is averaged over splits for each policy pair. Invert
        each pair's rate, compose its concordant floor, then average over pairs.
        Neither pooled inversion nor the mean of the two self-rates is this
        estimator. The shared implementation is also used by calibration.
        """
        estimate = pairwise_split_estimate(self.values[a], self.values[b], cols, reps, rng)
        return estimate["cross"], estimate["floor"]

    def independence_null(self, a: str, b: str, reps: int, rng) -> float:
        """Permute metric b across policies; expected flip rate is 0.5."""
        cols = np.arange(len(self.seeds))
        sa, sb = self.score(a, cols), self.score(b, cols)
        return float(np.mean([_flips(sa, rng.permutation(sb), self.ii, self.jj) for _ in range(reps)]))


def cell(g: "Grid", a: str, b: str, cols: np.ndarray, args, rng) -> dict:
    """One metric-pair cell: excess over the floor, jackknife SE, one-sided p.

    Factored out so the --drop-metric sensitivity prices exactly the same
    estimator as the main table rather than a re-implementation of it.
    """
    from math import erf, sqrt

    estimate = pairwise_split_estimate(g.values[a], g.values[b], cols, args.reps, rng)
    cross, self_ = estimate["cross"], estimate["floor"]
    point = cross - self_
    jk = []
    for i in range(len(cols)):
        c2, s2 = g.excess(a, b, np.delete(cols, i), args.jack_reps, rng)
        jk.append(c2 - s2)
    jk = np.array(jk)
    n = len(cols)
    se = float(np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum()))
    z = point / se if se > 0 else 0.0
    return {**estimate, "excess": point, "jackknife_se": se,
            "jackknife_split_count": len(_balanced_partitions(n - 1)[0]) if n - 1 <= 10 else args.jack_reps,
            "jackknife_half_size": (n - 1) // 2,
            "ci": [float(point - 1.96 * se), float(point + 1.96 * se)],
            "p_one_sided": float(1.0 - 0.5 * (1.0 + erf(z / sqrt(2))))}


def bh_reject(pvals: list[float], alpha: float) -> list[bool]:
    """Benjamini-Hochberg step-up."""
    n = len(pvals)
    order = np.argsort(pvals)
    keep = np.zeros(n, dtype=bool)
    kmax = -1
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= alpha * rank / n:
            kmax = rank
    if kmax > 0:
        keep[order[:kmax]] = True
    return keep.tolist()


def run_power(g: Grid, metric: str, rng, n_sim: int = 300, splits: int = 60) -> list[dict]:
    """What true rank separation is detectable at this seed count?

    A null result is only informative if the design could have found an effect.
    We simulate two metrics that share a true ranking up to a controlled number
    of reversed pairs, give each the per-seed noise scale observed for `metric`,
    and ask how often the test fires. The decision threshold is calibrated from
    the zero-separation case rather than fixed by hand. Independent draws at
    separation zero estimate false-positive rate. This is a single uncorrected
    Gaussian-model diagnostic, not power for every cell in the table.
    """
    v = g.values[metric]
    truth = np.sort(v.mean(axis=1))[::-1]
    # The statistic compares policies WITHIN a half, so a per-seed shift common
    # to every policy cancels -- which is the whole point of the paired-contrast
    # section. Total per-seed SD is dominated by exactly that shared effect
    # (the seed share of return variance runs 0.38-0.91 here), so using it
    # injects noise the statistic never sees and makes the design look far
    # weaker than it is. The scale that matters is the spread of policy
    # differences.
    diffs = [(v[i] - v[j]).std(ddof=1) / np.sqrt(2.0) for i, j in zip(g.ii, g.jj)]
    noise = float(np.mean(diffs))
    n_pol, n_seed = v.shape
    ii, jj = g.ii, g.jj

    def one(partner: np.ndarray) -> float:
        a = truth[:, None] + rng.normal(0, noise, (n_pol, n_seed))
        b = partner[:, None] + rng.normal(0, noise, (n_pol, n_seed))
        estimate = pairwise_split_estimate(a, b, np.arange(n_seed), splits, rng)
        return estimate["cross"] - estimate["floor"]

    # calibrate on the no-difference case
    null_draws = np.array([one(truth.copy()) for _ in range(n_sim)])
    thresh = float(np.quantile(null_draws, 0.95))

    out = []
    for k_rev in (0, 1, 2, 3, 4):
        partner = truth.copy()
        # reverse the top 2*k_rev entries -> a controlled number of discordant pairs
        if k_rev:
            m = min(2 * k_rev, n_pol)
            partner[:m] = partner[:m][::-1]
        disc = float(np.mean((truth[ii] > truth[jj]) != (partner[ii] > partner[jj])))
        draws = np.array([one(partner) for _ in range(n_sim)])
        out.append({"reversed_block": int(2 * k_rev),
                    "true_pair_disagreement": disc,
                    "detection_rate": float(np.mean(draws > thresh)),
                    "threshold": thresh, "calibration_draws": n_sim,
                    "evaluation_draws": n_sim, "single_uncorrected_test": True,
                    "noise_sd": noise,
                    "model": "Independent homoscedastic Gaussian metric observations; return-derived latent spacing.",
                    "truth_a": truth.tolist(), "truth_b": partner.tolist()})
    return out


def run_floor(g: Grid, rng, reps: int = 400) -> dict:
    """Self-disagreement as a function of half size, to extrapolate to full n."""
    n = len(g.seeds)
    cols = np.arange(n)
    out = {}
    for c in METRICS:
        by_h = {}
        for h in range(2, n // 2 + 1):
            vals = []
            for _ in range(reps):
                p = rng.permutation(cols)
                h1, h2 = p[:h], p[h:2 * h]
                vals.append(_flips(g.score(c, h1), g.score(c, h2), g.ii, g.jj))
            by_h[h] = float(np.mean(vals))
        hs = np.array(sorted(by_h))
        ys = np.array([by_h[h] for h in hs])
        # self-disagreement decays roughly as a power of the half size
        pos = ys > 1e-9
        if pos.sum() >= 2:
            slope, intercept = np.polyfit(np.log(hs[pos]), np.log(ys[pos]), 1)
            extrap = float(np.exp(intercept + slope * np.log(n)))
        else:
            slope, extrap = float("nan"), 0.0
        out[c] = {"by_half_size": {int(k): v for k, v in by_h.items()},
                  "log_log_slope": float(slope),
                  "extrapolated_to_full_n": extrap}
    return out


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    dirs = [d for d in sorted(glob.glob(os.path.join(args.results, "*/")))
            if any(p in d for p in args.pattern)]

    records = []
    print("cross = A(h1) vs B(h2); floor = mean of separately inverted policy-pair floors")
    print("Wald intervals and BH flags are nominal exploratory diagnostics; see")
    print("results/inference_calibration.json for model calibration and its scope.\n")
    print(f"{'env':11s} {'A vs B':44s} {'cross':>6s} {'floor':>6s} {'excess':>8s} {'jk 95% CI':>18s} {'p':>7s}")

    for d in dirs:
        env = os.path.basename(d.rstrip("/")).split("_")[1]
        files = glob.glob(os.path.join(d, "*_rows.csv"))
        if not files:
            continue
        g = Grid(list(csv.DictReader(open(files[0]))))
        n = len(g.seeds)
        if n // 2 < 2:
            print(f"  {env}: {n} seeds, split-half not meaningful; skipped")
            continue
        cols = np.arange(n)
        for a, b in itertools.combinations(METRICS, 2):
            c = cell(g, a, b, cols, args, rng)
            lo, hi = c["ci"]
            records.append({"env": env, "metric_a": a, "metric_b": b, "n_seeds": n,
                            "policy_pairs": [[g.pol[i], g.pol[j]] for i, j in zip(g.ii, g.jj)], **c,
                            "independence_null": g.independence_null(a, b, 400, rng)})
            print(f"  {env:9s} {a + ' vs ' + b:44s} {c['cross']:6.3f} {c['floor']:6.3f} "
                  f"{c['excess']:+8.3f} [{lo:+.3f},{hi:+.3f}] {c['p_one_sided']:7.3f}")

    if records:
        pv = [r["p_one_sided"] for r in records]
        keep = bh_reject(pv, args.alpha)
        for r, k in zip(records, keep):
            r["bh_significant"] = bool(k)
        raw = sum(p < args.alpha for p in pv)
        print(f"\n  {raw} of {len(pv)} nominal one-sided flags before correction")
        print(f"  {sum(keep)} nominal Benjamini-Hochberg flags at alpha={args.alpha}")

    payload = {"seed": args.seed, "reps": args.reps, "jack_reps": args.jack_reps,
               "estimator": "mean_pair_specific_concordant_floor_v2",
               "inference_scope": "Nominal exploratory Wald/BH flags; model calibration is in inference_calibration.json.",
               "alpha": args.alpha, "results": records}

    if args.floor or args.power:
        g0 = Grid(list(csv.DictReader(open(glob.glob(os.path.join(dirs[0], "*_rows.csv"))[0]))))
        env0 = os.path.basename(dirs[0].rstrip("/")).split("_")[1]
        if args.floor:
            print(f"\n  NOISE FLOOR vs HALF SIZE ({env0})")
            fl = run_floor(g0, rng)
            payload["floor_sweep"] = {env0: fl}
            for c, v in fl.items():
                sizes = "  ".join(f"h={k}:{x:.3f}" for k, x in sorted(v["by_half_size"].items()))
                print(f"    {c:28s} {sizes}   -> n={len(g0.seeds)}: {v['extrapolated_to_full_n']:.3f}")
            print("    Descriptive half-size sensitivity only; the log-log extrapolation")
            print("    is unvalidated and is not an inferential floor or exclusion bound.")
        if args.power:
            print(f"\n  POWER ({env0}, {len(g0.seeds)} seeds, metric return_mean)")
            pw = run_power(g0, "return_mean", rng)
            payload["power"] = {env0: pw}
            for row in pw:
                print(f"    true pair disagreement {row['true_pair_disagreement']:.3f}"
                      f"  ->  detected {row['detection_rate']:.2f}")
            print("    Threshold uses separate no-difference calibration draws; the first")
            print("    row estimates false-positive rate on independent validation draws.")
            print("    Single uncorrected test in one Gaussian model, not a table-wide bound.")

    if args.drop_metric:
        drop = args.drop_metric
        if drop not in METRICS:
            raise SystemExit(f"--drop-metric: {drop!r} is not one of {list(METRICS)}")
        # Its own generator, so the sensitivity cannot perturb the table above.
        srng = np.random.default_rng(args.seed)
        kept = [m for m in METRICS if m != drop]
        srecs = []
        for d in dirs:
            env = os.path.basename(d.rstrip("/")).split("_")[1]
            files = glob.glob(os.path.join(d, "*_rows.csv"))
            if not files:
                continue
            g = Grid(list(csv.DictReader(open(files[0]))))
            if len(g.seeds) // 2 < 2:
                continue
            cols = np.arange(len(g.seeds))
            for a, b in itertools.combinations(kept, 2):
                srecs.append({"env": env, "metric_a": a, "metric_b": b,
                              **cell(g, a, b, cols, args, srng)})
        skeep = bh_reject([r["p_one_sided"] for r in srecs], args.alpha)
        for r, k in zip(srecs, skeep):
            r["bh_significant"] = bool(k)
        payload["drop_metric_sensitivity"] = {
            "dropped": drop, "n_tests": len(srecs), "n_bh": int(sum(skeep)),
            "results": srecs,
        }
        print(f"\n  SENSITIVITY: dropping {drop}")
        print(f"    {len(srecs)} comparisons remain, {sum(skeep)} survive "
              f"Benjamini-Hochberg at alpha={args.alpha}")
        for r in (x for x in srecs if x["bh_significant"]):
            print(f"      {r['env']:11s} {r['metric_a']} vs {r['metric_b']}"
                  f"   excess {r['excess']:+.3f}  p {r['p_one_sided']:.3f}")
        print("    Under the equalised budget improvement_per_1k_updates is a within-seed")
        print("    affine transform of return_mean. Across-seed IQM trimming can nevertheless")
        print("    change rankings; this sensitivity removes the related metric family.")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  wrote {args.json_out}")


if __name__ == "__main__":
    main()
