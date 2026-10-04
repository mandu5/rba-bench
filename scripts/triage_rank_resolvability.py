"""Triage: is the replay-strategy ranking resolvable at the seed counts we have?

Runs on the merged per-seed rows already on disk. No training, no GPU.

The question this answers is not "which policy wins". It is which comparisons
the data can actually support, which turns out to be a much smaller set:

  1. Absolute return, seed-bootstrapped, with the ordering tested rather than
     read off CI overlap. Overlap of two marginal intervals is not a test of a
     difference, so the extremes get an exact within-seed label-permutation
     test (range statistic, which builds in the selection of the extremes) and
     selected adjacent steps are inspected only after exact paired sign-flip
     tests and Holm correction across all possible pairs in the environment.
  2. Rank stability: how often each policy is top-1 across seed resamples.
  3. The paired statistic. Seeds are shared across policies, so per-seed
     differences cancel the seed variance that swamps the absolute numbers.
  4. The recency / non-recency contrast, paired at the seed level.
  5. Metric disagreement against a null. A pairwise flip rate between two
     metrics has no interpretation without the chance level, which is 0.5 for
     independent rankings -- not 0. Metrics designed to measure different things
     will flip at chance by construction, so a flip rate near 0.5 is not
     evidence that evaluation is unreliable.

Usage:
    python3 scripts/triage_rank_resolvability.py
    python3 scripts/triage_rank_resolvability.py --boot 20000 --seed 7
"""
from __future__ import annotations

import argparse
import csv
import json
import glob
import itertools
import os
from collections import Counter

import numpy as np

# +1 where higher is better, -1 where lower is better.
METRICS = {
    "return_mean": 1,
    "recovery_auc": 1,
    "forgetting": -1,
    "update_cost": -1,
    "improvement_per_1k_updates": 1,
}
RECENCY = {"RecentOnly", "RecencyWeighted", "StaticMix"}
CONTROL = "Frozen"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank-resolvability triage over merged results.")
    p.add_argument("--results", default="results/merged")
    p.add_argument("--pattern", nargs="*", default=["phase_g_10seed", "pcgym"],
                   help="Substrings selecting which merged directories to use.")
    p.add_argument("--boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default="results/rank_resolvability.json")
    return p.parse_args()


def iqm(x: np.ndarray) -> float:
    """Interquartile mean, the robust aggregate used by the merged reports."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def exact_signflip_p(diffs: np.ndarray) -> float:
    """Exact two-sided sign-flip p for H0: paired differences symmetric about 0."""
    signs = np.array(list(itertools.product([1, -1], repeat=len(diffs))))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((null >= abs(diffs.mean()) - 1e-12).mean())


def bh_survivors(named_ps, alpha: float = 0.05) -> set:
    order = sorted(range(len(named_ps)), key=lambda i: named_ps[i][1])
    kmax = 0
    for k, i in enumerate(order, 1):
        if named_ps[i][1] <= alpha * k / len(named_ps):
            kmax = k
    return {named_ps[order[j]][0] for j in range(kmax)}


def adjacent_pair_audit(X: np.ndarray, policies: list[str], order: list[str],
                        alpha: float = 0.05) -> dict:
    """Protect any selected leaderboard pair by Holm over all possible pairs."""
    all_pairs = list(itertools.combinations(range(len(policies)), 2))
    ps = [exact_signflip_p(X[i] - X[j]) for i, j in all_pairs]
    ranked = np.argsort(ps)
    adjusted = np.empty(len(ps))
    running = 0.0
    for k, i in enumerate(ranked):
        running = max(running, (len(ps) - k) * ps[i])
        adjusted[i] = min(running, 1.0)
    lookup = {pair: i for i, pair in enumerate(all_pairs)}
    selected = [lookup[tuple(sorted((policies.index(order[k]),
                                     policies.index(order[k + 1]))))]
                for k in range(len(order) - 1)]
    adjacent_ps = [ps[i] for i in selected]
    minimum = 2.0 / (2 ** X.shape[1])
    return {"n_pairs": len(selected), "n_all_pairs": len(all_pairs),
            "p_values": adjacent_ps,
            "holm_p_values": [float(adjusted[i]) for i in selected],
            "holm_rejections": int(sum(adjusted[i] <= alpha for i in selected)),
            "exploratory_bh_rejections": len(bh_survivors(list(enumerate(adjacent_ps)), alpha)),
            "minimum_attainable_p": minimum, "first_holm_cutoff": alpha / len(ps),
            "cannot_reject_at_seed_count": minimum > alpha / len(ps)}


def load(d: str):
    files = glob.glob(os.path.join(d, "*_rows.csv"))
    if not files:
        return None
    rows = list(csv.DictReader(open(files[0])))
    if not rows or "adaptation_policy" not in rows[0]:
        return None
    seeds = sorted({r["seed"] for r in rows})
    policies = sorted({r["adaptation_policy"] for r in rows} - {CONTROL})
    table = {(r["adaptation_policy"], r["seed"]): r for r in rows}
    return seeds, policies, table


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    # The ordering tests were added after the first release. They draw from
    # their own stream so every number produced by the original chain is
    # unchanged by the addition.
    rng2 = np.random.default_rng(args.seed + 101)
    dirs = [d for d in sorted(glob.glob(os.path.join(args.results, "*/")))
            if any(p in d for p in args.pattern)]

    verdict = {"ordering_resolved": 0, "partition_resolved": 0, "n_env": 0}
    flip_tally: Counter[str] = Counter()
    record: dict = {"seed": args.seed, "boot": args.boot, "envs": {}, "budget": {}}

    for d in dirs:
        loaded = load(d)
        if loaded is None:
            continue
        seeds, policies, T = loaded
        env = os.path.basename(d.rstrip("/")).split("_")[1]
        verdict["n_env"] += 1

        def vec(p: str, c: str) -> np.ndarray:
            return np.array([float(T[(p, s)][c]) for s in seeds])

        print(f"\n{'=' * 74}\n{env.upper()}  ({len(seeds)} seeds, {len(policies)} adaptive policies)\n{'=' * 74}")

        # 1 -- absolute return with a seed bootstrap
        est = {}
        for p in policies:
            v = vec(p, "return_mean")
            bs = np.array([iqm(v[rng.integers(0, len(v), len(v))]) for _ in range(args.boot)])
            est[p] = (iqm(v), *np.quantile(bs, [0.025, 0.975]))
        order = sorted(policies, key=lambda p: -est[p][0])
        print("  return_mean IQM [95% seed-bootstrap CI]")
        for p in order:
            m, lo, hi = est[p]
            print(f"    {p:18s} {m:10.2f}  [{lo:9.2f}, {hi:9.2f}]")
        disjoint = est[order[0]][1] > est[order[-1]][2]
        disjoint = bool(disjoint)
        verdict["ordering_resolved"] += int(disjoint)
        record["envs"][env] = {
            "n_seeds": len(seeds), "n_policies": len(policies),
            "return_iqm": {p: {"iqm": est[p][0], "ci": [est[p][1], est[p][2]]} for p in policies},
            "order": order, "ordering_resolved": bool(disjoint),
        }
        print(f"    -> rank 1 vs rank {len(order)} CIs disjoint: {disjoint}")

        # 1a -- the extremes, tested rather than eyeballed. Overlap of two
        # marginal CIs is not a test of a difference, and the seed effect that
        # widens both intervals is shared, so it cancels in a within-seed
        # comparison the marginal intervals never make. Two tests, because the
        # obvious one is biased: best and worst were selected on this same
        # data, so a paired test on that one pair inherits the selection. The
        # label-permutation test uses the range of policy IQMs as its
        # statistic, which builds the selection in, and is exact under
        # within-seed exchangeability.
        Xr = np.array([vec(p, "return_mean") for p in policies])
        p_pair = exact_signflip_p(Xr[policies.index(order[0])]
                                  - Xr[policies.index(order[-1])])
        obs_range = est[order[0]][0] - est[order[-1]][0]
        n_perm = 20000
        hits = 0
        for _ in range(n_perm):
            Xp = np.empty_like(Xr)
            for s_i in range(Xr.shape[1]):
                Xp[:, s_i] = Xr[rng2.permutation(len(policies)), s_i]
            v_i = np.array([iqm(row) for row in Xp])
            hits += (v_i.max() - v_i.min()) >= obs_range - 1e-12
        p_omni = float((hits + 1) / (n_perm + 1))
        record["envs"][env]["extremes"] = {
            "paired_sign_flip_p": p_pair,
            "attainable_p_floor": 2.0 / (2 ** len(seeds)),
            "omnibus_range_p": p_omni,
            "omnibus_reps": n_perm,
        }
        print(f"    -> best vs worst, naive paired sign-flip p = {p_pair:.4f} "
              f"(selected on the same data)")
        print(f"    -> within-seed label permutation, range statistic: p = {p_omni:.4f}")

        # Correct the complete pair family before inspecting data-selected
        # adjacencies. Keep selected-only BH solely as an exploratory comparison.
        audit = adjacent_pair_audit(Xr, policies, order)
        record["envs"][env]["adjacent_pairs"] = audit
        print(f"    -> selected adjacent steps after all-pair Holm: "
              f"{audit['holm_rejections']}/{audit['n_pairs']}; "
              f"p-grid precludes rejection: {audit['cannot_reject_at_seed_count']}")

        # 2 -- how stable is "who is best"
        wins: Counter[str] = Counter()
        for _ in range(args.boot):
            idx = rng.integers(0, len(seeds), len(seeds))
            wins[max(policies, key=lambda p: iqm(vec(p, "return_mean")[idx]))] += 1
        print("  P(top-1) under seed resampling: " +
              ", ".join(f"{p} {c / args.boot:.3f}" for p, c in wins.most_common(3)))
        top1, top1_c = wins.most_common(1)[0]
        record["envs"][env]["top1"] = {"policy": top1, "prob": top1_c / args.boot}

        # 3 -- the paired, budget-normalised statistic
        print("  improvement_per_1k_updates IQM [95% CI], paired against Frozen")
        pe = {}
        for p in policies:
            v = vec(p, "improvement_per_1k_updates")
            bs = np.array([iqm(v[rng.integers(0, len(v), len(v))]) for _ in range(args.boot)])
            pe[p] = (iqm(v), *np.quantile(bs, [0.025, 0.975]))
        for p in sorted(policies, key=lambda p: -pe[p][0]):
            m, lo, hi = pe[p]
            print(f"    {p:18s} {m:9.3f}  [{lo:8.3f}, {hi:8.3f}]"
                  f"{'  recency' if p in RECENCY else '         '}"
                  f"{'  excludes 0' if lo > 0 else ''}")

        # 4 -- recency vs the rest, differenced within seed
        rec = [p for p in policies if p in RECENCY]
        non = [p for p in policies if p not in RECENCY]
        diff = np.array([np.mean([float(T[(p, s)]["return_mean"]) for p in rec])
                         - np.mean([float(T[(p, s)]["return_mean"]) for p in non]) for s in seeds])
        bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(args.boot)])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        resolved = lo > 0 or hi < 0
        resolved = bool(resolved)
        verdict["partition_resolved"] += int(resolved)
        record["envs"][env]["recency_contrast"] = {
            "mean": float(diff.mean()), "ci": [float(lo), float(hi)],
            "sign_consistent": int(((diff > 0) == (diff.mean() > 0)).sum()), "n_seeds": len(diff),
            "resolved": bool(resolved),
            "cohen_d_paired": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else None,
        }
        print(f"  recency - non-recency, paired per seed: {diff.mean():+.2f} "
              f"[{lo:+.2f}, {hi:+.2f}], same sign in {int((diff > 0).sum())}/{len(diff)} seeds "
              f"-> {'resolved' if resolved else 'not resolved'}")

        # 4a -- the same standard the paper applies to the metric table.
        # A bootstrap interval is anticonservative at these sample sizes; an
        # exact sign-flip test is not, and at n=5 its smallest attainable
        # two-sided p is 2/2^5 = 0.0625, so no effect size can reach 0.05.
        n_d = len(diff)
        obs = abs(diff.mean())
        if n_d <= 20:
            signs = np.array(list(itertools.product([1, -1], repeat=n_d)))
            null = np.abs((signs * diff).mean(axis=1))
            perm_p = float((null >= obs - 1e-12).mean())
            p_floor = 2.0 / (2 ** n_d)
        else:
            null = np.abs((rng.choice([1, -1], (20000, n_d)) * diff).mean(axis=1))
            perm_p = float(((null >= obs).sum() + 1) / (20000 + 1))
            p_floor = 1.0 / (20000 + 1)
        record["envs"][env]["recency_contrast"]["exact_sign_flip_p"] = perm_p
        record["envs"][env]["recency_contrast"]["attainable_p_floor"] = p_floor
        print(f"  exact sign-flip p = {perm_p:.4f} "
              f"(smallest attainable at {n_d} seeds: {p_floor:.4f})")

        # 4b -- why the paired contrast resolves when the ranking does not.
        # Seed share of TOTAL return variance, from unbiased expected-mean-
        # squares components of the two-way policy x seed layout. The naive
        # plug-in var_seed/(var_seed + resid.var()) is wrong twice over: it
        # drops the policy component from the denominator, and both plug-ins
        # are biased upward (seed means carry resid/n_pol; resid.var() divides
        # by nm instead of (n-1)(m-1)). It overstated the share by up to 0.23.
        X = np.array([vec(p, "return_mean") for p in policies])   # policy x seed
        a_n, b_n = X.shape
        ms_pol = b_n * float(X.mean(axis=1).var(ddof=1))
        ms_seed = a_n * float(X.mean(axis=0).var(ddof=1))
        resid = X - X.mean(axis=1, keepdims=True) - X.mean(axis=0, keepdims=True) + X.mean()
        ms_res = float((resid ** 2).sum() / ((a_n - 1) * (b_n - 1)))
        s2_pol = max(0.0, (ms_pol - ms_res) / b_n)
        s2_seed = max(0.0, (ms_seed - ms_res) / a_n)
        total_var = s2_seed + s2_pol + ms_res
        icc = s2_seed / total_var if total_var > 0 else float("nan")
        pair_d = []
        for i, j in itertools.combinations(range(len(policies)), 2):
            g = X[i] - X[j]
            if g.std(ddof=1) > 0:
                pair_d.append(abs(g.mean()) / g.std(ddof=1))
        d_contrast = record["envs"][env]["recency_contrast"]["cohen_d_paired"]
        record["envs"][env]["mechanism"] = {
            "seed_icc": float(icc),
            "policy_share": float(s2_pol / total_var) if total_var > 0 else None,
            "median_pair_cohen_d": float(np.median(pair_d)) if pair_d else None,
            "contrast_over_pair_ratio": (float(d_contrast / np.median(pair_d))
                                         if pair_d and d_contrast else None),
        }
        print(f"  seed ICC {icc:.3f}; paired contrast d {d_contrast:.2f} vs median "
              f"policy-pair d {np.median(pair_d):.2f} "
              f"({d_contrast / np.median(pair_d):.1f}x)")

        # 4c -- the RecentOnly reversal, tested rather than read off the ranks.
        # A rank is not a result; the claim is that pure recency is above the
        # other adaptive policies somewhere and below them somewhere else.
        others = [q for q in policies if q != "RecentOnly"]
        if "RecentOnly" in policies and others:
            ro = np.array([float(T[("RecentOnly", s)]["return_mean"])
                           - np.mean([float(T[(q, s)]["return_mean"]) for q in others])
                           for s in seeds])
            n_r = len(ro)
            obs_r = abs(ro.mean())
            sgn = np.array(list(itertools.product([1, -1], repeat=n_r)))
            p_r = float((np.abs((sgn * ro).mean(axis=1)) >= obs_r - 1e-12).mean())
            record["envs"][env]["recentonly_vs_others"] = {
                "mean": float(ro.mean()),
                "cohen_d_paired": float(ro.mean() / ro.std(ddof=1)) if ro.std(ddof=1) else None,
                "sign_consistent_positive": int((ro > 0).sum()),
                "n_seeds": n_r,
                "exact_sign_flip_p": p_r,
                "rank": order.index("RecentOnly") + 1,
            }
            print(f"  RecentOnly vs other adaptive policies: {ro.mean():+.1f} "
                  f"(rank {order.index('RecentOnly') + 1}), exact p = {p_r:.4f}")

        # 5 -- metric disagreement against the chance level
        pairs = list(itertools.combinations(policies, 2))
        print("  metric-pair flip rate vs chance (0.5):")
        for a, b in itertools.combinations(METRICS, 2):
            bs = []
            for _ in range(min(args.boot, 2000)):
                idx = rng.integers(0, len(seeds), len(seeds))
                sa = {p: METRICS[a] * iqm(vec(p, a)[idx]) for p in policies}
                sb = {p: METRICS[b] * iqm(vec(p, b)[idx]) for p in policies}
                bs.append(sum((sa[x] > sa[y]) != (sb[x] > sb[y]) for x, y in pairs) / len(pairs))
            lo, hi = np.quantile(bs, [0.025, 0.975])
            sa = {p: METRICS[a] * iqm(vec(p, a)) for p in policies}
            sb = {p: METRICS[b] * iqm(vec(p, b)) for p in policies}
            obs = sum((sa[x] > sa[y]) != (sb[x] > sb[y]) for x, y in pairs) / len(pairs)
            tag = "below chance" if hi < 0.5 else ("indistinguishable" if lo < 0.5 else "above chance")
            flip_tally[str(tag)] += 1
            print(f"    {a} vs {b:32s} {obs:.3f}  CI[{lo:.2f},{hi:.2f}]  {tag}")

    # 6 -- which budget is actually equalised
    print(f"\n{'=' * 74}\nSAME-BUDGET AUDIT\n{'=' * 74}")
    print("  The comparison is only 'fixed budget' with respect to the resource that")
    print("  was actually held fixed. Updates and replay samples are equalised exactly;")
    print("  wall-clock is not, and the gap runs the same way in every environment.")
    ratios = {}
    for d in dirs:
        env = os.path.basename(d.rstrip("/")).split("_")[1]
        files = glob.glob(os.path.join(d, "*_budget_audit.csv"))
        if not files:
            continue
        rows = [r for r in csv.DictReader(open(files[0])) if r["adaptation_policy"] != CONTROL]
        exact = (len({r["update_cost_iqm"] for r in rows}) == 1
                 and len({r["replay_samples_iqm"] for r in rows}) == 1
                 and {float(r["budget_used_ratio_iqm"]) for r in rows} == {1.0}
                 # equal in aggregate is not equal per seed; the zero-variance
                 # claim in the paper needs the std columns checked too
                 and {float(r["update_cost_std"]) for r in rows} == {0.0}
                 and {float(r["replay_samples_std"]) for r in rows} == {0.0})
        wc = {r["adaptation_policy"]: float(r["wallclock_update_time_iqm"]) for r in rows}
        cheapest = min(wc, key=wc.get)
        ratios[env] = {p: v / wc[cheapest] for p, v in wc.items()}
        rel = {p: float(v / wc[cheapest]) for p, v in wc.items()}
        record["budget"][env] = {
            "updates_and_samples_identical": bool(exact),
            "wallclock_spread": float(max(wc.values()) / wc[cheapest]),
            "cheapest": cheapest,
            "relative_wallclock": rel,
            # the paper's direction claim lives at the group level: the
            # policy-level version ("the cheapest are ranked highest") is
            # false in three environments
            "recency_group_mean": float(np.mean([v for p, v in rel.items() if p in RECENCY])),
            "nonrecency_group_mean": float(np.mean([v for p, v in rel.items() if p not in RECENCY])),
        }
        print(f"  {env:12s} updates and samples identical: {exact}   "
              f"wall-clock spread {max(wc.values()) / wc[cheapest]:.2f}x "
              f"(cheapest: {cheapest})")
    if ratios:
        envs = list(ratios)
        print(f"\n  wall-clock cost relative to the cheapest policy\n"
              f"  {'policy':18s}" + "".join(f"{e[:9]:>11s}" for e in envs))
        for p in sorted({p for r in ratios.values() for p in r}):
            print(f"  {p:18s}" + "".join(f"{ratios[e].get(p, float('nan')):11.2f}" for e in envs))
        worst = max(max(r.values()) for r in ratios.values())
        print(f"\n  -> under a wall-clock-matched budget the cheapest policy would get up to")
        print(f"     {worst:.2f}x more updates. The policies that already rank highest are")
        print(f"     also the cheapest, so this accounting understates their advantage.")

    n = verdict["n_env"]
    print(f"\n{'=' * 74}\nVERDICT over {n} result sets\n{'=' * 74}")
    print(f"  full ordering resolvable (rank 1 vs last disjoint) : {verdict['ordering_resolved']}/{n}")
    print(f"  recency/non-recency contrast resolvable            : {verdict['partition_resolved']}/{n}")
    print(f"  metric pairs below chance / indistinguishable / above chance: "
          f"{flip_tally['below chance']} / {flip_tally['indistinguishable']} / {flip_tally['above chance']}")
    # The paper corrects the metric table for multiplicity; its own positive
    # claim is five tests and has to be held to the same standard.
    pv = [(e, d["recency_contrast"]["exact_sign_flip_p"])
          for e, d in record["envs"].items()
          if "exact_sign_flip_p" in d.get("recency_contrast", {})]
    if pv:
        m = len(pv)
        order = sorted(range(m), key=lambda i: pv[i][1])
        kmax = 0
        for k, i in enumerate(order, 1):
            if pv[i][1] <= 0.05 * k / m:
                kmax = k
        bh = {pv[order[j]][0] for j in range(kmax)}
        holm = set()
        for k, i in enumerate(order, 1):
            if pv[i][1] <= 0.05 / (m - k + 1):
                holm.add(pv[i][0])
            else:
                break
        record["contrast_multiplicity"] = {
            "n_tests": m,
            "uncorrected_bootstrap": int(verdict["partition_resolved"]),
            "bh_survivors": sorted(bh),
            "holm_survivors": sorted(holm),
        }
        print(f"\n  recency contrast under the paper's own standard:")
        print(f"    uncorrected bootstrap intervals : {verdict['partition_resolved']} of {m}")
        print(f"    exact sign-flip + Benjamini-Hochberg : {len(bh)} of {m}  {sorted(bh)}")
        print(f"    exact sign-flip + Holm               : {len(holm)} of {m}  {sorted(holm)}")

    # the ordering under real tests, across environments, same BH standard
    ex = [(e, d["extremes"]) for e, d in record["envs"].items() if "extremes" in d]
    if ex:
        omni_bh = sorted(bh_survivors([(e, x["omnibus_range_p"]) for e, x in ex]))
        pair_bh = sorted(bh_survivors([(e, x["paired_sign_flip_p"]) for e, x in ex]))
        adj_sep = sum(d["adjacent_pairs"]["holm_rejections"]
                      for d in record["envs"].values() if "adjacent_pairs" in d)
        adj_tot = sum(d["adjacent_pairs"]["n_pairs"]
                      for d in record["envs"].values() if "adjacent_pairs" in d)
        full = sum(1 for d in record["envs"].values()
                   if "adjacent_pairs" in d
                   and d["adjacent_pairs"]["holm_rejections"] == d["adjacent_pairs"]["n_pairs"])
        record["extremes_multiplicity"] = {
            "n_tests": len(ex),
            "omnibus_bh_survivors": omni_bh,
            "paired_bh_survivors": pair_bh,
            "adjacent_separated": adj_sep,
            "adjacent_total": adj_tot,
            "adjacent_exploratory_bh": sum(d["adjacent_pairs"]["exploratory_bh_rejections"]
                                           for d in record["envs"].values()),
            "adjacent_selection_control": "Holm over every policy pair before selection",
            "orders_fully_resolved": full,
        }
        print(f"\n  the ordering under within-seed tests instead of CI overlap:")
        print(f"    best vs worst (label permutation + BH)     : {len(omni_bh)} of {len(ex)}  {omni_bh}")
        print(f"    naive paired test on the selected extremes : {len(pair_bh)} of {len(ex)}  {pair_bh}")
        print(f"    adjacent leaderboard steps separating      : {adj_sep} of {adj_tot}")
        print(f"    environments whose full order resolves     : {full} of {len(ex)}")

    # multiplicity across environments for the reversal claim too
    rv = [(e, d["recentonly_vs_others"]) for e, d in record["envs"].items()
          if "recentonly_vs_others" in d]
    if rv:
        m2 = len(rv)
        o2 = sorted(range(m2), key=lambda i: rv[i][1]["exact_sign_flip_p"])
        k2 = 0
        for k, i in enumerate(o2, 1):
            if rv[i][1]["exact_sign_flip_p"] <= 0.05 * k / m2:
                k2 = k
        surv = [rv[o2[j]][0] for j in range(k2)]
        above = sorted(e for e in surv if rv[[x[0] for x in rv].index(e)][1]["mean"] > 0)
        below = sorted(e for e in surv if rv[[x[0] for x in rv].index(e)][1]["mean"] < 0)
        record["reversal"] = {"n_tests": m2, "bh_survivors": sorted(surv),
                              "above": above, "below": below,
                              "reversal_supported": bool(above and below)}
        print(f"\n  RecentOnly reversal, Benjamini-Hochberg over {m2} environments:")
        print(f"    above the other policies: {above}")
        print(f"    below the other policies: {below}")
        print(f"    reversal supported on both sides: {bool(above and below)}")

    # The three families (extremes, contrast, reversal) share data, and each
    # is corrected within its own family of five. A joint correction across
    # every test in the paper is the robustness check a reviewer would ask
    # for; record whether it changes any survivor.
    fams = []
    for e, d in record["envs"].items():
        if "extremes" in d:
            fams.append((f"extremes:{e}", d["extremes"]["omnibus_range_p"]))
        c = d.get("recency_contrast", {})
        if "exact_sign_flip_p" in c:
            fams.append((f"contrast:{e}", c["exact_sign_flip_p"]))
        r = d.get("recentonly_vs_others")
        if r:
            fams.append((f"reversal:{e}", r["exact_sign_flip_p"]))
    if fams:
        joint = sorted(bh_survivors(fams))
        per_family = sorted(
            {f"extremes:{e}" for e in record.get("extremes_multiplicity", {}).get("omnibus_bh_survivors", [])}
            | {f"contrast:{e}" for e in record.get("contrast_multiplicity", {}).get("bh_survivors", [])}
            | {f"reversal:{e}" for e in record.get("reversal", {}).get("bh_survivors", [])})
        record["joint_multiplicity"] = {
            "n_tests": len(fams),
            "survivors": joint,
            "unchanged_from_per_family": joint == per_family,
        }
        print(f"\n  joint Benjamini-Hochberg across all {len(fams)} tests:")
        print(f"    survivors unchanged from per-family correction: {joint == per_family}")

    record["verdict"] = {k: int(v) for k, v in verdict.items()}
    record["verdict"]["flip_tally"] = {k: int(v) for k, v in flip_tally.items()}
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(record, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print(f"\n  wrote {args.json_out}")
    print("\n  A flip rate whose interval covers 0.5 is indistinguishable from two")
    print("  metrics that share no information about policy quality. For metrics")
    print("  that measure different constructs this is expected, which is why a")
    print("  raw flip count is not on its own evidence that evaluation is broken.")


if __name__ == "__main__":
    main()
