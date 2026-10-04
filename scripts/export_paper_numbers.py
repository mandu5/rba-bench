"""Emit every numeric claim in the manuscript as a LaTeX macro.

The manuscript never hard-codes a number. It inputs ``paper/numbers.tex`` and
refers to macros, so a figure quoted in the text cannot drift from the artifact
it came from. Rerun the two analysis scripts, rerun this, and the paper updates.
Anything the paper needs that is not derivable from a committed artifact does
not get a macro, which makes such cases obvious.

Usage:
    python3 scripts/export_paper_numbers.py

Reads:
    results/rank_resolvability.json
    results/metric_disagreement_null.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOLV = ROOT / "results" / "rank_resolvability.json"
NULLJS = ROOT / "results" / "metric_disagreement_null.json"
CALIBRATION = ROOT / "results" / "inference_calibration.json"
TRAINING = ROOT / "results" / "training_setup.json"
DEFAULT_OUT = ROOT / "paper" / "numbers.tex"

_DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
           "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
# LaTeX control sequences take letters only, so metric and policy names are
# folded into CamelCase identifiers.
_SHORT = {
    "return_mean": "Ret", "recovery_auc": "Rec", "forgetting": "Forg",
    "improvement_per_1k_updates": "Imp", "update_cost": "Cost",
    "RecentOnly": "RecentOnly", "RecencyWeighted": "RecencyWeighted",
    "StaticMix": "StaticMix", "Uniform": "Uniform", "PER": "PER",
    "Budgeted-Grad": "BudGrad", "Budgeted-Novelty": "BudNov", "Budgeted-TD": "BudTD",
}
# Environment keys are internal slugs; the manuscript, its tables and its figure
# captions all spell the environments the usual way, so anything that reaches
# the PDF as text goes through this map. Macro *names* keep using the slug.
_DISPLAY = {
    "ant": "Ant", "halfcheetah": "HalfCheetah", "hopper": "Hopper",
    "walker2d": "Walker2d", "pcgym": "PC-Gym",
}


def disp(env: str) -> str:
    return _DISPLAY.get(str(env), str(env))


# Body text spells metrics out; the short codes are for the appendix table.
_FULL = {
    "return_mean": "return", "recovery_auc": "recovery",
    "forgetting": "forgetting", "update_cost": "update cost",
    "improvement_per_1k_updates": "budget-normalised improvement",
}


def macroname(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isdigit():
            out.append(_DIGITS[ch])
        elif ch.isalpha():
            out.append(ch)
    return "".join(out)


def fmt(v, places: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}".replace(",", "{,}")
    return f"{v:.{places}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Export manuscript numbers as LaTeX macros.")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    rz = json.loads(RESOLV.read_text())
    nl = json.loads(NULLJS.read_text())
    calibration = json.loads(CALIBRATION.read_text())
    training = json.loads(TRAINING.read_text())
    M: dict[str, str] = {}

    # ---------- headline verdict ----------
    v = rz["verdict"]
    M["NEnv"] = fmt(v["n_env"])
    M["OrderingResolved"] = fmt(v["ordering_resolved"])
    M["OrderingUnresolved"] = fmt(v["n_env"] - v["ordering_resolved"])
    M["PartitionResolved"] = fmt(v["partition_resolved"])

    # ---------- per environment ----------
    for env, e in rz["envs"].items():
        E = macroname(env.capitalize())
        M[f"Seeds{E}"] = fmt(e["n_seeds"])
        M[f"NPol{E}"] = fmt(e["n_policies"])
        M[f"OrderResolved{E}"] = fmt(e["ordering_resolved"])
        best, worst = e["order"][0], e["order"][-1]
        M[f"Best{E}"] = best
        M[f"Worst{E}"] = worst
        for tag, p in (("Best", best), ("Worst", worst)):
            r = e["return_iqm"][p]
            M[f"{tag}Iqm{E}"] = fmt(r["iqm"], 1)
            M[f"{tag}Lo{E}"] = fmt(r["ci"][0], 1)
            M[f"{tag}Hi{E}"] = fmt(r["ci"][1], 1)
        c = e["recency_contrast"]
        M[f"Contrast{E}"] = fmt(c["mean"], 1)
        M[f"ContrastLo{E}"] = fmt(c["ci"][0], 1)
        M[f"ContrastHi{E}"] = fmt(c["ci"][1], 1)
        M[f"ContrastSigns{E}"] = fmt(c["sign_consistent"])
        M[f"ContrastResolved{E}"] = fmt(c["resolved"])
        M[f"ContrastD{E}"] = fmt(c["cohen_d_paired"], 2)
        if "exact_sign_flip_p" in c:
            M[f"ContrastP{E}"] = fmt(c["exact_sign_flip_p"], 4)
            M[f"ContrastPFloor{E}"] = fmt(c["attainable_p_floor"], 4)
        m = e.get("mechanism", {})
        M[f"Icc{E}"] = fmt(m.get("seed_icc"), 3)
        M[f"PairD{E}"] = fmt(m.get("median_pair_cohen_d"), 2)
        M[f"Ratio{E}"] = fmt(m.get("contrast_over_pair_ratio"), 2)
        # rank of RecentOnly, the reversal that motivates the domain result
        if "RecentOnly" in e["order"]:
            M[f"RecentOnlyRank{E}"] = fmt(e["order"].index("RecentOnly") + 1)
        x = e.get("extremes")
        if x:
            M[f"OmniP{E}"] = fmt(x["omnibus_range_p"], 2)
            M[f"ExtPairP{E}"] = fmt(x["paired_sign_flip_p"], 4)
        a = e.get("adjacent_pairs")
        if a:
            M[f"AdjSep{E}"] = fmt(a["holm_rejections"])
            M[f"AdjTot{E}"] = fmt(a["n_pairs"])
        t = e.get("top1")
        if t:
            M[f"TopOne{E}"] = t["policy"]
            M[f"TopOneProb{E}"] = fmt(t["prob"], 2)

    # ---------- the paper's own claim, held to its own standard ----------
    cm = rz.get("contrast_multiplicity")
    if cm:
        M["ContrastNTests"] = fmt(cm["n_tests"])
        M["ContrastUncorrected"] = fmt(cm["uncorrected_bootstrap"])
        M["ContrastBH"] = fmt(len(cm["bh_survivors"]))
        M["ContrastHolm"] = fmt(len(cm["holm_survivors"]))
        M["ContrastBHList"] = ", ".join(disp(e) for e in cm["bh_survivors"])
        M["ContrastHolmList"] = ", ".join(disp(e) for e in cm["holm_survivors"])

    # ---------- the ordering, tested rather than eyeballed ----------
    em = rz.get("extremes_multiplicity")
    if em:
        M["ExtremesNTests"] = fmt(em["n_tests"])
        M["ExtremesBH"] = fmt(len(em["omnibus_bh_survivors"]))
        M["ExtremesBHList"] = ", ".join(disp(e) for e in em["omnibus_bh_survivors"])
        M["ExtremesPairedBH"] = fmt(len(em["paired_bh_survivors"]))
        M["AdjacentSeparated"] = fmt(em["adjacent_separated"])
        M["AdjacentTotal"] = fmt(em["adjacent_total"])
        M["OrdersFullyResolved"] = fmt(em["orders_fully_resolved"])
        M["AdjacentExploratory"] = fmt(em["adjacent_exploratory_bh"])
        adjacent = rz["envs"]["hopper"]["adjacent_pairs"]
        M["AllPolicyPairs"] = fmt(adjacent["n_all_pairs"])
        M["AdjacentPFloor"] = fmt(adjacent["minimum_attainable_p"], 6)
        M["AdjacentHolmCutoff"] = fmt(adjacent["first_holm_cutoff"], 6)
    tops = [(env, e["top1"]["prob"]) for env, e in rz["envs"].items() if "top1" in e]
    if tops:
        lo_env, lo_p = min(tops, key=lambda t: t[1])
        hi_env, hi_p = max(tops, key=lambda t: t[1])
        M["TopOneProbMin"] = fmt(lo_p, 2)
        M["TopOneProbMinEnv"] = disp(lo_env)
        M["TopOneProbMax"] = fmt(hi_p, 2)
        M["TopOneProbMaxEnv"] = disp(hi_env)
    jm = rz.get("joint_multiplicity")
    if jm:
        M["JointNTests"] = fmt(jm["n_tests"])
        M["JointUnchanged"] = fmt(jm["unchanged_from_per_family"])
    iccs = [(env, e["mechanism"]["seed_icc"]) for env, e in rz["envs"].items()
            if e.get("mechanism", {}).get("seed_icc") is not None]
    if iccs:
        lo_env, lo_v = min(iccs, key=lambda t: t[1])
        hi_env, hi_v = max(iccs, key=lambda t: t[1])
        M["IccMin"], M["IccMinEnv"] = fmt(lo_v, 2), disp(lo_env)
        M["IccMax"], M["IccMaxEnv"] = fmt(hi_v, 2), disp(hi_env)

    # ---------- the RecentOnly reversal, tested ----------
    rv = rz.get("reversal")
    if rv:
        M["RevAbove"] = ", ".join(disp(e) for e in rv["above"]) or "none"
        M["RevBelow"] = ", ".join(disp(e) for e in rv["below"]) or "none"
        M["RevSupported"] = fmt(rv["reversal_supported"])
        M["RevNTests"] = fmt(rv["n_tests"])
    for env, ev in rz["envs"].items():
        r = ev.get("recentonly_vs_others")
        if not r:
            continue
        E = macroname(env.capitalize())
        M[f"Ro{E}Mean"] = fmt(r["mean"], 1)
        M[f"Ro{E}D"] = fmt(r["cohen_d_paired"], 2)
        M[f"Ro{E}P"] = fmt(r["exact_sign_flip_p"], 4)
        M[f"Ro{E}Pos"] = fmt(r["sign_consistent_positive"])
        M[f"Ro{E}Rank"] = fmt(r["rank"])

    # ---------- budget accounting ----------
    spreads = []
    for env, b in rz["budget"].items():
        E = macroname(env.capitalize())
        M[f"WallSpread{E}"] = fmt(b["wallclock_spread"], 2)
        M[f"Cheapest{E}"] = b["cheapest"]
        M[f"BudgetExact{E}"] = fmt(b["updates_and_samples_identical"])
        if "recency_group_mean" in b:
            M[f"RecWall{E}"] = fmt(b["recency_group_mean"], 2)
            M[f"NonWall{E}"] = fmt(b["nonrecency_group_mean"], 2)
        spreads.append(b["wallclock_spread"])
    if spreads:
        M["WallSpreadMin"] = fmt(min(spreads), 2)
        M["WallSpreadMax"] = fmt(max(spreads), 2)

    # ---------- the null instrument ----------
    res = nl["results"]
    M["NTests"] = fmt(len(res))
    M["NRaw"] = fmt(sum(r["p_one_sided"] < nl["alpha"] for r in res))
    M["NBH"] = fmt(sum(r.get("bh_significant", False) for r in res))
    M["NClamped"] = fmt(sum(r["clamped_pair_count"] > 0 for r in res))
    M["NClampedPairs"] = fmt(sum(r["clamped_pair_count"] for r in res))
    M["NCellPairs"] = fmt(sum(len(r["pair_floors"]) for r in res))
    M["FullSplitCount"] = fmt(res[0]["split_count"])
    M["JackSplitCount"] = fmt(res[0]["jackknife_split_count"])
    M["MetricHalfSize"] = fmt(res[0]["half_size"])
    M["JackHalfSize"] = fmt(res[0]["jackknife_half_size"])
    M["Alpha"] = fmt(nl["alpha"], 2)
    M["ChanceExpected"] = fmt(nl["alpha"] * len(res), 1)
    surv = [r for r in res if r.get("bh_significant")]
    descriptions = []
    for r in surv:
        a, b = _FULL.get(r["metric_a"], r["metric_a"]), _FULL.get(r["metric_b"], r["metric_b"])
        descriptions.append(f"{disp(r['env'])}: {a} versus {b}, excess {r['excess']:+.3f} "
                            f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")
    M["MetricFlagDescriptions"] = "; ".join(descriptions) or "none"
    for i, r in enumerate(surv[:3], start=1):
        w = ["", "First", "Second", "Third"][i]
        M[f"Surv{w}Env"] = disp(r["env"])
        M[f"Surv{w}A"] = _FULL.get(r["metric_a"], r["metric_a"])
        M[f"Surv{w}B"] = _FULL.get(r["metric_b"], r["metric_b"])
        M[f"Surv{w}Excess"] = fmt(r["excess"], 3)
        M[f"Surv{w}Lo"] = fmt(r["ci"][0], 3)
        M[f"Surv{w}Hi"] = fmt(r["ci"][1], 3)
    # ---------- redundancy sensitivity ----------
    # improvement_per_1k_updates is a within-seed affine transform of
    # return_mean under the equalised budget, so its cells duplicate the
    # return cells. This prices that duplication instead of asserting it.
    if "drop_metric_sensitivity" in nl:
        ds = nl["drop_metric_sensitivity"]
        M["DropMetric"] = _FULL.get(ds["dropped"], ds["dropped"])
        M["NDistinct"] = fmt(ds["n_tests"])
        M["NBHDrop"] = fmt(ds["n_bh"])
        dsurv = [r for r in ds["results"] if r.get("bh_significant")]
        M["DropFlagDescriptions"] = ", ".join(
            f"{disp(r['env'])} ({_FULL.get(r['metric_a'], r['metric_a'])} versus "
            f"{_FULL.get(r['metric_b'], r['metric_b'])})" for r in dsurv) or "none"
        if dsurv:
            M["DropSurvEnv"] = disp(dsurv[0]["env"])
            M["DropSurvA"] = _FULL.get(dsurv[0]["metric_a"], dsurv[0]["metric_a"])
            M["DropSurvB"] = _FULL.get(dsurv[0]["metric_b"], dsurv[0]["metric_b"])

    # every cell, for the appendix table
    for r in res:
        k = macroname(r["env"].capitalize()) + _SHORT.get(r["metric_a"], "") + _SHORT.get(r["metric_b"], "")
        M[f"Cross{k}"] = fmt(r["cross"], 3)
        M[f"Floor{k}"] = fmt(r["floor"], 3)
        M[f"Excess{k}"] = fmt(r["excess"], 3)
        M[f"ExLo{k}"] = fmt(r["ci"][0], 3)
        M[f"ExHi{k}"] = fmt(r["ci"][1], 3)

    # ---------- power ----------
    if "power" in nl:
        env, rows = next(iter(nl["power"].items()))
        M["PowerEnv"] = disp(env)
        for row in rows:
            k = macroname(f"{row['true_pair_disagreement']:.3f}".replace("0.", "D"))
            M[f"Power{k}"] = fmt(row["detection_rate"], 2)
            M[f"PowerSep{k}"] = fmt(row["true_pair_disagreement"], 3)
        M["PowerFullReversal"] = fmt(max(rows, key=lambda r: r["true_pair_disagreement"])["detection_rate"], 2)
        M["PowerDraws"] = fmt(rows[0]["evaluation_draws"])
        M["PowerNoise"] = fmt(rows[0]["noise_sd"], 2)
        # the separation at which detection first becomes reliable, and the
        # largest separation that is still effectively invisible
        rel = [r for r in rows if r["detection_rate"] >= 0.8]
        if rel:
            M["PowerReliableAt"] = fmt(min(r["true_pair_disagreement"] for r in rel), 2)
        weak = [r for r in rows if r["detection_rate"] < 0.5 and r["true_pair_disagreement"] > 0]
        if weak:
            w = max(weak, key=lambda r: r["true_pair_disagreement"])
            M["PowerWeakSep"] = fmt(w["true_pair_disagreement"], 2)
            M["PowerWeakRate"] = fmt(w["detection_rate"], 2)
        mx = max(res, key=lambda r: r["excess"])
        M["LargestObservedExcess"] = fmt(mx["excess"], 3)
        M["PowerFalsePos"] = fmt(min(rows, key=lambda r: r["true_pair_disagreement"])["detection_rate"], 2)

    # ---------- noise floor ----------
    if "floor_sweep" in nl:
        env, fl = next(iter(nl["floor_sweep"].items()))
        M["FloorEnv"] = disp(env)
        for metric, d in fl.items():
            k = _SHORT.get(metric, macroname(metric))
            M[f"FloorHalf{k}"] = fmt(max(d["by_half_size"].values()), 3)
            M[f"FloorFull{k}"] = fmt(d["extrapolated_to_full_n"], 3)
        worst = max(fl.items(), key=lambda kv: kv[1]["extrapolated_to_full_n"])
        M["FloorWorstMetric"] = _FULL.get(worst[0], worst[0])
        M["FloorWorstValue"] = fmt(worst[1]["extrapolated_to_full_n"], 3)

    # ---------- nonsmooth estimator calibration ----------
    M["CalibrationDatasets"] = fmt(calibration["cases"][0]["datasets"])
    M["CalibrationCases"] = fmt(len(calibration["cases"]))
    M["CalibrationPolicies"] = fmt(calibration["policies"])
    M["CalibrationSeeds"] = fmt(calibration["seeds"])
    M["CalibrationHalf"] = fmt(calibration["half_size"])
    labels = ["Equal noise, concordant", "Heterogeneous, concordant",
              "Heterogeneous, unequal", "Heterogeneous, reversal"]
    for tag, row in zip(("Equal", "Hetero", "Unequal", "Reversal"), calibration["cases"]):
        M[f"CalibrationCoverage{tag}"] = fmt(row["interval_coverage"], 3)
        M[f"CalibrationRejection{tag}"] = fmt(row["one_sided_rejection_rate"], 3)
    cal_lines = [r"\begin{tabular}{lrrr}", r"\toprule",
                 r"Model & Bias & Coverage [MC CI] & Rejection [MC CI] \\", r"\midrule"]
    for label, row in zip(labels, calibration["cases"]):
        k = macroname(row["name"])
        M[f"CalCov{k}"] = fmt(row["interval_coverage"], 3)
        M[f"CalReject{k}"] = fmt(row["one_sided_rejection_rate"], 3)
        M[f"CalBias{k}"] = fmt(row["bias"], 3)
        cov, reject = row["coverage_mc_interval"], row["rejection_mc_interval"]
        cal_lines.append(f"{label} & {row['bias']:+.3f} & {row['interval_coverage']:.3f} "
                         f"[{cov[0]:.3f}, {cov[1]:.3f}] & {row['one_sided_rejection_rate']:.3f} "
                         f"[{reject[0]:.3f}, {reject[1]:.3f}] \\\\")
    reversal = calibration["cases"][-1]
    M["CalibrationPopulationDraws"] = fmt(reversal["population_target"]["draws"])
    M["CalibrationPopulationTarget"] = fmt(reversal["population_target"]["excess"], 4)
    M["CalibrationPopulationSE"] = fmt(reversal["population_target"]["monte_carlo_se"], 4)
    unequal = calibration["cases"][2]
    equal_sigma = calibration["cases"][0]["sigma_a"][0]
    M["CalibrationSigma"] = fmt(equal_sigma, 2)
    M["CalibrationSigmaLow"] = fmt(min(unequal["sigma_a"]) / equal_sigma, 2)
    M["CalibrationSigmaHigh"] = fmt(max(unequal["sigma_a"]) / equal_sigma, 1)
    M["CalibrationNoiseRatio"] = fmt(unequal["sigma_b"][-1] / unequal["sigma_a"][0], 1)
    cal_lines.extend([r"\bottomrule", r"\end{tabular}"])
    Path(args.output).with_name("table_calibration.tex").write_text("\n".join(cal_lines) + "\n")

    # ---------- declared training settings, checked against observed rows ----------
    common = training["envs"]["hopper"]
    for key in ("hidden_sizes", "gamma", "tau", "actor_lr", "critic_lr", "autotune",
                "batch_size", "learning_starts", "policy_frequency", "target_network_frequency"):
        if any(e["model"][key] != common["model"][key] for e in training["envs"].values()):
            raise ValueError(f"Training table assumes common model setting {key}")
    model = common["model"]
    M["HiddenLayers"] = fmt(len(model["hidden_sizes"]))
    M["HiddenWidth"] = fmt(model["hidden_sizes"][0])
    M["ActorLR"] = fmt(model["actor_lr"], 4)
    M["CriticLR"] = fmt(model["critic_lr"], 4)
    M["TrainGamma"] = fmt(model["gamma"], 2)
    M["TrainTau"] = fmt(model["tau"], 3)
    M["TrainBatch"] = fmt(model["batch_size"])
    M["LearningStarts"] = fmt(model["learning_starts"])
    M["ActorFrequency"] = fmt(model["policy_frequency"])
    M["TargetFrequency"] = fmt(model["target_network_frequency"])
    M["EvaluationEpisodes"] = fmt(common["eval"]["eval_episodes"])
    M["EvalSeedOffset"] = fmt(common["eval"]["eval_seed_offset"])
    train_lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
                   r"Environment & Seeds & Pretrain/deploy & Buffer/recent & Window/updates & Eval interval \\",
                   r"\midrule"]
    for env, e in training["envs"].items():
        k = macroname(env.capitalize())
        M[f"TrainingSeedMin{k}"] = fmt(min(e["seeds"]))
        M[f"TrainingSeedMax{k}"] = fmt(max(e["seeds"]))
        M[f"ObservedUpdates{k}"] = fmt(e["observed_updates"])
        train_lines.append(f"{disp(env)} & {len(e['seeds'])} & {e['train']['pretrain_steps']:,}/{e['deploy_steps']:,} & "
                           f"{e['model']['buffer_size']:,}/{e['model']['recent_buffer_size']:,} & "
                           f"{e['budget']['window_steps']:,}/{e['budget']['max_updates_per_window']:,} & "
                           f"{e['eval']['eval_interval_steps']:,} \\\\")
    train_lines.extend([r"\bottomrule", r"\end{tabular}"])
    Path(args.output).with_name("table_training.tex").write_text("\n".join(train_lines) + "\n")

    # The 24-cell appendix table is generated rather than hand-written, so it
    # cannot fall out of step with the artifact. The whole tabular lives in the
    # fragment: \input inside an alignment breaks at the file boundary.
    pretty = {"Ret": "return", "Rec": "recovery", "Forg": "forgetting",
              "Imp": "improvement/1k", "Cost": "update cost"}
    lines_tex = [
        "% Generated by scripts/export_paper_numbers.py -- do not edit by hand.",
        r"\begin{tabular}{llrrrlc}",
        r"\toprule",
        r"Environment & Metric pair & cross & floor & excess & nominal 95\% CI & \\",
        r"\midrule",
    ]
    for r in sorted(res, key=lambda r: (r["env"], r["metric_a"], r["metric_b"])):
        a = pretty.get(_SHORT.get(r["metric_a"], ""), r["metric_a"])
        b = pretty.get(_SHORT.get(r["metric_b"], ""), r["metric_b"])
        star = r"$\star$" if r.get("bh_significant") else ""
        lines_tex.append(
            f"{r['env']} & {a} vs.\\ {b} & {r['cross']:.3f} & {r['floor']:.3f} & "
            f"{r['excess']:+.3f} & [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] & {star} \\\\")
    lines_tex += [r"\bottomrule", r"\end{tabular}"]
    Path(args.output).with_name("table_cells.tex").write_text("\n".join(lines_tex) + "\n")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by scripts/export_paper_numbers.py -- do not edit by hand.",
             "% Every numeric claim in the manuscript resolves through these macros.", ""]
    for k in sorted(M):
        lines.append(f"\\newcommand{{\\{macroname(k)}}}{{{M[k]}}}")
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out} with {len(M)} macros")


if __name__ == "__main__":
    main()
