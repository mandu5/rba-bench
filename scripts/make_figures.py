"""Generate the manuscript figures directly from the committed JSON artifacts.

Same discipline as the numbers: a figure is never drawn by hand, so it cannot
disagree with the analysis it illustrates. Rerun the analysis scripts, rerun
this, and the figures update.

Figures (named by content, not number, so LaTeX ordering cannot contradict
the filenames)
  fig_power.pdf      Detection in one Hopper-derived Gaussian model, using a
                     single uncorrected test; not a bound on other table cells.
  fig_mechanism.pdf  Why the paired contrast resolves and the ranking does not,
                     in common effect-size units so environments with very
                     different return scales sit on one axis. Filled circles
                     mark the contrasts that survive the exact test + BH.

Usage:
    python3 scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESOLV = ROOT / "results" / "rank_resolvability.json"
NULLJS = ROOT / "results" / "metric_disagreement_null.json"
OUT = ROOT / "paper" / "figures"

# Validated categorical slots 1 and 2 (protan dE 24.7, normal dE 33.6 against
# each other). Identity is additionally carried by marker shape, so the figures
# survive greyscale printing.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # Type 3 is matplotlib's default and is what the NeurIPS instructions
    # tell authors not to ship; 42 embeds TrueType instead.
    "pdf.fonttype": 42,
})


# The manuscript and its captions spell the environments the usual way, so the
# axis labels have to as well; the JSON keys are internal slugs.
_DISPLAY = {"ant": "Ant", "halfcheetah": "HalfCheetah", "hopper": "Hopper",
            "walker2d": "Walker2d", "pcgym": "PC-Gym"}


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def fig_power(nl: dict) -> None:
    env, rows = next(iter(nl["power"].items()))
    rows = sorted(rows, key=lambda r: r["true_pair_disagreement"])
    x = [r["true_pair_disagreement"] for r in rows]
    y = [r["detection_rate"] for r in rows]
    fpr = y[0]

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.axhline(fpr, color=MUTED, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    # lower right is empty: the curve is above it and the axis labels are below
    ax.text(1.05, fpr + 0.035, f"false-positive rate {fpr:.2f}",
            ha="right", va="bottom", color=MUTED, fontsize=7)

    ax.plot(x, y, color=BLUE, linewidth=2.0, marker="o", markersize=5,
            markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=0.8, zorder=3)

    ax.annotate(f"{y[-1]:.2f}", xy=(x[-1], y[-1]), xytext=(-4, -4),
                textcoords="offset points", ha="right", va="top", color=INK, fontsize=7.5)

    # shade up to the first measured point with reliable (>= 0.8) detection,
    # the same quantity the text quotes as "becomes reliable only around";
    # annotation sits clear of the y-axis spine
    rel = [r["true_pair_disagreement"] for r in rows if r["detection_rate"] >= 0.8]
    weak = [r["true_pair_disagreement"] for r in rows
            if r["detection_rate"] < 0.8 and r["true_pair_disagreement"] > 0]
    if weak:
        edge = min(rel) if rel else max(weak)
        ax.axvspan(-0.03, edge, color=MUTED, alpha=0.07, zorder=0)
        ax.annotate("not reliably resolvable\nat ten seeds", xy=(0.03, 0.97),
                    ha="left", va="top", color=MUTED, fontsize=7)
    ax.annotate("full reversal detected\nin this model",
                xy=(x[-1] - 0.02, y[-1] - 0.10), color=INK, fontsize=7,
                va="top", ha="right")

    ax.set_xlabel("true fraction of pairs ordered differently")
    ax.set_ylabel("detection rate")
    ax.set_title(f"{_DISPLAY.get(env, env)} model; single uncorrected test", fontsize=8)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlim(-0.03, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    _despine(ax)
    fig.savefig(OUT / "fig_power.pdf")
    plt.close(fig)


def fig_mechanism(rz: dict) -> None:
    """Paired contrast effect size against a typical single-pair effect size.

    Both are paired Cohen d, so environments whose returns differ by an order of
    magnitude share one axis. The contrast interval is converted to the same
    units from the reported mean, interval and d. Filled circles are the
    contrasts that survive the exact test + BH; open circles are the
    uncorrected-only ones, so the figure cannot overstate the corrected count.
    """
    survivors = set(rz.get("contrast_multiplicity", {}).get("bh_survivors", []))
    envs, contrast_d, lo_d, hi_d, pair_d = [], [], [], [], []
    for env, e in rz["envs"].items():
        c, m = e["recency_contrast"], e.get("mechanism", {})
        if not c.get("cohen_d_paired") or not m.get("median_pair_cohen_d"):
            continue
        d, mean = c["cohen_d_paired"], c["mean"]
        scale = d / mean if mean else 0.0     # d per raw unit
        envs.append(env)
        contrast_d.append(d)
        lo_d.append(c["ci"][0] * scale)
        hi_d.append(c["ci"][1] * scale)
        pair_d.append(m["median_pair_cohen_d"])

    order = sorted(range(len(envs)), key=lambda i: contrast_d[i])
    ypos = range(len(order))

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)

    for y, i in zip(ypos, order):
        ax.plot([lo_d[i], hi_d[i]], [y, y], color=BLUE, linewidth=2.0,
                solid_capstyle="round", zorder=3)
    kept = [(y, i) for y, i in zip(ypos, order) if envs[i] in survivors]
    rest = [(y, i) for y, i in zip(ypos, order) if envs[i] not in survivors]
    ax.scatter([contrast_d[i] for _, i in kept], [y for y, _ in kept], s=34,
               color=BLUE, marker="o", edgecolor="white", linewidth=0.8,
               zorder=4, label="recency contrast (filled: survives BH)")
    ax.scatter([contrast_d[i] for _, i in rest], [y for y, _ in rest], s=34,
               facecolor="white", marker="o", edgecolor=BLUE, linewidth=1.2,
               zorder=4)
    # diamonds sit off the row centreline so a near-equal pair d (Ant) cannot
    # cover the contrast marker
    ax.scatter([pair_d[i] for i in order], [y + 0.22 for y in ypos], s=34,
               color=ORANGE, marker="D", edgecolor="white", linewidth=0.8,
               zorder=4, label="median single policy pair")

    ax.set_yticks(list(ypos))
    ax.set_yticklabels([_DISPLAY.get(envs[i], envs[i]) for i in order])
    ax.set_xlabel("paired effect size (Cohen $d$)")
    span = max(hi_d) - min(min(lo_d), 0.0)
    ax.set_xlim(min(min(lo_d), 0.0) - 0.04 * span, max(hi_d) + 0.06 * span)
    # above the axes: "lower right" sat on top of the smallest environment
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, handletextpad=0.4, borderpad=0.2, columnspacing=1.2)
    _despine(ax)
    fig.savefig(OUT / "fig_mechanism.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rz = json.loads(RESOLV.read_text())
    nl = json.loads(NULLJS.read_text())
    if "power" in nl:
        fig_power(nl)
        print(f"wrote {OUT / 'fig_power.pdf'}")
    else:
        print("no power block in the artifact; run metric_disagreement_null.py --power")
    fig_mechanism(rz)
    print(f"wrote {OUT / 'fig_mechanism.pdf'}")


if __name__ == "__main__":
    main()
