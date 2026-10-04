# What Ten Seeds Can and Cannot Resolve About Replay Strategy Selection Under Distribution Shift

Code, configurations, per-seed results and manuscript for RBA-Bench, a
fixed-budget comparison of replay strategies under distribution shift.
Accepted at the NeurIPS 2026 Workshop on Continual Learning in the Era of
Foundation Models and Embodied Agents (CL4FMAgents), non-archival.

Source: https://github.com/mandu5/rba-bench

Paper record: https://openreview.net/forum?id=haLv7d3OFU

Camera-ready deadline: October 10, 2026; the workshop does not list a time or timezone.

## Scope and findings

Eight adaptive policies plus Frozen, four MuJoCo tasks with ten seeds each,
and a process-control reactor with five seeds. Deployment gradient updates
and replay samples match exactly across adaptive policies.

- Selection-aware within-seed label permutations separate best from worst in
  3/5 environments after BH. The modal winner's bootstrap rank-one probability
  can be as low as 0.47.
- Adjacent pairs are selected by sorting observed IQMs. We test all 28 pairs
  per environment and apply Holm before inspecting adjacencies. The resulting
  0/35 count is forced by the exact two-sided p-value grid at these seed counts,
  not evidence that every comparison is equal. Selected-pair-only BH yields
  2/35 exploratory flags and does not account for selection.
- A pair-specific concordant metric-disagreement floor is obtained by inverting
  each policy pair's self-disagreement before averaging. The revised analysis
  gives 3/24 nominal BH flags; removing the related improvement metric gives
  2/12. Wald intervals and BH flags remain exploratory and model-dependent.
- Four predeclared Gaussian calibration cases use 400 fresh datasets each.
  Concordant-null type-I rates are 0.005, 0.030 and 0.050. Nominal 95% coverage
  is 0.978, 0.960, 0.960 and 0.930; the partial-reversal case undercovers the
  target in this simulation. Monte Carlo intervals are reported, and the
  cases do not validate inference for all benchmark environments.
- Hopper-derived simulated power covers one uncorrected Gaussian-model test.
  It supplies no uniform exclusion bound for unflagged cells in other
  environments.
- The paired recency-group contrast survives exact sign-flip plus BH in 2/5
  environments (1/5 under Holm). Purely recent replay reverses its advantage
  between HalfCheetah and Ant. Matched update/sample budgets still leave
  wall-clock spread of 1.21–1.45× in this implementation.

The manuscript and generated JSON are the complete account.
`docs/camera_ready_changes.md` maps the official reviews to the corrections.

## Reproduce the analysis and paper

Use Python 3.11 or 3.12 and Tectonic on `PATH`. No GPU, simulator or training
is needed for analysis, calibration, figures or the PDF:

```bash
python3 -m pip install -e .
python3 -m pip install -r requirements-analysis.txt pytest
make
make verify
```

For an uninstalled checkout with the dependencies already present, use
`PYTHONPATH=src make` and `PYTHONPATH=src make verify`.

`make` regenerates the rank audit, metric floor, four-case calibration,
training-setup audit, numerical macros, OpenReview text, figures and
`paper/main.pdf`. All numerical manuscript results and added training settings
are exported through `paper/numbers.tex` and generated table fragments.
`make verify` runs tests and fails if regenerated artifacts differ from the
git baseline. The regression cases check nonlinear pair-specific inversion,
selection correction, the fixed population calibration target and observed
seed/budget scope.

A ZIP export has no git history. Before using `make verify` there, initialise
a local baseline with `git init` and `git add -f .`. This checks reproduction
of the supplied snapshot; it does not independently establish scientific
validity. The verified build uses pinned NumPy/Matplotlib versions from
`requirements-analysis.txt`. Training dependencies are listed in
`pyproject.toml`.

## Training setup

The paper appendix and `results/training_setup.json` report configurations
and audit their seed grids and deployment update/sample counts against the
tracked CSVs. Actor and critics use two 256-unit ReLU layers, Adam learning
rates 0.0003, discount 0.99, target mixing 0.005, automatic entropy tuning,
batch size 256 and a 5,000-step random-action warmup. Policies within a seed
share the pretrained agent and replay state.

| Environment | Config | Seeds | Pretrain / deployment steps |
| --- | --- | --- | --- |
| Hopper-v5 | `configs/train/hopper_v5_v21_main.yaml` | 21–30 | 50,000 / 70,000 |
| Walker2d-v5 | `configs/train/walker2d_v5_v21_main.yaml` | 31–40 | 60,000 / 90,000 |
| HalfCheetah-v5 | `configs/train/halfcheetah_v5_v21_main.yaml` | 41–50 | 60,000 / 90,000 |
| Ant-v5 | `configs/train/ant_v5_v21_main.yaml` | 51–60 | 60,000 / 90,000 |
| PC-Gym CSTR | `configs/train/pcgym_case_study.yaml` | 61–65 | 40,000 / 40,000 |

To run a training sweep after installing the simulator dependencies, use the
v2.1 policy roster and the exact seed set. For example:

```bash
python3 -m rl_research.cli sweep-seeds \
  --config configs/train/hopper_v5_v21_main.yaml \
  --seeds 21 22 23 24 25 26 27 28 29 30 --v21 --device cuda
```

Select the config and seed set above for the other environments; choose
`cpu`, `mps` or `cuda` to match available hardware. This can require substantial
compute. Reportable runs used a mixture of GPU sessions and local hardware;
the source/CSV audit does not claim bitwise reproducibility of retraining.
The reactor has only five training runs per policy and is a small case study.

The raw-run path is `sweep-seeds` → `merge-seed-sweeps` / `phase-g-intake`
→ merged per-seed CSVs. The released CSVs are sufficient for the paper rebuild.
Raw logs and checkpoints are not included; `RELEASE.md` lists the release scope.

## Repository map

| Path | Content |
| --- | --- |
| `paper/` | manuscript, generated numbers/tables, figures and PDF |
| `scripts/` | analysis, calibration, setup audit, number/text exporters and figure builder |
| `results/merged/*/*_rows.csv` | tracked per-seed metrics used by analyses |
| `results/*.json` | reproducible derived analysis artifacts |
| `src/rl_research/` | learner, environments, drift, replay, policies and experiment runner |
| `configs/train/` | experiment configurations |
| `tests/` | unit and scientific-regression checks |
| `docs/abstract.txt`, `docs/tldr.txt` | generated OpenReview form text |
| `docs/camera_ready_changes.md` | review-to-change mapping |

Software license: MIT. The OpenReview manuscript submission uses CC BY 4.0.
The vendored NeurIPS template retains its original attribution.
