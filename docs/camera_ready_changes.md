# RBA-Bench camera-ready revision, October 4, 2026

The scientific revision addresses both official reviews for CL4FMAgents
submission #77. Reviewer Bnsn's review is
[ov4HF8w0Fb](https://openreview.net/forum?id=haLv7d3OFU&noteId=ov4HF8w0Fb);
Reviewer 4KEN's review is
[l73szlBL8z](https://openreview.net/forum?id=haLv7d3OFU&noteId=l73szlBL8z).
The reviews were read from the authenticated author view on October 4.
No review text was inferred from ratings.

## Point-by-point changes

| Review request | Revision | Evidence and location |
| --- | --- | --- |
| Bnsn: specify pair-specific floor versus inversion of averaged self-disagreement | Corrected the submitted pooled inversion. Estimate each policy pair's self-rate over splits, invert that pair's rate, compose its concordant floor, then average floors across pairs. Added the indexed estimator equation and assumptions. | Main §5; `scripts/metric_disagreement_null.py::pairwise_split_estimate`; pair-level rates, floors and policy-pair labels in `results/metric_disagreement_null.json`. |
| Bnsn: calibrate nonsmooth ten-seed inference under heterogeneous pairwise noise | Added four fixed Gaussian cases with 400 fresh datasets each. Concordant-null targets are analytically zero, so coverage is not checked against a biased sampling mean. The reversal uses a separately seeded 100,000-half-sample population target. Reported bias, type-I/power, coverage and Wilson Monte Carlo intervals. Wald/BH flags remain nominal exploratory diagnostics. | Main §5 and Limitations; Appendix B; `scripts/validate_inference.py`; `results/inference_calibration.json`; generated `paper/table_calibration.tex`. |
| Bnsn: account for adjacent pairs selected after sorting observed ranking | Test all 28 policy pairs per environment, apply Holm familywise correction, then inspect selected adjacencies. Retained selected-only BH solely as exploratory. Explicitly explain why the exact p-value grid prevents rejection at the current seed counts. | Main §3–4, abstract and conclusion; `scripts/triage_rank_resolvability.py::adjacent_pair_audit`; `results/rank_resolvability.json`. |
| Bnsn: do not turn Hopper's single uncorrected power into a uniform exclusion bound | Removed benchmark-wide gross-disagreement exclusion claims from abstract, introduction, results, limitations, conclusion and README. Labelled the curve as a Hopper-derived homoscedastic Gaussian model for one uncorrected test. The zero-separation point now uses independent evaluation draws. | Main §6 and Figure 2; `scripts/metric_disagreement_null.py::run_power`; `scripts/make_figures.py`. |
| 4KEN: improve training reproducibility | Added architecture, optimiser, learning rates, entropy tuning, warmup, update frequencies, replay capacities, lengths, window budgets, evaluation schedule and exact seed sets. Audit observed CSV seed grids and update/sample counts against declared configs. Documented the `--v21` sweep command and analysis-only rebuild. | Main §2; Appendix C and Table 4; README; `scripts/export_training_setup.py`; `results/training_setup.json`. |
| 4KEN: make five-environment/five-run scope explicit | Four locomotion tasks have ten runs per policy; the reactor has five. The metric-floor table uses only the four locomotion tasks. Half-samples and delete-one partitions are subsets of existing runs, not additional training replicates. Generalisation and retraining limits are explicit. | Abstract, Setup, Limitations, Appendix C and README. |

## Changed outputs versus the submitted version

| Quantity | Submitted analysis | Revised analysis |
| --- | --- | --- |
| Metric floor estimator | Invert self-rates pooled across policy pairs | Invert each policy pair's self-rates, then average its floor |
| Ten-seed split integration | Random splits; random delete-one splits | All 252 ordered balanced partitions; all 630 disjoint delete-one partitions |
| Adjacent steps | 2/35 selected-only BH flags, presented as resolved comparisons | 0/35 all-pair Holm rejections; 2/35 selected-only BH flags explicitly exploratory |
| Nominal metric flags before BH | 3/24 | 5/24 |
| Nominal metric BH flags | 2/24 | 3/24 |
| Removing the related improvement metric | 1/12 | 2/12 |
| Hopper-model full-reversal detection | 0.99 | 1.00 |
| Hopper-model detection at true separation 0.214 | 0.31 | 0.27 |
| Hopper-model zero-separation detection | Calibration draws reused, 0.05 by construction | Independent evaluation draws, 0.06 |
| Seed/run scope and raw metrics | Four ten-seed locomotion datasets and one five-seed reactor dataset | Unchanged; no new RL training |

The revised metric BH flags are Hopper return versus forgetting (+0.267),
Hopper forgetting versus improvement (+0.279), and HalfCheetah recovery versus
forgetting (+0.090). Removing improvement leaves the Hopper return/forgetting
and HalfCheetah recovery/forgetting flags. They are candidate differences under
nominal, model-dependent inference. Across-seed IQM trimming means the related
return and improvement metrics need not have identical aggregate rankings.

At ten seeds the minimum two-sided sign-flip p-value is 2/1024 = 0.001953125,
above the first Holm cutoff 0.05/28 = 0.001785714. Thus the formal 0/35 count
is a limitation of this testing procedure, not evidence that every comparison
is equal or fundamentally undetectable. Coarse contrast BH 2/5 and Holm 1/5,
selection-aware extremes BH 3/5, reversal findings and budget inputs are
unchanged.

## Calibration findings and limits

| Gaussian model | Type-I error or power | Nominal 95% coverage | Coverage Monte Carlo 95% interval |
| --- | --- | --- | --- |
| Equal noise, concordant | Type-I 0.005 | 0.9775 | [0.9578, 0.9881] |
| Policy-specific heterogeneous, concordant | Type-I 0.030 | 0.960 | [0.9360, 0.9752] |
| Heterogeneous with unequal metric noise, concordant | Type-I 0.050 | 0.960 | [0.9360, 0.9752] |
| Heterogeneous partial reversal | Power 0.390 | 0.930 | [0.9007, 0.9511] |

The reversal's empirical coverage is below 0.95. Each case has only 400 datasets
and therefore material Monte Carlo uncertainty. Observing a largest null
rejection rate of 0.05 does not prove universal calibration. The model assumes
independent Gaussian observations across policy, metric and seed, and does not
exhaust dependence or heterogeneity in actual environments. Every calibration
dataset and every benchmark cell has at least one pair requiring finite-sample
clipping. The paper discloses clipping, model assumptions and inferential limits;
it does not claim distribution-free jackknife validity, arbitrary-dependence FDR
control or a uniform exclusion bound.

## Reproduction and format

Numerical results and added training settings are emitted into
`paper/numbers.tex` and generated table fragments. The OpenReview abstract and
TLDR are generated from resolved macros at `docs/abstract.txt` and
`docs/tldr.txt`, with 1,620 and 199 characters respectively. README documents
the dependency setup, source build and training scope.

The named final-mode manuscript retains Youngmin Ko, The Pennsylvania State
University, and the footnote “Independent research. Current affiliation:
KRAFTON.” No unverified email is included. Software is MIT; the manuscript's
OpenReview license is CC BY 4.0.

The original submitted PDF and the October 4 formatting-only draft remain
preserved separately. Figure bytes require three distinct comparisons:
the original pre-preparation baseline `00b8664`, the formatting draft
`426052e`, and this scientific revision. The mechanism figure is byte-identical
to `426052e` when regenerated with the fixed source epoch, but differs from
`00b8664`. The power figure changes both content and bytes for this revision.
A byte-identical export rebuild compares the revised artifacts with the
independently exported source, not with the original submission.

The clean export includes code, configs, tests, tracked merged CSVs, derived
JSON, manuscript, generated text and this review map. It excludes local git
history, internal planning/submission records, application materials, raw review
captures, raw logs, checkpoints and credentials. Source repository:
https://github.com/mandu5/rba-bench.

Verified package hashes, source-file list, build logs, page/font checks and
the independent rebuild comparison are supplied in the separate camera-ready
artifact manifest. Rebuild success establishes snapshot consistency; it does
not verify the reduction from unavailable raw-run logs or bitwise retraining.
