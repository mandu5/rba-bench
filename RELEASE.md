# What is in this repository, and what is not

The working tree this was cut from is about 18 GB. Almost none of that is needed
to check the paper, so this repository tracks roughly 1 MB: the code, the
configs, and the ~180 KB of merged per-seed results the analysis actually reads.

## Reproducing the paper

```bash
make            # analysis -> numbers -> figures -> paper/main.pdf
make verify     # asserts the manuscript's numbers still match the artifacts
```

No GPU, training or environment simulator is needed for the paper rebuild.
Analysis uses the CSVs below; model calibration uses deterministic simulation,
and the training-setup audit reads the declared configs and observed CSV grids.
Manuscript results and added training settings use generated `paper/numbers.tex`
and table fragments. Rebuild consistency checks establish agreement with the
supplied artifacts, not independent scientific validity or retraining equality.

## What is tracked

| Path | Why |
| --- | --- |
| `src/`, `configs/` | the experiment code and the run configurations |
| `scripts/` | the analysis, the number export, the figures |
| `tests/` | unit tests |
| `results/merged/*_phase_g_10seed_merged/*_rows.csv` | per-seed metrics, 4 MuJoCo environments x 9 policies x 10 seeds |
| `results/merged/*pcgym*/*_rows.csv` | the same for the process-control case study, 5 seeds |
| `results/merged/*/*_budget_audit.csv` | the same-budget audit: update counts, replay samples, wall-clock |
| `results/*.json` | rank audit, metric floor, model calibration and training-setup audit |
| `paper/` | manuscript source, generated numbers, figures, and the built PDF |
| `docs/abstract.txt`, `docs/tldr.txt` | generated OpenReview text |
| `docs/camera_ready_changes.md` | review-to-change mapping |

`results/` is ignored by default and these files are tracked with `git add -f`.
That is deliberate: it means a new run cannot silently enlarge the repository,
and the set of inputs the paper depends on stays an explicit, reviewable list.

## What is not tracked, and why

| Excluded | Size | Reason |
| --- | --- | --- |
| `RL_checkpoints/` | 9.3 GB | model weights; outputs of training, not inputs to the analysis |
| `artifacts/` | 8.3 GB | raw per-run logs; the merged CSVs are the reduction of these |
| `*.zip` | ~640 MB | dated snapshots of earlier states of this same tree |
| `site/` | 28 MB | generated static site |
| `results/` (rest) | ~243 MB | sweeps, plots and smoke runs not referenced by the paper |

None of it is required to check a claim in the paper. If a reviewer wants the
raw per-run logs behind the merged CSVs, they should be published separately
with a DOI rather than carried in git history, where they cannot be removed.

## Caveat on the excluded data

The merged CSVs are a reduction, and this repository does not let you verify the
reduction itself — only everything downstream of it. The merge code is in `src/`
and the raw logs exist, so the step is reproducible in principle, but a reader
who wants to audit it needs the archive rather than this repository.

The clean source export excludes local git history, internal planning and
submission records, application documents, credentials, raw logs and checkpoints.
Software is MIT-licensed; the submitted manuscript uses CC BY 4.0. The vendored
NeurIPS template retains its original attribution.
