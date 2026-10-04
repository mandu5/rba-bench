"""Command line entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_research.config import dump_experiment_config, load_experiment_config
from rl_research.evaluation.plots import plot_sweep_summary
from rl_research.experiments.artifact import build_artifact_page
from rl_research.experiments.budget_sweep import apply_budget_multiplier, budget_metadata_payload, positive_budget_multiplier
from rl_research.experiments.cross_env import compare_environments
from rl_research.experiments.merge import merge_seed_sweep_inputs
from rl_research.experiments.phase_g import intake_phase_g_result
from rl_research.experiments.runner import run_experiment
from rl_research.experiments.sweep import DEFAULT_POLICIES, DEFAULT_POLICIES_V21, run_policy_sweep, run_seed_sweep


def _add_execution_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        default=None,
        help="Override train.device, e.g. cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override config.output_dir without editing the YAML file.",
    )
    parser.add_argument(
        "--budget-multiplier",
        type=positive_budget_multiplier,
        default=1.0,
        help="Scale budget.max_updates_per_window for controlled budget sweeps.",
    )


def _apply_execution_overrides(config, args: argparse.Namespace) -> dict[str, object]:
    device = getattr(args, "device", None)
    output_dir = getattr(args, "output_dir", None)
    budget_multiplier = getattr(args, "budget_multiplier", 1.0)
    if device is not None:
        config.train.device = str(device)
    if output_dir is not None:
        config.output_dir = str(output_dir)
    return apply_budget_multiplier(config, budget_multiplier)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rba-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Validate and print a config.")
    validate.add_argument("--config", required=True)

    show = subparsers.add_parser("show-config", help="Print a config as JSON.")
    show.add_argument("--config", required=True)

    run = subparsers.add_parser("run", help="Run an experiment.")
    run.add_argument("--config", required=True)
    run.add_argument("--name", default=None)
    run.add_argument("--override-total-steps", type=int, default=None)
    run.add_argument("--dry-run", action="store_true")
    _add_execution_overrides(run)

    smoke = subparsers.add_parser("smoke", help="Run a short smoke test.")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--steps", type=int, default=256)
    _add_execution_overrides(smoke)

    sweep = subparsers.add_parser("sweep-policies", help="Run same-budget baseline policy comparisons.")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--policies", nargs="*", default=DEFAULT_POLICIES)
    sweep.add_argument("--v21", action="store_true",
                       help="Use the v2.1 default policy roster (Uniform/PER/Budgeted-{TD,Grad,Novelty}).")
    sweep.add_argument("--override-total-steps", type=int, default=None)
    sweep.add_argument("--dry-run", action="store_true")
    _add_execution_overrides(sweep)

    seed_sweep = subparsers.add_parser("sweep-seeds", help="Run policy sweeps across multiple seeds.")
    seed_sweep.add_argument("--config", required=True)
    seed_sweep.add_argument("--seeds", nargs="+", type=int, required=True)
    seed_sweep.add_argument("--policies", nargs="*", default=DEFAULT_POLICIES)
    seed_sweep.add_argument("--v21", action="store_true",
                            help="Use the v2.1 default policy roster.")
    seed_sweep.add_argument("--override-total-steps", type=int, default=None)
    seed_sweep.add_argument("--dry-run", action="store_true")
    _add_execution_overrides(seed_sweep)

    plot = subparsers.add_parser("plot-sweep", help="Create HTML plots from a policy sweep JSON summary.")
    plot.add_argument("--summary", required=True)
    plot.add_argument("--output-dir", default=None)

    merge = subparsers.add_parser(
        "merge-seed-sweeps",
        help="Merge split seed-sweep zips/folders and build robust summaries.",
    )
    merge.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Seed-sweep zip files, folders, or rows CSV files to merge.",
    )
    merge.add_argument("--output-dir", default="results/merged")
    merge.add_argument("--name", default="merged_seed_sweep")
    merge.add_argument("--bootstrap-resamples", type=int, default=5000)
    merge.add_argument("--bootstrap-seed", type=int, default=0)

    compare = subparsers.add_parser(
        "compare-envs",
        help="Build cross-environment tables and figures from merged seed-sweep outputs.",
    )
    compare.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Merged output directories or summary JSON files.",
    )
    compare.add_argument(
        "--env-labels",
        nargs="*",
        default=None,
        help="Optional labels matching --inputs order, e.g. Hopper-v5 Walker2d-v5.",
    )
    compare.add_argument("--output-dir", default="results/cross_env")
    compare.add_argument("--name", default="cross_env_comparison")

    phase_g_intake = subparsers.add_parser(
        "phase-g-intake",
        help="Merge a Phase G add-5 light bundle with the existing 5-seed base result.",
    )
    phase_g_intake.add_argument("--env", required=True, help="hopper, walker2d, halfcheetah, or ant.")
    phase_g_intake.add_argument("--add-input", required=True, help="Phase G add-5 zip/folder/rows CSV.")
    phase_g_intake.add_argument(
        "--base-input",
        default=None,
        help="Optional existing 5-seed base rows CSV/zip/folder. If omitted, the latest known base rows are used.",
    )
    phase_g_intake.add_argument("--output-dir", default="results/merged")
    phase_g_intake.add_argument("--progress-path", default="docs/phase_g_progress.json")
    phase_g_intake.add_argument("--bootstrap-resamples", type=int, default=5000)
    phase_g_intake.add_argument("--bootstrap-seed", type=int, default=0)

    artifact_page = subparsers.add_parser(
        "build-artifact-page",
        help="Build a static RBA-Bench HTML artifact page from cross-environment outputs.",
    )
    artifact_page.add_argument("--cross-env-dir", required=True)
    artifact_page.add_argument("--output-dir", default="results/artifacts")
    artifact_page.add_argument("--name", default="rba_bench_artifact")
    artifact_page.add_argument(
        "--phase-status",
        default="preliminary",
        help="Label shown on the page, e.g. preliminary or phase-g-10seed.",
    )
    artifact_page.add_argument(
        "--copy-assets",
        action="store_true",
        help="Copy linked HTML/CSV/JSON assets into the generated page directory for static hosting.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "validate-config":
        config = load_experiment_config(Path(args.config))
        print(json.dumps(dump_experiment_config(config), indent=2))
        return

    if args.command == "show-config":
        config = load_experiment_config(Path(args.config))
        print(json.dumps(dump_experiment_config(config), indent=2))
        return

    if args.command == "run":
        config = load_experiment_config(Path(args.config))
        budget_metadata = _apply_execution_overrides(config, args)
        summary = run_experiment(
            config=config,
            run_name_override=args.name,
            total_steps_override=args.override_total_steps,
            dry_run=args.dry_run,
        )
        summary.update(budget_metadata_payload(budget_metadata))
        print(json.dumps(summary, indent=2))
        return

    if args.command == "smoke":
        config = load_experiment_config(Path(args.config))
        budget_metadata = _apply_execution_overrides(config, args)
        config.train.total_steps = int(args.steps)
        config.train.pretrain_steps = min(max(args.steps // 2, 32), config.train.total_steps)
        config.model.batch_size = min(config.model.batch_size, 64)
        config.model.learning_starts = min(config.model.learning_starts, max(16, config.train.pretrain_steps // 4))
        summary = run_experiment(
            config=config,
            run_name_override=f"{config.name}_smoke",
            total_steps_override=args.steps,
            dry_run=False,
        )
        summary.update(budget_metadata_payload(budget_metadata))
        print(json.dumps(summary, indent=2))
        return

    if args.command == "sweep-policies":
        config = load_experiment_config(Path(args.config))
        budget_metadata = _apply_execution_overrides(config, args)
        policies = list(DEFAULT_POLICIES_V21) if getattr(args, "v21", False) else list(args.policies)
        summary = run_policy_sweep(
            config=config,
            policies=policies,
            total_steps_override=args.override_total_steps,
            dry_run=args.dry_run,
            budget_metadata=budget_metadata,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "sweep-seeds":
        config = load_experiment_config(Path(args.config))
        budget_metadata = _apply_execution_overrides(config, args)
        policies = list(DEFAULT_POLICIES_V21) if getattr(args, "v21", False) else list(args.policies)
        summary = run_seed_sweep(
            config=config,
            seeds=list(args.seeds),
            policies=policies,
            total_steps_override=args.override_total_steps,
            dry_run=args.dry_run,
            budget_metadata=budget_metadata,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "plot-sweep":
        paths = plot_sweep_summary(args.summary, args.output_dir)
        print(json.dumps(paths, indent=2))
        return

    if args.command == "merge-seed-sweeps":
        output = merge_seed_sweep_inputs(
            input_paths=list(args.inputs),
            output_dir=args.output_dir,
            name=args.name,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(json.dumps(output, indent=2))
        return

    if args.command == "compare-envs":
        env_labels = list(args.env_labels) if args.env_labels else None
        output = compare_environments(
            input_paths=list(args.inputs),
            env_labels=env_labels,
            output_dir=args.output_dir,
            name=args.name,
        )
        print(json.dumps(output, indent=2))
        return

    if args.command == "phase-g-intake":
        output = intake_phase_g_result(
            env_key=args.env,
            add_input=args.add_input,
            base_input=args.base_input,
            output_dir=args.output_dir,
            progress_path=args.progress_path,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(json.dumps(output, indent=2))
        return

    if args.command == "build-artifact-page":
        output = build_artifact_page(
            cross_env_dir=args.cross_env_dir,
            output_dir=args.output_dir,
            name=args.name,
            phase_status=args.phase_status,
            copy_assets=args.copy_assets,
        )
        print(json.dumps(output, indent=2))
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
