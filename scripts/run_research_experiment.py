"""Run one additive research experiment by ID."""

import argparse
import importlib

from voc_sar.research_experiments.common import load_study_config


RUNNERS = {
    "E01": "e01_message_conditioned",
    "E02": "e02_role_only",
    "E03": "e03_particle_diagnostics",
    "E04": "e04_noise_crn",
    "E05": "e05_continuation_bias",
    "E06": "e06_robust_null",
    "E07": "e07_cost_pareto",
    "E08": "e08_hypothesis_robustness",
    "E09": "e09_horizon_sweep",
    "E10": "e10_staleness",
    "E11": "e11_robustness",
    "E12": "e12_failures_runtime",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one additive SAR research study")
    parser.add_argument("experiment_id", choices=sorted(RUNNERS))
    parser.add_argument("--extended-seeds", action="store_true",
                        help="append seeds 10 through 29 after the existing ten")
    args = parser.parse_args()
    config = load_study_config(args.experiment_id, extended_seeds=args.extended_seeds)
    module = importlib.import_module(f"voc_sar.research_experiments.{RUNNERS[args.experiment_id]}")
    summary = module.run(config)
    print(f"Contexts: {summary['context_count']}")
    print(f"Results: {summary['run_directory']}")
    print(f"Reports: {summary['report_directory']}")


if __name__ == "__main__":
    main()
