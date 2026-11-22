"""Run an explicitly message-budget-matched H2 comparison."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import yaml

from voc_sar.evaluation.metrics import bootstrap_mean_interval, communication_reduction
from voc_sar.experiments.runner import _run_episode


def run_h2_budget_experiment(
    config: dict[str, Any],
    *,
    budget: int,
    seeds: list[int],
) -> dict[str, Any]:
    """Compare aware and agnostic selection at one exact episode message quota.

    Each selector may send value-positive messages at any decision during the
    episode, up to the fixed quota. If quota remains and only that many steps
    are left, the selector sends its best available pair even when net value
    is non-positive. Episodes that terminate early without meeting the quota
    are reported and excluded from the exact-budget paired test.
    """
    if budget < 1 or budget > int(config["environment"]["horizon"]):
        raise ValueError("budget must be between 1 and the episode horizon")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")

    horizon = int(config["communication"]["nonmyopic_horizon"])
    costs = 0.0
    methods = (
        "aware_nonmyopic_fixed_budget",
        "agnostic_nonmyopic_fixed_budget",
        "full",
        "never",
    )
    jobs = [
        (seed, method)
        for method in methods
        for seed in seeds
    ]
    worker_count = int(config["evaluation"].get("parallel_workers", 1))
    episode_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    completed = 0
    _progress(completed, len(jobs))

    if worker_count == 1:
        for seed, method in jobs:
            episode_rows.append(
                _run_h2_job(config, seed, method, costs, horizon, budget)
            )
            completed += 1
            _progress(completed, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _run_h2_job, config, seed, method, costs, horizon, budget
                ): (seed, method)
                for seed, method in jobs
            }
            for future in as_completed(futures):
                episode_rows.append(future.result())
                completed += 1
                _progress(completed, len(jobs))

    indexed = {(row["method"], row["seed"]): row for row in episode_rows}
    aware = [indexed[(methods[0], seed)] for seed in seeds]
    agnostic = [indexed[(methods[1], seed)] for seed in seeds]
    full = [indexed[("full", seed)] for seed in seeds]
    exact_budget_seeds = [
        seed for seed, aware_row, agnostic_row in zip(seeds, aware, agnostic)
        if aware_row["message_budget_met"]
        and agnostic_row["message_budget_met"]
        and aware_row["messages"] == agnostic_row["messages"] == budget
    ]
    paired_differences = [
        indexed[(methods[0], seed)]["discounted_task_return"]
        - indexed[(methods[1], seed)]["discounted_task_return"]
        for seed in exact_budget_seeds
    ]
    confidence = float(config["evaluation"].get("confidence_level", 0.95))
    bootstrap_samples = int(config["evaluation"].get("bootstrap_samples", 1000))
    bootstrap_seed = int(seeds[0]) + 50_123
    paired_interval = (
        bootstrap_mean_interval(
            paired_differences, confidence, bootstrap_samples, bootstrap_seed
        )
        if paired_differences else None
    )

    return_tolerance = float(config["evaluation"].get("return_match_tolerance", 1.0))
    matched_return_reductions = [
        communication_reduction(aware_row["messages"], full_row["messages"])
        for seed, aware_row, full_row in zip(seeds, aware, full)
        if seed in exact_budget_seeds
        and full_row["messages"] > 0
        and abs(
            aware_row["discounted_task_return"]
            - full_row["discounted_task_return"]
        ) <= return_tolerance
    ]
    matched_return_interval = (
        bootstrap_mean_interval(
            matched_return_reductions,
            confidence,
            bootstrap_samples,
            bootstrap_seed + 1,
        )
        if matched_return_reductions else None
    )

    condition_summaries = []
    for method in methods:
        rows = [indexed[(method, seed)] for seed in seeds]
        condition_summaries.append({
            "method": method,
            "episodes": len(rows),
            "mean_task_return": statistics.fmean(
                row["discounted_task_return"] for row in rows
            ),
            "mean_messages": statistics.fmean(row["messages"] for row in rows),
            "mean_rescued_victims": statistics.fmean(
                row["rescued_victims"] for row in rows
            ),
            "message_budget_met_episodes": sum(
                bool(row["message_budget_met"]) for row in rows
            ) if method.endswith("_fixed_budget") else None,
            "mean_forced_budget_sends": statistics.fmean(
                row["forced_budget_sends"] for row in rows
            ) if method.endswith("_fixed_budget") else None,
        })

    summary: dict[str, Any] = {
        "experiment": "H2 exact message-budget comparison",
        "interpretation": (
            "Receiver/message selection quality under a fixed whole-episode "
            "budget. Value-positive messages may be sent throughout the episode; "
            "remaining messages are forced only when required to meet the quota. "
            "Early-terminating episodes that miss quota are reported separately."
        ),
        "budget_messages_per_episode": budget,
        "message_schedule": (
            "Value-based sends are allowed at every decision. If quota remains "
            "and the number of horizon steps left equals the remaining quota, "
            "the best available pair is sent even if its estimated value is non-positive."
        ),
        "seeds": seeds,
        "paired_budget_comparison": {
            "aware_method": methods[0],
            "agnostic_method": methods[1],
            "attempted_seed_count": len(seeds),
            "paired_seed_count": len(exact_budget_seeds),
            "exact_message_match_count": len(exact_budget_seeds),
            "mean_aware_minus_agnostic_return": (
                statistics.fmean(paired_differences) if paired_differences else None
            ),
            "confidence_level": confidence,
            "confidence_interval": paired_interval,
            "positive_aware_advantage_supported": (
                paired_interval[0] > 0 if paired_interval else False
            ),
        },
        "aware_vs_full_at_matched_return": {
            "return_match_tolerance": return_tolerance,
            "matched_seed_count": len(matched_return_reductions),
            "mean_communication_reduction": (
                statistics.fmean(matched_return_reductions)
                if matched_return_reductions else None
            ),
            "confidence_interval": matched_return_interval,
            "target_reduction": float(
                config["evaluation"].get("thresholds", {}).get(
                    "communication_reduction", 0.20
                )
            ),
        },
        "conditions": condition_summaries,
        "runtime_seconds": time.perf_counter() - started,
        "config": config,
    }

    output_root = Path(config["evaluation"].get("output_dir", "results"))
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_root / f"h2_fixed_budget_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    summary["run_directory"] = str(output_dir.resolve())
    (output_dir / "config_used.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "episodes.json").write_text(
        json.dumps(sorted(episode_rows, key=lambda row: (row["method"], row["seed"])),
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _run_h2_job(
    config: dict[str, Any],
    seed: int,
    method: str,
    cost_weight: float,
    horizon: int,
    budget: int,
) -> dict[str, Any]:
    quota = budget if method.endswith("_fixed_budget") else None
    return _run_episode(
        config,
        seed,
        method,
        cost_weight,
        horizon,
        None,
        forced_message_budget=quota,
    )


def _progress(completed: int, total: int) -> None:
    if completed in {0, total} or completed % max(1, total // 20) == 0:
        print(f"H2 fixed-budget episodes: {completed}/{total}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=10)
    parser.add_argument("--seed-count", type=int, default=50)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    summary = run_h2_budget_experiment(config, budget=args.budget, seeds=seeds)
    print(f"Completed {len(seeds) * 4} episodes.")
    print(f"Results: {summary['run_directory']}")
    print(f"Runtime: {summary['runtime_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
