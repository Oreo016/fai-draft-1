"""Run paired-seed H4 cost sensitivity and a secondary cost sweep."""

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

from voc_sar.evaluation.metrics import bootstrap_mean_interval
from voc_sar.experiments.runner import _run_episode, _validate_config


def run_h4_cost_experiment(
    config: dict[str, Any], *, seeds: list[int], costs: list[float]
) -> dict[str, Any]:
    """Run aware nonmyopic episodes at paired costs on identical seeds."""
    _validate_config(config)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if 0.0 not in costs or 0.5 not in costs or len(set(costs)) != len(costs):
        raise ValueError("costs must uniquely include the primary pair 0.0 and 0.5")
    if any(cost < 0 for cost in costs):
        raise ValueError("communication costs cannot be negative")

    horizon = int(config["communication"]["nonmyopic_horizon"])
    jobs = [(seed, float(cost)) for cost in costs for seed in seeds]
    worker_count = int(config["evaluation"].get("parallel_workers", 1))
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    _progress(0, len(jobs))

    if worker_count == 1:
        for seed, cost in jobs:
            rows.append(_run_h4_job(config, seed, cost, horizon))
            _progress(len(rows), len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_run_h4_job, config, seed, cost, horizon): (seed, cost)
                for seed, cost in jobs
            }
            for future in as_completed(futures):
                rows.append(future.result())
                _progress(len(rows), len(jobs))

    indexed = {(float(row["cost_weight"]), int(row["seed"])): row for row in rows}
    confidence = float(config["evaluation"].get("confidence_level", 0.95))
    samples = int(config["evaluation"].get("bootstrap_samples", 1000))
    bootstrap_seed = int(seeds[0]) + 84_011
    ordered_costs = sorted(float(cost) for cost in costs)
    base_cost = 0.0
    by_cost: list[dict[str, Any]] = []
    paired_changes: dict[str, dict[str, Any]] = {}

    metrics = {
        "discounted_task_return": "discounted_task_return",
        "messages": "messages",
        "rescued_victims": "rescued_victims",
        "positive_voc_rate": "positive_voc_rate",
    }
    for cost_index, cost in enumerate(ordered_costs):
        condition_rows = [indexed[(cost, seed)] for seed in seeds]
        metric_summaries = {}
        for public_name, key in metrics.items():
            values = [float(row[key]) for row in condition_rows]
            interval = bootstrap_mean_interval(
                values, confidence, samples,
                bootstrap_seed + 100 * cost_index + len(public_name),
            )
            metric_summaries[public_name] = {
                "mean": statistics.fmean(values),
                "confidence_interval": interval,
            }
        by_cost.append({
            "cost_weight": cost,
            "episodes": len(condition_rows),
            "metrics": metric_summaries,
        })

        if cost != base_cost:
            changes_for_cost = {}
            for metric_index, (public_name, key) in enumerate(metrics.items()):
                differences = [
                    float(indexed[(cost, seed)][key])
                    - float(indexed[(base_cost, seed)][key])
                    for seed in seeds
                ]
                changes_for_cost[public_name] = {
                    "mean_change_vs_cost_0": statistics.fmean(differences),
                    "paired_confidence_interval": bootstrap_mean_interval(
                        differences, confidence, samples,
                        bootstrap_seed + 2000 + 100 * cost_index + metric_index,
                    ),
                }
            paired_changes[str(cost)] = changes_for_cost

    low_rates = [
        float(indexed[(0.0, seed)]["positive_voc_rate"]) for seed in seeds
    ]
    high_rates = [
        float(indexed[(0.5, seed)]["positive_voc_rate"]) for seed in seeds
    ]
    paired_rate_drops = [low - high for low, high in zip(low_rates, high_rates)]
    paired_relative_reductions = [
        (low - high) / low
        for low, high in zip(low_rates, high_rates)
        if low > 0
    ]
    low_mean = statistics.fmean(low_rates)
    high_mean = statistics.fmean(high_rates)
    aggregate_relative_reduction = (
        (low_mean - high_mean) / low_mean if low_mean > 0 else 0.0
    )
    h4_interval = (
        bootstrap_mean_interval(
            paired_relative_reductions,
            confidence,
            samples,
            bootstrap_seed + 8001,
        )
        if paired_relative_reductions else None
    )
    thresholds = config["evaluation"].get("thresholds", {})
    target = float(thresholds.get("high_cost_rate_reduction", 0.15))
    h4 = {
        "primary_low_cost": 0.0,
        "primary_high_cost": 0.5,
        "seed_count": len(seeds),
        "low_cost_mean_positive_voc_rate": low_mean,
        "high_cost_mean_positive_voc_rate": high_mean,
        "absolute_rate_drop": low_mean - high_mean,
        "aggregate_relative_rate_reduction": aggregate_relative_reduction,
        "paired_seed_relative_rate_reduction_mean": (
            statistics.fmean(paired_relative_reductions)
            if paired_relative_reductions else None
        ),
        "paired_seed_relative_rate_reduction_confidence_interval": h4_interval,
        "paired_seed_count_with_positive_low_cost_rate": len(paired_relative_reductions),
        "target_relative_reduction": target,
        "point_estimate_meets_target": aggregate_relative_reduction >= target,
        "paired_interval_supports_decrease": bool(h4_interval and h4_interval[0] > 0),
    }

    summary: dict[str, Any] = {
        "experiment": "H4 paired-seed cost sensitivity",
        "scope": "Aware nonmyopic policy; costs 0 and 0.5 are primary; other costs are secondary.",
        "seeds": [int(seed) for seed in seeds],
        "paired_seed_count": len(seeds),
        "confidence_level": confidence,
        "bootstrap_samples": samples,
        "primary_h4": h4,
        "cost_conditions": by_cost,
        "paired_changes_vs_zero_cost": paired_changes,
        "runtime_seconds": time.perf_counter() - started,
        "config": config,
    }

    output_root = Path(config["evaluation"].get("output_dir", "results"))
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_root / f"h4_cost_sensitivity_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    summary["run_directory"] = str(output_dir.resolve())
    (output_dir / "config_used.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "episodes.json").write_text(
        json.dumps(sorted(rows, key=lambda row: (row["cost_weight"], row["seed"])),
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _run_h4_job(
    config: dict[str, Any], seed: int, cost: float, horizon: int
) -> dict[str, Any]:
    return _run_episode(
        config,
        int(seed),
        "aware_nonmyopic",
        float(cost),
        horizon,
        None,
    )


def _progress(completed: int, total: int) -> None:
    if completed in {0, total} or completed % max(1, total // 20) == 0:
        print(f"H4 paired episodes: {completed}/{total}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed-start", type=int, default=610)
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--costs", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    config["evaluation"]["seeds"] = seeds
    summary = run_h4_cost_experiment(config, seeds=seeds, costs=args.costs)
    print(f"Completed {len(seeds) * len(args.costs)} paired episodes.")
    print(f"H4 primary summary: {summary['primary_h4']}")
    print(f"Results: {summary['run_directory']}")
    print(f"Runtime: {summary['runtime_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
