"""Run a focused, seed-clustered H1 receiver-differentiation evaluation."""

from datetime import datetime
import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import yaml

from voc_sar.communication.selector import shortlist
from voc_sar.environment.sar_grid import SARGrid
from voc_sar.evaluation.metrics import bootstrap_cluster_mean_interval
from voc_sar.evaluation.oracle import CounterfactualOracle
from voc_sar.experiments.runner import _validate_config
from voc_sar.policies.task import candidate_messages, task_policies
from voc_sar.types import Role


def run_h1_evaluation(
    config: dict[str, Any], *, seeds: list[int]
) -> dict[str, Any]:
    """Measure H1 only and retain eligible-context counts for each seed."""
    _validate_config(config)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")

    env_config = config["environment"]
    comm = config["communication"]
    evaluation = config["evaluation"]
    thresholds = evaluation.get("thresholds", {})
    requested_contexts = int(evaluation["held_out_contexts"])
    eligible_contexts_per_seed = max(1, math.ceil(requested_contexts / len(seeds)))
    h_eval = int(comm["oracle_horizon"])
    shortlist_size = int(comm["shortlist_size"])
    value_difference_threshold = float(
        thresholds.get("receiver_value_difference", 0.5)
    )

    per_seed: list[dict[str, Any]] = []
    flags_by_seed: list[list[float]] = []
    all_flags: list[float] = []
    total_diagnostic_contexts = 0
    started = time.perf_counter()
    _progress(0, len(seeds))

    for seed_index, seed in enumerate(seeds, 1):
        env = SARGrid(env_config)
        observations = env.reset(int(seed) + 77_777)
        policies = task_policies(env.roles)
        sender_id = next(
            (index for index, role in enumerate(env.roles) if role == Role.SCOUT),
            0,
        )
        receivers = [index for index in range(len(env.roles)) if index != sender_id]
        oracle = CounterfactualOracle(
            policies,
            rollout_count=int(comm["rollout_count"]),
            gamma=float(env_config.get("gamma", 0.99)),
            cost_weight=0.0,
            unit_cost=float(comm.get("unit_cost", 1.0)),
            seed=int(seed) + 77_777,
        )

        seed_flags: list[float] = []
        diagnostic_contexts = 0
        while not env.terminated and len(seed_flags) < eligible_contexts_per_seed:
            observation = observations[sender_id]
            planned_action = policies[sender_id].act(observation)
            messages = candidate_messages(observation, planned_action)
            top_pairs = shortlist(
                sender_id,
                messages,
                receivers,
                shortlist_size,
                current_step=observation.time_step,
                receiver_roles=env.roles,
                teammate_tracks=observation.teammates,
                receiver_summaries=observation.receiver_summaries,
                hazard_move_probability=float(
                    env_config.get("hazard_move_probability", 0.0)
                ),
                hazard_toggle_probability=float(
                    env_config.get("hazard_toggle_probability", 0.0)
                ),
            )
            if top_pairs:
                diagnostic_contexts += 1
                total_diagnostic_contexts += 1
                primary_message = next(
                    (message for message in messages if message.kind != "intention"),
                    None,
                )
                if primary_message is not None and len(receivers) >= 2:
                    receiver_values = [
                        oracle.evaluate(
                            env.clone(), sender_id, receiver_id,
                            primary_message, h_eval,
                        ).marginal_value
                        for receiver_id in receivers
                    ]
                    flag = float(
                        abs(receiver_values[0] - receiver_values[1])
                        > value_difference_threshold
                    )
                    seed_flags.append(flag)
                    all_flags.append(flag)

            actions = [
                policy.act(item)
                for policy, item in zip(policies, observations)
            ]
            observations = env.step(actions).observations

        flags_by_seed.append(seed_flags)
        per_seed.append({
            "seed": int(seed),
            "requested_h1_eligible_contexts": eligible_contexts_per_seed,
            "observed_diagnostic_contexts": diagnostic_contexts,
            "h1_eligible_contexts": len(seed_flags),
            "receiver_differentiated_contexts": int(sum(seed_flags)),
            "receiver_differentiation_fraction": (
                sum(seed_flags) / len(seed_flags) if seed_flags else None
            ),
        })
        _progress(seed_index, len(seeds))

    if not all_flags:
        raise RuntimeError("H1 evaluation produced no eligible contexts")
    confidence = float(evaluation.get("confidence_level", 0.95))
    bootstrap_samples = int(evaluation.get("bootstrap_samples", 1000))
    interval = bootstrap_cluster_mean_interval(
        flags_by_seed,
        confidence,
        bootstrap_samples,
        int(seeds[0]) + 311,
    )
    fraction = sum(all_flags) / len(all_flags)
    target_fraction = float(thresholds.get("receiver_differentiation_fraction", 0.30))
    summary: dict[str, Any] = {
        "experiment": "H1 focused held-out receiver differentiation",
        "seeds": [int(seed) for seed in seeds],
        "independent_seed_clusters": len(seeds),
        "requested_h1_eligible_contexts": requested_contexts,
        "requested_h1_eligible_contexts_per_seed": eligible_contexts_per_seed,
        "observed_diagnostic_contexts": total_diagnostic_contexts,
        "h1_eligible_contexts": len(all_flags),
        "seed_clusters_meeting_context_target": sum(
            item["h1_eligible_contexts"] >= eligible_contexts_per_seed
            for item in per_seed
        ),
        "receiver_differentiated_contexts": int(sum(all_flags)),
        "receiver_differentiation_fraction": fraction,
        "confidence_level": confidence,
        "seed_cluster_bootstrap_confidence_interval": interval,
        "value_difference_threshold": value_difference_threshold,
        "target_fraction": target_fraction,
        "point_estimate_meets_target": fraction >= target_fraction,
        "lower_confidence_bound_meets_target": interval[0] >= target_fraction,
        "per_seed_counts": per_seed,
        "runtime_seconds": time.perf_counter() - started,
        "config": config,
    }

    output_root = Path(evaluation.get("output_dir", "results"))
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_root / f"h1_evaluation_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    summary["run_directory"] = str(output_dir.resolve())
    (output_dir / "config_used.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "per_seed_counts.json").write_text(
        json.dumps(per_seed, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _progress(completed: int, total: int) -> None:
    if completed in {0, total} or completed % max(1, total // 20) == 0:
        print(f"H1 diagnostic seeds: {completed}/{total}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed-start", type=int, default=610)
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--contexts", type=int, default=500)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if args.contexts < 1 or args.seed_count < 1:
        parser.error("--contexts and --seed-count must be positive")
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    config["evaluation"]["seeds"] = seeds
    config["evaluation"]["held_out_contexts"] = args.contexts
    summary = run_h1_evaluation(config, seeds=seeds)
    print(f"H1 eligible contexts: {summary['h1_eligible_contexts']}")
    print(f"H1 fraction: {summary['receiver_differentiation_fraction']:.4f}")
    print(f"95% CI: {summary['seed_cluster_bootstrap_confidence_interval']}")
    print(f"Results: {summary['run_directory']}")
    print(f"Runtime: {summary['runtime_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
