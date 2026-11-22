"""E04: Rollout-count and common-random-number ablation."""

import random
import time
from typing import Any

from voc_sar.communication.estimator import ParticleReceiverEstimator, _rollout_return
from voc_sar.research_experiments.common import mean, run_study
from voc_sar.policies.task import task_policies


def _estimate(context, config, rollouts: int, common_random_numbers: bool, *,
              rollout_seed_offset: int = 0, paired_returns: list | None = None):
    base_seed = int(context["seed"])
    step = int(context["step"])
    sender_id, receiver_id = int(context["sender_id"]), int(context["receiver_id"])
    base = ParticleReceiverEstimator(config["environment"], seed=base_seed)
    worlds = base.sample_states(
        sender_id, receiver_id, int(config["experiment"]["particles"]),
        context["observation"],
    )
    rollout_seed = base_seed + rollout_seed_offset + step * 65_537
    rng = random.Random(rollout_seed + sender_id * 257 + receiver_id)
    policies = task_policies(context["roles"])
    horizon = int(config["communication"]["oracle_horizon"])
    gamma = float(config["environment"].get("gamma", 0.99))
    differences = []
    started = time.perf_counter()
    for index in range(rollouts):
        particle = worlds[rng.randrange(len(worlds))]
        baseline = particle.clone()
        intervention = particle.clone()
        send_seed = rollout_seed + index
        no_send_seed = send_seed if common_random_numbers else send_seed + 1_000_003
        baseline.reseed_dynamics(no_send_seed)
        intervention.reseed_dynamics(send_seed)
        no_send_return = _rollout_return(baseline, policies, horizon, gamma)
        send_return = _rollout_return(
            intervention, policies, horizon, gamma,
            (sender_id, receiver_id, context["message"]),
        )
        differences.append(send_return - no_send_return)
        if paired_returns is not None:
            paired_returns.append((send_return, no_send_return))
    elapsed = time.perf_counter() - started
    value = mean(differences) or 0.0
    se = 0.0 if len(differences) < 2 else (
        sum((item - value) ** 2 for item in differences) / (len(differences) - 1) / len(differences)
    ) ** 0.5
    return value, se, elapsed


def analyze(config: dict[str, Any], contexts: list[dict[str, Any]]):
    rollouts = [int(value) for value in config["experiment"]["rollout_counts"]]
    reference_count = int(config["experiment"]["reference_rollouts"])
    rows = []
    for context in contexts:
        reference, _, _ = _estimate(context, config, reference_count, True, rollout_seed_offset=7_919_111)
        for common in (True, False):
            for count in rollouts:
                branches = []
                value, se, elapsed = _estimate(context, config, count, common, paired_returns=branches)
                rows.append({
                    "seed": context["seed"], "step": context["step"],
                    "rollouts": count, "crn": common, "estimate": value,
                    "reference": reference, "bias": value - reference,
                    "paired_returns_send_no_send": branches,
                    "standard_error": se, "runtime_seconds_per_decision": elapsed,
                })
    headline = {}
    by_setting = {}
    for common in (True, False):
        for count in rollouts:
            group = [row for row in rows if row["crn"] == common and row["rollouts"] == count]
            label = f"n{count}_{'crn' if common else 'independent'}"
            by_setting[label] = {
                "mean_se": mean([row["standard_error"] for row in group]),
                "mean_bias_to_high_n": mean([row["bias"] for row in group]),
                "mean_runtime_seconds_per_decision": mean([row["runtime_seconds_per_decision"] for row in group]),
            }
            headline[f"{label}_se"] = by_setting[label]["mean_se"]
            headline[f"{label}_bias"] = by_setting[label]["mean_bias_to_high_n"]
    return rows, {"headline": headline, "by_setting": by_setting,
                  "bias_definition": "signed discrepancy from an independently sampled same-model reference",
                  "reference_rollouts": reference_count}


def run(config: dict[str, Any]):
    return run_study("E04", config, analyze)
