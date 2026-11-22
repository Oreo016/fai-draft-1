"""E06: Noise-robust selection rules and redundant-message null calibration."""

import math
from typing import Any

from voc_sar.research_experiments.common import mean, run_study
from voc_sar.research_experiments.e04_noise_crn import _estimate
from voc_sar.types import Message


def lower_confidence_bound(marginal_value: float, standard_error: float, z_value: float) -> float:
    return marginal_value - z_value * standard_error


def send_by_lcb(marginal_value: float, standard_error: float, z_value: float) -> bool:
    return lower_confidence_bound(marginal_value, standard_error, z_value) > 0


def send_by_epsilon(marginal_value: float, epsilon: float) -> bool:
    return marginal_value > epsilon


def analyze(config: dict[str, Any], contexts: list[dict[str, Any]]):
    alpha = float(config["experiment"]["confidence_level_lcb"])
    z_value = 1.6448536269514722 if math.isclose(alpha, 0.90) else 1.959963984540054
    epsilons = [float(value) for value in config["experiment"]["epsilon_grid"]]
    costs = [float(value) for value in config["experiment"]["cost_sweep"]]
    null_replicates = int(config["experiment"]["null_replicates"])
    null_rollouts = int(config["experiment"]["null_rollouts"])
    rows = []
    null_trials = []
    null_false_positives = {"strict": [], "lcb": [], **{f"epsilon_{e:g}": [] for e in epsilons}}
    positive_rates = {rule: {cost: [] for cost in costs} for rule in null_false_positives}
    for context in contexts:
        message = context["message"]
        receiver = int(context["receiver_id"])
        for replicate in range(null_replicates):
            null_message = Message(
                "null", message.location, message.observed_at, "redundant", context["sender_id"]
            )
            null_context = {**context, "message": null_message, "seed": int(context["seed"]) + replicate * 131}
            value, se, _ = _estimate(null_context, config, null_rollouts, True)
            null_trials.append({
                "seed": context["seed"], "step": context["step"], "replicate": replicate,
                "marginal_value": value, "standard_error": se,
            })
            null_false_positives["strict"].append(float(value > 0))
            null_false_positives["lcb"].append(float(send_by_lcb(value, se, z_value)))
            for epsilon in epsilons:
                null_false_positives[f"epsilon_{epsilon:g}"].append(float(send_by_epsilon(value, epsilon)))

        estimate, se, _ = _estimate(context, config, int(config["experiment"]["rollout_count"]), True)
        marginal = estimate
        standard_error = se
        for cost in costs:
            voc = marginal - cost * float(config["communication"]["unit_cost"])
            positive_rates["strict"][cost].append(float(voc > 0))
            positive_rates["lcb"][cost].append(float(send_by_lcb(voc, standard_error, z_value)))
            for epsilon in epsilons:
                positive_rates[f"epsilon_{epsilon:g}"][cost].append(float(send_by_epsilon(voc, epsilon)))
        for rule, rule_costs in positive_rates.items():
            rows.append({
                "seed": context["seed"], "step": context["step"],
                "rule": rule, "mv_hat": marginal, "standard_error": standard_error,
                "positive_rate_lambda_0": rule_costs[0.0][-1],
                "positive_rate_lambda_0_5": rule_costs[0.5][-1],
            })
    h4 = {}
    headline = {}
    for rule, rates in positive_rates.items():
        low, high = mean(rates[0.0]), mean(rates[0.5])
        reduction = (low - high) / low if low else None
        h4[rule] = {"lambda_0_rate": low, "lambda_0_5_rate": high,
                    "relative_drop": reduction}
        headline[f"{rule}_h4_relative_drop"] = reduction
        headline[f"{rule}_lambda_0_rate"] = low
        headline[f"{rule}_lambda_0_5_rate"] = high
    headline["null_false_positive_strict"] = mean(null_false_positives["strict"])
    headline["null_false_positive_lcb"] = mean(null_false_positives["lcb"])
    for epsilon in epsilons:
        key = f"epsilon_{epsilon:g}"
        headline[f"null_false_positive_{key}"] = mean(null_false_positives[key])
    return rows, {"headline": headline, "h4_by_rule": h4,
                  "null_trials": null_trials,
                  "null_false_positive_rates": {key: mean(value) for key, value in null_false_positives.items()},
                  "null_definition": "message kind null is ignored by task policies, so its true task marginal value is zero",
                  "lcb_confidence_level": alpha}


def run(config: dict[str, Any]):
    return run_study("E06", config, analyze)
