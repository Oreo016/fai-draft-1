"""E10: Information age versus observed marginal value."""

from collections import defaultdict
from typing import Any

from voc_sar.research_experiments.common import (
    mean, oracle_estimate, policy_choice, rollout_task_return, run_study, spearman,
)


def analyze(config: dict[str, Any], contexts: list[dict[str, Any]]):
    env = config["environment"]
    p_move = float(env.get("hazard_move_probability", 0.0))
    p_toggle = float(env.get("hazard_toggle_probability", 0.0))
    rows = []
    ages, observed = [], []
    grouped = defaultdict(lambda: {"heuristic": [], "observed": []})
    for context in contexts:
        message = context["message"]
        age = max(0, int(context["step"]) - int(message.observed_at))
        heuristic = ((1.0 - p_move) * (1.0 - p_toggle)) ** age
        value = oracle_estimate(config, context).marginal_value
        ages.append(heuristic)
        observed.append(value)
        grouped[age]["heuristic"].append(heuristic)
        grouped[age]["observed"].append(value)
        messages = context["messages"]
        with_freshness = policy_choice(config, context, messages, aware=True)
        without_freshness = policy_choice(config, context, messages, aware=True, freshness=False)
        returns = [
            rollout_task_return(context, (messages[choice[0]], choice[1]) if choice else None, config)
            for choice in (with_freshness, without_freshness)
        ]
        rows.append({"seed": context["seed"], "step": context["step"],
                     "information_age": age, "survival_heuristic": heuristic,
                     "observed_marginal_value": value,
                     "message_kind": message.kind, "message_status": message.status,
                     "with_freshness_choice": with_freshness, "without_freshness_choice": without_freshness,
                     "return_with_freshness": returns[0], "return_without_freshness": returns[1],
                     "freshness_return_difference": returns[0] - returns[1]})
    by_age = {
        str(age): {"mean_heuristic": mean(values["heuristic"]),
                  "mean_marginal_value": mean(values["observed"]),
                  "count": len(values["observed"])}
        for age, values in sorted(grouped.items())
    }
    headline = {
        "spearman_heuristic_vs_marginal_value": spearman(ages, observed),
        "mean_information_age": mean([row["information_age"] for row in rows]),
        "mean_absolute_marginal_value": mean([abs(value) for value in observed]),
        "paired_mean_freshness_return_difference": mean([row["freshness_return_difference"] for row in rows]),
    }
    return rows, {"headline": headline, "by_information_age": by_age,
                  "heuristic_definition": "[(1-p_move)(1-p_toggle)]^age",
                  "p_move": p_move, "p_toggle": p_toggle}


def run(config: dict[str, Any]):
    return run_study("E10", config, analyze)
