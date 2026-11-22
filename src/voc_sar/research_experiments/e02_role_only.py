"""E02: Compare a static role compatibility baseline with value selectors."""

from typing import Any

from voc_sar.evaluation.metrics import pair_regret
from voc_sar.research_experiments.common import policy_choice
from voc_sar.research_experiments.common import (
    bootstrap,
    evaluate_pairs,
    mean,
    rollout_task_return,
    run_study,
)
from voc_sar.types import Role


def _compatibility(message, role: Role, table: dict[str, Any]) -> float:
    key = "intention" if message.kind == "intention" else f"{message.kind}_{message.status}"
    scores = table.get(key, {})
    return float(scores.get(role.value, 0.0))


def select_role_only(messages, receiver_ids, roles, table):
    ranked = [
        (_compatibility(message, roles[receiver], table), -index, -receiver, index, receiver)
        for index, message in enumerate(messages)
        for receiver in receiver_ids
    ]
    if not ranked:
        return None
    best = max(ranked)
    return (best[-2], best[-1]) if best[0] > 0 else None


def analyze(config: dict[str, Any], contexts: list[dict[str, Any]]):
    seed_rows: dict[int, dict[str, list[dict[str, float]]]] = {}
    metric_values: dict[str, dict[str, list[float]]] = {
        name: {key: [] for key in ("return", "regret", "messages")}
        for name in ("aware", "agnostic", "role_only", "role_only_budget_matched")
    }
    h1_values: list[float] = []
    rows: list[dict[str, Any]] = []
    table = config["experiment"]["compatibility"]
    for context in contexts:
        messages = list(context["messages"][:3])
        truth, estimates = evaluate_pairs(config, context, messages=messages)
        aware_key = policy_choice(config, context, messages, aware=True)
        receiver_ids = [key[1] for key in estimates if key[0] == 0]
        agnostic_key = policy_choice(config, context, messages, aware=False)
        role_key = select_role_only(messages, receiver_ids, context["roles"], table)
        aware_sends = aware_key is not None
        matched_role_key = role_key if aware_sends else None
        choices = {
            "aware": aware_key,
            "agnostic": agnostic_key,
            "role_only": role_key,
            "role_only_budget_matched": matched_role_key,
        }
        for name, chosen in choices.items():
            predicted = truth.get(chosen, 0.0) if chosen is not None else 0.0
            regret = max(0.0, max([0.0, *truth.values()]) - predicted)
            task_choice = (messages[chosen[0]], chosen[1]) if chosen is not None else None
            task_return = rollout_task_return(context, task_choice, config)
            values = metric_values[name]
            values["return"].append(task_return)
            values["regret"].append(regret)
            values["messages"].append(float(chosen is not None))
            seed_entry = seed_rows.setdefault(int(context["seed"]), {})
            seed_entry.setdefault(name, []).append({
                "return": task_return,
                "messages": float(chosen is not None),
            })
            rows.append({
                "seed": context["seed"], "step": context["step"], "method": name,
                "return": task_return, "regret": regret,
                "messages": int(chosen is not None),
            })
        if receiver_ids:
            primary = [truth.get((0, rid), 0.0) for rid in receiver_ids]
            h1_values.append(float(abs(primary[0] - primary[1]) > 0.5) if len(primary) > 1 else 0.0)

    aware_vs_agnostic = []
    matched_aware_vs_agnostic_by_seed = []
    matched_aware_vs_role = []
    for per_seed in seed_rows.values():
        if "aware" in per_seed and "agnostic" in per_seed:
            paired = list(zip(per_seed["aware"], per_seed["agnostic"]))
            aware_vs_agnostic.append(mean([a["return"] - b["return"] for a, b in paired]))
            budget_pairs = [a["return"] - b["return"] for a, b in paired if a["messages"] == b["messages"]]
            if budget_pairs:
                matched_aware_vs_agnostic_by_seed.append(mean(budget_pairs))
        if "aware" in per_seed and "role_only_budget_matched" in per_seed:
            matched_aware_vs_role.extend(
                a["return"] - b["return"]
                for a, b in zip(per_seed["aware"], per_seed["role_only_budget_matched"])
            )
    headline = {}
    for name, metrics in metric_values.items():
        for metric, values in metrics.items():
            headline[f"{name}_{metric}"] = mean(values)
    headline["h1_receiver_differentiation_fraction"] = mean(h1_values)
    headline["h2_aware_minus_agnostic_return"] = mean(aware_vs_agnostic)
    headline["h2_aware_minus_agnostic_ci_low"] = (bootstrap(aware_vs_agnostic, config, 821) or (None, None))[0]
    headline["h2_aware_minus_agnostic_ci_high"] = (bootstrap(aware_vs_agnostic, config, 821) or (None, None))[1]
    headline["h2_aware_minus_agnostic_budget_matched"] = mean(matched_aware_vs_agnostic_by_seed)
    headline["h2_budget_matched_seed_count"] = len(matched_aware_vs_agnostic_by_seed)
    headline["h2_budget_matched_ci_low"] = (bootstrap(matched_aware_vs_agnostic_by_seed, config, 822) or (None, None))[0]
    headline["h2_budget_matched_ci_high"] = (bootstrap(matched_aware_vs_agnostic_by_seed, config, 822) or (None, None))[1]
    headline["budget_matched_aware_minus_role_only"] = mean(matched_aware_vs_role)
    return rows, {"headline": headline, "h1_h2_without_role_only": {
        "h1_fraction": mean(h1_values), "h2_aware_minus_agnostic": mean(aware_vs_agnostic),
    }, "h1_h2_with_role_only": {
        "role_only_mean_regret": mean(metric_values["role_only"]["regret"]),
        "budget_matched_aware_minus_role_return": mean(matched_aware_vs_role),
    }}


def run(config: dict[str, Any]):
    return run_study("E02", config, analyze)
