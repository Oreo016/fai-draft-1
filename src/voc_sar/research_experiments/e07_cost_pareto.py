"""E07: Finer communication-cost sweep and return/message Pareto frontier."""

import random
from typing import Any

from voc_sar.research_experiments.common import policy_choice
from voc_sar.research_experiments.common import evaluate_pairs, mean, rollout_task_return, run_study
from voc_sar.research_experiments.e02_role_only import _compatibility


def _distance_cost(context, receiver, config) -> float:
    experiment = config["experiment"]
    if not experiment.get("receiver_dependent_cost", False):
        return 1.0
    track = next((track for track in context["observation"].teammates if track.agent_id == receiver), None)
    distance = (
        abs(track.position[0] - context["observation"].position[0])
        + abs(track.position[1] - context["observation"].position[1])
        if track is not None else float(experiment["distance_scale_L"])
    )
    scale = float(experiment["distance_scale_L"])
    return receiver_cost_multiplier(distance, scale)


def receiver_cost_multiplier(distance: float, scale: float) -> float:
    return 1.0 + distance / scale if scale > 0 else 1.0


def analyze(config: dict[str, Any], contexts: list[dict[str, Any]]):
    experiment = config["experiment"]
    lambdas = [float(value) for value in experiment["lambda_grid"]]
    c0 = float(experiment["unit_cost_c0"])
    compatibility = experiment.get("compatibility", {})
    policy_config = {**config, "communication": {**config["communication"], "unit_cost": c0}}
    rows = []
    for context in contexts:
        messages = list(context["messages"][:3])
        truth, estimates = evaluate_pairs(config, context, messages=messages)
        for lam in lambdas:
            methods: dict[str, tuple[int, int] | None] = {}
            adjusted = {}
            for pair, estimate in estimates.items():
                multiplier = _distance_cost(context, pair[1], config)
                cost = lam * c0 * multiplier
                adjusted[pair] = estimate.marginal_value - cost
            methods["aware"] = (
                (max(adjusted, key=adjusted.get) if adjusted and max(adjusted.values()) > 0 else None)
                if experiment.get("receiver_dependent_cost", False)
                else policy_choice(policy_config, context, messages, aware=True, cost_weight=lam)
            )
            receiver_ids = sorted({receiver for _, receiver in estimates})
            methods["agnostic"] = policy_choice(policy_config, context, messages, aware=False, cost_weight=lam)
            role_rank = [
                (_compatibility(message, context["roles"][receiver], compatibility), -index, -receiver, index, receiver)
                for index, message in enumerate(messages) for receiver in receiver_ids
            ]
            best_role = max(role_rank) if role_rank else None
            methods["role_only"] = (best_role[-2], best_role[-1]) if best_role and best_role[0] > 0 else None
            methods["full"] = (0, receiver_ids[0]) if messages and receiver_ids else None
            methods["fixed_frequency"] = methods["role_only"] if int(context["step"]) % int(config["communication"].get("fixed_frequency", 3)) == 0 else None
            methods["random"] = (random.Random(int(context["seed"]) * 1009 + int(context["step"])).randrange(len(messages)), receiver_ids[0]) if messages and receiver_ids else None
            for method, key in methods.items():
                choice = (messages[key[0]], key[1]) if key is not None else None
                rows.append({
                    "seed": context["seed"], "step": context["step"], "lambda": lam,
                    "method": method,
                    "messages": int(key is not None),
                    "return": rollout_task_return(context, choice, config),
                    "regret": max([0.0, *truth.values()]) - (truth.get(key, 0.0) if key is not None else 0.0),
                })
    points = []
    for lam in lambdas:
        for method in ("aware", "agnostic", "role_only", "full", "fixed_frequency", "random"):
            group = [row for row in rows if row["lambda"] == lam and row["method"] == method]
            points.append({"lambda": lam, "method": method,
                           "mean_messages": mean([row["messages"] for row in group]),
                           "mean_return": mean([row["return"] for row in group])})
    frontier = [point for point in points if not any(
        other["mean_messages"] <= point["mean_messages"]
        and other["mean_return"] >= point["mean_return"]
        and (other["mean_messages"] < point["mean_messages"] or other["mean_return"] > point["mean_return"])
        for other in points
    )]
    headline = {f"{point['method']}_lambda_{point['lambda']:g}_return": point["mean_return"] for point in points}
    headline.update({f"{point['method']}_lambda_{point['lambda']:g}_messages": point["mean_messages"] for point in points})
    return rows, {"headline": headline, "pareto_points": points,
                  "pareto_frontier": frontier, "unit_cost_c0_reward_units": c0,
                  "receiver_dependent_cost": bool(config["experiment"].get("receiver_dependent_cost", False))}


def run(config: dict[str, Any]):
    return run_study("E07", config, analyze)
