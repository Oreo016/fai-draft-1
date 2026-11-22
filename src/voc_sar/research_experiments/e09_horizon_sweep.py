"""E09: Decision-horizon sweep against the fixed evaluation horizon."""

from typing import Any

from voc_sar.research_experiments.common import evaluate_pairs, mean, policy_choice, rollout_task_return, run_study


def analyze(config: dict[str, Any], contexts: list[dict[str, Any]]):
    horizons = [int(value) for value in config["experiment"]["horizons"]]
    eval_horizon = int(config["communication"]["oracle_horizon"])
    rows = []
    for context in contexts:
        messages = list(context["messages"][:3])
        receivers = [i for i in range(len(context["roles"])) if i != context["sender_id"]]
        for horizon in horizons:
            truth, estimates = evaluate_pairs(
                config, context, messages=messages, receiver_ids=receivers,
                horizon=eval_horizon, estimator_horizon=horizon,
            )
            oracle_best = max(truth, key=truth.get) if truth and max(truth.values()) > 0 else None
            chosen = policy_choice(config, context, messages, aware=True, horizon=horizon)
            error_values = [abs(estimates[key].marginal_value - truth[key]) for key in truth]
            choice = (messages[chosen[0]], chosen[1]) if chosen else None
            rows.append({
                "seed": context["seed"], "step": context["step"], "horizon": horizon,
                "agreement": float(chosen == oracle_best),
                "mae_voc": mean(error_values),
                "return": rollout_task_return(context, choice, config),
                "messages": int(chosen is not None),
            })
    by_horizon = {}
    headline = {}
    for horizon in horizons:
        group = [row for row in rows if row["horizon"] == horizon]
        by_horizon[str(horizon)] = {
            key: mean([row[key] for row in group])
            for key in ("agreement", "mae_voc", "return", "messages")
        }
        for metric, value in by_horizon[str(horizon)].items():
            headline[f"h{horizon}_{metric}"] = value
    return rows, {"headline": headline, "by_horizon": by_horizon,
                  "evaluation_horizon": eval_horizon}


def run(config: dict[str, Any]):
    return run_study("E09", config, analyze)
