"""E12: Offline failure traces and per-decision runtime decomposition."""

import time
from typing import Any

from voc_sar.research_experiments.common import policy_choice
from voc_sar.research_experiments.common import evaluate_pairs, make_estimator, mean, oracle_estimate, rollout_task_return, run_study
from voc_sar.research_experiments.e02_role_only import select_role_only


def analyze(config: dict[str, Any], contexts: list[dict[str, Any]]):
    rows = []
    failures = []
    runtime = {"particle_filter_seconds": [], "rollout_estimate_seconds": [], "oracle_seconds": []}
    for context in contexts:
        message = context["message"]
        receiver = int(context["receiver_id"])
        estimator = make_estimator(config, seed=int(context["seed"]), record_rollouts=True)
        estimate = estimator.estimate(context["sender_id"], receiver, message,
                                      int(config["communication"]["nonmyopic_horizon"]), context["observation"])
        filter_seconds = estimate.particle_seconds
        rollout_seconds = estimate.rollout_seconds
        start = time.perf_counter()
        truth = oracle_estimate(config, context).marginal_value
        oracle_seconds = time.perf_counter() - start
        runtime["particle_filter_seconds"].append(filter_seconds)
        runtime["rollout_estimate_seconds"].append(rollout_seconds)
        runtime["oracle_seconds"].append(oracle_seconds)
        messages = list(context["messages"][:3])
        truth_values, estimates = evaluate_pairs(config, context, messages=messages)
        start = time.perf_counter()
        aware = policy_choice(config, context, messages, aware=True)
        decision_seconds = time.perf_counter() - start
        receiver_ids = sorted({target for _, target in estimates})
        agnostic = policy_choice(config, context, messages, aware=False)
        role_only = select_role_only(
            messages, receiver_ids, context["roles"], config["experiment"]["compatibility"]
        )
        choices = {"aware": aware, "agnostic": agnostic, "role_only": role_only}
        task_returns = {
            name: rollout_task_return(context, (messages[key[0]], key[1]) if key is not None else None, config)
            for name, key in choices.items()
        }
        regret = max([0.0, *truth_values.values()]) - (truth_values.get(aware, 0.0) if aware else 0.0)
        hazards_here = [status for position, status in context["world"].hazards if position == message.location]
        trace = {
            "seed": context["seed"], "step": context["step"], "message_kind": message.kind,
            "message_status": message.status, "receiver_role": context["roles"][receiver].value,
            "information_age": max(0, int(context["step"]) - int(message.observed_at)),
            "hazard_state_at_location": hazards_here[0] if hazards_here else "none",
            "vochat": estimate.value_of_communication, "oracle_mv": truth,
            "regret": regret, "selected_pair": aware,
            "sampled_message_location": message.location, "sampled_receiver": receiver,
            "aware_minus_agnostic_return": task_returns["aware"] - task_returns["agnostic"],
            "aware_minus_role_only_return": task_returns["aware"] - task_returns["role_only"],
        }
        rows.append({**trace,
                     "particle_filter_seconds": filter_seconds,
                     "rollout_estimate_seconds": rollout_seconds,
                     "aware_decision_seconds": decision_seconds,
                     "oracle_seconds": oracle_seconds})
        if task_returns["aware"] < task_returns["agnostic"] or task_returns["aware"] < task_returns["role_only"]:
            failures.append(trace)
    failures.sort(key=lambda item: (item["regret"], -item["information_age"]), reverse=True)
    examples = failures[:int(config["experiment"]["failure_trace_count"])]
    headline = {f"mean_{key}": mean(values) for key, values in runtime.items()}
    headline["mean_aware_decision_seconds"] = mean([row["aware_decision_seconds"] for row in rows])
    headline["failure_context_count"] = len(failures)
    headline["mean_regret"] = mean([row["regret"] for row in rows])
    headline["mean_aware_minus_agnostic_return"] = mean([row["aware_minus_agnostic_return"] for row in rows])
    headline["mean_aware_minus_role_only_return"] = mean([row["aware_minus_role_only_return"] for row in rows])
    return rows, {"headline": headline, "runtime_breakdown": headline.copy(),
                  "failure_traces": examples,
                  "failure_count": len(failures)}


def run(config: dict[str, Any]):
    return run_study("E12", config, analyze)
