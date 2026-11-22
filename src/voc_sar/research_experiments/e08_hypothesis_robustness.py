"""E08: Exploratory threshold and intervention checks on study contexts."""

from collections import defaultdict
from dataclasses import asdict
from typing import Any

from voc_sar.evaluation.metrics import bootstrap_paired_relative_reduction
from voc_sar.research_experiments.common import (
    bootstrap, clustered_bootstrap, mean, oracle_estimate, policy_choice,
    rollout_task_return, run_study,
)


def analyze(config: dict[str, Any], contexts: list[dict[str, Any]]):
    experiment = config["experiment"]
    evaluation = config["evaluation"]
    thresholds = evaluation["thresholds"]
    communication = config["communication"]
    margin = float(experiment["delayed_value_margin"])
    h_myo = int(communication["myopic_horizon"])
    h_eval = int(communication["oracle_horizon"])
    primary_delta = float(thresholds.get("receiver_value_difference", 0.5))
    deltas = sorted({primary_delta, *map(float, experiment["delta1_grid"])})
    rows = []
    h1_by_seed = defaultdict(list)
    h3_by_seed = defaultdict(list)
    h2_by_seed = defaultdict(list)
    h2_matched_by_seed = defaultdict(list)
    rates_by_seed = defaultdict(list)
    matches_by_delta = defaultdict(list)
    for context in contexts:
        messages = list(context["messages"])
        receivers = [i for i in range(len(context["roles"])) if i != context["sender_id"]]
        if not messages or not receivers:
            continue
        seed = int(context["seed"])
        identity = {"seed": seed, "step": context["step"], "layout_seed": context["layout_seed"]}
        pairs = {}
        for index, message in enumerate(messages):
            for receiver in receivers:
                intervention = {**context, "message": message, "receiver_id": receiver}
                pairs[index, receiver] = {
                    horizon: oracle_estimate(config, intervention, horizon=horizon).marginal_value
                    for horizon in (h_myo, h_eval)
                }
        primary_index = next(
            (index for index, message in enumerate(messages) if message.kind != "intention"), None
        )
        if primary_index is not None and len(receivers) >= 2:
            first, second = [pairs[primary_index, receiver][h_eval] for receiver in receivers[:2]]
            difference = abs(first - second)
            for delta in deltas:
                flag = int(difference > delta)
                matches_by_delta[delta].append(flag)
                if delta == primary_delta:
                    h1_by_seed[seed].append(flag)

        delayed = any(abs(values[h_eval] - values[h_myo]) > margin for values in pairs.values())
        oracle = max(pairs, key=lambda pair: pairs[pair][h_eval])
        oracle = oracle if pairs[oracle][h_eval] > 0 else None
        myopic = policy_choice(config, context, messages, aware=True, horizon=h_myo)
        aware = policy_choice(config, context, messages, aware=True)
        agnostic = policy_choice(config, context, messages, aware=False)
        high_cost = policy_choice(config, context, messages, aware=True, cost_weight=0.5)
        gain = int(aware == oracle) - int(myopic == oracle)
        if delayed:
            h3_by_seed[seed].append(gain)

        def task_return(choice):
            return rollout_task_return(
                context, (messages[choice[0]], choice[1]) if choice else None, config
            )

        aware_assigned = aware or policy_choice(config, context, messages, aware=True, force_send=True)
        agnostic_assigned = agnostic or policy_choice(config, context, messages, aware=False, force_send=True)
        aware_return, agnostic_return = task_return(aware), task_return(agnostic)
        assigned_aware_return = task_return(aware_assigned)
        assigned_agnostic_return = task_return(agnostic_assigned)
        h2_by_seed[seed].append(aware_return - agnostic_return)
        h2_matched_by_seed[seed].append(assigned_aware_return - assigned_agnostic_return)
        rates_by_seed[seed].append((int(aware is not None), int(high_cost is not None)))
        rows.append({
            **identity,
            "messages": [asdict(message) for message in messages],
            "primary_message_index": primary_index,
            "oracle_pairs": [
                {"message_index": index, "receiver_id": receiver, "marginal_values": values}
                for (index, receiver), values in pairs.items()
            ],
            "myopic_choice": myopic, "nonmyopic_choice": aware, "oracle_choice": oracle,
            "delayed": delayed, "agreement_gain": gain,
            "myopic_agreement": int(myopic == oracle), "nonmyopic_agreement": int(aware == oracle),
            "aware_return": aware_return, "agnostic_return": agnostic_return,
            "assigned_aware_choice": aware_assigned, "assigned_agnostic_choice": agnostic_assigned,
            "assigned_aware_return": assigned_aware_return,
            "assigned_agnostic_return": assigned_agnostic_return,
            "low_cost_send": int(aware is not None), "high_cost_send": int(high_cost is not None),
        })

    h1_curve = {f"{delta:g}": mean(flags) for delta, flags in matches_by_delta.items()}
    h2_seed = [mean(values) for values in h2_by_seed.values()]
    h2_matched_seed = [mean(values) for values in h2_matched_by_seed.values()]
    rates = [(mean([pair[0] for pair in values]), mean([pair[1] for pair in values]))
             for values in rates_by_seed.values()]
    low, high = (map(mean, zip(*rates)) if rates else (None, None))
    relative = 1 - high / low if low else None
    h4_interval, undefined_draws = bootstrap_paired_relative_reduction(
        rates, float(evaluation.get("confidence_level", 0.95)),
        int(evaluation.get("bootstrap_samples", 1000)), 8025,
    )
    return rows, {
        "scope": "exploratory context interventions",
        "headline": {
            "h1_primary_fraction": h1_curve.get(f"{primary_delta:g}"),
            "h2_aware_minus_agnostic_return": mean(h2_seed),
            "h2_budget_matched_return": mean(h2_matched_seed),
            "h2_budget_matched_seed_count": len(h2_matched_seed),
            "h3_delayed_context_gain": mean([value for values in h3_by_seed.values() for value in values]),
            "h3_delayed_seed_count": len(h3_by_seed),
            "h4_positive_rate_drop_relative": relative,
        },
        "h1_threshold_curve": h1_curve,
        "h1_primary_ci": clustered_bootstrap(list(h1_by_seed.values()), config, 8021),
        "h2_paired_ci": bootstrap(h2_seed, config, 8022),
        "h2_budget_matched_paired_ci": bootstrap(h2_matched_seed, config, 8023),
        "h3_delayed_value_margin": margin,
        "h3_delayed_context_count": sum(map(len, h3_by_seed.values())),
        "h3_delayed_paired_ci": clustered_bootstrap(list(h3_by_seed.values()), config, 8024),
        "h4_low_rate": low, "h4_high_rate": high, "h4_paired_ci": h4_interval,
        "h4_undefined_bootstrap_draws": undefined_draws,
    }


def run(config: dict[str, Any]):
    return run_study("E08", config, analyze)

