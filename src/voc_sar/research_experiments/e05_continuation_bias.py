"""E05: Offline continuation-policy sensitivity for send/no-send values."""

import random
from typing import Any

from voc_sar.communication.estimator import _rollout_return
from voc_sar.communication.selector import select_communication, shortlist
from voc_sar.environment.sar_grid import SARGrid
from voc_sar.policies.task import candidate_messages, task_policies
from voc_sar.research_experiments.common import make_estimator, mean, oracle_estimate, run_study, spearman
from voc_sar.types import CommunicationChoice, Role


def _continued_return(config, context, message, receiver, mode, intervention, seed):
    world: SARGrid = context["world"].clone()
    world.reseed_dynamics(seed)
    policies = task_policies(world.roles)
    observations = world.observe()
    if intervention:
        slots = [None] * len(observations)
        slots[context["sender_id"]] = CommunicationChoice(receiver, message)
        observations = world.deliver(slots)
    total, discount = 0.0, 1.0
    horizon = int(config["communication"]["oracle_horizon"])
    gamma = float(config["environment"].get("gamma", 0.99))
    scout_id = next((i for i, role in enumerate(world.roles) if role == Role.SCOUT), context["sender_id"])
    other_receivers = [i for i in range(len(world.roles)) if i != scout_id]
    estimator = make_estimator(config, seed=seed, particles=8, rollout_count=4)
    for step in range(horizon):
        if world.terminated:
            break
        if step > 0 and mode == "fixed_frequency" and step % int(config["communication"].get("fixed_frequency", 3)) == 0:
            obs = observations[scout_id]
            candidates = candidate_messages(obs, policies[scout_id].act(obs))
            ranked = shortlist(scout_id, candidates, other_receivers, 1,
                               current_step=obs.time_step, receiver_roles=world.roles,
                               teammate_tracks=obs.teammates, receiver_summaries=obs.receiver_summaries)
            slots = [None] * len(observations)
            if ranked:
                fact, target = ranked[0]
                slots[scout_id] = CommunicationChoice(target, fact)
                observations = world.deliver(slots)
        elif step > 0 and mode == "aware":
            obs = observations[scout_id]
            candidates = candidate_messages(obs, policies[scout_id].act(obs))
            choice = select_communication(
                scout_id, candidates, other_receivers, 1, estimator,
                observation=obs, receiver_roles=world.roles,
                shortlist_size=min(3, int(config["communication"]["shortlist_size"])),
                seed=seed + step,
            )
            slots = [None] * len(observations)
            slots[scout_id] = choice
            observations = world.deliver(slots)
        actions = [policy.act(observation) for policy, observation in zip(policies, observations)]
        result = world.step(actions)
        total += discount * result.reward
        discount *= gamma
        observations = result.observations
    return total


def analyze(config: dict[str, Any], contexts: list[dict[str, Any]]):
    modes = list(config["experiment"]["continuations"])
    rows = []
    values_by_mode = {mode: [] for mode in modes}
    standard_values = []
    top1_matches = {mode: [] for mode in modes}
    rollout_count = int(config["experiment"]["rollout_count"])
    for context in contexts:
        message, receiver = context["message"], int(context["receiver_id"])
        standard = oracle_estimate(config, context).marginal_value
        standard_values.append(standard)
        for mode in modes:
            differences = []
            send_returns, no_send_returns = [], []
            for index in range(rollout_count):
                seed = int(context["seed"]) * 10007 + index
                if mode == "disabled":
                    world0 = context["world"].clone()
                    world1 = context["world"].clone()
                    world0.reseed_dynamics(seed)
                    world1.reseed_dynamics(seed)
                    policies = task_policies(context["roles"])
                    no_send = _rollout_return(world0, policies, int(config["communication"]["oracle_horizon"]), float(config["environment"].get("gamma", .99)))
                    sent = _rollout_return(world1, policies, int(config["communication"]["oracle_horizon"]), float(config["environment"].get("gamma", .99)), (context["sender_id"], receiver, message))
                else:
                    no_send = _continued_return(config, context, message, receiver, mode, False, seed)
                    sent = _continued_return(config, context, message, receiver, mode, True, seed)
                no_send_returns.append(no_send)
                send_returns.append(sent)
                differences.append(sent - no_send)
            value = mean(differences) or 0.0
            values_by_mode[mode].append(value)
            top1_matches[mode].append(float((value > 0) == (standard > 0)))
            rows.append({
                "seed": context["seed"], "step": context["step"], "continuation": mode,
                "standard_value": standard, "continuation_value": value,
                "mae": abs(value - standard), "spearman_input_value": value,
                "mean_send_return": mean(send_returns), "mean_no_send_return": mean(no_send_returns),
                "top1_agreement": top1_matches[mode][-1],
            })
    headline = {}
    summary_by_mode = {}
    for mode in modes:
        group = [row for row in rows if row["continuation"] == mode]
        summary_by_mode[mode] = {
            "mae_vs_disabled_oracle": mean([row["mae"] for row in group]),
            "spearman_vs_disabled_oracle": spearman(standard_values, values_by_mode[mode]),
            "top1_agreement": mean(top1_matches[mode]),
        }
        for metric, value in summary_by_mode[mode].items():
            if isinstance(value, (float, int)):
                headline[f"{mode}_{metric}"] = value
    return rows, {"headline": headline, "by_continuation": summary_by_mode}


def run(config: dict[str, Any]):
    return run_study("E05", config, analyze)
