"""Reproducible baseline, horizon, cost, and held-out oracle experiments."""

from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from voc_sar.communication.estimator import (
    PairedRolloutEstimator,
    ParticleReceiverEstimator,
)
from voc_sar.communication.selector import (
    SelectionTrace,
    select_communication,
    shortlist,
)
from voc_sar.environment.sar_grid import SARGrid
from voc_sar.evaluation.metrics import (
    action_agreement,
    bootstrap_cluster_mean_interval,
    bootstrap_mean_interval,
    communication_reduction,
    filter_recall,
    pair_regret,
    receiver_regret,
    voc_mean_absolute_error,
)
from voc_sar.evaluation.oracle import CounterfactualOracle
from voc_sar.policies.task import candidate_messages, task_policies
from voc_sar.types import CommunicationChoice, Role
from voc_sar.visualizations import create_search_animation, create_visualizations


def run_experiments(config: dict[str, Any]) -> dict[str, Any]:
    """Run matched seed comparisons and write JSON episode and summary results."""
    _validate_config(config)
    communication_config = config["communication"]
    evaluation_config = config["evaluation"]
    seeds = [int(seed) for seed in evaluation_config["seeds"]]
    costs = [float(cost) for cost in communication_config["cost_sweep"]]
    myopic_horizon = int(communication_config["myopic_horizon"])
    nonmyopic_horizon = int(communication_config["nonmyopic_horizon"])
    context_limit = max(1, math.ceil(
        int(evaluation_config["held_out_contexts"]) / len(seeds)
    ))

    conditions: list[tuple[str, float, int]] = [
        ("never", 0.0, nonmyopic_horizon),
        ("random", 0.0, nonmyopic_horizon),
        ("full", 0.0, nonmyopic_horizon),
        ("fixed_frequency", 0.0, nonmyopic_horizon),
    ]
    for cost in costs:
        conditions.extend((
            ("aware_myopic", cost, myopic_horizon),
            ("aware_nonmyopic", cost, nonmyopic_horizon),
            ("agnostic_nonmyopic", cost, nonmyopic_horizon),
        ))

    episode_rows: list[dict[str, Any]] = []
    search_frames_by_method: dict[str, list[dict[str, Any]]] = {}
    started_at = time.perf_counter()
    for method, cost_weight, horizon in conditions:
        for seed in seeds:
            capture_search = method not in search_frames_by_method
            method_frames: list[dict[str, Any]] | None = [] if capture_search else None
            episode_rows.append(_run_episode(
                config,
                seed,
                method,
                cost_weight,
                horizon,
                context_limit,
                method_frames,
            ))
            if method_frames is not None:
                search_frames_by_method[method] = method_frames

    diagnostics = _run_held_out_diagnostics(config, seeds)
    summary = _summarize(config, episode_rows, diagnostics)
    summary["runtime_seconds"] = time.perf_counter() - started_at
    summary["config"] = config
    output_root = Path(evaluation_config.get("output_dir", "results"))
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_root / f"run_{run_stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["run_directory"] = str(output_dir.resolve())
    summary["visualizations_directory"] = str((output_dir / "graphs").resolve())
    (output_dir / "episodes.json").write_text(
        json.dumps(episode_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    create_visualizations(episode_rows, summary, output_dir / "graphs")
    for method, frames in search_frames_by_method.items():
        create_search_animation(frames, output_dir / f"search-operations-{method}.gif")
    return summary


def _run_episode(
    config: dict[str, Any],
    seed: int,
    method: str,
    cost_weight: float,
    valuation_horizon: int,
    context_limit: int,
    search_frames: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    episode_started = time.perf_counter()
    env_config = config["environment"]
    comm_config = config["communication"]
    env = SARGrid(env_config)
    observations = env.reset(seed)
    policies = task_policies(env.roles)
    sender_id = next(
        (index for index, role in enumerate(env.roles) if role == Role.SCOUT),
        0,
    )
    receivers = [index for index in range(len(env.roles)) if index != sender_id]
    estimator = PairedRolloutEstimator(
        ParticleReceiverEstimator(env_config, seed=seed),
        policies,
        particles=int(comm_config["particles"]),
        rollout_count=int(comm_config["rollout_count"]),
        gamma=float(env_config.get("gamma", 0.99)),
        cost_weight=cost_weight,
        unit_cost=float(comm_config.get("unit_cost", 1.0)),
        seed=seed,
    )
    rng = random.Random(seed * 1009 + _method_number(method))
    rewards: list[float] = []
    sent_count = 0
    decision_count = 0
    valuation_contexts = 0
    estimator_standard_errors: list[float] = []
    selected_standard_errors: list[float] = []
    step = 0
    if search_frames is not None:
        _capture_search_frame(search_frames, env, [], None, method, seed)
    while not env.terminated:
        sender_observation = observations[sender_id]
        planned_action = policies[sender_id].act(sender_observation)
        messages = candidate_messages(sender_observation, planned_action)
        choice: CommunicationChoice | None = None

        if method in {"aware_myopic", "aware_nonmyopic", "agnostic_nonmyopic"}:
            if valuation_contexts < context_limit and messages and receivers:
                decision_count += 1
                valuation_contexts += 1
                trace = SelectionTrace()
                choice = select_communication(
                    sender_id,
                    messages,
                    receivers,
                    valuation_horizon,
                    estimator,
                    observation=sender_observation,
                    shortlist_size=int(comm_config["shortlist_size"]),
                    receiver_roles=env.roles,
                    aware=method != "agnostic_nonmyopic",
                    seed=seed,
                    hazard_move_probability=float(
                        env_config.get("hazard_move_probability", 0.0)
                    ),
                    hazard_toggle_probability=float(
                        env_config.get("hazard_toggle_probability", 0.0)
                    ),
                    trace=trace,
                )
                estimator_standard_errors.extend(
                    estimate.standard_error
                    for _, _, estimate in trace.evaluations
                    if estimate.standard_error is not None
                )
                if choice is not None:
                    selected_standard_errors.extend(
                        estimate.standard_error
                        for message, receiver_id, estimate in trace.evaluations
                        if message == choice.message
                        and receiver_id == choice.receiver_id
                        and estimate.standard_error is not None
                    )
        elif method == "random" and messages and receivers:
            decision_count += 1
            choice = CommunicationChoice(
                rng.choice(receivers),
                rng.choice(messages),
            )
        elif method == "full" and messages and receivers:
            decision_count += 1
            ranked = shortlist(
                sender_id,
                messages,
                receivers,
                1,
                current_step=sender_observation.time_step,
                receiver_roles=env.roles,
                teammate_tracks=sender_observation.teammates,
                receiver_summaries=sender_observation.receiver_summaries,
            )
            if ranked:
                message, receiver_id = ranked[0]
                choice = CommunicationChoice(receiver_id, message)
        elif method == "fixed_frequency":
            if (
                step % max(1, int(comm_config.get("fixed_frequency", 3))) == 0
                and messages
                and receivers
            ):
                decision_count += 1
                ranked = shortlist(
                    sender_id,
                    messages,
                    receivers,
                    1,
                    current_step=sender_observation.time_step,
                    receiver_roles=env.roles,
                    teammate_tracks=sender_observation.teammates,
                    receiver_summaries=sender_observation.receiver_summaries,
                )
                if ranked:
                    message, receiver_id = ranked[0]
                    choice = CommunicationChoice(receiver_id, message)
        elif method == "never":
            decision_count += 1
        else:
            raise ValueError(f"unknown communication condition: {method}")

        communication = [None] * len(observations)
        if choice is not None:
            communication[sender_id] = choice
            sent_count += 1
        observations = env.deliver(communication)
        actions = [
            policy.act(observation)
            for policy, observation in zip(policies, observations)
        ]
        result = env.step(actions)
        if search_frames is not None:
            _capture_search_frame(search_frames, env, actions, choice, method, seed)
        rewards.append(result.reward)
        observations = result.observations
        step += 1

    remaining_victims = sum(status != "rescued" for _, status in env.victims)
    return {
        "seed": seed,
        "method": method,
        "cost_weight": cost_weight,
        "horizon": valuation_horizon,
        "discounted_task_return": sum(
            (float(env_config.get("gamma", 0.99)) ** index) * reward
            for index, reward in enumerate(rewards)
        ),
        "messages": sent_count,
        "decisions": decision_count,
        "valuation_contexts": valuation_contexts,
        "communication_rate": sent_count / decision_count if decision_count else 0.0,
        "positive_voc_rate": (
            sent_count / decision_count
            if decision_count and method in {
                "aware_myopic", "aware_nonmyopic", "agnostic_nonmyopic"
            }
            else None
        ),
        "mean_candidate_estimator_standard_error": (
            _mean(estimator_standard_errors) if estimator_standard_errors else None
        ),
        "mean_selected_estimator_standard_error": (
            _mean(selected_standard_errors) if selected_standard_errors else None
        ),
        "runtime_seconds": time.perf_counter() - episode_started,
        "steps": step,
        "rescued_victims": len(env.victims) - remaining_victims,
        "unrescued_victims": remaining_victims,
        "failure_case": "unrescued_victims" if remaining_victims else None,
    }


def _capture_search_frame(
    frames: list[dict[str, Any]], env: SARGrid, actions: list[str],
    choice: CommunicationChoice | None, method: str, seed: int,
) -> None:
    frames.append({
        "step": env.step_count,
        "size": env.grid_size,
        "roles": [role.value for role in env.roles],
        "positions": [list(position) for position in env.positions],
        "victims": [[list(position), status] for position, status in env.victims],
        "hazards": [[list(position), status] for position, status in env.hazards],
        "actions": list(actions),
        "message": (
            f"Agent 0 -> {choice.receiver_id}: {choice.message.kind} at {choice.message.location}"
            if choice is not None else "No message sent"
        ),
        "method": method,
        "seed": seed,
    })


def _run_held_out_diagnostics(
    config: dict[str, Any], seeds: list[int]
) -> dict[str, Any]:
    env_config = config["environment"]
    comm_config = config["communication"]
    evaluation_config = config["evaluation"]
    thresholds = evaluation_config.get("thresholds", {})
    contexts_per_seed = max(1, math.ceil(
        int(evaluation_config["held_out_contexts"]) / len(seeds)
    ))
    gamma = float(env_config.get("gamma", 0.99))
    h_myopic = int(comm_config["myopic_horizon"])
    h_nonmyopic = int(comm_config["nonmyopic_horizon"])
    h_eval = int(comm_config["oracle_horizon"])
    rollout_count = int(comm_config["rollout_count"])
    particle_count = int(comm_config["particles"])
    shortlist_size = int(comm_config["shortlist_size"])
    useful_threshold = float(comm_config["useful_message_threshold"])
    oracle_threshold = float(comm_config["oracle_value_threshold"])

    receiver_differences: list[float] = []
    receiver_differentiation_flags: list[float] = []
    receiver_flags_by_seed: list[list[float]] = []
    pair_regrets: list[float] = []
    receiver_regrets: list[float] = []
    mae_predictions: list[float] = []
    mae_targets: list[float] = []
    useful_messages: list[bool] = []
    recalls: list[float] = []
    myopic_actions: list[object] = []
    nonmyopic_actions: list[object] = []
    oracle_actions: list[object] = []
    horizon_agreement_gains: list[float] = []
    horizon_gains_by_seed: list[list[float]] = []
    context_count = 0

    for seed in seeds:
        env = SARGrid(env_config)
        observations = env.reset(seed + 77_777)
        policies = task_policies(env.roles)
        sender_id = next(
            (index for index, role in enumerate(env.roles) if role == Role.SCOUT),
            0,
        )
        receivers = [index for index in range(len(env.roles)) if index != sender_id]
        if len(receivers) < 1:
            continue
        estimator = PairedRolloutEstimator(
            ParticleReceiverEstimator(env_config, seed=seed + 77_777),
            policies,
            particles=particle_count,
            rollout_count=rollout_count,
            gamma=gamma,
            cost_weight=0.0,
            unit_cost=float(comm_config.get("unit_cost", 1.0)),
            seed=seed + 77_777,
        )
        oracle = CounterfactualOracle(
            policies,
            rollout_count=rollout_count,
            gamma=gamma,
            cost_weight=0.0,
            unit_cost=float(comm_config.get("unit_cost", 1.0)),
            seed=seed + 77_777,
        )
        contexts_for_seed = 0
        receiver_flags_for_seed: list[float] = []
        horizon_gains_for_seed: list[float] = []
        while not env.terminated and contexts_for_seed < contexts_per_seed:
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
            all_pairs = shortlist(
                sender_id,
                messages,
                receivers,
                max(1, len(messages) * len(receivers)),
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
                context_count += 1
                contexts_for_seed += 1
                oracle_values: dict[tuple[int, int], float] = {}
                predictions: dict[int, dict[tuple[int, int], float]] = {
                    h_myopic: {},
                    h_nonmyopic: {},
                }
                for message_index, (message, receiver_id) in enumerate(all_pairs):
                    pair = (message_index, receiver_id)
                    oracle_values[pair] = oracle.evaluate(
                        env.clone(),
                        sender_id,
                        receiver_id,
                        message,
                        h_eval,
                    ).marginal_value
                for horizon in dict.fromkeys((h_myopic, h_nonmyopic)):
                    for message_index, (message, receiver_id) in enumerate(top_pairs):
                        pair = (message_index, receiver_id)
                        predictions[horizon][pair] = estimator.estimate(
                            sender_id,
                            receiver_id,
                            message,
                            horizon,
                            observation,
                        ).marginal_value

                top_to_all_indices = {}
                for top_index, (message, receiver_id) in enumerate(top_pairs):
                    all_index = next(
                        index for index, (all_message, all_receiver) in enumerate(all_pairs)
                        if all_message == message and all_receiver == receiver_id
                    )
                    top_to_all_indices[(top_index, receiver_id)] = all_index
                top_predictions: dict[int, dict[tuple[int, int], float]] = {}
                for horizon, values in predictions.items():
                    top_predictions[horizon] = {
                        (top_to_all_indices[(top_index, receiver_id)], receiver_id): value
                        for (top_index, receiver_id), value in values.items()
                    }
                all_pair_values = {
                    (index, receiver_id): oracle_values[(index, receiver_id)]
                    for index, (_, receiver_id) in enumerate(all_pairs)
                }
                recalls.append(filter_recall(
                    all_pair_values,
                    [
                        (top_to_all_indices[(index, receiver_id)], receiver_id)
                        for index, (_, receiver_id) in enumerate(top_pairs)
                    ],
                    oracle_threshold,
                ))
                for horizon, predicted_values in top_predictions.items():
                    if horizon == h_nonmyopic:
                        mae_predictions.extend(predicted_values.values())
                        mae_targets.extend(
                            oracle_values[pair] for pair in predicted_values
                        )

                def best_action(
                    values: dict[tuple[int, int], float]
                ) -> tuple[int, int] | None:
                    if not values:
                        return None
                    key, value = max(values.items(), key=lambda item: (item[1], item[0]))
                    return key if value > 0 else None

                myopic_action = best_action(top_predictions[h_myopic])
                nonmyopic_action = best_action(top_predictions[h_nonmyopic])
                oracle_action = best_action(oracle_values)
                myopic_actions.append(myopic_action)
                nonmyopic_actions.append(nonmyopic_action)
                oracle_actions.append(oracle_action)
                horizon_gain = (
                    float(nonmyopic_action == oracle_action)
                    - float(myopic_action == oracle_action)
                )
                horizon_agreement_gains.append(horizon_gain)
                horizon_gains_for_seed.append(horizon_gain)
                pair_regrets.append(pair_regret(oracle_values, nonmyopic_action))
                if nonmyopic_action is not None:
                    useful_messages.append(
                        oracle_values[nonmyopic_action] > useful_threshold
                    )
                primary_message = next(
                    (message for message in messages if message.kind != "intention"),
                    None,
                )
                if primary_message is not None and len(receivers) >= 2:
                    values = [
                        oracle.evaluate(
                            env.clone(), sender_id, receiver_id,
                            primary_message, h_eval,
                        ).marginal_value
                        for receiver_id in receivers
                    ]
                    receiver_differences.append(abs(values[0] - values[1]))
                    differentiation_flag = float(
                        abs(values[0] - values[1])
                        > float(thresholds.get("receiver_value_difference", 0.5))
                    )
                    receiver_differentiation_flags.append(differentiation_flag)
                    receiver_flags_for_seed.append(differentiation_flag)
                    chosen_receiver = (
                        nonmyopic_action[1]
                        if isinstance(nonmyopic_action, tuple)
                        and any(
                            message == primary_message and receiver == nonmyopic_action[1]
                            and index == nonmyopic_action[0]
                            for index, (message, receiver) in enumerate(all_pairs)
                        )
                        else None
                    )
                    receiver_regrets.append(receiver_regret(
                        dict(zip(receivers, values)),
                        chosen_receiver,
                    ))

            actions = [
                policy.act(item)
                for policy, item in zip(policies, observations)
            ]
            result = env.step(actions)
            observations = result.observations
        if receiver_flags_for_seed:
            receiver_flags_by_seed.append(receiver_flags_for_seed)
        if horizon_gains_for_seed:
            horizon_gains_by_seed.append(horizon_gains_for_seed)

    mae = (
        voc_mean_absolute_error(mae_predictions, mae_targets)
        if mae_predictions else 0.0
    )
    receiver_fraction = (
        _mean(receiver_differentiation_flags)
    )
    bootstrap_samples = int(evaluation_config.get("bootstrap_samples", 1000))
    confidence_level = float(evaluation_config.get("confidence_level", 0.95))
    seed = int(seeds[0])
    receiver_fraction_interval = (
        bootstrap_cluster_mean_interval(
            receiver_flags_by_seed,
            confidence_level,
            bootstrap_samples,
            seed + 311,
        )
        if receiver_differentiation_flags else (0.0, 0.0)
    )
    horizon_gain_interval = (
        bootstrap_cluster_mean_interval(
            horizon_gains_by_seed,
            confidence_level,
            bootstrap_samples,
            seed + 313,
        )
        if horizon_agreement_gains else (0.0, 0.0)
    )
    return {
        "held_out_contexts": context_count,
        "receiver_differentiation_contexts": len(receiver_differences),
        "receiver_value_differences": receiver_differences,
        "receiver_differentiation_fraction": receiver_fraction,
        "receiver_differentiation_confidence_interval": receiver_fraction_interval,
        "receiver_regret_mean": _mean(receiver_regrets),
        "pair_regret_mean": _mean(pair_regrets),
        "voc_mae_nonmyopic": mae,
        "filter_recall": _mean(recalls),
        "useful_message_rate": _mean([float(value) for value in useful_messages]),
        "agreement_myopic_to_oracle": action_agreement(myopic_actions, oracle_actions),
        "agreement_nonmyopic_to_oracle": action_agreement(nonmyopic_actions, oracle_actions),
        "agreement_gain_nonmyopic_over_myopic": (
            action_agreement(nonmyopic_actions, oracle_actions)
            - action_agreement(myopic_actions, oracle_actions)
        ),
        "agreement_gain_confidence_interval": horizon_gain_interval,
        "hypothesis_targets": {
            "H1_receiver_differentiation_fraction": float(
                thresholds.get("receiver_differentiation_fraction", 0.30)
            ),
            "H1_receiver_value_difference": float(
                thresholds.get("receiver_value_difference", 0.5)
            ),
            "H3_horizon_agreement_gain": float(
                thresholds.get("horizon_agreement_gain", 0.10)
            ),
        },
    }


def _summarize(
    config: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    confidence = float(evaluation.get("confidence_level", 0.95))
    bootstrap_samples = int(evaluation.get("bootstrap_samples", 1000))
    seed = int(evaluation["seeds"][0])
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        groups[(row["method"], row["cost_weight"])].append(row)
    conditions: list[dict[str, Any]] = []
    for (method, cost), rows in sorted(groups.items()):
        returns = [row["discounted_task_return"] for row in rows]
        interval = bootstrap_mean_interval(
            returns, confidence, bootstrap_samples, seed + _method_number(method)
        )
        conditions.append({
            "method": method,
            "cost_weight": cost,
            "episodes": len(rows),
            "mean_task_return": _mean(returns),
            "task_return_confidence_interval": interval,
            "mean_messages": _mean([float(row["messages"]) for row in rows]),
            "mean_message_count": sum(row["messages"] for row in rows),
            "mean_valuation_contexts": _mean(
                [float(row["valuation_contexts"]) for row in rows]
            ),
            "communication_rate": _mean(
                [float(row["communication_rate"]) for row in rows]
            ),
            "positive_voc_communication_rate": (
                _mean([
                    float(row["positive_voc_rate"])
                    for row in rows
                    if row["positive_voc_rate"] is not None
                ])
                if any(row["positive_voc_rate"] is not None for row in rows)
                else None
            ),
            "mean_candidate_estimator_standard_error": _optional_mean(
                [
                    float(row["mean_candidate_estimator_standard_error"])
                    for row in rows
                    if row["mean_candidate_estimator_standard_error"] is not None
                ]
            ),
            "mean_selected_estimator_standard_error": _optional_mean(
                [
                    float(row["mean_selected_estimator_standard_error"])
                    for row in rows
                    if row["mean_selected_estimator_standard_error"] is not None
                ]
            ),
            "mean_episode_runtime_seconds": _mean(
                [float(row["runtime_seconds"]) for row in rows]
            ),
            "mean_unrescued_victims": _mean(
                [float(row["unrescued_victims"]) for row in rows]
            ),
        })

    by_seed = {
        (row["method"], row["cost_weight"], row["seed"]): row
        for row in episode_rows
    }
    h2_differences_by_seed: dict[int, list[float]] = defaultdict(list)
    h2_budget_differences_by_seed: dict[int, list[float]] = defaultdict(list)
    budget_match_tolerance = int(
        evaluation.get("message_budget_match_tolerance", 0)
    )
    for cost in config["communication"]["cost_sweep"]:
        for episode_seed in evaluation["seeds"]:
            aware = by_seed.get(("aware_nonmyopic", float(cost), int(episode_seed)))
            agnostic = by_seed.get(("agnostic_nonmyopic", float(cost), int(episode_seed)))
            if aware is not None and agnostic is not None:
                difference = (
                    aware["discounted_task_return"] - agnostic["discounted_task_return"]
                )
                h2_differences_by_seed[int(episode_seed)].append(difference)
                if (
                    aware["messages"] > 0
                    and agnostic["messages"] > 0
                    and abs(aware["messages"] - agnostic["messages"])
                    <= budget_match_tolerance
                ):
                    h2_budget_differences_by_seed[int(episode_seed)].append(difference)
    h2_differences = [
        _mean(h2_differences_by_seed[episode_seed])
        for episode_seed in sorted(h2_differences_by_seed)
    ]
    h2_budget_matched_differences = [
        _mean(h2_budget_differences_by_seed[episode_seed])
        for episode_seed in sorted(h2_budget_differences_by_seed)
    ]
    h2_interval = (
        bootstrap_mean_interval(
            h2_differences, confidence, bootstrap_samples, seed + 9001
        )
        if h2_differences else (0.0, 0.0)
    )
    h2_budget_interval = (
        bootstrap_mean_interval(
            h2_budget_matched_differences,
            confidence,
            bootstrap_samples,
            seed + 9007,
        )
        if h2_budget_matched_differences else (0.0, 0.0)
    )
    full_rows = [
        row for row in episode_rows
        if row["method"] == "full"
    ]
    aware_rows = [
        row for row in episode_rows
        if row["method"] == "aware_nonmyopic" and row["cost_weight"] == 0.0
    ]
    per_seed_reductions = []
    matched_return_reductions = []
    return_match_tolerance = float(
        evaluation.get("return_match_tolerance", 1.0)
    )
    if full_rows:
        full_by_seed = {row["seed"]: row for row in full_rows}
        for row in aware_rows:
            baseline = full_by_seed.get(row["seed"])
            if baseline is not None and baseline["messages"] > 0:
                reduction = communication_reduction(row["messages"], baseline["messages"])
                per_seed_reductions.append(reduction)
                if abs(
                    row["discounted_task_return"]
                    - baseline["discounted_task_return"]
                ) <= return_match_tolerance:
                    matched_return_reductions.append(reduction)

    low_cost = 0.0
    high_cost = 0.5
    low_rate = _condition_value(conditions, "aware_nonmyopic", low_cost)
    high_rate = _condition_value(conditions, "aware_nonmyopic", high_cost)
    cost_rate_reduction = (
        (low_rate - high_rate) / low_rate if low_rate > 0 else 0.0
    )
    low_rate_by_seed = {
        row["seed"]: float(row["positive_voc_rate"])
        for row in episode_rows
        if row["method"] == "aware_nonmyopic"
        and row["cost_weight"] == low_cost
        and row["positive_voc_rate"] is not None
    }
    high_rate_by_seed = {
        row["seed"]: float(row["positive_voc_rate"])
        for row in episode_rows
        if row["method"] == "aware_nonmyopic"
        and row["cost_weight"] == high_cost
        and row["positive_voc_rate"] is not None
    }
    paired_cost_reductions = [
        (low_rate_by_seed[episode_seed] - high_rate_by_seed[episode_seed])
        / low_rate_by_seed[episode_seed]
        for episode_seed in sorted(low_rate_by_seed.keys() & high_rate_by_seed.keys())
        if low_rate_by_seed[episode_seed] > 0
    ]
    cost_reduction_interval = (
        bootstrap_mean_interval(
            paired_cost_reductions,
            confidence,
            bootstrap_samples,
            seed + 9011,
        )
        if paired_cost_reductions else (0.0, 0.0)
    )
    targets = evaluation.get("thresholds", {})
    pareto_frontier = [
        {
            "method": condition["method"],
            "cost_weight": condition["cost_weight"],
            "mean_messages": condition["mean_messages"],
            "mean_task_return": condition["mean_task_return"],
        }
        for condition in conditions
        if not any(
            other["mean_messages"] <= condition["mean_messages"]
            and other["mean_task_return"] >= condition["mean_task_return"]
            and (
                other["mean_messages"] < condition["mean_messages"]
                or other["mean_task_return"] > condition["mean_task_return"]
            )
            for other in conditions
        )
    ]
    hypotheses = {
        "H1_receiver_differentiation": {
            "observed_fraction": diagnostics["receiver_differentiation_fraction"],
            "confidence_interval": diagnostics[
                "receiver_differentiation_confidence_interval"
            ],
            "target_fraction": float(targets.get("receiver_differentiation_fraction", 0.30)),
            "value_difference_threshold": float(
                targets.get("receiver_value_difference", 0.5)
            ),
            "meets_target": diagnostics["receiver_differentiation_fraction"]
            >= float(targets.get("receiver_differentiation_fraction", 0.30)),
        },
        "H2_selectivity": {
            "mean_aware_minus_agnostic_return": _mean(h2_differences),
            "paired_confidence_interval": h2_interval,
            "confidence_interval_excludes_zero": (
                h2_interval[0] > 0 or h2_interval[1] < 0
            ),
            "mean_aware_minus_agnostic_return_at_matched_budget": _mean(
                h2_budget_matched_differences
            ),
            "matched_budget_confidence_interval": h2_budget_interval,
            "matched_budget_episode_count": len(h2_budget_matched_differences),
            "message_budget_match_tolerance": budget_match_tolerance,
            "matched_budget_confidence_interval_excludes_zero": (
                h2_budget_interval[0] > 0 or h2_budget_interval[1] < 0
            ) if h2_budget_matched_differences else False,
            "mean_communication_reduction_vs_full": _mean(per_seed_reductions),
            "communication_reduction_at_matched_return": (
                _mean(matched_return_reductions)
                if matched_return_reductions else None
            ),
            "matched_return_episode_count": len(matched_return_reductions),
            "return_match_tolerance": return_match_tolerance,
            "matched_return_communication_reduction_target": float(
                targets.get("communication_reduction", 0.20)
            ),
            "matched_return_reduction_meets_target": (
                bool(matched_return_reductions)
                and _mean(matched_return_reductions)
                >= float(targets.get("communication_reduction", 0.20))
            ),
            "meets_both_selectivity_tests": (
                bool(matched_return_reductions)
                and _mean(matched_return_reductions)
                >= float(targets.get("communication_reduction", 0.20))
                and bool(h2_budget_matched_differences)
                and (h2_budget_interval[0] > 0 or h2_budget_interval[1] < 0)
            ),
            "communication_reduction_target": float(
                targets.get("communication_reduction", 0.20)
            ),
        },
        "H3_downstream_value": {
            "myopic_oracle_agreement": diagnostics["agreement_myopic_to_oracle"],
            "nonmyopic_oracle_agreement": diagnostics["agreement_nonmyopic_to_oracle"],
            "agreement_gain": diagnostics["agreement_gain_nonmyopic_over_myopic"],
            "agreement_gain_confidence_interval": diagnostics[
                "agreement_gain_confidence_interval"
            ],
            "gain_target": float(targets.get("horizon_agreement_gain", 0.10)),
            "meets_target": diagnostics["agreement_gain_nonmyopic_over_myopic"]
            >= float(targets.get("horizon_agreement_gain", 0.10)),
            "confidence_interval_supports_positive_gain": (
                diagnostics["agreement_gain_confidence_interval"][0] > 0
            ),
        },
        "H4_cost_sensitivity": {
            "low_cost": low_cost,
            "high_cost": high_cost,
            "low_cost_positive_voc_rate": low_rate,
            "high_cost_positive_voc_rate": high_rate,
            "relative_rate_reduction": cost_rate_reduction,
            "paired_relative_rate_reduction_mean": _mean(paired_cost_reductions),
            "paired_relative_rate_reduction_confidence_interval": cost_reduction_interval,
            "reduction_target": float(targets.get("high_cost_rate_reduction", 0.15)),
            "meets_target": cost_rate_reduction
            >= float(targets.get("high_cost_rate_reduction", 0.15)),
            "confidence_interval_supports_decrease": cost_reduction_interval[0] > 0,
        },
    }
    return {
        "conditions": conditions,
        "pareto_frontier": sorted(
            pareto_frontier,
            key=lambda item: (item["mean_messages"], -item["mean_task_return"]),
        ),
        "diagnostics": diagnostics,
        "hypotheses": hypotheses,
        "failure_cases": [
            row for row in episode_rows if row["failure_case"] is not None
        ],
        "episode_count": len(episode_rows),
    }


def _validate_config(config: dict[str, Any]) -> None:
    for section in ("environment", "communication", "evaluation"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"config.{section} must be a mapping")
    environment = config["environment"]
    communication = config["communication"]
    evaluation = config["evaluation"]
    for key in ("grid_size", "horizon", "victims", "hazards"):
        if key not in environment:
            raise ValueError(f"config.environment.{key} is required")
    for key in (
        "particles", "shortlist_size", "rollout_count", "myopic_horizon",
        "nonmyopic_horizon", "oracle_horizon", "cost_sweep",
        "max_messages_per_sender_step", "unit_cost",
        "useful_message_threshold", "oracle_value_threshold",
    ):
        if key not in communication:
            raise ValueError(f"config.communication.{key} is required")
    if not evaluation.get("seeds"):
        raise ValueError("config.evaluation.seeds must contain at least one seed")
    if len(set(evaluation["seeds"])) != len(evaluation["seeds"]):
        raise ValueError("config.evaluation.seeds must be unique")
    if not communication["cost_sweep"]:
        raise ValueError("communication.cost_sweep must contain at least one cost")
    costs = [float(cost) for cost in communication["cost_sweep"]]
    if len(set(costs)) != len(costs):
        raise ValueError("communication.cost_sweep values must be unique")
    if 0.0 not in costs or 0.5 not in costs:
        raise ValueError("communication.cost_sweep must include the paper's 0.0 and 0.5 settings")
    if int(evaluation.get("held_out_contexts", 0)) < 1:
        raise ValueError("config.evaluation.held_out_contexts must be positive")
    if not 0.0 < float(evaluation.get("confidence_level", 0.95)) < 1.0:
        raise ValueError("evaluation.confidence_level must be strictly between 0 and 1")
    thresholds = evaluation.get("thresholds", {})
    if not isinstance(thresholds, dict):
        raise ValueError("evaluation.thresholds must be a mapping")
    significance_level = float(thresholds.get(
        "significance_level",
        1.0 - float(evaluation.get("confidence_level", 0.95)),
    ))
    if not 0.0 < significance_level < 1.0:
        raise ValueError("evaluation.thresholds.significance_level must be between 0 and 1")
    if not math.isclose(
        float(evaluation.get("confidence_level", 0.95)),
        1.0 - significance_level,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("confidence_level must equal 1 - significance_level")
    if int(evaluation.get("bootstrap_samples", 1000)) < 1:
        raise ValueError("evaluation.bootstrap_samples must be positive")
    if float(evaluation.get("return_match_tolerance", 1.0)) < 0:
        raise ValueError("evaluation.return_match_tolerance cannot be negative")
    if int(evaluation.get("message_budget_match_tolerance", 0)) < 0:
        raise ValueError("evaluation.message_budget_match_tolerance cannot be negative")
    if int(communication["shortlist_size"]) < 1:
        raise ValueError("communication.shortlist_size must be positive")
    if int(communication["rollout_count"]) < 1:
        raise ValueError("communication.rollout_count must be positive")
    if int(communication["particles"]) < 1:
        raise ValueError("communication.particles must be positive")
    if not math.isfinite(float(communication["unit_cost"])) or float(
        communication["unit_cost"]
    ) < 0:
        raise ValueError("communication.unit_cost must be finite and non-negative")
    for key in ("useful_message_threshold", "oracle_value_threshold"):
        if float(communication[key]) < 0:
            raise ValueError(f"communication.{key} cannot be negative")
    if int(communication.get("max_messages_per_sender_step", 1)) != 1:
        raise ValueError("the primary experiment requires one message slot per sender and step")
    roles = tuple(Role(role) for role in environment.get(
        "roles", [role.value for role in Role]
    ))
    if len(roles) != len(set(roles)) or set(roles) != set(Role):
        raise ValueError("the primary experiment requires one Scout, one Carrier, and one Medic")
    for key in ("myopic_horizon", "nonmyopic_horizon", "oracle_horizon"):
        if int(communication[key]) < 1:
            raise ValueError(f"communication.{key} must be positive")
    if int(communication["oracle_horizon"]) < int(communication["nonmyopic_horizon"]):
        raise ValueError("communication.oracle_horizon must be at least nonmyopic_horizon")
    if int(communication["myopic_horizon"]) >= int(communication["nonmyopic_horizon"]):
        raise ValueError("communication.myopic_horizon must be shorter than nonmyopic_horizon")
    for key in (
        "receiver_differentiation_fraction",
        "communication_reduction",
        "horizon_agreement_gain",
        "high_cost_rate_reduction",
    ):
        value = float(thresholds.get(key, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"evaluation.thresholds.{key} must be between 0 and 1")
    if float(thresholds.get("receiver_value_difference", 0.0)) < 0:
        raise ValueError("evaluation.thresholds.receiver_value_difference cannot be negative")
    for cost in costs:
        if not math.isfinite(float(cost)) or float(cost) < 0:
            raise ValueError("communication costs cannot be negative")


def _condition_value(conditions: list[dict[str, Any]], method: str, cost: float) -> float:
    return next(
        (
            item["positive_voc_communication_rate"]
            for item in conditions
            if item["method"] == method and item["cost_weight"] == cost
        ),
        0.0,
    )


def _method_number(method: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(method))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _optional_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
