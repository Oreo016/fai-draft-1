"""Reproducible baseline, horizon, cost, and held-out oracle experiments."""

from collections import defaultdict
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import sys
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
    bootstrap_paired_relative_reduction,
    communication_reduction,
    filter_recall,
    pair_regret,
)
from voc_sar.evaluation.oracle import CounterfactualOracle
from voc_sar.policies.task import candidate_messages, task_policies
from voc_sar.types import CommunicationChoice, Message, Role
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
    context_limit = evaluation_config.get("max_valuation_contexts_per_episode")
    if context_limit is not None:
        context_limit = int(context_limit)
    worker_count = int(evaluation_config.get("parallel_workers", 1))

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

    jobs: list[tuple[dict[str, Any], int, str, float, int, int | None, bool, int | None]] = []
    captured_methods: set[str] = set()
    for method, cost_weight, horizon in conditions:
        for seed in seeds:
            capture_search = method not in captured_methods
            if capture_search:
                captured_methods.add(method)
            jobs.append((
                config, seed, method, cost_weight, horizon,
                context_limit, capture_search, None,
            ))

    for budget in evaluation_config.get("assigned_message_budgets", []):
        for method in ("aware_nonmyopic", "agnostic_nonmyopic"):
            for seed in seeds:
                jobs.append((config, seed, method, 0.0, nonmyopic_horizon, None, False, budget))

    started_at = time.perf_counter()
    output_root = Path(evaluation_config.get("output_dir", "results"))
    run_stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_root / f"run_{run_stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    config_text = json.dumps(config, indent=2, sort_keys=True)
    (output_dir / "config_used.json").write_text(config_text, encoding="utf-8")
    source_root = Path(__file__).resolve().parents[1]
    provenance = {
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
        "source_sha256": {
            file.relative_to(source_root).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(source_root.rglob("*.py"))
        },
        "python_version": sys.version,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    episode_rows: list[dict[str, Any] | None] = [None] * len(jobs)
    search_frames_by_method: dict[str, list[dict[str, Any]]] = {}
    _report_progress("Episodes", 0, len(jobs))
    completed = 0
    progress_path = output_dir / "progress.json"

    def write_progress(status: str, failure: str | None = None) -> None:
        payload = {
            "status": status, "completed_episodes": completed,
            "requested_episodes": len(jobs), "failure": failure,
        }
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(progress_path)

    write_progress("episodes")

    def save_episode_result(
        index: int,
        result: tuple[dict[str, Any], str | None, list[dict[str, Any]] | None],
    ) -> None:
        nonlocal completed
        row, captured_method, frames = result
        episode_rows[index] = row
        if captured_method is not None and frames is not None:
            search_frames_by_method[captured_method] = frames
        completed += 1
        write_progress("episodes")
        _report_progress(
            "Episodes", completed, len(jobs),
            f"{row['method']} seed={row['seed']}",
        )

    try:
        if worker_count == 1:
            for index, job in enumerate(jobs):
                save_episode_result(index, _run_episode_job(job))
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(_run_episode_job, job): index
                    for index, job in enumerate(jobs)
                }
                for future in as_completed(futures):
                    save_episode_result(futures[future], future.result())
        completed_episode_rows = [row for row in episode_rows if row is not None]
        (output_dir / "episodes.json").write_text(
            json.dumps(completed_episode_rows, indent=2, sort_keys=True), encoding="utf-8"
        )
        write_progress("diagnostics")
        diagnostics = _run_held_out_diagnostics(config, seeds)
    except Exception as error:
        write_progress("failed", f"{type(error).__name__}: {error}")
        raise
    context_rows = diagnostics.pop("context_rows")
    summary = _summarize(config, completed_episode_rows, diagnostics)
    summary["decision_scope"] = {
        "max_valuation_contexts_per_episode": context_limit,
        "valuation_contexts_are_capped": context_limit is not None,
        "held_out_contexts_are_reserved_for": "offline diagnostics only",
    }
    summary["execution"] = {
        "parallel_workers": worker_count,
        "episode_execution": "parallel" if worker_count > 1 else "sequential",
    }
    summary["runtime_seconds"] = time.perf_counter() - started_at
    summary["config"] = config
    summary["run_directory"] = str(output_dir.resolve())
    summary["visualizations_directory"] = str((output_dir / "graphs").resolve())
    summary["config_file"] = str((output_dir / "config_used.json").resolve())
    (output_dir / "contexts.json").write_text(
        json.dumps(context_rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    create_visualizations(
        [row for row in completed_episode_rows if row["assigned_message_budget"] is None],
        summary, output_dir / "graphs",
    )
    for method, frames in search_frames_by_method.items():
        create_search_animation(frames, output_dir / f"search-operations-{method}.gif")
    write_progress("complete")
    return summary


def _report_progress(
    phase: str, completed: int, total: int, detail: str = ""
) -> None:
    if total < 1:
        return
    percentage = 100 * completed // total
    message = f"{phase}: {completed}/{total} ({percentage}%)"
    if detail:
        message += f" | {detail}"
    if sys.stderr.isatty():
        print(f"\r{message:<100}", end="\n" if completed == total else "", file=sys.stderr, flush=True)
    else:
        interval = max(1, math.ceil(total / 20))
        if completed in {0, 1, total} or completed % interval == 0:
            print(message, file=sys.stderr, flush=True)


def _run_episode_job(
    job: tuple[dict[str, Any], int, str, float, int, int | None, bool, int | None],
) -> tuple[dict[str, Any], str | None, list[dict[str, Any]] | None]:
    config, seed, method, cost_weight, horizon, context_limit, capture_search, budget = job
    frames: list[dict[str, Any]] | None = [] if capture_search else None
    row = _run_episode(
        config, seed, method, cost_weight, horizon, context_limit, frames, budget
    )
    return row, method if capture_search else None, frames


def _run_episode(
    config: dict[str, Any],
    seed: int,
    method: str,
    cost_weight: float,
    valuation_horizon: int,
    context_limit: int | None,
    search_frames: list[dict[str, Any]] | None = None,
    assigned_budget: int | None = None,
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
    step_rows = []
    decision_seconds = 0.0
    evaluation = config["evaluation"]
    actionable_budget = (
        assigned_budget is not None and evaluation.get("budget_schedule") == "actionable_facts"
    )
    sent_facts: set[tuple] = set()
    budget_send_steps: list[int] = []
    assigned_steps = (
        _budget_send_steps(config["evaluation"], assigned_budget)
        if assigned_budget is not None else ()
    )
    sent_count = 0
    decision_count = 0
    valuation_contexts = 0
    estimator_standard_errors: list[float] = []
    selected_standard_errors: list[float] = []
    step = 0
    if search_frames is not None:
        _capture_search_frame(search_frames, env, [], None, method, seed)
    while not env.terminated:
        decision_started = time.perf_counter()
        sender_observation = observations[sender_id]
        planned_action = policies[sender_id].act(sender_observation)
        messages = candidate_messages(sender_observation, planned_action)
        choice: CommunicationChoice | None = None
        assigned_send = step in assigned_steps
        if actionable_budget:
            messages = _actionable_budget_messages(messages, step, sent_facts)
            assigned_send = sent_count < assigned_budget and bool(messages)

        if method in {"aware_myopic", "aware_nonmyopic", "agnostic_nonmyopic"}:
            if (
                (context_limit is None or valuation_contexts < context_limit)
                and (assigned_budget is None or assigned_send)
                and messages
                and receivers
            ):
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
                    force_send=assigned_budget is not None,
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

        decision_elapsed = time.perf_counter() - decision_started
        communication = [None] * len(observations)
        if choice is not None:
            communication[sender_id] = choice
            sent_count += 1
            if assigned_budget is not None:
                budget_send_steps.append(step)
                sent_facts.add(_fact_key(choice.message))
        observations = env.deliver(communication)
        actions = [
            policy.act(observation)
            for policy, observation in zip(policies, observations)
        ]
        result = env.step(actions)
        if search_frames is not None:
            _capture_search_frame(search_frames, env, actions, choice, method, seed)
        rewards.append(result.reward)
        decision_seconds += decision_elapsed
        step_rows.append({
            "step": step, "reward": result.reward, "events": dict(result.info),
            "task_actions": actions, "send": asdict(choice) if choice is not None else None,
            "eligible_budget_facts": len(messages) if actionable_budget else None,
            "communication_decision_seconds": decision_elapsed,
        })
        observations = result.observations
        step += 1

    remaining_victims = sum(status != "rescued" for _, status in env.victims)
    return {
        "seed": seed,
        "method": method,
        "cost_weight": cost_weight,
        "horizon": valuation_horizon,
        "assigned_message_budget": assigned_budget,
        "budget_send_steps": budget_send_steps if assigned_budget is not None else None,
        "step_trace": step_rows,
        "mean_communication_decision_seconds": decision_seconds / step if step else None,
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
            if assigned_budget is None and decision_count and method in {
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
    requested_contexts = int(evaluation_config["held_out_contexts"])
    base_contexts, extra_contexts = divmod(requested_contexts, len(seeds))
    delayed_margin = float(evaluation_config.get("delayed_value_margin", 0.5))
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
    useful_messages: list[bool] = []
    recalls: list[float] = []
    myopic_actions: list[object] = []
    nonmyopic_actions: list[object] = []
    oracle_actions: list[object] = []
    horizon_agreement_gains: list[float] = []
    horizon_gains_by_seed: list[list[float]] = []
    context_count = 0
    context_rows = []
    delayed_agreements = []

    _report_progress("Diagnostics", 0, len(seeds))
    for seed_index, seed in enumerate(seeds, 1):
        contexts_per_seed = base_contexts + (seed_index <= extra_contexts)
        context_steps = None
        if evaluation_config.get("diagnostic_context_sampling") == "evenly_spaced":
            last_step = int(env_config["horizon"]) - 1
            context_steps = {
                index * last_step // max(1, contexts_per_seed - 1)
                for index in range(contexts_per_seed)
            }
        env = SARGrid(env_config)
        observations = env.reset(seed + 77_777)
        policies = task_policies(env.roles)
        sender_id = next(
            (index for index, role in enumerate(env.roles) if role == Role.SCOUT),
            0,
        )
        receivers = [index for index in range(len(env.roles)) if index != sender_id]
        if len(receivers) < 1:
            _report_progress("Diagnostics", seed_index, len(seeds), f"seed={seed}")
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
            record_rollouts=True,
        )
        oracle = CounterfactualOracle(
            policies,
            rollout_count=int(evaluation_config.get("oracle_rollout_count", rollout_count)),
            gamma=gamma,
            cost_weight=0.0,
            unit_cost=float(comm_config.get("unit_cost", 1.0)),
            seed=seed + 155_555,
            record_rollouts=True,
        )
        contexts_for_seed = 0
        receiver_flags_for_seed: list[float] = []
        horizon_gains_for_seed: list[float] = []
        while not env.terminated and contexts_for_seed < contexts_per_seed:
            observation = observations[sender_id]
            planned_action = policies[sender_id].act(observation)
            messages = (
                candidate_messages(observation, planned_action)
                if context_steps is None or observation.time_step in context_steps else []
            )
            if messages:
                context_count += 1
                contexts_for_seed += 1
                all_pairs = [(index, receiver) for index in range(len(messages)) for receiver in receivers]
                oracle_estimates = {}
                oracle_myopic_estimates = {}
                for index, receiver in all_pairs:
                    message = messages[index]
                    oracle_estimates[index, receiver] = oracle.evaluate(
                        env, sender_id, receiver, message, h_eval
                    )
                    oracle_myopic_estimates[index, receiver] = oracle.evaluate(
                        env, sender_id, receiver, message, h_myopic
                    )
                oracle_values = {
                    pair: estimate.value_of_communication for pair, estimate in oracle_estimates.items()
                }
                best = max(oracle_values, key=oracle_values.get)
                oracle_action = best if oracle_values[best] > 0 else None
                delayed = any(
                    abs(oracle_estimates[pair].marginal_value
                        - oracle_myopic_estimates[pair].marginal_value) > delayed_margin
                    for pair in all_pairs
                )

                choices = {}
                predictions = {}
                for horizon in (h_myopic, h_nonmyopic):
                    trace = SelectionTrace()
                    choice = select_communication(
                        sender_id, messages, receivers, horizon, estimator,
                        observation=observation, shortlist_size=shortlist_size,
                        receiver_roles=env.roles, seed=seed,
                        hazard_move_probability=float(env_config.get("hazard_move_probability", 0)),
                        hazard_toggle_probability=float(env_config.get("hazard_toggle_probability", 0)),
                        trace=trace,
                    )
                    choices[horizon] = (
                        (messages.index(choice.message), choice.receiver_id) if choice else None
                    )
                    predictions[horizon] = {
                        (messages.index(message), receiver): estimate
                        for message, receiver, estimate in trace.evaluations
                    }

                myopic_action = choices[h_myopic]
                nonmyopic_action = choices[h_nonmyopic]
                myopic_actions.append(myopic_action)
                nonmyopic_actions.append(nonmyopic_action)
                oracle_actions.append(oracle_action)
                if delayed:
                    agreement = (int(myopic_action == oracle_action), int(nonmyopic_action == oracle_action))
                    delayed_agreements.append(agreement)
                    gain = agreement[1] - agreement[0]
                    horizon_agreement_gains.append(gain)
                    horizon_gains_for_seed.append(gain)

                nonmyopic = predictions[h_nonmyopic]
                if any(value > oracle_threshold for value in oracle_values.values()):
                    recalls.append(filter_recall(oracle_values, list(nonmyopic), oracle_threshold))
                mae_predictions.append(_mean([
                    abs(estimate.value_of_communication - oracle_values[pair])
                    for pair, estimate in nonmyopic.items()
                ]))
                pair_regrets.append(pair_regret(oracle_values, nonmyopic_action))
                if nonmyopic_action is not None:
                    useful_messages.append(oracle_estimates[nonmyopic_action].marginal_value > useful_threshold)
                    selected_index, selected_receiver = nonmyopic_action
                    receiver_regrets.append(
                        max(oracle_values[selected_index, receiver] for receiver in receivers)
                        - oracle_values[selected_index, selected_receiver]
                    )
                primary_index = next(
                    (index for index, message in enumerate(messages) if message.kind != "intention"), None
                )
                if primary_index is not None and len(receivers) >= 2:
                    difference = abs(
                        oracle_estimates[primary_index, receivers[0]].marginal_value
                        - oracle_estimates[primary_index, receivers[1]].marginal_value
                    )
                    receiver_differences.append(difference)
                    flag = float(difference > float(thresholds.get("receiver_value_difference", 0.5)))
                    receiver_differentiation_flags.append(flag)
                    receiver_flags_for_seed.append(flag)

                context_rows.append({
                    "seed": seed, "layout_seed": seed + 77_777, "step": observation.time_step,
                    "sender_id": sender_id, "receiver_ids": receivers, "messages": [asdict(m) for m in messages],
                    "delayed": delayed, "primary_message_index": primary_index,
                    "myopic_action": myopic_action, "nonmyopic_action": nonmyopic_action,
                    "oracle_action": oracle_action,
                    "pairs": [
                        {
                            "message_index": index, "receiver_id": receiver,
                            "oracle": asdict(oracle_estimates[index, receiver]),
                            "oracle_myopic": asdict(oracle_myopic_estimates[index, receiver]),
                            "predictions": {
                                str(horizon): asdict(estimates[index, receiver])
                                for horizon, estimates in predictions.items() if (index, receiver) in estimates
                            },
                        }
                        for index, receiver in all_pairs
                    ],
                })

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
        _report_progress("Diagnostics", seed_index, len(seeds), f"seed={seed}")

    mae = _optional_mean(mae_predictions)
    receiver_fraction = _optional_mean(receiver_differentiation_flags)
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
        if receiver_differentiation_flags else None
    )
    horizon_gain_interval = (
        bootstrap_cluster_mean_interval(
            horizon_gains_by_seed,
            confidence_level,
            bootstrap_samples,
            seed + 313,
        )
        if horizon_agreement_gains else None
    )
    return {
        "held_out_contexts": context_count,
        "requested_held_out_contexts": requested_contexts,
        "unobserved_held_out_contexts": requested_contexts - context_count,
        "context_rows": context_rows,
        "delayed_contexts": len(delayed_agreements),
        "delayed_seed_count": len(horizon_gains_by_seed),
        "delayed_value_margin": delayed_margin,
        "delayed_myopic_agreement": _optional_mean([pair[0] for pair in delayed_agreements]),
        "delayed_nonmyopic_agreement": _optional_mean([pair[1] for pair in delayed_agreements]),
        "delayed_agreement_gain": _optional_mean(horizon_agreement_gains),
        "receiver_differentiation_contexts": len(receiver_differences),
        "receiver_differentiation_seed_count": len(receiver_flags_by_seed),
        "receiver_value_differences": receiver_differences,
        "receiver_differentiation_fraction": receiver_fraction,
        "receiver_differentiation_confidence_interval": receiver_fraction_interval,
        "receiver_regret_mean": _optional_mean(receiver_regrets),
        "pair_regret_mean": _mean(pair_regrets),
        "voc_mae_nonmyopic": mae,
        "filter_recall": _optional_mean(recalls),
        "filter_recall_eligible_contexts": len(recalls),
        "useful_message_rate": _optional_mean([float(value) for value in useful_messages]),
        "useful_message_send_count": len(useful_messages),
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
    total_episode_count = len(episode_rows)
    budget_rows = [row for row in episode_rows if row.get("assigned_message_budget") is not None]
    episode_rows = [row for row in episode_rows if row.get("assigned_message_budget") is None]
    budget_results = _assigned_budget_comparisons(budget_rows, evaluation, confidence, bootstrap_samples, seed)
    primary_budget = next((row for row in budget_results if row["primary"]), None)
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        groups[(row["method"], row["cost_weight"])].append(row)
    conditions: list[dict[str, Any]] = []
    for (method, cost), rows in sorted(groups.items()):
        returns = [row["discounted_task_return"] for row in rows]
        rescues = [row["rescued_victims"] for row in rows]
        interval = bootstrap_mean_interval(
            returns, confidence, bootstrap_samples, seed + _method_number(method)
        )
        conditions.append({
            "method": method,
            "cost_weight": cost,
            "episodes": len(rows),
            "mean_task_return": _mean(returns),
            "task_return_confidence_interval": interval,
            "mean_rescued_victims": _mean(rescues),
            "total_rescued_victims": sum(rescues),
            "rescued_victims_confidence_interval": (
                bootstrap_mean_interval(rescues, confidence, bootstrap_samples, seed + _method_number(method))
                if len(rescues) > 1 else None
            ),
            "mean_messages": _mean([float(row["messages"]) for row in rows]),
            "mean_message_count": _mean([row["messages"] for row in rows]),
            "total_messages": sum(row["messages"] for row in rows),
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
    paired_comparisons = _paired_return_comparisons(
        episode_rows,
        config["communication"]["cost_sweep"],
        evaluation["seeds"],
        confidence,
        bootstrap_samples,
        seed,
    )
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
    matched_return_counts = []
    full_return_differences = []
    full_rescue_differences = []
    return_match_tolerance = float(
        evaluation.get("return_match_tolerance", 1.0)
    )
    if full_rows:
        full_by_seed = {row["seed"]: row for row in full_rows}
        for row in aware_rows:
            baseline = full_by_seed.get(row["seed"])
            if baseline is not None and baseline["messages"] > 0:
                full_return_differences.append(row["discounted_task_return"] - baseline["discounted_task_return"])
                full_rescue_differences.append(row["rescued_victims"] - baseline["rescued_victims"])
                reduction = communication_reduction(row["messages"], baseline["messages"])
                per_seed_reductions.append(reduction)
                if abs(
                    row["discounted_task_return"]
                    - baseline["discounted_task_return"]
                ) <= return_match_tolerance:
                    matched_return_counts.append((row["messages"], baseline["messages"]))

    full_return_interval = (
        bootstrap_mean_interval(full_return_differences, confidence, bootstrap_samples, seed + 9009)
        if len(full_return_differences) > 1 else None
    )
    equivalent_return = bool(
        full_return_interval
        and -return_match_tolerance < full_return_interval[0]
        and full_return_interval[1] < return_match_tolerance
    )
    full_total = sum(row["messages"] for row in full_rows)
    overall_reduction = (
        communication_reduction(sum(row["messages"] for row in aware_rows), full_total)
        if full_total else None
    )
    descriptive_matched_reduction = (
        communication_reduction(
            sum(pair[0] for pair in matched_return_counts),
            sum(pair[1] for pair in matched_return_counts),
        )
        if matched_return_counts else None
    )

    low_cost = 0.0
    high_cost = 0.5
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
    cost_pairs = [
        (low_rate_by_seed[episode_seed], high_rate_by_seed[episode_seed])
        for episode_seed in sorted(low_rate_by_seed.keys() & high_rate_by_seed.keys())
    ]
    low_rate = _optional_mean([low for low, _ in cost_pairs])
    high_rate = _optional_mean([high for _, high in cost_pairs])
    cost_rate_reduction = 1 - high_rate / low_rate if low_rate else None
    cost_reduction_interval, undefined_bootstrap_samples = bootstrap_paired_relative_reduction(
        cost_pairs, confidence, bootstrap_samples, seed + 9011
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
            "meets_target": (
                diagnostics["receiver_differentiation_fraction"] is not None
                and diagnostics["receiver_differentiation_fraction"]
                >= float(targets.get("receiver_differentiation_fraction", 0.30))
            ),
            "confidence_interval_supports_target": (
                diagnostics["receiver_differentiation_seed_count"] > 1
                and diagnostics["receiver_differentiation_confidence_interval"] is not None
                and diagnostics["receiver_differentiation_confidence_interval"][0]
                >= float(targets.get("receiver_differentiation_fraction", 0.30))
            ),
        },
        "H2_selectivity": {
            "mean_aware_minus_agnostic_return": _mean(h2_differences),
            "paired_confidence_interval": h2_interval,
            "confidence_interval_excludes_zero": (
                h2_interval[0] > 0 or h2_interval[1] < 0
            ),
            "posthoc_positive_equal_count_mean_return_difference": _optional_mean(
                h2_budget_matched_differences
            ),
            "posthoc_positive_equal_count_confidence_interval": h2_budget_interval,
            "message_budget_match_tolerance": budget_match_tolerance,
            "posthoc_positive_equal_count_confidence_interval_excludes_zero": (
                h2_budget_interval[0] > 0 or h2_budget_interval[1] < 0
            ) if h2_budget_matched_differences else False,
            "matched_budget_positive_pair_count": sum(map(len, h2_budget_differences_by_seed.values())),
            "matched_budget_seed_count": len(h2_budget_matched_differences),
            "assigned_budget_primary": primary_budget,
            "assigned_budget_comparisons": budget_results,
            "mean_communication_reduction_vs_full": overall_reduction,
            "mean_seed_communication_reduction_vs_full": _optional_mean(per_seed_reductions),
            "communication_reduction_at_matched_return": overall_reduction if equivalent_return else None,
            "descriptive_posthoc_matched_return_reduction": descriptive_matched_reduction,
            "matched_return_episode_count": len(matched_return_counts),
            "return_match_tolerance": return_match_tolerance,
            "full_vs_aware_return_difference_ci": full_return_interval,
            "aware_minus_full_mean_rescue_difference": _optional_mean(full_rescue_differences),
            "aware_minus_full_rescue_difference_ci": (
                bootstrap_mean_interval(full_rescue_differences, confidence, bootstrap_samples, seed + 9010)
                if len(full_rescue_differences) > 1 else None
            ),
            "equivalent_return": equivalent_return,
            "matched_return_reduction_meets_target": (
                equivalent_return and overall_reduction is not None
                and overall_reduction >= float(targets.get("communication_reduction", 0.20))
            ),
            "meets_both_selectivity_tests": (
                equivalent_return and overall_reduction is not None
                and overall_reduction >= float(targets.get("communication_reduction", 0.20))
                and bool(primary_budget and primary_budget["confidence_interval_excludes_zero"])
            ),
        },
        "H3_downstream_value": {
            "myopic_oracle_agreement": diagnostics["delayed_myopic_agreement"],
            "nonmyopic_oracle_agreement": diagnostics["delayed_nonmyopic_agreement"],
            "agreement_gain": diagnostics["delayed_agreement_gain"],
            "agreement_gain_confidence_interval": diagnostics["agreement_gain_confidence_interval"],
            "delayed_contexts": diagnostics["delayed_contexts"],
            "delayed_seed_count": diagnostics["delayed_seed_count"],
            "gain_target": float(targets.get("horizon_agreement_gain", 0.10)),
            "meets_target": (
                diagnostics["delayed_agreement_gain"] is not None
                and diagnostics["delayed_agreement_gain"] >= float(targets.get("horizon_agreement_gain", 0.10))
            ),
            "confidence_interval_supports_target": (
                diagnostics["delayed_seed_count"] > 1
                and diagnostics["agreement_gain_confidence_interval"] is not None
                and diagnostics["agreement_gain_confidence_interval"][0]
                >= float(targets.get("horizon_agreement_gain", 0.10))
            ),
        },
        "H4_cost_sensitivity": {
            "low_cost": low_cost,
            "high_cost": high_cost,
            "low_cost_positive_voc_rate": low_rate,
            "high_cost_positive_voc_rate": high_rate,
            "relative_rate_reduction": cost_rate_reduction,
            "zero_low_rate_seeds": sum(low == 0 for low, _ in cost_pairs),
            "paired_seed_count": len(cost_pairs),
            "undefined_bootstrap_samples": undefined_bootstrap_samples,
            "paired_relative_rate_reduction_confidence_interval": cost_reduction_interval,
            "reduction_target": float(targets.get("high_cost_rate_reduction", 0.15)),
            "meets_target": (
                cost_rate_reduction is not None
                and cost_rate_reduction >= float(targets.get("high_cost_rate_reduction", 0.15))
            ),
            "confidence_interval_supports_target": (
                cost_reduction_interval is not None
                and undefined_bootstrap_samples == 0
                and cost_reduction_interval[0] >= float(targets.get("high_cost_rate_reduction", 0.15))
            ),
        },
    }
    return {
        "conditions": conditions,
        "paired_return_comparisons": paired_comparisons,
        "pareto_frontier": sorted(
            pareto_frontier,
            key=lambda item: (item["mean_messages"], -item["mean_task_return"]),
        ),
        "diagnostics": diagnostics,
        "hypotheses": hypotheses,
        "failure_cases": [
            {key: value for key, value in row.items() if key != "step_trace"}
            for row in episode_rows if row["failure_case"] is not None
        ],
        "episode_count": total_episode_count,
        "assigned_budget_episode_count": len(budget_rows),
    }


def _assigned_budget_comparisons(rows, evaluation, confidence, samples, seed):
    """Compare assigned arms and report adherence separately from outcomes."""
    indexed = {
        (row["assigned_message_budget"], row["seed"], row["method"]): row for row in rows
    }
    results = []
    budgets = evaluation.get("assigned_message_budgets", [])
    for budget in budgets:
        pairs = []
        for episode_seed in evaluation["seeds"]:
            aware = indexed[budget, episode_seed, "aware_nonmyopic"]
            agnostic = indexed[budget, episode_seed, "agnostic_nonmyopic"]
            pairs.append({
                "seed": episode_seed,
                "aware_messages": aware["messages"],
                "agnostic_messages": agnostic["messages"],
                "aware_send_steps": aware.get("budget_send_steps"),
                "agnostic_send_steps": agnostic.get("budget_send_steps"),
                "aware_rescues": aware["rescued_victims"],
                "agnostic_rescues": agnostic["rescued_victims"],
                "rescue_difference": aware["rescued_victims"] - agnostic["rescued_victims"],
                "return_difference": aware["discounted_task_return"] - agnostic["discounted_task_return"],
            })
        adherence = all(pair["aware_messages"] == pair["agnostic_messages"] == budget for pair in pairs)
        differences = [pair["return_difference"] for pair in pairs]
        rescue_differences = [pair["rescue_difference"] for pair in pairs]
        assigned_return_interval = (
            bootstrap_mean_interval(differences, confidence, samples, seed + budget + 9021)
            if len(pairs) > 1 else None
        )
        interval = assigned_return_interval if adherence else None
        schedule = evaluation.get("budget_schedule", "fixed_steps")
        results.append({
            "assigned_messages": budget,
            "schedule_type": schedule,
            "send_schedule": list(_budget_send_steps(evaluation, budget)) if schedule == "fixed_steps" else None,
            "primary": budget == evaluation.get("primary_message_budget", budgets[0]),
            "protocol_adherence": adherence,
            "fully_adherent_pair_count": sum(
                pair["aware_messages"] == pair["agnostic_messages"] == budget for pair in pairs
            ),
            "aware_unused_slots": sum(budget - pair["aware_messages"] for pair in pairs),
            "agnostic_unused_slots": sum(budget - pair["agnostic_messages"] for pair in pairs),
            "pairs": pairs,
            "mean_return_difference": _optional_mean(differences),
            "assigned_policy_return_difference_ci": assigned_return_interval,
            "mean_rescue_difference": _optional_mean(rescue_differences),
            "assigned_policy_rescue_difference_ci": (
                bootstrap_mean_interval(rescue_differences, confidence, samples, seed + budget + 9022)
                if len(pairs) > 1 else None
            ),
            "confidence_interval": interval,
            "confidence_interval_excludes_zero": bool(
                interval is not None and (interval[0] > 0 or interval[1] < 0)
            ),
        })
    return results


def _budget_send_steps(evaluation: dict[str, Any], budget: int) -> tuple[int, ...]:
    steps = evaluation.get("budget_send_steps")
    return tuple(steps[:budget]) if steps is not None else tuple(range(budget))


def _fact_key(message: Message) -> tuple:
    return message.kind, message.location, message.status


def _actionable_budget_messages(
    messages: list[Message], step: int, sent_facts: set[tuple],
) -> list[Message]:
    """Admit task-relevant observations using the same sender-local rule for both arms."""
    return [
        message for message in messages
        if _fact_key(message) not in sent_facts
        and (
            (message.kind == "victim" and message.status in {"located", "stabilized"})
            or (message.kind == "hazard" and message.status == "active" and message.observed_at == step)
        )
    ]


def _paired_return_comparisons(
    episode_rows: list[dict[str, Any]],
    costs: list[float],
    seeds: list[int],
    confidence: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Report same-seed return effects and exact realized-budget subsets."""
    indexed = {
        (row["method"], float(row["cost_weight"]), int(row["seed"])): row
        for row in episode_rows
    }
    pairs: list[tuple[str, str, float]] = []
    for cost in costs:
        pairs.append(("aware_nonmyopic", "agnostic_nonmyopic", float(cost)))
    pairs.extend((
        ("aware_nonmyopic", "random", 0.0),
        ("agnostic_nonmyopic", "random", 0.0),
    ))
    reports: list[dict[str, Any]] = []
    for pair_index, (first_method, second_method, cost) in enumerate(pairs):
        differences: list[dict[str, Any]] = []
        budget_matched: list[dict[str, Any]] = []
        for episode_seed in seeds:
            first = indexed.get((first_method, cost, int(episode_seed)))
            second = indexed.get((second_method, cost, int(episode_seed)))
            if first is None or second is None:
                continue
            difference = float(first["discounted_task_return"]) - float(
                second["discounted_task_return"]
            )
            pair_record = {
                "seed": int(episode_seed),
                "return_difference": difference,
                "first_messages": int(first["messages"]),
                "second_messages": int(second["messages"]),
            }
            differences.append(pair_record)
            if (
                int(first["messages"]) == int(second["messages"])
                and int(first["messages"]) > 0
            ):
                budget_matched.append(pair_record)
        paired_values = [item["return_difference"] for item in differences]
        budget_values = [item["return_difference"] for item in budget_matched]
        reports.append({
            "first_method": first_method,
            "second_method": second_method,
            "cost_weight": cost,
            "paired_seed_count": len(differences),
            "mean_first_minus_second_return": _mean(paired_values),
            "paired_seed_return_differences": differences,
            "paired_confidence_interval": (
                bootstrap_mean_interval(
                    paired_values, confidence, bootstrap_samples,
                    bootstrap_seed + 10000 + pair_index,
                ) if paired_values else None
            ),
            "exact_budget_matched_seed_count": len(budget_matched),
            "mean_return_difference_at_exact_matched_budget": (
                _mean(budget_values) if budget_values else None
            ),
            "exact_budget_matched_confidence_interval": (
                bootstrap_mean_interval(
                    budget_values, confidence, bootstrap_samples,
                    bootstrap_seed + 11000 + pair_index,
                ) if budget_values else None
            ),
            "exact_budget_matched_seed_differences": budget_matched,
        })
    return reports


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
    if evaluation.get("diagnostic_context_sampling", "first_eligible") not in ("first_eligible", "evenly_spaced"):
        raise ValueError("evaluation.diagnostic_context_sampling must be first_eligible or evenly_spaced")
    if int(evaluation.get("parallel_workers", 1)) < 1:
        raise ValueError("config.evaluation.parallel_workers must be at least 1")
    budgets = evaluation.get("assigned_message_budgets", [])
    if (
        not isinstance(budgets, list)
        or any(type(budget) is not int or not 1 <= budget <= int(environment["horizon"]) for budget in budgets)
        or len(set(budgets)) != len(budgets)
    ):
        raise ValueError("evaluation.assigned_message_budgets must contain unique positive step counts")
    if "primary_message_budget" in evaluation and evaluation["primary_message_budget"] not in budgets:
        raise ValueError("evaluation.primary_message_budget must be an assigned budget")
    send_steps = evaluation.get("budget_send_steps")
    schedule = evaluation.get("budget_schedule", "fixed_steps")
    if schedule not in ("fixed_steps", "actionable_facts"):
        raise ValueError("evaluation.budget_schedule must be fixed_steps or actionable_facts")
    if schedule == "actionable_facts" and send_steps is not None:
        raise ValueError("evaluation.budget_send_steps applies only to fixed_steps")
    if send_steps is not None and (
        not isinstance(send_steps, list)
        or len(send_steps) < max(budgets, default=0)
        or any(type(step) is not int or not 0 <= step < int(environment["horizon"]) for step in send_steps)
        or send_steps != sorted(set(send_steps))
    ):
        raise ValueError("evaluation.budget_send_steps must be unique, ordered steps covering the largest budget")
    if int(evaluation.get("oracle_rollout_count", communication["rollout_count"])) < 1:
        raise ValueError("evaluation.oracle_rollout_count must be positive")
    delayed_margin = float(evaluation.get("delayed_value_margin", 0.5))
    if not math.isfinite(delayed_margin) or delayed_margin < 0:
        raise ValueError("evaluation.delayed_value_margin must be finite and non-negative")
    valuation_cap = evaluation.get("max_valuation_contexts_per_episode")
    if valuation_cap is not None and int(valuation_cap) < 1:
        raise ValueError(
            "config.evaluation.max_valuation_contexts_per_episode must be positive or null"
        )
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


def _method_number(method: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(method))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _optional_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
