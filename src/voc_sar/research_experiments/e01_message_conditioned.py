"""E01: Compare particle rollouts with and without candidate-fact conditioning."""

from typing import Any, Sequence

from voc_sar.communication.estimator import ParticleReceiverEstimator
from voc_sar.evaluation.metrics import pair_regret, receiver_regret
from voc_sar.research_experiments.common import make_estimator, mean, oracle_estimate, rollout_task_return, run_study
from voc_sar.types import Message, Observation


class MessageConditionedParticles:
    """Condition sampled particles on the supplied candidate, never on truth."""

    def __init__(self, base: ParticleReceiverEstimator, message: Message) -> None:
        self.base = base
        self.message = message

    def sample_states(
        self, sender_id: int, receiver_id: int, particles: int, observation: Observation
    ) -> Sequence[Any]:
        worlds = [
            world.clone()
            for world in self.base.sample_states(sender_id, receiver_id, particles, observation)
        ]
        for world in worlds:
            _force_candidate_fact(world, self.message)
        return worlds


def _force_candidate_fact(world: Any, message: Message) -> None:
    row, col = message.location
    if not (0 <= row < world.grid_size and 0 <= col < world.grid_size):
        return
    if message.kind == "victim" and world._victims:
        item = next(
            (victim for victim in world._victims if victim.position == message.location),
            world._victims[0],
        )
        item.position = message.location
        item.status = message.status
    elif message.kind == "hazard" and world._hazards:
        item = next(
            (hazard for hazard in world._hazards if hazard.position == message.location),
            world._hazards[0],
        )
        item.position = message.location
        item.active = message.status == "active"
        item.disabled_remaining = (
            max(1, int(world.config.get("hazard_disable_steps", 5)))
            if message.status == "disabled" else 0
        )


def analyze(config: dict[str, Any], contexts: list[dict[str, Any]]):
    errors: dict[str, list[float]] = {"unconditioned": [], "conditioned": []}
    receiver_regrets: dict[str, list[float]] = {"unconditioned": [], "conditioned": []}
    pair_regrets: dict[str, list[float]] = {"unconditioned": [], "conditioned": []}
    returns: dict[str, list[float]] = {"unconditioned": [], "conditioned": []}
    rows: list[dict[str, Any]] = []
    horizon = int(config["communication"]["oracle_horizon"])
    max_messages = int(config["experiment"].get("messages_per_context", 3))
    for context in contexts:
        messages = list(context["messages"][:max_messages])
        receivers = [index for index in range(len(context["roles"])) if index != context["sender_id"]]
        truth: dict[tuple[int, int], float] = {}
        estimates: dict[str, dict[tuple[int, int], Any]] = {"unconditioned": {}, "conditioned": {}}
        for message_index, message in enumerate(messages):
            for receiver_id in receivers:
                target = oracle_estimate(
                    config, {**context, "message": message, "receiver_id": receiver_id}, horizon=horizon
                ).marginal_value
                key = (message_index, receiver_id)
                truth[key] = target
                base = ParticleReceiverEstimator(
                    config["environment"], seed=int(context["seed"]) + int(context["step"])
                )
                estimates["unconditioned"][key] = make_estimator(
                    config, seed=int(context["seed"]) + 31, state_estimator=base
                ).estimate(context["sender_id"], receiver_id, message, horizon, context["observation"])
                conditioned = MessageConditionedParticles(base, message)
                estimates["conditioned"][key] = make_estimator(
                    config, seed=int(context["seed"]) + 31, state_estimator=conditioned
                ).estimate(context["sender_id"], receiver_id, message, horizon, context["observation"])

        primary_values = {
            receiver: truth[(0, receiver)] for receiver in receivers
        }
        for label in ("unconditioned", "conditioned"):
            predicted = estimates[label]
            errors[label].extend(
                abs(predicted[key].marginal_value - value)
                for key, value in truth.items()
            )
            receiver_values = {
                receiver: predicted[(0, receiver)].marginal_value for receiver in receivers
            }
            receiver_choice = max(receiver_values, key=receiver_values.get)
            receiver_regrets[label].append(receiver_regret(primary_values, receiver_choice))
            selected = max(predicted, key=lambda key: predicted[key].value_of_communication)
            chosen = selected if predicted[selected].value_of_communication > 0 else None
            pair_regrets[label].append(pair_regret(truth, chosen))
            choice = (messages[chosen[0]], chosen[1]) if chosen is not None else None
            returns[label].append(rollout_task_return(context, choice, config))
            rows.append({
                "seed": context["seed"], "step": context["step"], "variant": label,
                "mae_voc": errors[label][-len(truth):] and mean(errors[label][-len(truth):]),
                "receiver_regret": receiver_regrets[label][-1],
                "pair_regret": pair_regrets[label][-1],
                "j0_return_from_context": returns[label][-1],
            })

    headline = {}
    for label in ("unconditioned", "conditioned"):
        headline[f"{label}_mae_voc"] = mean(errors[label])
        headline[f"{label}_receiver_regret"] = mean(receiver_regrets[label])
        headline[f"{label}_pair_regret"] = mean(pair_regrets[label])
        headline[f"{label}_j0_return"] = mean(returns[label])
    headline["mae_change_conditioned_minus_unconditioned"] = (
        headline["conditioned_mae_voc"] - headline["unconditioned_mae_voc"]
    )
    return rows, {"headline": headline}


def run(config: dict[str, Any]):
    return run_study("E01", config, analyze)
