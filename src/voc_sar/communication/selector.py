"""Cheap candidate filtering and receiver-aware/agnostic selection."""

import random
from dataclasses import dataclass, field, replace
from typing import Protocol, Sequence

from voc_sar.communication.estimator import ValueEstimate
from voc_sar.types import (
    AgentTrack,
    CommunicationChoice,
    Message,
    Observation,
    ReceiverSummary,
    Role,
)


_DEFAULT_ROLES = (Role.SCOUT, Role.CARRIER, Role.MEDIC)


class ValueEstimator(Protocol):
    cost_weight: float
    unit_cost: float

    def estimate_message(
        self, message: Message, horizon: int, observation: Observation
    ) -> ValueEstimate: ...

    def estimate(
        self,
        sender_id: int,
        receiver_id: int,
        message: Message,
        horizon: int,
        observation: Observation,
    ) -> ValueEstimate:
        """Estimate paired send/no-send value for one receiver and message."""
        ...


@dataclass
class SelectionTrace:
    """Optional estimates emitted by one selector call for uncertainty reporting."""

    evaluations: list[tuple[Message, int, ValueEstimate]] = field(default_factory=list)


def shortlist(
    sender_id: int,
    messages: list[Message],
    receiver_ids: list[int],
    limit: int,
    *,
    current_step: int = 0,
    receiver_roles: Sequence[Role] | None = None,
    teammate_tracks: Sequence[AgentTrack] = (),
    receiver_summaries: Sequence[ReceiverSummary] = (),
    hazard_move_probability: float = 0.0,
    hazard_toggle_probability: float = 0.0,
) -> list[tuple[Message, int]]:
    """Rank message/receiver pairs by role utility, freshness, and proximity."""
    if limit < 0:
        raise ValueError("shortlist limit cannot be negative")
    if not 0 <= hazard_move_probability <= 1 or not 0 <= hazard_toggle_probability <= 1:
        raise ValueError("hazard probabilities must be between 0 and 1")
    if limit == 0:
        return []
    roles = tuple(receiver_roles) if receiver_roles is not None else _DEFAULT_ROLES
    track_by_id = {track.agent_id: track for track in teammate_tracks}
    summary_by_id = {summary.receiver_id: summary for summary in receiver_summaries}
    pairs: list[tuple[float, int, int, Message]] = []
    for message_index, message in enumerate(messages):
        age = max(0, current_step - message.observed_at)
        freshness = ((1 - hazard_move_probability) * (1 - hazard_toggle_probability)) ** age
        for receiver_id in receiver_ids:
            if receiver_id == sender_id:
                continue
            role = roles[receiver_id] if receiver_id < len(roles) else Role.SCOUT
            role_utility = _role_utility(message, role)
            novelty = _novelty(
                message,
                summary_by_id.get(receiver_id),
            )
            track = track_by_id.get(receiver_id)
            proximity = 1.0
            if track is not None:
                distance = abs(track.position[0] - message.location[0]) + abs(
                    track.position[1] - message.location[1]
                )
                proximity = 1.0 / (1.0 + distance)
            score = role_utility * freshness * novelty * (0.5 + 0.5 * proximity)
            pairs.append((score, message_index, receiver_id, message))
    pairs.sort(
        key=lambda item: (-item[0], item[2], item[1], item[3].observed_at, item[3].location)
    )
    return [(message, receiver_id) for _, _, receiver_id, message in pairs[:limit]]


def select_communication(
    sender_id: int,
    messages: list[Message],
    receiver_ids: list[int],
    horizon: int,
    estimator: ValueEstimator | None = None,
    *,
    observation: Observation | None = None,
    shortlist_size: int = 8,
    current_step: int | None = None,
    receiver_roles: Sequence[Role] | None = None,
    cost_weight: float | None = None,
    unit_cost: float | None = None,
    aware: bool = True,
    seed: int = 0,
    hazard_move_probability: float = 0.0,
    hazard_toggle_probability: float = 0.0,
    trace: SelectionTrace | None = None,
    force_send: bool = False,
) -> CommunicationChoice | None:
    """Greedily choose the best positive-net-value directed message."""
    if estimator is None:
        raise ValueError("an estimator is required for value-based selection")
    if observation is None:
        raise ValueError("the sender's local observation is required")
    if horizon < 1:
        raise ValueError("horizon must be at least one")
    if sender_id != observation.agent_id:
        raise ValueError("sender_id must match the sender observation")
    step = observation.time_step if current_step is None else current_step
    if aware:
        pairs = shortlist(
            sender_id,
            messages,
            receiver_ids,
            shortlist_size,
            current_step=step,
            receiver_roles=receiver_roles,
            teammate_tracks=observation.teammates,
            receiver_summaries=observation.receiver_summaries,
            hazard_move_probability=hazard_move_probability,
            hazard_toggle_probability=hazard_toggle_probability,
        )
        if not pairs:
            return None
        scored: list[tuple[float, int, int, Message]] = []
        for index, (message, receiver_id) in enumerate(pairs):
            estimate = _estimate(
                estimator, sender_id, receiver_id, message, horizon, observation,
                cost_weight, unit_cost,
            )
            if trace is not None:
                trace.evaluations.append((message, receiver_id, estimate))
            scored.append((estimate.value_of_communication, index, receiver_id, message))
        value, _, receiver_id, message = max(
            scored, key=lambda item: (item[0], -item[1])
        )
        return CommunicationChoice(receiver_id, message) if value > 0 or force_send else None

    shortlisted_messages = _message_only_shortlist(
        messages,
        shortlist_size,
        step,
        hazard_move_probability,
        hazard_toggle_probability,
    )
    receiver_pool = sorted({receiver for receiver in receiver_ids if receiver != sender_id})
    if not receiver_pool or not shortlisted_messages:
        return None
    tie_break_receiver = random.Random(seed).choice(receiver_pool)
    scored_messages: list[tuple[float, int, Message]] = []
    for message_index, message in enumerate(shortlisted_messages):
        estimate = _with_cost(
            estimator.estimate_message(message, horizon, observation),
            estimator, message, cost_weight, unit_cost,
        )
        if trace is not None:
            trace.evaluations.append((message, tie_break_receiver, estimate))
        scored_messages.append((
            estimate.value_of_communication,
            message_index,
            message,
        ))
    value, _, selected_message = max(
        scored_messages, key=lambda item: (item[0], -item[1])
    )
    if value <= 0 and not force_send:
        return None
    return CommunicationChoice(tie_break_receiver, selected_message)


def _estimate(
    estimator: ValueEstimator,
    sender_id: int,
    receiver_id: int,
    message: Message,
    horizon: int,
    observation: Observation,
    cost_weight: float | None,
    unit_cost: float | None,
):
    estimate = estimator.estimate(
        sender_id, receiver_id, message, horizon, observation
    )
    return _with_cost(estimate, estimator, message, cost_weight, unit_cost)


def _with_cost(
    estimate: ValueEstimate,
    estimator: ValueEstimator,
    message: Message,
    cost_weight: float | None,
    unit_cost: float | None,
) -> ValueEstimate:
    if cost_weight is None and unit_cost is None:
        return estimate
    weight = estimator.cost_weight if cost_weight is None else cost_weight
    unit = estimator.unit_cost if unit_cost is None else unit_cost
    cost = weight * unit
    if message.metadata is not None and "cost" in message.metadata:
        cost *= float(message.metadata["cost"])
    return replace(
        estimate,
        communication_cost=cost,
        value_of_communication=estimate.marginal_value - cost,
    )


def _role_utility(message: Message, receiver_role: Role) -> float:
    if message.kind == "intention":
        return 1.0
    if message.kind == "victim":
        values = {
            "hidden": {Role.SCOUT: 1.0, Role.CARRIER: 0.5, Role.MEDIC: 0.5},
            "located": {Role.SCOUT: 0.25, Role.CARRIER: 1.5, Role.MEDIC: 2.0},
            "stabilized": {Role.SCOUT: 0.1, Role.CARRIER: 3.0, Role.MEDIC: 0.25},
            "rescued": {Role.SCOUT: 0.05, Role.CARRIER: 0.05, Role.MEDIC: 0.05},
        }
        return values.get(message.status, {}).get(receiver_role, 0.1)
    if message.kind == "hazard":
        if message.status == "active":
            return 2.5 if receiver_role == Role.MEDIC else 1.25
        return 1.0 if message.status == "clear" else 0.05
    return 0.1


def _message_only_shortlist(
    messages: list[Message],
    limit: int,
    current_step: int,
    hazard_move_probability: float,
    hazard_toggle_probability: float,
) -> list[Message]:
    """Rank facts without receiver role, position, or belief features."""
    kind_utility = {
        ("victim", "located"): 2.0,
        ("victim", "stabilized"): 2.0,
        ("victim", "hidden"): 1.0,
        ("victim", "rescued"): 0.1,
        ("hazard", "active"): 1.5,
        ("hazard", "clear"): 1.0,
        ("hazard", "disabled"): 0.1,
        ("intention", "scan"): 0.75,
        ("intention", "stabilize"): 0.75,
        ("intention", "rescue"): 0.75,
        ("intention", "disable"): 0.75,
    }
    indexed = []
    for index, message in enumerate(messages):
        age = max(0, current_step - message.observed_at)
        freshness = ((1 - hazard_move_probability) * (1 - hazard_toggle_probability)) ** age
        score = kind_utility.get((message.kind, message.status), 0.25) * freshness
        indexed.append((score, index, message))
    indexed.sort(
        key=lambda item: (-item[0], item[1], item[2].observed_at, item[2].location)
    )
    return [message for _, _, message in indexed[:limit]]


def _novelty(message: Message, summary: ReceiverSummary | None) -> float:
    if summary is None:
        return 1.0
    previous = next(
        (
            fact for fact in summary.known_facts
            if fact.kind == message.kind and fact.location == message.location
        ),
        None,
    )
    if previous is None or previous.status != message.status:
        return 1.0
    return 0.0 if previous.observed_at >= message.observed_at else 0.65
