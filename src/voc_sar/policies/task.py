"""Deterministic role policies driven exclusively by local observations."""

from collections import deque
from typing import Protocol, Sequence

from voc_sar.types import Message, Observation, Position, Role

TaskAction = str


class TaskPolicy(Protocol):
    def act(self, observation: Observation) -> TaskAction:
        """Choose a task action from one agent's local observation."""
        ...


class ScoutPolicy:
    def act(self, observation: Observation) -> TaskAction:
        if not set(observation.visible_cells).issubset(observation.scanned_cells):
            return "scan"
        target = _nearest_unvisited(observation)
        return _move_toward(observation, target) if target is not None else "scan"


class CarrierPolicy:
    def act(self, observation: Observation) -> TaskAction:
        if observation.carrying_victim and observation.evacuation_point is not None:
            if observation.position == observation.evacuation_point:
                return "rescue"
            return _move_toward(observation, observation.evacuation_point)
        victim = _nearest_fact(observation, "victim", {"stabilized"})
        if victim is not None:
            if _distance(observation.position, victim.location) <= 1:
                return "rescue"
            return _move_toward(observation, victim.location)
        medic_intention = _nearest_intention(
            observation, "medic", {"stabilize"}
        )
        if medic_intention is not None:
            return _move_toward(
                observation, medic_intention.location, stop_adjacent=True
            )
        target = _nearest_unvisited(observation)
        return _move_toward(observation, target) if target is not None else "wait"


class MedicPolicy:
    def act(self, observation: Observation) -> TaskAction:
        victim = _nearest_fact(observation, "victim", {"located"})
        if victim is not None:
            if _distance(observation.position, victim.location) <= 1:
                return "stabilize"
            return _move_toward(observation, victim.location)
        hazard = _nearest_fact(observation, "hazard", {"active"})
        if hazard is not None:
            if _distance(observation.position, hazard.location) <= 1:
                return "disable"
            return _move_toward(observation, hazard.location, stop_adjacent=True)
        scout_intention = _nearest_intention(observation, "scout", {"scan"})
        if scout_intention is not None:
            if _distance(observation.position, scout_intention.location) <= 1:
                return "wait"
            return _move_toward(
                observation, scout_intention.location, stop_adjacent=True
            )
        target = _nearest_unvisited(observation)
        return _move_toward(observation, target) if target is not None else "wait"


def task_policies(roles: Sequence[Role]) -> tuple[TaskPolicy, ...]:
    """Create one fixed role policy per agent in the configured team."""
    policy_by_role: dict[Role, TaskPolicy] = {
        Role.SCOUT: ScoutPolicy(),
        Role.CARRIER: CarrierPolicy(),
        Role.MEDIC: MedicPolicy(),
    }
    return tuple(policy_by_role[role] for role in roles)


def candidate_messages(
    observation: Observation, planned_action: TaskAction | None = None
) -> list[Message]:
    """List existing local facts and the agent's current planned intention."""
    messages = list(observation.known_facts)
    if planned_action is not None:
        target = _intention_target(observation, planned_action)
        messages.append(Message(
            kind="intention",
            location=target,
            observed_at=observation.time_step,
            status=planned_action,
            sender_id=observation.agent_id,
            metadata={
                "role": observation.role.value,
                "agent_position": observation.position,
            },
        ))
    return messages


def _intention_target(observation: Observation, action: str) -> Position:
    if action in {"up", "down", "left", "right"}:
        movement = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }[action]
        target = (
            observation.position[0] + movement[0],
            observation.position[1] + movement[1],
        )
        if 0 <= target[0] < observation.grid_size and 0 <= target[1] < observation.grid_size:
            return target
    elif action == "stabilize":
        victim = _nearest_fact(observation, "victim", {"located"})
        if victim is not None:
            return victim.location
    elif action == "rescue":
        victim = _nearest_fact(observation, "victim", {"stabilized"})
        if victim is not None:
            return victim.location
    elif action == "disable":
        hazard = _nearest_fact(observation, "hazard", {"active"})
        if hazard is not None:
            return hazard.location
    return observation.position


def _nearest_intention(
    observation: Observation, role: str, statuses: set[str]
) -> Message | None:
    candidates = [
        fact for fact in observation.known_facts
        if fact.kind == "intention"
        and fact.status in statuses
        and fact.metadata is not None
        and fact.metadata.get("role") == role
    ]
    return min(
        candidates,
        key=lambda fact: (_distance(observation.position, fact.location), -fact.observed_at),
        default=None,
    )


def _nearest_fact(
    observation: Observation, kind: str, statuses: set[str]
) -> Message | None:
    candidates = [
        fact for fact in observation.known_facts
        if fact.kind == kind and fact.status in statuses
    ]
    return min(
        candidates,
        key=lambda fact: (_distance(observation.position, fact.location), -fact.observed_at),
        default=None,
    )


def _nearest_unvisited(observation: Observation) -> Position | None:
    visited = set(observation.visited_cells)
    candidates = [
        (row, col)
        for row in range(observation.grid_size)
        for col in range(observation.grid_size)
        if (row, col) not in visited
    ]
    return min(
        candidates,
        key=lambda point: (_distance(observation.position, point), point),
        default=None,
    )


def _move_toward(
    observation: Observation, target: Position, stop_adjacent: bool = False
) -> TaskAction:
    if _distance(observation.position, target) <= int(stop_adjacent):
        return "wait"
    known_blocked = {
        fact.location
        for fact in observation.known_facts
        if fact.kind == "hazard" and fact.status == "active"
    }
    directions: tuple[tuple[str, Position], ...] = (
        ("up", (-1, 0)),
        ("left", (0, -1)),
        ("right", (0, 1)),
        ("down", (1, 0)),
    )
    queue = deque([(observation.position, None)])
    visited = {observation.position}
    while queue:
        position, first_action = queue.popleft()
        if _distance(position, target) <= int(stop_adjacent):
            return first_action or "wait"
        for action, (dr, dc) in directions:
            next_position = (position[0] + dr, position[1] + dc)
            if not (0 <= next_position[0] < observation.grid_size):
                continue
            if not (0 <= next_position[1] < observation.grid_size):
                continue
            if (
                next_position in known_blocked
                or next_position in visited
            ) and _distance(next_position, target) > int(stop_adjacent):
                continue
            visited.add(next_position)
            queue.append((next_position, first_action or action))
    return "wait"


def _distance(first: Position, second: Position) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])
