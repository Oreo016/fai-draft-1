"""Cooperative, partially observed search-and-rescue grid simulator."""

from copy import deepcopy
from dataclasses import dataclass, field
import random
from typing import Sequence

from voc_sar.types import (
    AgentTrack,
    CommunicationChoice,
    Message,
    Observation,
    Position,
    ReceiverSummary,
    Role,
    StepResult,
)

MOVEMENTS: dict[str, Position] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "stay": (0, 0),
    "wait": (0, 0),
}


@dataclass
class _Victim:
    position: Position
    status: str = "hidden"


@dataclass
class _Hazard:
    position: Position
    active: bool = True
    disabled_remaining: int = 0


@dataclass
class SARGrid:
    """Grid world whose global state is reserved for offline diagnostics.

    The decentralized interface consists only of :class:`Observation`,
    :meth:`deliver`, and the actions passed to :meth:`step`. Messages are
    delivered before task actions; callers should therefore call ``deliver``
    before asking policies to act.
    """

    config: dict
    _rng: random.Random = field(init=False, repr=False)
    _positions: list[Position] = field(default_factory=list, init=False, repr=False)
    _victims: list[_Victim] = field(default_factory=list, init=False, repr=False)
    _hazards: list[_Hazard] = field(default_factory=list, init=False, repr=False)
    _step_count: int = field(default=0, init=False, repr=False)
    _done: bool = field(default=False, init=False, repr=False)
    _known_facts: list[list[Message]] = field(default_factory=list, init=False, repr=False)
    _visited: list[set[Position]] = field(default_factory=list, init=False, repr=False)
    _scanned: list[set[Position]] = field(default_factory=list, init=False, repr=False)
    _teammates: list[dict[int, AgentTrack]] = field(default_factory=list, init=False, repr=False)
    _sent_facts: list[dict[int, list[Message]]] = field(default_factory=list, init=False, repr=False)

    @property
    def grid_size(self) -> int:
        return int(self.config.get("grid_size", 10))

    @property
    def positions(self) -> tuple[Position, ...]:
        """Current positions, exposed only for simulator/oracle diagnostics."""
        return tuple(self._positions)

    @property
    def victims(self) -> tuple[tuple[Position, str], ...]:
        """Victim positions and statuses; never include these in policy inputs."""
        return tuple((victim.position, victim.status) for victim in self._victims)

    @property
    def hazards(self) -> tuple[tuple[Position, str], ...]:
        """Hazard state; never include this property in policy inputs."""
        return tuple(
            (hazard.position, self._hazard_status(hazard))
            for hazard in self._hazards
        )

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def terminated(self) -> bool:
        return self._done

    @property
    def roles(self) -> tuple[Role, ...]:
        configured = self.config.get("roles", [role.value for role in Role])
        try:
            return tuple(Role(role) for role in configured)
        except ValueError as error:
            raise ValueError(f"unknown role in environment.roles: {error}") from error

    @classmethod
    def from_sender_observation(
        cls, config: dict, observation: Observation, seed: int,
        receiver_id: int | None = None,
    ) -> "SARGrid":
        """Sample one generative-model world conditioned only on local evidence."""
        world = cls(config)
        world.reset(seed)
        if observation.agent_id >= len(world._positions):
            raise ValueError("observation agent_id is outside the configured team")

        size = world.grid_size
        if not (0 <= observation.position[0] < size and 0 <= observation.position[1] < size):
            raise ValueError("observation position is outside the configured grid")

        occupied: set[Position] = {observation.position}
        positions = [observation.position] * len(world.roles)
        assigned_agent_ids = {observation.agent_id}
        for track in observation.teammates:
            if track.agent_id == observation.agent_id or track.agent_id >= len(positions):
                continue
            if (
                0 <= track.agent_id < len(positions)
                and 0 <= track.position[0] < size
                and 0 <= track.position[1] < size
                and track.position not in occupied
            ):
                positions[track.agent_id] = track.position
                occupied.add(track.position)
                assigned_agent_ids.add(track.agent_id)
        available = [
            (row, col)
            for row in range(size)
            for col in range(size)
            if (row, col) not in occupied
        ]
        world._rng.shuffle(available)
        for agent_id in range(len(positions)):
            if agent_id in assigned_agent_ids:
                continue
            positions[agent_id] = available.pop()
        world._positions = positions

        known_victims: dict[Position, Message] = {}
        known_hazards: dict[Position, Message] = {}
        known_clear_hazard_locations: set[Position] = set()
        for fact in observation.known_facts:
            if fact.kind == "victim":
                previous = known_victims.get(fact.location)
                if previous is None or fact.observed_at >= previous.observed_at:
                    known_victims[fact.location] = fact
            elif fact.kind == "hazard":
                if fact.status == "clear":
                    known_hazards.pop(fact.location, None)
                    known_clear_hazard_locations.add(fact.location)
                    continue
                known_clear_hazard_locations.discard(fact.location)
                previous = known_hazards.get(fact.location)
                if previous is None or fact.observed_at >= previous.observed_at:
                    known_hazards[fact.location] = fact

        occupied_entities: set[Position] = set()
        victims: list[_Victim] = []
        for location, fact in sorted(known_victims.items()):
            if len(victims) >= len(world._victims):
                break
            victims.append(_Victim(location, fact.status))
            occupied_entities.add(location)
        unknown_victim_count = len(world._victims) - len(victims)
        sampled_victims = [
            position
            for position in available
            if position not in occupied_entities
        ]
        world._rng.shuffle(sampled_victims)
        for position in sampled_victims[:unknown_victim_count]:
            victims.append(_Victim(position))
            occupied_entities.add(position)
        world._victims = victims

        hazards: list[_Hazard] = []
        for location, fact in sorted(known_hazards.items()):
            if len(hazards) >= len(world._hazards):
                break
            hazards.append(_Hazard(location, fact.status == "active"))
            occupied_entities.add(location)
        unknown_hazard_count = len(world._hazards) - len(hazards)
        sampled_hazards = [
            position
            for position in available
            if position not in occupied_entities
            and position not in known_clear_hazard_locations
        ]
        world._rng.shuffle(sampled_hazards)
        for position in sampled_hazards[:unknown_hazard_count]:
            hazards.append(_Hazard(position))
            occupied_entities.add(position)
        world._hazards = hazards

        world._step_count = observation.time_step
        world._done = False
        world._known_facts[observation.agent_id] = list(observation.known_facts)
        if receiver_id is not None:
            summary = next(
                (
                    item for item in observation.receiver_summaries
                    if item.receiver_id == receiver_id
                ),
                None,
            )
            if summary is not None:
                world._known_facts[receiver_id] = list(summary.known_facts)
        world._visited[observation.agent_id] = set(observation.visited_cells)
        world._scanned[observation.agent_id] = set(observation.scanned_cells)
        world._observe_all()
        world._known_facts[observation.agent_id] = list(observation.known_facts)
        world._visited[observation.agent_id].update(observation.visited_cells)
        world._scanned[observation.agent_id].update(observation.scanned_cells)
        world._teammates[observation.agent_id] = {
            track.agent_id: track for track in observation.teammates
        }
        return world

    def reset(self, seed: int | None = None) -> tuple[Observation, ...]:
        size = self.grid_size
        if size < 1:
            raise ValueError("grid_size must be at least 1")
        if int(self.config.get("horizon", 100)) < 1:
            raise ValueError("horizon must be at least 1")
        if not 0.0 <= float(self.config.get("gamma", 0.99)) <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if int(self.config.get("hazard_disable_steps", 5)) < 1:
            raise ValueError("hazard_disable_steps must be at least 1")
        if int(self.config.get("max_messages_per_sender_step", 1)) != 1:
            raise ValueError("SARGrid supports exactly one message slot per sender per step")
        roles = self.roles
        if not roles:
            raise ValueError("at least one agent role is required")
        victim_count = int(self.config.get("victims", 3))
        hazard_count = int(self.config.get("hazards", 2))
        if victim_count < 0 or hazard_count < 0:
            raise ValueError("victims and hazards must be non-negative")
        if len(roles) + victim_count + hazard_count > size * size:
            raise ValueError("grid does not have enough cells for the configured entities")
        self._validate_probability("hazard_move_probability")
        self._validate_probability("hazard_toggle_probability")

        self._rng = random.Random(seed)
        cells = [(row, col) for row in range(size) for col in range(size)]
        self._rng.shuffle(cells)
        self._positions = cells[:len(roles)]
        self._victims = [
            _Victim(position)
            for position in cells[len(roles):len(roles) + victim_count]
        ]
        hazard_start = len(roles) + victim_count
        self._hazards = [
            _Hazard(position)
            for position in cells[hazard_start:hazard_start + hazard_count]
        ]
        self._step_count = 0
        self._done = False
        self._known_facts = [[] for _ in roles]
        self._visited = [set() for _ in roles]
        self._scanned = [set() for _ in roles]
        self._teammates = [{} for _ in roles]
        self._sent_facts = [{} for _ in roles]
        return self._observations()

    def observe(self) -> tuple[Observation, ...]:
        """Return fresh local observations without exposing simulator state."""
        self._observe_all()
        return self._observations()

    def deliver(
        self, communication: Sequence[CommunicationChoice | None]
    ) -> tuple[Observation, ...]:
        """Deliver at most one existing fact per sender before agents act."""
        if not self._positions:
            raise RuntimeError("call reset() before deliver()")
        if self._done:
            raise RuntimeError("episode has terminated; call reset() before deliver()")
        if len(communication) != len(self._positions):
            raise ValueError(
                f"expected {len(self._positions)} communication slots, got {len(communication)}"
            )
        for sender_id, choice in enumerate(communication):
            if choice is None:
                continue
            if not 0 <= choice.receiver_id < len(self._positions):
                raise ValueError(f"invalid receiver_id: {choice.receiver_id}")
            if choice.receiver_id == sender_id:
                raise ValueError("a sender cannot send a message to itself")
            if not self._is_available_message(sender_id, choice.message):
                if not self._is_current_intention(sender_id, choice.message):
                    raise ValueError("communication must use a fact in the sender's local history")
            self._merge_fact(choice.receiver_id, choice.message)
            self._merge_sent_fact(sender_id, choice.receiver_id, choice.message)
        return self._observations()

    def step(
        self,
        task_actions: Sequence[object],
        communication: Sequence[CommunicationChoice | None] | None = None,
    ) -> StepResult:
        """Apply one task action per agent, transition hazards, and return local views.

        Communication is a separate pre-action phase. ``communication`` is
        accepted for interface compatibility but must be delivered through
        :meth:`deliver` first so receiver policies can observe it before acting.
        """
        if not self._positions:
            raise RuntimeError("call reset() before step()")
        if self._done:
            raise RuntimeError("episode has terminated; call reset() before step()")
        if communication is not None and any(choice is not None for choice in communication):
            raise ValueError("call deliver() before choosing task actions")
        if communication is not None and len(communication) != len(self._positions):
            raise ValueError(
                f"expected {len(self._positions)} communication slots, got {len(communication)}"
            )
        if len(task_actions) != len(self._positions):
            raise ValueError(f"expected {len(self._positions)} task actions, got {len(task_actions)}")

        validated: list[str] = []
        for agent_id, action in enumerate(task_actions):
            if not isinstance(action, str):
                raise ValueError(f"invalid task action for agent {agent_id}: {action!r}")
            normalized = action.lower()
            if normalized not in MOVEMENTS and normalized not in {
                "scan", "stabilize", "rescue", "disable"
            }:
                choices = ", ".join((*MOVEMENTS, "scan", "stabilize", "rescue", "disable"))
                raise ValueError(f"invalid task action {action!r}; expected one of {choices}")
            self._validate_role_action(agent_id, normalized)
            validated.append(normalized)

        for agent_id, action in enumerate(validated):
            if action not in MOVEMENTS:
                continue
            dr, dc = MOVEMENTS[action]
            row, col = self._positions[agent_id]
            destination = (row + dr, col + dc)
            if not self._in_bounds(destination) or self._blocked(destination):
                continue
            self._positions[agent_id] = destination

        events = {"located": 0, "stabilized": 0, "rescued": 0, "disabled": 0}
        for agent_id, action in enumerate(validated):
            if action not in MOVEMENTS:
                self._apply_task_action(agent_id, action, events)

        self._step_count += 1
        self._transition_hazards()
        self._observe_all()
        reward_cfg = self.config.get("reward", {})
        reward = (
            float(reward_cfg.get("rescue", 10.0)) * events["rescued"]
            + float(reward_cfg.get("locate", 1.0)) * events["located"]
            + float(reward_cfg.get("stabilize", 2.0)) * events["stabilized"]
            - float(reward_cfg.get("step_penalty", 0.05)) * len(self._positions)
        )
        horizon = int(self.config.get("horizon", 100))
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        all_rescued = bool(self._victims) and all(v.status == "rescued" for v in self._victims)
        self._done = self._step_count >= horizon or (
            bool(self.config.get("terminate_when_all_rescued", True)) and all_rescued
        )
        return StepResult(
            observations=self._observations(),
            reward=reward,
            terminated=self._done,
            info={"step": self._step_count, **events},
        )

    def clone(self) -> "SARGrid":
        """Deep-copy simulator state for paired offline counterfactuals."""
        return deepcopy(self)

    def reseed_dynamics(self, seed: int) -> None:
        """Set a rollout's future transition randomness without changing its state."""
        self._rng.seed(seed)

    def _validate_probability(self, name: str) -> None:
        value = float(self.config.get(name, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    def _validate_role_action(self, agent_id: int, action: str) -> None:
        role = self.roles[agent_id]
        allowed = {
            Role.SCOUT: {"scan"},
            Role.MEDIC: {"stabilize", "disable"},
            Role.CARRIER: {"rescue"},
        }
        if action in {"scan", "stabilize", "rescue", "disable"} and action not in allowed[role]:
            raise ValueError(f"{role.value} cannot perform {action}")

    def _apply_task_action(self, agent_id: int, action: str, events: dict[str, int]) -> None:
        position = self._positions[agent_id]
        if action == "scan":
            visible = set(self._visible_cells(position))
            self._scanned[agent_id].update(visible)
            for victim in self._victims:
                if victim.position in visible and victim.status == "hidden":
                    victim.status = "located"
                    events["located"] += 1
            for hazard in self._hazards:
                if hazard.position in visible:
                    self._record_fact(agent_id, "hazard", hazard.position, self._hazard_status(hazard))
        elif action == "stabilize":
            victim = self._closest_victim(position, "located")
            if victim is not None:
                victim.status = "stabilized"
                events["stabilized"] += 1
                self._record_fact(agent_id, "victim", victim.position, victim.status)
        elif action == "rescue":
            victim = self._closest_victim(position, "stabilized")
            if victim is not None:
                victim.status = "rescued"
                events["rescued"] += 1
                self._record_fact(agent_id, "victim", victim.position, victim.status)
        elif action == "disable":
            hazard = self._closest_hazard(position)
            if hazard is not None and hazard.active:
                hazard.active = False
                hazard.disabled_remaining = max(1, int(self.config.get("hazard_disable_steps", 5)))
                events["disabled"] += 1
                self._record_fact(agent_id, "hazard", hazard.position, "disabled")

    def _closest_victim(self, position: Position, status: str) -> _Victim | None:
        candidates = [
            victim for victim in self._victims
            if victim.status == status and self._adjacent(position, victim.position)
        ]
        return min(candidates, key=lambda victim: self._distance(position, victim.position), default=None)

    def _closest_hazard(self, position: Position) -> _Hazard | None:
        candidates = [
            hazard for hazard in self._hazards
            if self._adjacent(position, hazard.position)
        ]
        return min(candidates, key=lambda hazard: self._distance(position, hazard.position), default=None)

    def _transition_hazards(self) -> None:
        move_probability = float(self.config.get("hazard_move_probability", 0.0))
        toggle_probability = float(self.config.get("hazard_toggle_probability", 0.0))
        random_draws = [
            (self._rng.random(), self._rng.random(), self._rng.random())
            for _ in self._hazards
        ]
        for hazard, (move_draw, toggle_draw, destination_draw) in zip(
            self._hazards, random_draws
        ):
            if hazard.disabled_remaining > 0:
                hazard.disabled_remaining -= 1
                if hazard.disabled_remaining == 0:
                    hazard.active = True
            elif toggle_draw < toggle_probability:
                hazard.active = not hazard.active
            if hazard.active and move_draw < move_probability:
                occupied = {
                    item.position for item in self._hazards if item is not hazard
                } | {victim.position for victim in self._victims}
                free = [
                    (row, col)
                    for row in range(self.grid_size)
                    for col in range(self.grid_size)
                    if (row, col) not in occupied
                ]
                if free:
                    hazard.position = free[min(int(destination_draw * len(free)), len(free) - 1)]

    def _observe_all(self) -> None:
        if not self._positions:
            return
        for agent_id, position in enumerate(self._positions):
            visible = set(self._visible_cells(position))
            self._visited[agent_id].update(visible)
            for victim in self._victims:
                if victim.status != "hidden" and victim.position in visible:
                    self._record_fact(agent_id, "victim", victim.position, victim.status)
            known_hazard_locations = {
                fact.location for fact in self._known_facts[agent_id] if fact.kind == "hazard"
            }
            observed_hazard_locations = {
                hazard.position
                for hazard in self._hazards
                if hazard.position in visible
            }
            for hazard in self._hazards:
                if hazard.position in visible and hazard.position in known_hazard_locations:
                    self._record_fact(agent_id, "hazard", hazard.position, self._hazard_status(hazard))
            for location in known_hazard_locations.intersection(visible):
                if (
                    location not in observed_hazard_locations
                    and not any(
                        fact.kind == "hazard"
                        and fact.location == location
                        and fact.status == "clear"
                        for fact in self._known_facts[agent_id]
                    )
                ):
                    self._record_fact(agent_id, "hazard", location, "clear")
            for teammate_id, teammate_position in enumerate(self._positions):
                if teammate_id == agent_id or teammate_position in visible:
                    self._teammates[agent_id][teammate_id] = AgentTrack(
                        teammate_id,
                        self.roles[teammate_id],
                        teammate_position,
                        self._step_count,
                    )

    def _observations(self) -> tuple[Observation, ...]:
        if not self._positions:
            return ()
        observations = []
        for agent_id, (role, position) in enumerate(zip(self.roles, self._positions)):
            observations.append(Observation(
                agent_id=agent_id,
                role=role,
                position=position,
                grid_size=self.grid_size,
                time_step=self._step_count,
                visible_cells=self._visible_cells(position),
                visited_cells=tuple(sorted(self._visited[agent_id])),
                scanned_cells=tuple(sorted(self._scanned[agent_id])),
                teammates=tuple(
                    self._teammates[agent_id][key]
                    for key in sorted(self._teammates[agent_id])
                ),
                receiver_summaries=tuple(
                    ReceiverSummary(receiver_id, tuple(facts))
                    for receiver_id, facts in sorted(self._sent_facts[agent_id].items())
                ),
                known_facts=tuple(self._known_facts[agent_id]),
            ))
        return tuple(observations)

    def _record_fact(self, agent_id: int, kind: str, location: Position, status: str) -> None:
        self._merge_fact(
            agent_id,
            Message(kind, location, self._step_count, status, agent_id),
        )

    def _merge_fact(self, agent_id: int, message: Message) -> None:
        self._merge_message(self._known_facts[agent_id], message)

    def _merge_sent_fact(self, sender_id: int, receiver_id: int, message: Message) -> None:
        receiver_facts = self._sent_facts[sender_id].setdefault(receiver_id, [])
        self._merge_message(receiver_facts, message)

    @staticmethod
    def _merge_message(facts: list[Message], message: Message) -> None:
        if message.kind == "intention":
            facts[:] = [
                fact for fact in facts
                if not (fact.kind == "intention" and fact.sender_id == message.sender_id)
            ]
            facts.append(message)
            return
        for index, fact in enumerate(facts):
            if fact.kind == message.kind and fact.location == message.location:
                if message.observed_at >= fact.observed_at:
                    facts[index] = message
                return
        facts.append(message)

    def _is_available_message(self, sender_id: int, message: Message) -> bool:
        return any(
            fact.kind == message.kind
            and fact.location == message.location
            and fact.observed_at == message.observed_at
            and fact.status == message.status
            for fact in self._known_facts[sender_id]
        )

    def _is_current_intention(self, sender_id: int, message: Message) -> bool:
        agent_position = (
            message.metadata.get("agent_position")
            if message.metadata is not None
            else None
        )
        return (
            message.kind == "intention"
            and message.sender_id == sender_id
            and agent_position == self._positions[sender_id]
            and message.observed_at == self._step_count
        )

    def _visible_cells(self, position: Position) -> tuple[Position, ...]:
        radius = max(0, int(self.config.get("vision_radius", 1)))
        row, col = position
        return tuple(
            (r, c)
            for r in range(max(0, row - radius), min(self.grid_size, row + radius + 1))
            for c in range(max(0, col - radius), min(self.grid_size, col + radius + 1))
        )

    def _visible_cells_for(self, position: Position) -> tuple[Position, ...]:
        return self._visible_cells(position)

    def _blocked(self, position: Position) -> bool:
        return any(hazard.position == position and hazard.active for hazard in self._hazards)

    def _in_bounds(self, position: Position) -> bool:
        row, col = position
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    @staticmethod
    def _distance(first: Position, second: Position) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    @classmethod
    def _adjacent(cls, first: Position, second: Position) -> bool:
        return cls._distance(first, second) <= 1

    @staticmethod
    def _hazard_status(hazard: _Hazard) -> str:
        if hazard.active:
            return "active"
        return "disabled" if hazard.disabled_remaining > 0 else "inactive"
