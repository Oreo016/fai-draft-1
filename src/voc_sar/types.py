"""Shared domain types. Keep simulator truth separate from policy inputs."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

Position = tuple[int, int]


class Role(str, Enum):
    SCOUT = "scout"
    CARRIER = "carrier"
    MEDIC = "medic"


@dataclass(frozen=True)
class AgentTrack:
    """Last locally observed teammate position; never the current global position."""

    agent_id: int
    role: Role
    position: Position
    observed_at: int


@dataclass(frozen=True)
class ReceiverSummary:
    """Facts this sender knows it has already delivered to one receiver."""

    receiver_id: int
    known_facts: tuple["Message", ...] = ()


@dataclass(frozen=True)
class Observation:
    """One agent's local view; must not contain the global simulator state."""

    agent_id: int
    role: Role
    position: Position
    grid_size: int
    time_step: int = 0
    visible_cells: tuple[Position, ...] = ()
    visited_cells: tuple[Position, ...] = ()
    scanned_cells: tuple[Position, ...] = ()
    teammates: tuple[AgentTrack, ...] = ()
    receiver_summaries: tuple[ReceiverSummary, ...] = ()
    known_facts: tuple["Message", ...] = ()


@dataclass(frozen=True)
class Message:
    """An existing structured information item considered for communication."""

    kind: str
    location: Position
    observed_at: int
    status: str
    sender_id: int
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CommunicationChoice:
    """A directed send decision; receiver and message are chosen jointly."""

    receiver_id: int
    message: Message


@dataclass(frozen=True)
class StepResult:
    """Public transition result. Keep private simulator state out of this type."""

    observations: tuple[Observation, ...]
    reward: float
    terminated: bool
    info: Mapping[str, Any]
