"""Sender-conditioned particle hypotheses and paired counterfactual rollouts."""

from dataclasses import dataclass
import math
import random
from typing import Protocol, Sequence

from voc_sar.environment.sar_grid import SARGrid
from voc_sar.policies.task import TaskPolicy
from voc_sar.types import CommunicationChoice, Message, Observation


@dataclass(frozen=True)
class ValueEstimate:
    marginal_value: float
    communication_cost: float
    value_of_communication: float
    standard_error: float | None = None


class ReceiverStateEstimator(Protocol):
    def sample_states(
        self, sender_id: int, receiver_id: int, particles: int, observation: Observation
    ) -> Sequence[SARGrid]:
        """Sample model worlds using only sender-observable evidence."""
        ...


class ParticleReceiverEstimator:
    """Fixed prior particle filter conditioned on sender facts and teammate tracks."""

    def __init__(self, config: dict, seed: int = 0) -> None:
        self.config = config
        self.seed = seed

    def sample_states(
        self, sender_id: int, receiver_id: int, particles: int, observation: Observation
    ) -> Sequence[SARGrid]:
        if sender_id != observation.agent_id:
            raise ValueError("sender_id must match the supplied local observation")
        roles = SARGrid(self.config).roles
        if not 0 <= receiver_id < len(roles):
            raise ValueError("receiver_id is outside the configured team")
        if sender_id == receiver_id:
            raise ValueError("sender and receiver must be different agents")
        if particles < 1:
            raise ValueError("particles must be at least one")
        context_seed = self.seed + observation.time_step * 1_000_003
        samples = []
        for index in range(particles):
            particle_seed = context_seed + index * 9_176
            samples.append(
                SARGrid.from_sender_observation(
                    self.config, observation, particle_seed, receiver_id
                )
            )
        return samples


class PairedRolloutEstimator:
    """Estimate send-minus-no-send task return on matched particle rollouts."""

    def __init__(
        self,
        state_estimator: ReceiverStateEstimator,
        task_policies: Sequence[TaskPolicy],
        *,
        particles: int = 64,
        rollout_count: int = 32,
        gamma: float = 0.99,
        cost_weight: float = 0.0,
        unit_cost: float = 1.0,
        seed: int = 0,
    ) -> None:
        if particles < 1 or rollout_count < 1:
            raise ValueError("particles and rollout_count must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if (
            not math.isfinite(cost_weight)
            or not math.isfinite(unit_cost)
            or cost_weight < 0
            or unit_cost < 0
        ):
            raise ValueError("communication costs must be non-negative")
        self.state_estimator = state_estimator
        self.task_policies = tuple(task_policies)
        self.particles = particles
        self.rollout_count = rollout_count
        self.gamma = gamma
        self.cost_weight = cost_weight
        self.unit_cost = unit_cost
        self.seed = seed

    def estimate(
        self,
        sender_id: int,
        receiver_id: int,
        message: Message,
        horizon: int,
        observation: Observation,
    ) -> ValueEstimate:
        if horizon < 1:
            raise ValueError("horizon must be at least one")
        if sender_id == receiver_id:
            raise ValueError("sender and receiver must be different agents")
        if sender_id != observation.agent_id:
            raise ValueError("sender_id must match the supplied local observation")
        worlds = self.state_estimator.sample_states(
            sender_id, receiver_id, self.particles, observation
        )
        if not worlds:
            raise ValueError("receiver-state estimator returned no particles")

        paired_differences: list[float] = []
        particle_rng = random.Random(
            self.seed + observation.time_step * 65_537 + sender_id * 257 + receiver_id
        )
        for rollout_index in range(self.rollout_count):
            particle = worlds[particle_rng.randrange(len(worlds))]
            paired_seed = self.seed + observation.time_step * 65_537 + rollout_index
            no_send_world = particle.clone()
            send_world = particle.clone()
            no_send_world.reseed_dynamics(paired_seed)
            send_world.reseed_dynamics(paired_seed)
            no_send_return = _rollout_return(
                no_send_world, self.task_policies, horizon, self.gamma
            )
            send_return = _rollout_return(
                send_world,
                self.task_policies,
                horizon,
                self.gamma,
                (sender_id, receiver_id, message),
            )
            paired_differences.append(send_return - no_send_return)

        marginal_value = sum(paired_differences) / len(paired_differences)
        standard_error = _standard_error(paired_differences)
        cost = _message_cost(message, self.cost_weight, self.unit_cost)
        return ValueEstimate(
            marginal_value=marginal_value,
            communication_cost=cost,
            value_of_communication=marginal_value - cost,
            standard_error=standard_error,
        )


def _rollout_return(
    world: SARGrid,
    task_policies: Sequence[TaskPolicy],
    horizon: int,
    gamma: float,
    intervention: tuple[int, int, Message] | None = None,
) -> float:
    observations = world.observe()
    if intervention is not None:
        sender_id, receiver_id, message = intervention
        # The particle world is reconstructed from the sender's observation,
        # but its observe pass may refresh or canonicalize facts. Preserve the
        # exact candidate that the sender supplied to the estimator so the
        # counterfactual delivery validates against the sampled sender history.
        world._merge_fact(sender_id, message)
        communication = [None] * len(observations)
        communication[sender_id] = _communication_choice(receiver_id, message)
        observations = world.deliver(communication)

    total = 0.0
    discount = 1.0
    for _ in range(horizon):
        if world.terminated:
            break
        actions = [
            task_policies[observation.agent_id].act(observation)
            for observation in observations
        ]
        result = world.step(actions)
        total += discount * result.reward
        discount *= gamma
        observations = result.observations
        if result.terminated:
            break
    return total


def _communication_choice(receiver_id: int, message: Message) -> CommunicationChoice:
    return CommunicationChoice(receiver_id=receiver_id, message=message)


def _message_cost(message: Message, cost_weight: float, unit_cost: float) -> float:
    metadata_cost = 1.0
    if message.metadata is not None and "cost" in message.metadata:
        metadata_cost = float(message.metadata["cost"])
    if not math.isfinite(metadata_cost) or metadata_cost < 0:
        raise ValueError("message cost metadata must be finite and non-negative")
    return cost_weight * unit_cost * metadata_cost


def _standard_error(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    sample_variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(sample_variance / len(values))
