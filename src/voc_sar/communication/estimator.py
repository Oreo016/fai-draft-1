"""Sender-conditioned particle hypotheses and paired counterfactual rollouts."""

from dataclasses import dataclass, replace
import math
import random
import time
from typing import Protocol, Sequence

from voc_sar.environment.sar_grid import SARGrid
from voc_sar.policies.task import TaskPolicy
from voc_sar.policies.task import task_policies
from voc_sar.types import CommunicationChoice, Message, Observation, Role


@dataclass(frozen=True)
class ValueEstimate:
    marginal_value: float
    communication_cost: float
    value_of_communication: float
    standard_error: float | None = None
    paired_returns: tuple[tuple[float, float], ...] = ()
    particle_seconds: float | None = None
    rollout_seconds: float | None = None


class ReceiverStateEstimator(Protocol):
    def sample_states(
        self, sender_id: int, receiver_id: int, particles: int, observation: Observation
    ) -> Sequence[SARGrid]:
        """Sample model worlds using only sender-observable evidence."""
        ...


class ParticleReceiverEstimator:
    """Sequential model-world filter conditioned on sender-local observations."""

    def __init__(self, config: dict, seed: int = 0) -> None:
        self.config = config
        self.seed = seed
        self._beliefs: dict[tuple[int, int, int], tuple[Observation, tuple[SARGrid, ...]]] = {}
        self.last_proposal_ess: float | None = None

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
        key = sender_id, receiver_id, particles
        previous = self._beliefs.get(key)
        if previous is not None and previous[0] == observation:
            return previous[1]
        context_seed = self.seed + observation.time_step * 1_000_003
        predicted = []
        if previous is not None and previous[0].time_step < observation.time_step:
            policies = task_policies(roles)
            for particle in previous[1]:
                world = particle.clone()
                observations = world.observe()
                while world.step_count < observation.time_step and not world.terminated:
                    actions = [policy.act(local) for policy, local in zip(policies, observations)]
                    observations = world.step(actions).observations
                if not world.terminated:
                    predicted.append(world)
        ancestor_count = particles - max(1, particles // 4)
        ancestors = []
        self.last_proposal_ess = None
        if predicted:
            weights = [world.local_observation_weight(observation) for world in predicted]
            self.last_proposal_ess = sum(weights) ** 2 / sum(weight ** 2 for weight in weights)
            ancestors = random.Random(context_seed + 1).choices(
                predicted, weights=weights, k=ancestor_count
            )
        samples = []
        for index in range(particles):
            particle_seed = context_seed + index * 9_176
            samples.append(
                SARGrid.from_sender_observation(
                    self.config, observation, particle_seed, receiver_id,
                    predicted_world=ancestors[index] if index < len(ancestors) else None,
                )
            )
        posterior = tuple(samples)
        self._beliefs[key] = observation, posterior
        return posterior


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
        record_rollouts: bool = False,
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
        self.record_rollouts = record_rollouts
        self._reference_estimators: dict[Role, PairedRolloutEstimator] = {}

    def estimate_message(
        self, message: Message, horizon: int, observation: Observation
    ) -> ValueEstimate:
        """Average over fixed synthetic receiver roles without real receiver features."""
        if observation.role not in self._reference_estimators:
            roles = (observation.role, Role.CARRIER, Role.MEDIC)
            config = {**self.state_estimator.config, "roles": [role.value for role in roles]}
            self._reference_estimators[observation.role] = PairedRolloutEstimator(
                ParticleReceiverEstimator(config, seed=self.seed),
                task_policies(roles),
                particles=self.particles,
                rollout_count=self.rollout_count,
                gamma=self.gamma,
                cost_weight=self.cost_weight,
                unit_cost=self.unit_cost,
                seed=self.seed,
                record_rollouts=self.record_rollouts,
            )
        local = replace(observation, agent_id=0, teammates=(), receiver_summaries=())
        reference = self._reference_estimators[observation.role]
        synthetic_message = replace(message, sender_id=0)
        if self.rollout_count == 1:
            receiver = 1 + (self.seed + observation.time_step) % 2
            return reference.estimate(0, receiver, synthetic_message, horizon, local)
        half = self.rollout_count // 2
        carrier = reference.estimate(0, 1, synthetic_message, horizon, local, rollout_count=half)
        medic = reference.estimate(
            0, 2, synthetic_message, horizon, local, rollout_count=self.rollout_count - half,
            rollout_seed_offset=7_919_117,
        )
        marginal = (carrier.marginal_value + medic.marginal_value) / 2
        return ValueEstimate(
            marginal, carrier.communication_cost, marginal - carrier.communication_cost,
            math.hypot(carrier.standard_error, medic.standard_error) / 2,
            carrier.paired_returns + medic.paired_returns,
            carrier.particle_seconds + medic.particle_seconds if self.record_rollouts else None,
            carrier.rollout_seconds + medic.rollout_seconds if self.record_rollouts else None,
        )

    def estimate(
        self,
        sender_id: int,
        receiver_id: int,
        message: Message,
        horizon: int,
        observation: Observation,
        *,
        rollout_count: int | None = None,
        rollout_seed_offset: int = 0,
    ) -> ValueEstimate:
        if horizon < 1:
            raise ValueError("horizon must be at least one")
        if sender_id == receiver_id:
            raise ValueError("sender and receiver must be different agents")
        if sender_id != observation.agent_id:
            raise ValueError("sender_id must match the supplied local observation")
        start = time.perf_counter()
        worlds = self.state_estimator.sample_states(
            sender_id, receiver_id, self.particles, observation
        )
        if not worlds:
            raise ValueError("receiver-state estimator returned no particles")

        rollout_start = time.perf_counter()
        paired_differences: list[float] = []
        paired_returns = []
        rollout_seed = self.seed + observation.time_step * 65_537 + rollout_seed_offset
        particle_rng = random.Random(rollout_seed + sender_id * 257 + receiver_id)
        for rollout_index in range(rollout_count if rollout_count is not None else self.rollout_count):
            particle = worlds[particle_rng.randrange(len(worlds))]
            paired_seed = rollout_seed + rollout_index
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
            if self.record_rollouts:
                paired_returns.append((send_return, no_send_return))

        marginal_value = sum(paired_differences) / len(paired_differences)
        standard_error = _standard_error(paired_differences)
        cost = _message_cost(message, self.cost_weight, self.unit_cost)
        return ValueEstimate(
            marginal_value=marginal_value,
            communication_cost=cost,
            value_of_communication=marginal_value - cost,
            standard_error=standard_error,
            paired_returns=tuple(paired_returns),
            particle_seconds=rollout_start - start if self.record_rollouts else None,
            rollout_seconds=time.perf_counter() - rollout_start if self.record_rollouts else None,
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
        # The caller validates the candidate against the original sender context.
        # Reconstructed worlds can refresh facts during their observation pass.
        communication = [None] * len(observations)
        communication[sender_id] = _communication_choice(receiver_id, message)
        observations = world._deliver_validated(communication)

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
