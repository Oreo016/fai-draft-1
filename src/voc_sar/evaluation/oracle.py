"""Offline-only counterfactual evaluator with privileged simulator context."""

import math
from typing import Sequence

from voc_sar.communication.estimator import ValueEstimate, _message_cost, _rollout_return
from voc_sar.environment.sar_grid import SARGrid
from voc_sar.policies.task import TaskPolicy
from voc_sar.types import Message


class CounterfactualOracle:
    """Evaluate matched send/no-send returns from complete simulator state."""

    def __init__(
        self,
        task_policies: Sequence[TaskPolicy],
        *,
        rollout_count: int = 32,
        gamma: float = 0.99,
        cost_weight: float = 0.0,
        unit_cost: float = 1.0,
        seed: int = 0,
    ) -> None:
        if rollout_count < 1:
            raise ValueError("rollout_count must be at least one")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if (
            not math.isfinite(cost_weight)
            or not math.isfinite(unit_cost)
            or cost_weight < 0
            or unit_cost < 0
        ):
            raise ValueError("communication costs must be non-negative")
        self.task_policies = tuple(task_policies)
        self.rollout_count = rollout_count
        self.gamma = gamma
        self.cost_weight = cost_weight
        self.unit_cost = unit_cost
        self.seed = seed

    def evaluate(
        self,
        context: SARGrid,
        sender_id: int,
        receiver_id: int,
        message: Message,
        horizon: int,
    ) -> ValueEstimate:
        """Return offline oracle value; never call this from decentralized policies."""
        if not isinstance(context, SARGrid):
            raise TypeError("oracle context must be a SARGrid instance")
        if horizon < 1:
            raise ValueError("horizon must be at least one")
        if sender_id == receiver_id:
            raise ValueError("sender and receiver must be different agents")
        if not 0 <= sender_id < len(context.positions) or not 0 <= receiver_id < len(context.positions):
            raise ValueError("sender_id and receiver_id must refer to configured agents")

        paired_differences: list[float] = []
        for rollout_index in range(self.rollout_count):
            paired_seed = self.seed + context.step_count * 65_537 + rollout_index
            no_send_world = context.clone()
            send_world = context.clone()
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
        if len(paired_differences) < 2:
            standard_error = 0.0
        else:
            mean = marginal_value
            variance = sum((value - mean) ** 2 for value in paired_differences) / (
                len(paired_differences) - 1
            )
            standard_error = math.sqrt(variance / len(paired_differences))
        cost = _message_cost(message, self.cost_weight, self.unit_cost)
        return ValueEstimate(
            marginal_value=marginal_value,
            communication_cost=cost,
            value_of_communication=marginal_value - cost,
            standard_error=standard_error,
        )
