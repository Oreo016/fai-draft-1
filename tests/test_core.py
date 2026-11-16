import json
from pathlib import Path
import tempfile
import unittest

from voc_sar.communication.estimator import (
    PairedRolloutEstimator,
    ParticleReceiverEstimator,
    ValueEstimate,
)
from voc_sar.communication.selector import select_communication, shortlist
from voc_sar.environment.sar_grid import SARGrid
from voc_sar.evaluation.metrics import (
    bootstrap_cluster_mean_interval,
    bootstrap_mean_interval,
    episode_return,
    filter_recall,
    pair_regret,
    receiver_regret,
    voc_mean_absolute_error,
)
from voc_sar.evaluation.oracle import CounterfactualOracle
from voc_sar.experiments.runner import run_experiments
from voc_sar.policies.task import task_policies
from voc_sar.types import (
    CommunicationChoice,
    Message,
    Observation,
    ReceiverSummary,
    Role,
)


def small_config() -> dict:
    return {
        "environment": {
            "grid_size": 6,
            "roles": ["scout", "carrier", "medic"],
            "victims": 1,
            "hazards": 0,
            "horizon": 8,
            "gamma": 1.0,
            "vision_radius": 0,
            "reward": {
                "rescue": 10.0,
                "locate": 1.0,
                "stabilize": 2.0,
                "step_penalty": 0.05,
            },
        },
        "communication": {
            "max_messages_per_sender_step": 1,
            "particles": 2,
            "shortlist_size": 2,
            "rollout_count": 1,
            "myopic_horizon": 1,
            "nonmyopic_horizon": 2,
            "oracle_horizon": 2,
            "unit_cost": 1.0,
            "cost_weight": 0.0,
            "cost_sweep": [0.0, 0.5],
            "useful_message_threshold": 0.0,
            "oracle_value_threshold": 0.5,
            "fixed_frequency": 2,
        },
        "evaluation": {
            "seeds": [0],
            "held_out_contexts": 1,
            "bootstrap_samples": 10,
            "confidence_level": 0.95,
            "thresholds": {},
            "output_dir": "results",
        },
    }


class StaticParticleEstimator:
    def __init__(self, world: SARGrid) -> None:
        self.world = world

    def sample_states(self, sender_id, receiver_id, particles, observation):
        if sender_id != observation.agent_id or sender_id == receiver_id:
            raise ValueError("invalid test particle request")
        return [self.world.clone() for _ in range(particles)]


class CoreTests(unittest.TestCase):
    def test_seeded_layout_and_local_observation_boundary(self) -> None:
        config = small_config()["environment"]
        world = SARGrid(config)
        first = world.reset(23)
        positions = world.positions
        second = world.reset(23)
        self.assertEqual(first, second)
        self.assertEqual(positions, world.positions)
        self.assertFalse({"victims", "hazards", "state"}.intersection(
            Observation.__dataclass_fields__
        ))

    def test_particle_prior_uses_only_sender_local_evidence(self) -> None:
        config = small_config()["environment"]
        first_world = SARGrid(config)
        first_world.reset(11)
        second_world = SARGrid(config)
        second_world.reset(11)
        scout_position = first_world.positions[0]
        hidden_locations = [
            (row, col)
            for row in range(config["grid_size"])
            for col in range(config["grid_size"])
            if (row, col) != scout_position
        ]
        first_world._victims[0].position = hidden_locations[0]
        second_world._victims[0].position = hidden_locations[-1]
        first_observation = first_world.observe()[0]
        second_observation = second_world.observe()[0]
        self.assertEqual(first_observation, second_observation)

        particle_filter = ParticleReceiverEstimator(config, seed=99)
        first_particles = particle_filter.sample_states(0, 1, 4, first_observation)
        second_particles = particle_filter.sample_states(0, 1, 4, second_observation)
        self.assertEqual(
            [(world.positions, world.victims, world.hazards) for world in first_particles],
            [(world.positions, world.victims, world.hazards) for world in second_particles],
        )

    def test_scan_delivery_and_transition_order(self) -> None:
        world = SARGrid(small_config()["environment"])
        world.reset(2)
        world._positions = [(2, 2), (4, 4), (0, 0)]
        world._victims[0].position = (2, 2)
        observations = world.observe()
        result = world.step(["scan", "wait", "wait"])
        self.assertEqual(world.victims[0][1], "located")
        self.assertAlmostEqual(result.reward, 1.0 - 3 * 0.05)
        fact = next(fact for fact in result.observations[0].known_facts if fact.kind == "victim")

        communication = [None, None, None]
        communication[0] = CommunicationChoice(1, fact)
        delivered = world.deliver(communication)
        self.assertIn(fact, delivered[1].known_facts)
        sender_summary = delivered[0].receiver_summaries[0]
        self.assertEqual(sender_summary.receiver_id, 1)
        self.assertIn(fact, sender_summary.known_facts)
        particle = SARGrid.from_sender_observation(
            small_config()["environment"], delivered[0], 91, receiver_id=1
        )
        self.assertIn(fact, particle.observe()[1].known_facts)
        self.assertEqual(observations[1].known_facts, ())
        with self.assertRaisesRegex(ValueError, "call deliver"):
            world.step(["wait", "wait", "wait"], communication)

    def test_role_restrictions_and_event_reward(self) -> None:
        world = SARGrid(small_config()["environment"])
        world.reset(3)
        with self.assertRaisesRegex(ValueError, "scout cannot perform rescue"):
            world.step(["rescue", "wait", "wait"])
        world._positions = [(2, 2), (3, 3), (0, 0)]
        world._victims[0].position = (2, 2)
        world.step(["scan", "wait", "wait"])
        world._positions[2] = (2, 3)
        world._record_fact(2, "victim", (2, 2), "located")
        result = world.step(["wait", "wait", "stabilize"])
        self.assertEqual(world.victims[0][1], "stabilized")
        self.assertAlmostEqual(result.reward, 2.0 - 3 * 0.05)

    def test_seeded_hazard_toggle_transition(self) -> None:
        config = small_config()["environment"]
        config["roles"] = ["scout"]
        config["victims"] = 0
        config["hazards"] = 1
        config["hazard_toggle_probability"] = 1.0
        first_world = SARGrid(config)
        first_world.reset(5)
        second_world = SARGrid(config)
        second_world.reset(5)
        first_world.step(["wait"])
        second_world.step(["wait"])
        self.assertEqual(first_world.hazards, second_world.hazards)
        self.assertEqual(first_world.hazards[0][1], "inactive")
        first_world.step(["wait"])
        self.assertEqual(first_world.hazards[0][1], "active")

    def test_paired_estimator_and_offline_oracle(self) -> None:
        world = SARGrid(small_config()["environment"])
        world.reset(4)
        world._positions = [(3, 5), (3, 1), (5, 5)]
        world._victims[0].position = (3, 3)
        world._victims[0].status = "stabilized"
        message = Message("victim", (3, 3), 0, "stabilized", 0)
        world._record_fact(0, "victim", (3, 3), "stabilized")
        policies = task_policies(world.roles)
        observation = world.observe()[0]

        estimator = PairedRolloutEstimator(
            StaticParticleEstimator(world),
            policies,
            particles=1,
            rollout_count=2,
            gamma=1.0,
            cost_weight=0.5,
            unit_cost=0.5,
            seed=10,
        )
        estimate = estimator.estimate(0, 1, message, 2, observation)
        oracle = CounterfactualOracle(
            policies,
            rollout_count=2,
            gamma=1.0,
            cost_weight=0.5,
            unit_cost=0.5,
            seed=10,
        )
        oracle_estimate = oracle.evaluate(world, 0, 1, message, 2)
        self.assertGreater(estimate.marginal_value, 0.0)
        self.assertAlmostEqual(estimate.marginal_value, oracle_estimate.marginal_value)
        self.assertAlmostEqual(estimate.communication_cost, 0.25)
        self.assertAlmostEqual(
            estimate.value_of_communication,
            estimate.marginal_value - estimate.communication_cost,
        )

    def test_receiver_aware_selector_uses_estimated_receiver_value(self) -> None:
        class FakeEstimator:
            cost_weight = 0.0
            unit_cost = 1.0

            def estimate(
                self, sender_id, receiver_id, message, horizon, observation
            ):
                value = (
                    2.0
                    if receiver_id == 1
                    and message.kind == "victim"
                    and sender_id == observation.agent_id
                    and horizon > 0
                    else 0.25
                )
                return ValueEstimate(value, 0.0, value, 0.0)

        observation = Observation(0, Role.SCOUT, (1, 1), 5, time_step=0)
        message = Message("victim", (2, 2), 0, "stabilized", 0)
        choice = select_communication(
            0,
            [message],
            [1, 2],
            3,
            FakeEstimator(),
            observation=observation,
            receiver_roles=(Role.SCOUT, Role.CARRIER, Role.MEDIC),
            shortlist_size=2,
        )
        self.assertIsNotNone(choice)
        self.assertEqual(choice.receiver_id, 1)

    def test_candidate_filter_uses_known_deliveries_but_agnostic_is_receiver_free(self) -> None:
        victim = Message("victim", (2, 2), 0, "stabilized", 0)
        hazard = Message("hazard", (1, 3), 1, "active", 0)
        ranked = shortlist(
            0,
            [victim, hazard],
            [1],
            1,
            current_step=1,
            receiver_roles=(Role.SCOUT, Role.CARRIER, Role.MEDIC),
            receiver_summaries=(ReceiverSummary(1, (victim,)),),
        )
        self.assertEqual(ranked, [(hazard, 1)])

        class FakeEstimator:
            cost_weight = 0.0
            unit_cost = 1.0

            def estimate(self, sender_id, receiver_id, message, horizon, observation):
                if sender_id != observation.agent_id or receiver_id < 0 or horizon < 1:
                    raise ValueError("invalid test estimator request")
                value = 2.0 if message.kind == "victim" else 3.0
                return ValueEstimate(value, 0.0, value, 0.0)

        observation = Observation(0, Role.SCOUT, (0, 0), 6)
        choices = [
            select_communication(
                0,
                [victim, hazard],
                [1, 2],
                2,
                FakeEstimator(),
                observation=observation,
                shortlist_size=1,
                receiver_roles=roles,
                aware=False,
                seed=7,
            )
            for roles in (
                (Role.SCOUT, Role.CARRIER, Role.MEDIC),
                (Role.SCOUT, Role.MEDIC, Role.CARRIER),
            )
        ]
        self.assertEqual(choices[0], choices[1])
        self.assertEqual(choices[0].message, victim)

    def test_metrics_and_bootstrap_are_deterministic(self) -> None:
        self.assertAlmostEqual(episode_return([1.0, 2.0], 0.5), 2.0)
        self.assertEqual(receiver_regret({1: 2.0, 2: 1.0}, 2), 1.0)
        self.assertEqual(pair_regret({("a", 1): 3.0}, None), 3.0)
        self.assertEqual(filter_recall({(0, 1): 1.0}, [(0, 1)], 0.5), 1.0)
        self.assertEqual(voc_mean_absolute_error([1.0, 2.0], [1.5, 2.0]), 0.25)
        interval = bootstrap_mean_interval([1.0, 2.0, 3.0], samples=50, seed=4)
        self.assertEqual(interval, bootstrap_mean_interval([1.0, 2.0, 3.0], samples=50, seed=4))
        cluster_interval = bootstrap_cluster_mean_interval(
            [[0.0, 1.0], [1.0, 1.0]], samples=50, seed=4
        )
        self.assertEqual(
            cluster_interval,
            bootstrap_cluster_mean_interval([[0.0, 1.0], [1.0, 1.0]], samples=50, seed=4),
        )

    def test_experiment_runner_writes_episode_and_summary_records(self) -> None:
        config = small_config()
        with tempfile.TemporaryDirectory() as output:
            config["evaluation"]["output_dir"] = output
            summary = run_experiments(config)
            episode_file = Path(output) / "episodes.json"
            summary_file = Path(output) / "summary.json"
            self.assertTrue(episode_file.is_file())
            self.assertTrue(summary_file.is_file())
            self.assertEqual(summary["episode_count"], 10)
            self.assertTrue(summary["diagnostics"]["held_out_contexts"] > 0)
            self.assertEqual(len(json.loads(episode_file.read_text(encoding="utf-8"))), 10)
            self.assertTrue(summary["hypotheses"]["H4_cost_sensitivity"]["high_cost"] == 0.5)


if __name__ == "__main__":
    unittest.main()
