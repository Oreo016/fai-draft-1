"""Focused unit and smoke tests for additive E01-E12 study code."""

import inspect
import json
from pathlib import Path
import tempfile
import unittest

from voc_sar.communication.estimator import PairedRolloutEstimator, ParticleReceiverEstimator, _rollout_return
from voc_sar.environment.sar_grid import SARGrid
from voc_sar.research_experiments.common import (
    collect_contexts, load_study_config, write_experiment_visualizations, write_reports,
)
from voc_sar.research_experiments.e01_message_conditioned import MessageConditionedParticles, _force_candidate_fact
from voc_sar.research_experiments.e02_role_only import select_role_only
from voc_sar.research_experiments.e03_particle_diagnostics import diversity_ess
from voc_sar.research_experiments.e04_noise_crn import _estimate
from voc_sar.research_experiments.e05_continuation_bias import _continued_return
from voc_sar.research_experiments.e06_robust_null import lower_confidence_bound, send_by_epsilon, send_by_lcb
from voc_sar.research_experiments.e07_cost_pareto import receiver_cost_multiplier
from voc_sar.research_experiments.e08_hypothesis_robustness import analyze as analyze_e08
from voc_sar.research_experiments.e08_hypothesis_robustness import run as run_e08
from voc_sar.research_experiments.e10_staleness import analyze as analyze_e10
from voc_sar.research_experiments.e11_robustness import _settings
from voc_sar.research_experiments.common import holm_adjust, spearman
from voc_sar.types import Message, Role


def _small_study_config(experiment_id: str):
    config = load_study_config(experiment_id)
    config["environment"].update({
        "grid_size": 6, "roles": ["scout", "carrier", "medic"],
        "victims": 1, "hazards": 0, "horizon": 5, "gamma": 1.0,
        "hazard_move_probability": 0.0, "hazard_toggle_probability": 0.0,
        "vision_radius": 0,
    })
    config["communication"].update({
        "particles": 2, "rollout_count": 1, "myopic_horizon": 1,
        "nonmyopic_horizon": 2, "oracle_horizon": 3, "shortlist_size": 2,
    })
    config["evaluation"].update({"seeds": [0], "bootstrap_samples": 20})
    config["experiment"]["contexts_per_seed"] = 1
    return config


class ResearchExperimentTests(unittest.TestCase):
    def test_seed_extension_preserves_existing_prefix(self) -> None:
        config = load_study_config("E01", extended_seeds=True)
        self.assertEqual(config["evaluation"]["seeds"][:10], list(range(10)))
        self.assertEqual(config["evaluation"]["seeds"], list(range(30)))
        self.assertEqual(config["evaluation"]["bootstrap_samples"], 10_000)

    def test_message_conditioning_changes_only_sampled_particle(self) -> None:
        config = _small_study_config("E01")
        world = SARGrid(config["environment"])
        observation = world.reset(7)[0]
        state_filter = ParticleReceiverEstimator(config["environment"], seed=11)
        candidate = Message("victim", (2, 2), 0, "located", 0)
        conditioned = MessageConditionedParticles(state_filter, candidate)
        particle = conditioned.sample_states(0, 1, 1, observation)[0]
        self.assertIn(((2, 2), "located"), particle.victims)
        self.assertEqual(observation.known_facts, ())
        self.assertNotEqual(world.victims, particle.victims)

    def test_role_only_selector_uses_roles_and_local_messages_only(self) -> None:
        config = _small_study_config("E02")
        message = Message("victim", (1, 1), 0, "stabilized", 0)
        choice = select_role_only(
            [message], [1, 2], (Role.SCOUT, Role.CARRIER, Role.MEDIC),
            config["experiment"]["compatibility"],
        )
        self.assertEqual(choice, (0, 1))
        source = inspect.getsource(select_role_only)
        for forbidden in (".positions", ".victims", ".hazards", "SARGrid"):
            self.assertNotIn(forbidden, source)

    def test_diversity_ess_and_noise_rule_helpers(self) -> None:
        world = SARGrid(_small_study_config("E03")["environment"])
        world.reset(1)
        self.assertEqual(diversity_ess([world.clone(), world.clone()]), 1.0)
        self.assertEqual(lower_confidence_bound(1.0, 0.5, 2.0), 0.0)
        self.assertFalse(send_by_lcb(1.0, 0.5, 2.0))
        self.assertTrue(send_by_epsilon(0.51, 0.5))
        self.assertFalse(send_by_epsilon(0.5, 0.5))

    def test_crn_default_matches_existing_paired_estimator_exactly(self) -> None:
        config = _small_study_config("E04")
        config["experiment"].update({"particles": 2, "rollout_counts": [4], "reference_rollouts": 4})
        world = SARGrid(config["environment"])
        observations = world.reset(3)
        obs = observations[0]
        message = Message("intention", obs.position, obs.time_step, "scan", 0,
                          {"role": "scout", "agent_position": obs.position})
        context = {"seed": 3, "step": 0, "sender_id": 0, "receiver_id": 1,
                   "observation": obs, "roles": world.roles, "message": message,
                   "world": world.clone()}
        estimate, se, _ = _estimate(context, config, 4, True)
        baseline = PairedRolloutEstimator(
            ParticleReceiverEstimator(config["environment"], seed=3),
            __import__("voc_sar.policies.task", fromlist=["task_policies"]).task_policies(world.roles),
            particles=2, rollout_count=4, gamma=1.0, seed=3,
        ).estimate(0, 1, message, 3, obs)
        self.assertEqual(estimate, baseline.marginal_value)
        self.assertEqual(se, baseline.standard_error)

    def test_disabled_continuation_matches_existing_rollout(self) -> None:
        config = _small_study_config("E05")
        context = collect_contexts(config)[0]
        message, receiver = context["message"], context["receiver_id"]
        actual = _continued_return(config, context, message, receiver, "disabled", False, 31)
        world = context["world"].clone()
        world.reseed_dynamics(31)
        from voc_sar.policies.task import task_policies
        expected = _rollout_return(world, task_policies(world.roles), 3, 1.0)
        self.assertEqual(actual, expected)

    def test_cost_holm_and_spearman_helpers(self) -> None:
        self.assertEqual(receiver_cost_multiplier(5, 10), 1.5)
        self.assertEqual(receiver_cost_multiplier(5, 0), 1.0)
        self.assertEqual(holm_adjust({"b": .02, "a": .01}), {"a": .02, "b": .02})
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)

    def test_every_study_analyzer_smoke_and_reports_are_from_stored_data(self) -> None:
        modules = {
            "E01": "e01_message_conditioned", "E02": "e02_role_only",
            "E03": "e03_particle_diagnostics", "E04": "e04_noise_crn",
            "E05": "e05_continuation_bias", "E06": "e06_robust_null",
            "E07": "e07_cost_pareto", "E08": "e08_hypothesis_robustness",
            "E09": "e09_horizon_sweep", "E10": "e10_staleness",
            "E11": "e11_robustness", "E12": "e12_failures_runtime",
        }
        import importlib
        for experiment_id, module_name in modules.items():
            config = _small_study_config(experiment_id)
            experiment = config["experiment"]
            experiment["rollout_count"] = 1
            experiment["particles"] = 2
            if experiment_id == "E03": experiment["particle_counts"] = [2, 3]
            if experiment_id == "E04": experiment.update({"rollout_counts": [1, 2], "reference_rollouts": 2})
            if experiment_id == "E05": experiment.update({"contexts_per_seed": 1, "rollout_count": 1})
            if experiment_id == "E06": experiment.update({"null_replicates": 1, "null_rollouts": 1, "epsilon_grid": [0.0, .25]})
            if experiment_id == "E07": experiment["lambda_grid"] = [0.0, .5]
            if experiment_id == "E08": experiment.update({"theta1_grid": [.5], "delta1_grid": [.5]})
            if experiment_id == "E09": experiment["horizons"] = [1, 2]
            if experiment_id == "E11": experiment.update({"grid_sizes": [6], "team_variants": [["scout", "carrier"]]})
            contexts = collect_contexts(config)
            module = importlib.import_module(f"voc_sar.research_experiments.{module_name}")
            episodes, summary = module.analyze(config, contexts)
            self.assertIsInstance(episodes, list, experiment_id)
            self.assertIn("headline", summary, experiment_id)

        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory) / "run"
            report_dir = Path(directory) / "reports"
            result_dir.mkdir()
            (result_dir / "episodes.json").write_text(json.dumps([{"seed": 0, "score": 1.0}]))
            (result_dir / "summary.json").write_text(json.dumps({"headline": {"score": 1.0}}))
            write_reports(result_dir, report_dir)
            self.assertTrue((report_dir / "episodes.csv").is_file())
            self.assertTrue((report_dir / "summary.tex").is_file())
            self.assertTrue((report_dir / "overview.png").is_file())
            self.assertTrue((report_dir / "overview.pdf").is_file())

    def test_each_experiment_writes_its_key_visualizations(self) -> None:
        examples = {
            "E01": ([{"variant": "conditioned", "mae_voc": 1, "pair_regret": 2, "j0_return_from_context": 3}], {}),
            "E02": ([{"method": "aware", "return": 1, "regret": 2, "messages": 1}], {}),
            "E03": ([{"particles": 8, "ess_fraction": .5, "mae_voc": 1, "regret": 2}], {}),
            "E04": ([{"rollouts": 8, "crn": True, "standard_error": .1, "bias": .2, "runtime_seconds_per_decision": .3}], {}),
            "E05": ([{"continuation": "aware", "mae": 1, "top1_agreement": .5, "continuation_value": 2}], {}),
            "E06": ([{"rule": "strict", "positive_rate_lambda_0": .8, "positive_rate_lambda_0_5": .4}], {"null_false_positive_rates": {"strict": .1}}),
            "E07": ([{"lambda": .5, "method": "aware", "return": 2, "messages": 1}], {"pareto_points": [{"method": "aware", "mean_messages": 1, "mean_return": 2}]}),
            "E08": ([], {"h1_threshold_curve": {"theta=0.5,delta=0.5": .5}}),
            "E09": ([{"horizon": 2, "agreement": .5, "mae_voc": 1, "return": 3}], {}),
            "E10": ([{"survival_heuristic": .5, "observed_marginal_value": 1, "message_kind": "victim"}], {"by_information_age": {"1": {"mean_marginal_value": 1}}}),
            "E11": ([{"setting": "low", "aware_minus_agnostic_return": 1, "aware_regret": 2}], {}),
            "E12": ([{"information_age": 1, "regret": 2, "message_kind": "victim", "aware_minus_agnostic_return": 1}], {}),
        }
        with tempfile.TemporaryDirectory() as directory:
            for experiment_id, (rows, extra) in examples.items():
                output = Path(directory) / experiment_id
                write_experiment_visualizations(
                    rows,
                    {"experiment_id": experiment_id,
                     "headline": {"mean_particle_filter_seconds": .1,
                                  "mean_rollout_estimate_seconds": .2,
                                  "mean_oracle_seconds": .3,
                                  "h2_aware_minus_agnostic_return": .1,
                                  "h2_budget_matched_return": .2,
                                  "h3_delayed_context_gain": .3,
                                  "h4_positive_rate_drop_absolute": .4,
                                  "H1_holm_p": .05}, **extra},
                    output,
                )
                pngs = list(output.glob("*.png"))
                self.assertTrue(pngs, experiment_id)
                self.assertEqual(len(pngs), len(list(output.glob("*.pdf"))), experiment_id)


if __name__ == "__main__":
    unittest.main()
