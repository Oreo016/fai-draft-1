"""E03: Particle diversity/ESS, value-estimation error, and regret by Np."""

from collections import Counter
from typing import Any

from voc_sar.communication.estimator import ParticleReceiverEstimator
from voc_sar.evaluation.metrics import pair_regret
from voc_sar.research_experiments.common import make_estimator, mean, oracle_estimate, run_study


def _fingerprint(world: Any) -> tuple[Any, ...]:
    return (
        world.positions,
        tuple(sorted(world.victims)),
        tuple(sorted(world.hazards)),
    )


def diversity_ess(particles: list[Any]) -> float:
    if not particles:
        return 0.0
    frequencies = Counter(_fingerprint(world) for world in particles)
    total = len(particles)
    return 1.0 / sum((count / total) ** 2 for count in frequencies.values())


def analyze(config: dict[str, Any], contexts: list[dict[str, Any]]):
    rows = []
    filters = {}
    counts = [int(value) for value in config["experiment"]["particle_counts"]]
    for context in contexts:
        message = context["message"]
        receiver = int(context["receiver_id"])
        horizon = int(config["communication"]["oracle_horizon"])
        truth = oracle_estimate(config, context, horizon=horizon).marginal_value
        for count in counts:
            key = int(context["seed"]), count
            if key not in filters:
                filters[key] = ParticleReceiverEstimator(config["environment"], seed=key[0])
            state_filter = filters[key]
            particles = state_filter.sample_states(
                int(context["sender_id"]), receiver, count, context["observation"]
            )
            position_counts = Counter(world.positions[receiver] for world in particles)
            true_position = context["world"].positions[receiver]
            true_position_mass = position_counts.get(true_position, 0) / count
            position_brier = 1 - 2 * true_position_mass + sum(
                (frequency / count) ** 2 for frequency in position_counts.values()
            )
            ess = diversity_ess(list(particles))
            estimate = make_estimator(
                config, seed=int(context["seed"]), particles=count, state_estimator=state_filter,
                rollout_count=int(config["experiment"]["rollout_count"]),
            ).estimate(
                int(context["sender_id"]), receiver, message, horizon, context["observation"]
            )
            regret = pair_regret(
                {(0, receiver): truth},
                (0, receiver) if estimate.value_of_communication > 0 else None,
            )
            rows.append({
                "seed": context["seed"], "step": context["step"],
                "particles": count, "ess": ess, "ess_fraction": ess / count,
                "proposal_weight_ess": state_filter.last_proposal_ess,
                "receiver_position_mass_at_truth": true_position_mass,
                "receiver_position_brier": position_brier,
                "degenerate": int(ess < count / 4),
                "mae_voc": abs(estimate.marginal_value - truth),
                "regret": regret,
            })
    by_count = {}
    headline = {}
    for count in counts:
        group = [row for row in rows if row["particles"] == count]
        by_count[str(count)] = {
            "mean_ess": mean([row["ess"] for row in group]),
            "mean_ess_fraction": mean([row["ess_fraction"] for row in group]),
            "degeneracy_rate": mean([row["degenerate"] for row in group]),
            "mae_voc": mean([row["mae_voc"] for row in group]),
            "regret": mean([row["regret"] for row in group]),
            "receiver_position_brier": mean([row["receiver_position_brier"] for row in group]),
        }
        headline[f"np_{count}_ess_fraction"] = by_count[str(count)]["mean_ess_fraction"]
        headline[f"np_{count}_degeneracy_rate"] = by_count[str(count)]["degeneracy_rate"]
        headline[f"np_{count}_mae_voc"] = by_count[str(count)]["mae_voc"]
        headline[f"np_{count}_regret"] = by_count[str(count)]["regret"]
    return rows, {"headline": headline, "by_particle_count": by_count,
                  "ess_definition": "inverse Simpson diversity ESS from duplicate sampled-state frequencies; not a weighted-posterior ESS"}


def run(config: dict[str, Any]):
    return run_study("E03", config, analyze)
