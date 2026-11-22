"""E11: Environment and team robustness with paired aware/agnostic effects."""

from copy import deepcopy
from typing import Any

from voc_sar.research_experiments.common import policy_choice
from voc_sar.research_experiments.common import (
    bootstrap,
    collect_contexts,
    evaluate_pairs,
    make_estimator,
    mean,
    rollout_task_return,
    run_study,
)


def _one_setting(base_config, setting):
    config = deepcopy(base_config)
    env = config["environment"]
    env["hazard_move_probability"] = setting["p_move"]
    env["hazard_toggle_probability"] = setting["p_toggle"]
    env["grid_size"] = setting["grid_size"]
    env["roles"] = setting["roles"]
    env["victims"] = min(int(env["victims"]), max(0, env["grid_size"] ** 2 - len(env["roles"]) - int(env["hazards"])))
    config["experiment"]["contexts_per_seed"] = 1
    return config


def _settings(config):
    exp, env = config["experiment"], config["environment"]
    settings = []
    for label, move, toggle in zip(("low", "default", "high"), exp["move_levels"], exp["toggle_levels"]):
        settings.append({"setting": f"dynamics_{label}", "p_move": float(move),
                         "p_toggle": float(toggle), "grid_size": env["grid_size"],
                         "roles": list(env["roles"])})
    default_move, default_toggle = float(env["hazard_move_probability"]), float(env["hazard_toggle_probability"])
    for grid_size in exp["grid_sizes"]:
        settings.append({"setting": f"grid_{grid_size}", "p_move": default_move,
                         "p_toggle": default_toggle, "grid_size": int(grid_size),
                         "roles": list(env["roles"])})
    for roles in exp["team_variants"]:
        settings.append({"setting": "team_" + "_".join(roles), "p_move": default_move,
                         "p_toggle": default_toggle, "grid_size": env["grid_size"],
                         "roles": list(roles)})
    return settings


def analyze(config: dict[str, Any], contexts: list[dict[str, Any]]):
    rows = []
    by_setting = {}
    for setting in _settings(config):
        local = _one_setting(config, setting)
        local_contexts = collect_contexts(local)
        paired_by_seed = {}
        for context in local_contexts:
            messages = list(context["messages"][:3])
            truth, _ = evaluate_pairs(
                local, context, messages=messages,
                horizon=int(local["communication"]["oracle_horizon"]),
                estimator_horizon=int(local["communication"]["nonmyopic_horizon"]),
                rollout_count=int(local["experiment"]["rollout_count"]),
            )
            aware = policy_choice(local, context, messages, aware=True)
            agnostic = policy_choice(local, context, messages, aware=False)
            choices = {"aware": aware, "agnostic": agnostic}
            returns = {}
            for method, choice in choices.items():
                returns[method] = rollout_task_return(
                    context, (messages[choice[0]], choice[1]) if choice else None, local
                )
            difference = returns["aware"] - returns["agnostic"]
            paired_by_seed.setdefault(int(context["seed"]), []).append(difference)
            rows.append({"setting": setting["setting"], "seed": context["seed"],
                         "p_move": setting["p_move"], "p_toggle": setting["p_toggle"],
                         "grid_size": setting["grid_size"], "roles": setting["roles"],
                         "aware_return": returns["aware"], "agnostic_return": returns["agnostic"],
                         "aware_minus_agnostic_return": difference,
                         "messages_aware": int(aware is not None),
                         "messages_agnostic": int(agnostic is not None),
                         "aware_regret": max([0.0, *truth.values()]) - (truth.get(aware, 0.0) if aware else 0.0)})
        paired = [mean(values) for values in paired_by_seed.values()]
        by_setting[setting["setting"]] = {
            "aware_minus_agnostic_return": mean(paired),
            "paired_seed_ci": bootstrap(paired, local, 11_211),
            "paired_seed_count": len(paired),
            "mean_messages_aware": mean([r["messages_aware"] for r in rows if r["setting"] == setting["setting"]]),
            "mean_messages_agnostic": mean([r["messages_agnostic"] for r in rows if r["setting"] == setting["setting"]]),
        }
    headline = {f"{name}_return_difference": values["aware_minus_agnostic_return"]
                for name, values in by_setting.items()}
    return rows, {"headline": headline, "by_setting": by_setting}


def run(config: dict[str, Any]):
    return run_study("E11", config, analyze)
