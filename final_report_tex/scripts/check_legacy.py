"""Recompute secondary results from the archived project/faiproj workspace."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from random import Random
from statistics import mean


REPORT = Path(__file__).resolve().parents[1]
ARCHIVE = REPORT.parent / "project" / "faiproj" / "results"
FIXED_ALL = "h2_fixed_budget_20261121_214955_584240"
WHOLE_HORIZON = (
    "h2_fixed_budget_20261121_223852_784456",
    "h2_fixed_budget_20261121_231120_101460",
)
H1_RUNS = (
    "h1_evaluation_20261121_225928_494352",
    "h1_evaluation_20261121_230305_562535",
)
H4_RUN = "h4_cost_sensitivity_20261122_005543_147362"
REPLICATES = 10_000
BOOTSTRAP_SEED = 2026


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def bootstrap(rows, statistic):
    rng = Random(BOOTSTRAP_SEED)
    values = sorted(
        result for _ in range(REPLICATES)
        if (result := statistic(rng.choices(rows, k=len(rows)))) is not None
    )
    return [values[int(q * (len(values) - 1))] for q in (0.025, 0.975)]


def proportion(pairs):
    successes, denominator = map(sum, zip(*pairs))
    return successes / denominator if denominator else None


def relative_drop(pairs):
    low, high = map(sum, zip(*pairs))
    return 1 - high / low if low else None


def h1_run(name):
    summary = read(ARCHIVE / name / "summary.json")
    rows = read(ARCHIVE / name / "per_seed_counts.json")
    pairs = [
        (row["receiver_differentiated_contexts"], row["h1_eligible_contexts"])
        for row in rows
    ]
    total, eligible = map(sum, zip(*pairs))
    assert total == summary["receiver_differentiated_contexts"]
    assert eligible == summary["h1_eligible_contexts"]
    return {
        "run": name, "seeds": [row["seed"] for row in rows],
        "observed_contexts": sum(row["observed_diagnostic_contexts"] for row in rows),
        "eligible_contexts": eligible, "hits": total,
        "fraction": proportion(pairs), "interval": bootstrap(pairs, proportion),
    }


def h2_run(name):
    summary = read(ARCHIVE / name / "summary.json")
    rows = read(ARCHIVE / name / "episodes.json")
    quota = summary["budget_messages_per_episode"]
    by_seed = defaultdict(dict)
    for row in rows:
        if row["method"] in ("aware_nonmyopic_fixed_budget", "agnostic_nonmyopic_fixed_budget"):
            role = row["method"].split("_", 1)[0]
            assert role not in by_seed[row["seed"]]
            by_seed[row["seed"]][role] = row
    paired = {}
    for seed, pair in sorted(by_seed.items()):
        aware, agnostic = pair["aware"], pair["agnostic"]
        if aware["messages"] == agnostic["messages"] == quota:
            paired[seed] = aware["discounted_task_return"] - agnostic["discounted_task_return"]
    assert set(by_seed) == set(summary["seeds"])
    assert len(paired) == summary["paired_budget_comparison"]["exact_message_match_count"]
    return {
        "run": name, "quota": quota, "attempted_seeds": len(by_seed),
        "exact_quota_seeds": len(paired), "paired_differences": paired,
        "mean_return_difference": mean(paired.values()),
        "interval": bootstrap(list(paired.values()), mean),
        "schedule": summary.get("message_schedule", "archived full-horizon fixed schedule"),
    }


def h4_run():
    rows = read(ARCHIVE / H4_RUN / "episodes.json")
    indexed = {(row["seed"], row["cost_weight"]): row for row in rows}
    seeds = sorted({row["seed"] for row in rows})
    costs = sorted({row["cost_weight"] for row in rows})
    assert len(indexed) == len(rows) == len(seeds) * len(costs)
    conditions = {}
    rates = {}
    for cost in costs:
        arm = [indexed[seed, cost] for seed in seeds]
        rates[cost] = [row["messages"] / row["decisions"] if row["decisions"] else 0 for row in arm]
        conditions[cost] = {
            "mean_positive_rate": mean(rates[cost]),
            "mean_return": mean(row["discounted_task_return"] for row in arm),
            "total_messages": sum(row["messages"] for row in arm),
        }
    pairs = list(zip(rates[0], rates[0.5]))
    return {
        "run": H4_RUN, "seeds": seeds, "episodes": len(rows), "conditions": conditions,
        "relative_reduction": relative_drop(pairs),
        "interval": bootstrap(pairs, relative_drop),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT / "data_summary" / "legacy_checks.json")
    args = parser.parse_args()
    all_h2 = [h2_run(path.name) for path in sorted(ARCHIVE.glob("h2_fixed_budget_*")) if path.is_dir()]
    h2_by_name = {run["run"]: run for run in all_h2}
    first, second = [h2_by_name[name] for name in WHOLE_HORIZON]
    configs = [read(ARCHIVE / name / "config_used.json") for name in WHOLE_HORIZON]
    for section in ("environment", "communication"):
        assert configs[0][section] == configs[1][section]
    assert first["schedule"] == second["schedule"] and first["quota"] == second["quota"] == 5
    assert not first["paired_differences"].keys() & second["paired_differences"].keys()
    cohort_differences = list(first["paired_differences"].values()) + list(second["paired_differences"].values())
    result = {
        "archive": ARCHIVE.relative_to(REPORT.parent).as_posix(),
        "environment": configs[0]["environment"],
        "bootstrap_replicates": REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
        "h1": [h1_run(name) for name in H1_RUNS],
        "h2": {
            "runs": all_h2, "fixed_schedule_reference": FIXED_ALL,
            "pooled_whole_horizon_quota_5": {
                "runs": WHOLE_HORIZON, "exact_quota_seeds": len(cohort_differences),
                "mean_return_difference": mean(cohort_differences),
                "interval": bootstrap(cohort_differences, mean),
            },
        },
        "h4": h4_run(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
