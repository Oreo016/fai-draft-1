"""Recompute paper results from archived traces and render tables and figures."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
PRIMARY = ROOT / "results/run_20261122_144033_944398"
FRESH_STUDIES = {
    "E01": "140035_246491", "E02": "140019_731832", "E03": "140141_343656",
    "E04": "140321_777612", "E05": "141611_297838", "E06": "140427_962155",
    "E07": "140247_607739", "E08": "140141_142268", "E09": "140459_256588",
    "E10": "140251_059893", "E11": "140425_455156", "E12": "140441_666903",
}
SWEEP = ROOT / "results/sweep_gamma_lambda/run_20261122_140807_648516"
LABEL = {
    "never": "Never", "random": "Random", "full": "Full",
    "fixed_frequency": "Fixed-frequency", "aware_myopic": "Aware ($H=1$)",
    "aware_nonmyopic": "Aware ($H=5$)", "agnostic_nonmyopic": "Agnostic ($H=5$)",
}
NBOOT = 10_000


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"Trace mismatch: expected {expected}, observed {actual}")


def bootstrap(units, statistic=mean):
    """Percentile interval, resampling independent seed units with replacement."""
    rng = random.Random(215)
    n = len(units)
    if n < 2:
        return None
    samples = sorted(
        value for _ in range(NBOOT)
        if (value := statistic(rng.choices(units, k=n))) is not None
    )
    if not samples:
        return None
    return [samples[int(tail * (len(samples) - 1))] for tail in (0.025, 0.975)]


def flatten(groups):
    return [x for group in groups for x in group]


def rate_reduction(pairs):
    denominator = sum(x[0] for x in pairs)
    return 1 - sum(x[1] for x in pairs) / denominator if denominator else None


def condition_summary(groups):
    output = []
    for (method, cost, budget), rows in groups.items():
        outcome = [row["discounted_task_return"] for row in rows]
        output.append(dict(
            method=method, cost=cost, budget=budget, seeds=len(rows),
            return_mean=mean(outcome), return_sd=stdev(outcome),
            return_se=stdev(outcome) / math.sqrt(len(rows)),
            return_ci=bootstrap(outcome),
            messages=sum(row["messages"] for row in rows),
            mean_messages=mean(row["messages"] for row in rows),
            rescues=sum(row["rescued_victims"] for row in rows),
            mean_rescues=mean(row["rescued_victims"] for row in rows),
            runtime_episode_mean_s=mean(row["runtime_seconds"] for row in rows),
            runtime_step_mean_s=mean(
                row["mean_communication_decision_seconds"] for row in rows
            ),
            decision_time_total_s=sum(
                step["communication_decision_seconds"]
                for row in rows for step in row["step_trace"]
            ),
            steps=sum(row["steps"] for row in rows),
        ))
    return output


def diagnostic_summary(contexts, communication, evaluation):
    receiver_flags, delayed_gains = defaultdict(list), defaultdict(list)
    receiver_regret, pair_regret, errors, recalls, usefulness = [], [], [], [], []
    strata = {"delayed": [0, 0, 0], "other": [0, 0, 0]}
    candidate_wise_eligible, candidate_wise_hits = [], []
    h5 = str(communication["nonmyopic_horizon"])
    h1 = str(communication["myopic_horizon"])
    h1_delta = evaluation["thresholds"]["receiver_value_difference"]
    for context in contexts:
        values, marginal_short, prediction = {}, {}, {h1: {}, h5: {}}
        for pair in context["pairs"]:
            key = (pair["message_index"], pair["receiver_id"])
            oracle, myopic = pair["oracle"], pair["oracle_myopic"]
            for saved in (oracle, myopic, *pair["predictions"].values()):
                differences = [a - b for a, b in saved["paired_returns"]]
                close(saved["marginal_value"], mean(differences))
                close(saved["value_of_communication"],
                      saved["marginal_value"] - saved["communication_cost"])
                close(saved["standard_error"], stdev(differences) / math.sqrt(len(differences)))
            values[key] = oracle["marginal_value"]  # Diagnostic cost is zero.
            marginal_short[key] = myopic["marginal_value"]
            for horizon, saved in pair["predictions"].items():
                prediction[horizon][key] = saved["value_of_communication"]
        m, a, oracle = [
            tuple(context[field]) if context[field] is not None else None
            for field in ("myopic_action", "nonmyopic_action", "oracle_action")
        ]
        for choice, admissible in ((m, prediction[h1]), (a, prediction[h5]), (oracle, values)):
            best = max([0.0, *admissible.values()])
            if choice is None:
                close(best, 0.0)
            else:
                if best <= 0:
                    raise ValueError("Found non-positive selected value in held-out diagnostics")
                close(admissible[choice], best)
        delayed = any(
            abs(values[key] - marginal_short[key]) > evaluation["delayed_value_margin"]
            for key in values
        )
        if delayed != context["delayed"]:
            raise ValueError("Inconsistent delayed context flag")
        bucket = strata["delayed" if delayed else "other"]
        bucket[0] += 1
        bucket[1] += int(m == oracle)
        bucket[2] += int(a == oracle)
        if delayed:
            delayed_gains[context["seed"]].append(int(a == oracle) - int(m == oracle))
        eligible_pairs = {p for p, value in values.items() if value > communication["oracle_value_threshold"]}
        if eligible_pairs:
            recalls.append(len(eligible_pairs & prediction[h5].keys()) / len(eligible_pairs))
        errors.append(mean(abs(prediction[h5][p] - values[p]) for p in prediction[h5]))
        pair_regret.append(max([0.0, *values.values()]) - (values[a] if a else 0.0))
        receivers = context["receiver_ids"]
        if a:
            receiver_regret.append(max(values[(a[0], j)] for j in receivers) - values[a])
            usefulness.append(values[a] > communication["useful_message_threshold"])
        index = next((i for i, msg in enumerate(context["messages"]) if msg["kind"] != "intention"), None)
        if index != context["primary_message_index"]:
            raise ValueError("Primary H1 message is inconsistent")
        if index is not None:
            flagged = abs(values[(index, receivers[0])] - values[(index, receivers[1])]) > h1_delta
            receiver_flags[context["seed"]].append(flagged)
            candidate_wise_eligible.append(1)
            candidate_wise_hits.append(any(
                abs(values[(i, receivers[0])] - values[(i, receivers[1])]) > h1_delta
                for i, msg in enumerate(context["messages"]) if msg["kind"] != "intention"
            ))
    h1_out = flatten(receiver_flags.values())
    gain = flatten(delayed_gains.values())
    return dict(
        contexts=len(contexts), h1_hits=sum(h1_out), h1_contexts=len(h1_out),
        h1_rate=mean(h1_out), h1_ci=bootstrap(list(receiver_flags.values()), lambda xs: mean(flatten(xs))),
        h1_per_seed={seed: [sum(flags), len(flags)] for seed, flags in receiver_flags.items()},
        any_candidate_h1_hits=sum(candidate_wise_hits), any_candidate_h1_contexts=sum(candidate_wise_eligible),
        h3_gain=mean(gain), h3_ci=bootstrap(list(delayed_gains.values()), lambda xs: mean(flatten(xs))),
        agreement_strata=strata, h3_delayed_seed_count=len(delayed_gains),
        receiver_regret=mean(receiver_regret), receiver_regret_n=len(receiver_regret),
        pair_regret=mean(pair_regret), shortlist_voic_mae=mean(errors),
        filter_recall=mean(recalls), filter_recall_contexts=len(recalls),
        useful_deliveries=sum(usefulness), evaluated_deliveries=len(usefulness),
    )


def primary_summary(primary):
    episodes, contexts = (load(primary / f"{name}.json") for name in ("episodes", "contexts"))
    config = load(primary / "config_used.json")
    env, comm, evaluation = (config[name] for name in ("environment", "communication", "evaluation"))
    seeds = evaluation["seeds"]
    groups = defaultdict(list)
    entries = {}
    for row in episodes:
        key = (row["method"], row["cost_weight"], row["assigned_message_budget"])
        if (*key, row["seed"]) in entries:
            raise ValueError(f"Duplicate episode: {key}, {row['seed']}")
        entries[(*key, row["seed"])] = row
        groups[key].append(row)
        trace = row["step_trace"]
        if len(trace) != row["steps"] or row["messages"] != sum(x["send"] is not None for x in trace):
            raise ValueError(f"Trace length/count mismatch: {key}, {row['seed']}")
        rewards = []
        sent_steps = []
        fact_keys = set()
        for step, record in enumerate(trace):
            assert step == record["step"]
            ev, r = record["events"], env["reward"]
            reward = (r["rescue"] * ev["rescued"] + r["locate"] * ev["located"]
                      + r["stabilize"] * ev["stabilized"] - r["step_penalty"] * len(record["task_actions"]))
            close(reward, record["reward"])
            rewards.append(reward * env["gamma"] ** step)
            if key[2] is None:
                continue
            if record["eligible_budget_facts"] and len(fact_keys) < key[2]:
                assert record["send"] is not None
            if record["send"] is not None:
                m = record["send"]["message"]
                fact = (m["kind"], tuple(m["location"]), m["status"])
                assert fact not in fact_keys
                assert (m["kind"] == "victim" and m["status"] in {"located", "stabilized"}) or (
                    m["kind"] == "hazard" and m["status"] == "active" and m["observed_at"] == step)
                fact_keys.add(fact)
                sent_steps.append(step)
        close(sum(rewards), row["discounted_task_return"])
        assert row["rescued_victims"] == sum(step["events"]["rescued"] for step in trace)
        if key[2] is not None:
            assert len(sent_steps) == key[2] and sent_steps == row["budget_send_steps"]
    assert all(sorted(r["seed"] for r in rows) == sorted(seeds) for rows in groups.values())
    assert len(episodes) == 120 and len(contexts) == evaluation["held_out_contexts"]
    assert len({(c["seed"], c["step"]) for c in contexts}) == len(contexts)

    returns, rescues, actual_budget_returns, actual_budget_rescues = [], [], [], []
    rates = []
    message_totals = [0, 0, 0, 0]
    budget = evaluation["primary_message_budget"]
    aware, full, agnostic = "aware_nonmyopic", "full", "agnostic_nonmyopic"
    for seed in seeds:
        a, f, high = (entries[(m, cost, None, seed)] for m, cost in ((aware, 0), (full, 0), (aware, 0.5)))
        ta, tg = (entries[(m, 0, budget, seed)] for m in (aware, agnostic))
        returns.append(a["discounted_task_return"] - f["discounted_task_return"])
        rescues.append(a["rescued_victims"] - f["rescued_victims"])
        actual_budget_returns.append(ta["discounted_task_return"] - tg["discounted_task_return"])
        actual_budget_rescues.append(ta["rescued_victims"] - tg["rescued_victims"])
        rates.append((a["messages"] / a["decisions"], high["messages"] / high["decisions"]))
        for row, rate in zip((a, high), rates[-1]):
            close(rate, row["positive_voc_rate"])
        counts = (a["messages"], a["decisions"], high["messages"], high["decisions"])
        message_totals = [x + y for x, y in zip(message_totals, counts)]
    contrast = dict(
        h2_crb=1-sum(r["messages"] for r in groups[aware, 0, None]) / sum(r["messages"] for r in groups[full, 0, None]),
        h2_return_delta=mean(returns), h2_return_ci=bootstrap(returns),
        h2_full_rescue_delta=mean(rescues), h2_full_rescue_ci=bootstrap(rescues),
        h2_budget_return_delta=mean(actual_budget_returns), h2_budget_return_ci=bootstrap(actual_budget_returns),
        h2_budget_rescue_delta=mean(actual_budget_rescues), h2_budget_rescue_ci=bootstrap(actual_budget_rescues),
        h2_budget_returns_by_seed=dict(zip(seeds, actual_budget_returns)),
        h2_budget_rescues_by_seed=dict(zip(seeds, actual_budget_rescues)),
        h4_rates=list(map(mean, zip(*rates))), h4_reduction=rate_reduction(rates),
        h4_ci=bootstrap(rates, rate_reduction), h4_message_and_opportunity_totals=message_totals,
    )
    conditions = condition_summary(groups)
    regular = [row for row in conditions if row["budget"] is None]
    frontier = [
        row for row in regular
        if not any(
            other["mean_messages"] <= row["mean_messages"]
            and other["return_mean"] >= row["return_mean"]
            and (other["mean_messages"] < row["mean_messages"] or other["return_mean"] > row["return_mean"])
            for other in regular
        )
    ]
    return dict(
        source=primary.relative_to(ROOT).as_posix(), seed_ids=seeds, episode_count=len(episodes), config=config,
        conditions=conditions, frontier=frontier,
        diagnostics=diagnostic_summary(contexts, comm, evaluation), contrasts=contrast,
        runtime_seconds=load(primary / "summary.json")["runtime_seconds"],
    )


def grouped_means(rows, keys, measures):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, group in groups.items():
        averages = {}
        for measure in measures:
            values = [item[measure] for item in group if item[measure] is not None]
            averages[measure] = mean(values) if values else None
        output.append({**dict(zip(keys, key)), "rows": len(group),
                       "seeds": len({r["seed"] for r in group}), **averages})
    return output


def secondary_summary():
    descriptive = {
        "E01": (("variant",), ("j0_return_from_context", "mae_voc", "pair_regret", "receiver_regret")),
        "E02": (("method",), ("messages", "regret", "return")),
        "E03": (("particles",), ("ess_fraction", "proposal_weight_ess", "receiver_position_brier", "mae_voc", "regret")),
        "E04": (("rollouts", "crn"), ("bias", "standard_error", "runtime_seconds_per_decision")),
        "E05": (("continuation",), ("top1_agreement", "mae")),
        "E06": (("rule",), ("positive_rate_lambda_0", "positive_rate_lambda_0_5")),
        "E07": (("lambda", "method"), ("messages", "regret", "return")),
        "E08": (("delayed",), ("agreement_gain", "myopic_agreement", "nonmyopic_agreement")),
        "E09": (("horizon",), ("agreement", "mae_voc", "messages", "return")),
        "E10": ((), ("information_age", "freshness_return_difference", "observed_marginal_value")),
        "E11": (("setting",), ("aware_minus_agnostic_return", "aware_regret", "messages_aware", "messages_agnostic")),
        "E12": ((), ("aware_decision_seconds", "oracle_seconds", "particle_filter_seconds", "rollout_estimate_seconds", "regret")),
    }
    output = {}
    for experiment, stamp in FRESH_STUDIES.items():
        path = ROOT / "results/experiments" / experiment / f"run_20261122_{stamp}"
        cfg = load(path / "config_used.json")
        raw = load(path / "episodes.json")
        keys, measures = descriptive[experiment]
        stats = dict(source=path.relative_to(ROOT).as_posix(), config=cfg, groups=grouped_means(raw, keys, measures))
        if experiment == "E04":
            for row in raw:
                differences = [a - b for a, b in row["paired_returns_send_no_send"]]
                close(row["estimate"], mean(differences))
                close(row["standard_error"], stdev(differences) / math.sqrt(len(differences)))
        elif experiment == "E10":
            stats["minimum_age"], stats["maximum_age"] = (f(x["information_age"] for x in raw) for f in (min, max))
            stats["changed_returns"] = sum(row["freshness_return_difference"] != 0 for row in raw)
        elif experiment == "E12":
            stats["failure_contexts"] = sum(
                row["aware_minus_agnostic_return"] < 0 or row["aware_minus_role_only_return"] < 0
                for row in raw
            )
        output[experiment] = stats
    pairs = defaultdict(lambda: defaultdict(dict))
    sweep_raw = load(SWEEP / "contexts.json")
    for row in sweep_raw:
        key = row["gamma"], row["lambda"]
        pairs[key][(row["seed"], row["step"])][row["method"]] = row["return"]
    intervals = []
    for (gamma, cost), observations in pairs.items():
        by_seed = defaultdict(list)
        for (seed, _), methods in observations.items():
            by_seed[seed].append(methods["aware"] - methods["agnostic"])
        differences = [mean(rows) for rows in by_seed.values()]
        intervals.append(dict(gamma=gamma, cost=cost, seed_count=len(by_seed),
                              difference=mean(differences), ci=bootstrap(differences)))
    output["gamma_lambda"] = dict(source=SWEEP.relative_to(ROOT).as_posix(), raw_rows=len(sweep_raw), intervals=intervals,
                                  excludes_zero=sum(lo > 0 or hi < 0 for item in intervals for lo, hi in [item["ci"]]))
    return output


def interval_tex(interval, digits=3, scale=1):
    return "[" + ", ".join(f"{x * scale:.{digits}f}" for x in interval) + "]"


def write_tables(data, tables):
    rows = [
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"Policy & $\lambda$ & Mean $J$ [95\% CI] & Messages & Rescues \\", r"\midrule",
    ]
    for row in data["conditions"]:
        if row["budget"] is not None:
            continue
        value = f"{row['return_mean']:.3f}" + r"\; " + interval_tex(row["return_ci"])
        rows.append(
            f"{LABEL[row['method']]} & {row['cost']:g} & " + "$" + value + "$"
            + f" & {row['messages']} & {row['rescues']} " + r"\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}", ""])
    (tables / "conditions.tex").write_text("\n".join(rows), encoding="utf-8")

    d, c = data["diagnostics"], data["contrasts"]
    rows = [
        r"\begin{tabular}{@{}lllr@{}}", r"\toprule",
        r"Test & Required & Observed [95\% CI where applicable] & Verdict \\", r"\midrule",
        f"H1 & " + r"$\geq 30\%$" + f" & {d['h1_hits']}/{d['h1_contexts']} = {d['h1_rate']*100:.2f}"
        + r"\%, " + interval_tex(d["h1_ci"], 2, 100) + r"\% & Fail \\",
        r"H2a & $\mathrm{CRB}\geq 20\%$ at matched $J$ & " + f"{c['h2_crb']*100:.2f}"
        + r"\% fewer messages & Met \\",
        r"H2b & CI excludes zero at matched budget & "
        + f"{c['h2_budget_return_delta']:.4f}, {interval_tex(c['h2_budget_return_ci'], 4)}" + r" & Fail \\",
        r"H3 & $\geq 10$ percentage points & " + f"+{d['h3_gain']*100:.2f} pp, "
        + interval_tex(d["h3_ci"], 2, 100) + r" pp & Pass \\",
        r"H4 & $\geq 15\%$ relative reduction & " + f"{c['h4_reduction']*100:.2f}" + r"\%, "
        + interval_tex(c["h4_ci"], 2, 100) + r"\% & Pass \\",
        r"\bottomrule", r"\end{tabular}", "",
    ]
    (tables / "hypotheses.tex").write_text("\n".join(rows), encoding="utf-8")


def render_findings(data, figures, screenshot_dir):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.15, 2.55), layout="constrained")
    frontier = sorted({(x["mean_messages"], x["return_mean"]) for x in data["frontier"]})
    ax.plot(*zip(*frontier), color="#546478", zorder=1, linewidth=1.3)
    selected = [r for r in data["conditions"] if r["budget"] is None and (
        r["cost"] == 0 or r["method"] == "aware_nonmyopic")]
    point_design = {
        "never": ("o", "#596475", "No-send ties", (3, -8)),
        "random": ("o", "#596475", "Random", (-26, 9)),
        "full": ("o", "#596475", "Full", (-10, -14)),
        "fixed_frequency": ("o", "#596475", "Fixed-freq.", (-19, -15)),
        "aware_nonmyopic": ("D", "#176A8C", "Aware", (4, 7)),
        "agnostic_nonmyopic": ("s", "#8D4B1D", "Agnostic", (-18, -13)),
    }
    for row in selected:
        method = row["method"]
        if method == "aware_myopic":
            continue
        marker, color, label, offset = point_design[method]
        if method == "aware_nonmyopic":
            label += ", " + r"$\lambda=" + f"{row['cost']:g}$"
            if row["cost"] > 0:
                offset = (18, 6)
        xy = row["mean_messages"], row["return_mean"]
        ax.scatter(*xy, marker=marker, color=color, s=25, zorder=2)
        ax.annotate(label, xy, xytext=offset, textcoords="offset points", fontsize=7, color=color)
    ax.set(xlabel="Mean messages per episode", ylabel="Mean discounted task return",
           title="(a) Return–communication trade-off", xlim=(-5, 103), ylim=(-0.484, -0.355))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#d8dde3", linewidth=.7)

    n1, a1, a5 = data["diagnostics"]["agreement_strata"]["delayed"]
    n2, b1, b5 = data["diagnostics"]["agreement_strata"]["other"]
    group = (0, 1, 2)
    group_sizes = (n1, n2, n1 + n2)
    for shift, totals, horizon, color in ((-.18, (a1, b1, a1+b1), 1, "#9BA8B7"),
                                          (.18, (a5, b5, a5+b5), 5, "#176A8C")):
        bars = bx.bar([j + shift for j in group], [100*y/n for y, n in zip(totals, group_sizes)],
                      width=.36, color=color, label=f"$H={horizon}$")
        bx.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    bx.set_xticks(group, [f"Delayed\n($n={n1}$)", f"Other\n($n={n2}$)", f"All\n($n={n1+n2}$)"])
    bx.set(ylabel="Agreement with $H=15$ (%)", title="(b) Held-out action agreement", ylim=(0, 83))
    bx.set_axisbelow(True)
    bx.grid(axis="y", color="#d8dde3", linewidth=.7)
    bx.legend(loc="upper right", frameon=False, fontsize=8, ncols=2)
    fig.savefig(figures / "findings.pdf")
    if screenshot_dir:
        fig.savefig(screenshot_dir / "paper_primary_findings.png", dpi=240)
    plt.close(fig)


def render_estimator(secondary, figures, screenshot_dir):
    data = secondary["E04"]["groups"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.2), layout="constrained")
    for crn, label, color, ls in ((True, "Shared draws", "#176A8C", "-"),
                                  (False, "Independent draws", "#8D4B1D", "--")):
        rows = sorted((row for row in data if row["crn"] == crn), key=lambda x: x["rollouts"])
        for plot, field in ((ax, "standard_error"), (bx, "runtime_seconds_per_decision")):
            plot.plot([r["rollouts"] for r in rows], [r[field] for r in rows],
                      marker="o", ms=3, linestyle=ls, color=color, label=label)
            plot.set_xscale("log", base=2)
            plot.set_xticks([r["rollouts"] for r in rows], [f"{r['rollouts']}" for r in rows])
            plot.set_xlabel("Rollout pairs")
            plot.grid(axis="y", color="#d8dde3", linewidth=.7)
    ax.set_ylabel("Mean within-estimator SE")
    ax.set_title("(a) Estimator precision")
    bx.set_ylabel("Mean runtime (s)")
    bx.set_title("(b) Decision runtime")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(figures / "estimator.pdf")
    if screenshot_dir:
        fig.savefig(screenshot_dir / "paper_estimator_ablation.png", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", action="store_true", help="Also save PNG result panels in results/screenshots.")
    args = parser.parse_args()
    dirs = {part: PAPER / part for part in ("figures", "tables", "data_summary")}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    screenshots = ROOT / "results/screenshots" if args.screenshots else None
    if screenshots:
        screenshots.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    primary = primary_summary(PRIMARY)
    secondary = secondary_summary()
    for name, payload in (("primary_metrics", primary), ("secondary_metrics", secondary)):
        (dirs["data_summary"] / f"{name}.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    cols = ["method", "cost", "budget", "seeds", "return_mean", "return_sd", "return_se", "return_ci",
            "messages", "mean_messages", "rescues", "mean_rescues", "runtime_episode_mean_s",
            "runtime_step_mean_s", "decision_time_total_s", "steps"]
    with (dirs["data_summary"] / "condition_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(primary["conditions"])
    write_tables(primary, dirs["tables"])
    render_findings(primary, dirs["figures"], screenshots)
    render_estimator(secondary, dirs["figures"], screenshots)
    print(f"Validated {primary['episode_count']} episodes and {primary['diagnostics']['contexts']} held-out contexts")


if __name__ == "__main__":
    main()
