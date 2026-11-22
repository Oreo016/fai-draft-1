"""Shared I/O and offline context helpers for the additive study runners."""

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
import csv
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import yaml

from voc_sar.communication.estimator import PairedRolloutEstimator, ParticleReceiverEstimator, _rollout_return
from voc_sar.communication.selector import select_communication
from voc_sar.environment.sar_grid import SARGrid
from voc_sar.evaluation.metrics import bootstrap_cluster_mean_interval, bootstrap_mean_interval
from voc_sar.evaluation.oracle import CounterfactualOracle
from voc_sar.policies.task import candidate_messages, task_policies
from voc_sar.types import CommunicationChoice, Message, Role


StudyFunction = Callable[[dict[str, Any], list[dict[str, Any]]], tuple[list[dict[str, Any]], dict[str, Any]]]


def load_study_config(experiment_id: str, *, extended_seeds: bool = False) -> dict[str, Any]:
    """Load a frozen shared study base and one experiment-specific config."""
    normalized_id = experiment_id.upper()
    config_dir = Path(__file__).resolve().parents[3] / "configs" / "experiments"
    config_files = sorted(config_dir.glob("e*.yaml"))
    config_path = next(
        (path for path in config_files if yaml.safe_load(path.read_text(encoding="utf-8")).get("id") == normalized_id),
        None,
    )
    if config_path is None:
        raise ValueError(f"unknown research experiment ID: {experiment_id}")
    config = yaml.safe_load((config_dir / "study_base.yaml").read_text(encoding="utf-8"))
    config["experiment"] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if extended_seeds:
        base_seeds = list(config["evaluation"]["seeds"])
        appended = [seed for seed in range(30) if seed not in base_seeds]
        config["evaluation"]["seeds"] = base_seeds + appended
    config["experiment"]["config_file"] = str(config_path.resolve())
    return config


def run_study(
    experiment_id: str,
    config: dict[str, Any],
    study: StudyFunction,
) -> dict[str, Any]:
    """Run an additive study and write it under an experiment-specific folder."""
    started = time.perf_counter()
    contexts = collect_contexts(config)
    episodes, summary = study(config, contexts)
    summary.update({
        "experiment_id": experiment_id,
        "context_count": len(contexts),
        "seed_count": len(config["evaluation"]["seeds"]),
        "seeds": list(config["evaluation"]["seeds"]),
        "runtime_seconds": time.perf_counter() - started,
    })
    result_root = Path(config["evaluation"].get("output_dir", "results"))
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    result_dir = result_root / "experiments" / experiment_id / f"run_{stamp}"
    report_dir = Path(config["evaluation"].get("report_dir", "reports")) / "experiments" / experiment_id / f"run_{stamp}"
    result_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    summary["run_directory"] = str(result_dir.resolve())
    summary["report_directory"] = str(report_dir.resolve())
    summary["config_file"] = str((result_dir / "config_used.json").resolve())
    (result_dir / "config_used.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (result_dir / "experiment_config.yaml").write_text(
        yaml.safe_dump(config["experiment"], sort_keys=False), encoding="utf-8"
    )
    (result_dir / "episodes.json").write_text(
        json.dumps(episodes, indent=2, sort_keys=True), encoding="utf-8"
    )
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if "failure_traces" in summary:
        (result_dir / "failure_traces.json").write_text(
            json.dumps(summary["failure_traces"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    write_reports(result_dir, report_dir)
    print(f"Experiment {experiment_id} complete: {result_dir}")
    print(f"Reports: {report_dir}")
    return summary


def collect_contexts(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect local observations plus offline world clones for evaluation."""
    environment = config["environment"]
    experiment = config["experiment"]
    requested = max(1, int(experiment.get("contexts_per_seed", 2)))
    contexts: list[dict[str, Any]] = []
    for seed in config["evaluation"]["seeds"]:
        env = SARGrid(environment)
        layout_seed = int(seed) + int(config["evaluation"].get("study_layout_seed_offset", 177_777))
        observations = env.reset(layout_seed)
        policies = task_policies(env.roles)
        sender_id = next(
            (index for index, role in enumerate(env.roles) if role == Role.SCOUT), 0
        )
        receivers = [index for index in range(len(env.roles)) if index != sender_id]
        if not receivers:
            continue
        seed_contexts = 0
        while not env.terminated and seed_contexts < requested:
            sender_observation = observations[sender_id]
            planned_action = policies[sender_id].act(sender_observation)
            messages = candidate_messages(sender_observation, planned_action)
            messages = [message for message in messages if message.kind != "intention"] or messages
            stale_messages = [
                message for message in messages if message.observed_at < sender_observation.time_step
            ]
            if experiment.get("prefer_stale_messages", False) and stale_messages:
                messages = stale_messages
            minimum_step = int(experiment.get("minimum_context_step", 0))
            if messages and env.step_count >= minimum_step:
                selected = messages[(seed_contexts + int(seed)) % len(messages)]
                receiver_id = receivers[(seed_contexts + int(seed)) % len(receivers)]
                contexts.append({
                    "seed": int(seed),
                    "layout_seed": layout_seed,
                    "step": env.step_count,
                    "sender_id": sender_id,
                    "receiver_id": receiver_id,
                    "message": selected,
                    "messages": tuple(messages),
                    "observation": sender_observation,
                    "world": env.clone(),
                    "roles": env.roles,
                })
                seed_contexts += 1
            actions = [policy.act(item) for policy, item in zip(policies, observations)]
            result = env.step(actions)
            observations = result.observations
    return contexts


def make_estimator(
    config: dict[str, Any],
    *,
    seed: int,
    particles: int | None = None,
    rollout_count: int | None = None,
    cost_weight: float = 0.0,
    state_estimator: Any | None = None,
    record_rollouts: bool = False,
) -> PairedRolloutEstimator:
    environment = config["environment"]
    communication = config["communication"]
    roles = SARGrid(environment).roles
    return PairedRolloutEstimator(
        state_estimator or ParticleReceiverEstimator(environment, seed=seed),
        task_policies(roles),
        particles=particles or int(communication["particles"]),
        rollout_count=rollout_count or int(communication["rollout_count"]),
        gamma=float(environment.get("gamma", 0.99)),
        cost_weight=cost_weight,
        unit_cost=float(communication.get("unit_cost", 1.0)),
        seed=seed,
        record_rollouts=record_rollouts,
    )


def oracle_estimate(
    config: dict[str, Any], context: Mapping[str, Any], *,
    horizon: int | None = None, rollout_count: int | None = None,
) -> Any:
    communication = config["communication"]
    oracle = CounterfactualOracle(
        task_policies(context["roles"]),
        rollout_count=rollout_count or int(communication["rollout_count"]),
        gamma=float(config["environment"].get("gamma", 0.99)),
        seed=int(context["seed"]),
    )
    return oracle.evaluate(
        context["world"], int(context["sender_id"]), int(context["receiver_id"]),
        context["message"], horizon or int(communication["oracle_horizon"]),
    )


def evaluate_pairs(
    config: dict[str, Any],
    context: Mapping[str, Any],
    *,
    messages: Sequence[Message] | None = None,
    receiver_ids: Sequence[int] | None = None,
    horizon: int | None = None,
    estimator_horizon: int | None = None,
    particles: int | None = None,
    rollout_count: int | None = None,
    estimator_factory: Callable[[Message], Any] | None = None,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], Any]]:
    """Evaluate local-observation-based estimates and offline oracle values."""
    candidate_messages = list(messages if messages is not None else context["messages"])
    receiver_pool = list(receiver_ids if receiver_ids is not None else range(len(context["roles"])))
    sender_id = int(context["sender_id"])
    receiver_pool = [receiver for receiver in receiver_pool if receiver != sender_id]
    oracle_horizon = horizon or int(config["communication"]["oracle_horizon"])
    estimate_horizon = estimator_horizon or int(config["communication"]["nonmyopic_horizon"])
    estimates: dict[tuple[int, int], Any] = {}
    truth: dict[tuple[int, int], float] = {}
    for message_index, message in enumerate(candidate_messages):
        for receiver_id in receiver_pool:
            if estimator_factory is None:
                estimator = make_estimator(
                    config,
                    seed=int(context["seed"]) + message_index * 101 + receiver_id,
                    particles=particles,
                    rollout_count=rollout_count,
                )
            else:
                estimator = estimator_factory(message)
            estimates[(message_index, receiver_id)] = estimator.estimate(
                sender_id, receiver_id, message, estimate_horizon, context["observation"]
            )
            truth[(message_index, receiver_id)] = oracle_estimate(
                config,
                {**context, "message": message, "receiver_id": receiver_id},
                horizon=oracle_horizon,
                rollout_count=rollout_count,
            ).marginal_value
    return truth, estimates


def policy_choice(
    config: dict[str, Any], context: Mapping[str, Any], messages: Sequence[Message],
    *, aware: bool, horizon: int | None = None, cost_weight: float = 0.0,
    force_send: bool = False, freshness: bool = True,
) -> tuple[int, int] | None:
    """Use the deployed selector for an offline context intervention."""
    sender_id = int(context["sender_id"])
    communication = config["communication"]
    environment = config["environment"]
    estimator = make_estimator(
        config, seed=int(context["seed"]), cost_weight=cost_weight,
        rollout_count=config["experiment"].get("rollout_count"),
    )
    candidates = list(messages)
    choice = select_communication(
        sender_id, candidates,
        [receiver for receiver in range(len(context["roles"])) if receiver != sender_id],
        horizon or int(communication["nonmyopic_horizon"]), estimator,
        observation=context["observation"], receiver_roles=context["roles"],
        shortlist_size=int(communication["shortlist_size"]), aware=aware,
        seed=int(context["seed"]), force_send=force_send,
        hazard_move_probability=float(environment.get("hazard_move_probability", 0)) if freshness else 0,
        hazard_toggle_probability=float(environment.get("hazard_toggle_probability", 0)) if freshness else 0,
    )
    return (candidates.index(choice.message), choice.receiver_id) if choice else None


def best_pair(
    estimates: Mapping[tuple[int, int], Any], *, positive_only: bool = True
) -> tuple[int, int] | None:
    if not estimates:
        return None
    key = max(estimates, key=lambda pair: estimates[pair].value_of_communication)
    if positive_only and estimates[key].value_of_communication <= 0:
        return None
    return key


def rollout_task_return(
    context: Mapping[str, Any], choice: tuple[Message, int] | None, config: dict[str, Any]
) -> float:
    """Offline task return from a context after an optional initial delivery."""
    world = context["world"].clone()
    observations = world.observe()
    if choice is not None:
        message, receiver_id = choice
        slots = [None] * len(observations)
        slots[int(context["sender_id"])] = CommunicationChoice(receiver_id, message)
        observations = world.deliver(slots)
    policies = task_policies(world.roles)
    gamma = float(config["environment"].get("gamma", 0.99))
    total, discount = 0.0, 1.0
    while not world.terminated:
        actions = [policy.act(observation) for policy, observation in zip(policies, observations)]
        result = world.step(actions)
        total += discount * result.reward
        discount *= gamma
        observations = result.observations
    return total


def bootstrap(values: Sequence[float], config: dict[str, Any], seed: int = 0) -> tuple[float, float] | None:
    if not values:
        return None
    evaluation = config["evaluation"]
    return bootstrap_mean_interval(
        values,
        float(evaluation.get("confidence_level", 0.95)),
        int(evaluation.get("bootstrap_samples", 10_000)),
        seed,
    )


def clustered_bootstrap(
    clusters: Sequence[Sequence[float]], config: dict[str, Any], seed: int = 0
) -> tuple[float, float] | None:
    nonempty = [cluster for cluster in clusters if cluster]
    if not nonempty:
        return None
    evaluation = config["evaluation"]
    return bootstrap_cluster_mean_interval(
        nonempty,
        float(evaluation.get("confidence_level", 0.95)),
        int(evaluation.get("bootstrap_samples", 10_000)),
        seed,
    )


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        return None
    left, right = rankdata(first), rankdata(second)
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_ss = sum((a - left_mean) ** 2 for a in left)
    right_ss = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else 0.0


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * p_value))
        adjusted[name] = running
    return adjusted


def paired_sign_flip_pvalue(values: Sequence[float], *, seed: int = 0, samples: int = 10_000) -> float | None:
    """Two-sided paired randomization p-value under a zero-mean null."""
    if not values:
        return None
    observed = abs(mean(values) or 0.0)
    rng = random.Random(seed)
    exceed = 0
    for _ in range(samples):
        statistic = abs(mean([value if rng.random() < 0.5 else -value for value in values]) or 0.0)
        exceed += statistic >= observed
    return (exceed + 1) / (samples + 1)


def write_reports(result_dir: Path, report_dir: Path) -> None:
    """Produce report artifacts only from persisted episode and summary JSON."""
    episodes = json.loads((result_dir / "episodes.json").read_text(encoding="utf-8"))
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = episodes or [summary.get("headline", {})]
    columns = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with (report_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})
    headline = summary.get("headline", {})
    with (report_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value"))
        writer.writerows((key, _csv_value(value)) for key, value in sorted(headline.items()))
    lines = [r"\begin{tabular}{lr}", r"\hline", r"Metric & Value \\", r"\hline"]
    for key, value in sorted(headline.items()):
        lines.append(f"{_latex(key)} & {_latex(_csv_value(value))} \\")
    lines.extend((r"\hline", r"\end{tabular}"))
    (report_dir / "summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_figure(headline, report_dir / "overview.png")
    image = Image.open(report_dir / "overview.png").convert("RGB")
    image.save(report_dir / "overview.pdf", "PDF", resolution=150)
    write_experiment_visualizations(episodes, summary, result_dir / "visualizations")
    write_experiment_visualizations(episodes, summary, report_dir / "visualizations")


_PLOT_COLORS = ("#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2")


def write_experiment_visualizations(
    episodes: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], output_dir: Path
) -> None:
    """Write experiment-specific PNG/PDF figures from persisted study results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = str(summary.get("experiment_id", ""))
    rows = list(episodes)
    headline = summary.get("headline", {})
    if experiment_id == "E01":
        for metric, title in (("mae_voc", "VoC estimate error"), ("pair_regret", "Oracle pair regret"),
                              ("j0_return_from_context", "Context task return")):
            _line_plot(rows, "variant", metric, output_dir / f"{metric}.png", title)
    elif experiment_id == "E02":
        for metric, title in (("return", "Task return by communication method"), ("regret", "Oracle regret by method"),
                              ("messages", "Communication rate by method")):
            _line_plot(rows, "method", metric, output_dir / f"{metric}.png", title)
    elif experiment_id == "E03":
        for metric, title in (("ess_fraction", "Particle diversity ESS fraction"), ("mae_voc", "VoC error by particle count"),
                              ("regret", "Decision regret by particle count")):
            _line_plot(rows, "particles", metric, output_dir / f"{metric}.png", title)
    elif experiment_id == "E04":
        for metric, title in (("standard_error", "Estimator standard error vs rollout count"),
                              ("bias", "Bias from high rollout reference"),
                              ("runtime_seconds_per_decision", "Runtime vs rollout count")):
            _line_plot(rows, "rollouts", metric, output_dir / f"{metric}.png", title, series="crn")
    elif experiment_id == "E05":
        for metric, title in (("mae", "Continuation value error"), ("top1_agreement", "Sign agreement with standard value"),
                              ("continuation_value", "Estimated continuation value")):
            _line_plot(rows, "continuation", metric, output_dir / f"{metric}.png", title)
    elif experiment_id == "E06":
        _line_plot(rows, "rule", "positive_rate_lambda_0", output_dir / "positive_rate_cost_0.png",
                   "Positive communication rate at zero cost")
        _line_plot(rows, "rule", "positive_rate_lambda_0_5", output_dir / "positive_rate_cost_high.png",
                   "Positive communication rate at higher cost")
        rates = summary.get("null_false_positive_rates", {})
        _bar_plot(list(rates), [rates[key] for key in rates], output_dir / "null_false_positives.png",
                  "Redundant-message false-positive rate")
    elif experiment_id == "E07":
        for metric, title in (("return", "Task return across cost weights"), ("messages", "Communication rate across cost weights")):
            _line_plot(rows, "lambda", metric, output_dir / f"{metric}_by_cost.png", title, series="method")
        points = summary.get("pareto_points", [])
        _scatter_plot(points, "mean_messages", "mean_return", "method", output_dir / "return_message_pareto.png",
                      "Return vs communication budget")
    elif experiment_id == "E08":
        curve = summary.get("h1_threshold_curve", {})
        curve_rows = []
        for label, value in curve.items():
            values = label.replace("theta=", "").replace("delta=", "").split(",")
            if len(values) == 2:
                curve_rows.append({"theta": float(values[0]), "delta": float(values[1]), "fraction": value})
            elif len(values) == 1:
                curve_rows.append({"delta": float(values[0]), "fraction": value})
        _line_plot(curve_rows, "delta", "fraction", output_dir / "h1_threshold_curve.png",
                   "H1 receiver differentiation curve",
                   series="theta" if curve_rows and "theta" in curve_rows[0] else None)
        effect_keys = ("h2_aware_minus_agnostic_return", "h2_budget_matched_return",
                       "h3_delayed_context_gain", "h4_positive_rate_drop_relative")
        _bar_plot(list(effect_keys), [headline.get(key) for key in effect_keys], output_dir / "hypothesis_effects.png",
                  "Paired hypothesis effects")
        p_keys = [key for key in headline if key.endswith("_holm_p")]
        if p_keys:
            _bar_plot(p_keys, [headline[key] for key in p_keys], output_dir / "holm_adjusted_p_values.png",
                  "Holm-adjusted hypothesis p-values")
    elif experiment_id == "E09":
        for metric, title in (("agreement", "Oracle agreement by estimator horizon"), ("mae_voc", "VoC error by estimator horizon"),
                              ("return", "Task return by estimator horizon")):
            _line_plot(rows, "horizon", metric, output_dir / f"{metric}_by_horizon.png", title)
    elif experiment_id == "E10":
        _scatter_plot(rows, "survival_heuristic", "observed_marginal_value", "message_kind",
                      output_dir / "heuristic_vs_oracle_value.png", "Information survival heuristic vs oracle value")
        ages = summary.get("by_information_age", {})
        age_rows = [{"age": int(age), **values} for age, values in ages.items()]
        _line_plot(age_rows, "age", "mean_marginal_value", output_dir / "value_by_information_age.png",
                   "Observed value by information age")
    elif experiment_id == "E11":
        _line_plot(rows, "setting", "aware_minus_agnostic_return", output_dir / "paired_return_by_setting.png",
                   "Aware minus agnostic return by environment setting")
        _line_plot(rows, "setting", "aware_regret", output_dir / "aware_regret_by_setting.png",
                   "Aware policy regret by environment setting")
    elif experiment_id == "E12":
        runtime_keys = ("particle_filter_seconds", "rollout_estimate_seconds", "oracle_seconds")
        _bar_plot(list(runtime_keys), [headline.get(f"mean_{key}") for key in runtime_keys],
                  output_dir / "runtime_breakdown.png", "Per-decision runtime breakdown (seconds)")
        _line_plot(rows, "information_age", "regret", output_dir / "regret_by_information_age.png",
                   "Policy regret by information age")
        _line_plot(rows, "message_kind", "aware_minus_agnostic_return", output_dir / "paired_return_advantage.png",
                   "Aware return advantage by message kind")


def _save_chart(image: Image.Image, output: Path) -> None:
    image.save(output)
    image.convert("RGB").save(output.with_suffix(".pdf"), "PDF", resolution=150)


def _chart_canvas(title: str, y_label: str):
    image = Image.new("RGB", (1100, 680), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((40, 24), title, fill="#111827", font=font)
    draw.text((20, 320), y_label, fill="#334155", font=font)
    draw.line((100, 70, 100, 580), fill="#64748b", width=2)
    draw.line((100, 580, 1050, 580), fill="#64748b", width=2)
    return image, draw, font


def _line_plot(rows, x_key, y_key, output, title, series=None):
    values = {}
    for row in rows:
        x, y = row.get(x_key), row.get(y_key)
        if x is None or not isinstance(y, (int, float)) or not math.isfinite(float(y)):
            continue
        group = str(row.get(series, "value")) if series else "value"
        values.setdefault(group, {}).setdefault(x, []).append(float(y))
    if not values:
        return
    categories = sorted({x for group in values.values() for x in group}, key=lambda item: (0, item) if isinstance(item, (int, float)) else (1, str(item)))
    all_y = [value for group in values.values() for bucket in group.values() for value in bucket]
    low, high = min(0.0, min(all_y)), max(0.0, max(all_y))
    if math.isclose(low, high):
        low, high = low - 1.0, high + 1.0
    image, draw, font = _chart_canvas(title, y_key)
    left, top, width, height = 120, 90, 880, 450
    def point(x_index, value):
        px = left + (width / max(1, len(categories) - 1)) * x_index
        py = top + height - (value - low) / (high - low) * height
        return int(px), int(py)
    for tick in range(5):
        value = low + (high - low) * tick / 4
        y = top + height - height * tick / 4
        draw.line((left, y, left + width, y), fill="#e2e8f0")
        draw.text((52, y - 5), f"{value:.3g}", fill="#475569", font=font)
    for index, category in enumerate(categories):
        x = left + (width / max(1, len(categories) - 1)) * index
        label = str(category)
        draw.text((int(x) - min(70, len(label) * 3), 590), label[:22], fill="#475569", font=font)
    for group_index, (group, buckets) in enumerate(sorted(values.items())):
        points = []
        color = _PLOT_COLORS[group_index % len(_PLOT_COLORS)]
        for index, category in enumerate(categories):
            bucket = buckets.get(category)
            if bucket:
                p = point(index, sum(bucket) / len(bucket))
                points.append(p)
                draw.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=color)
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        legend_y = 80 + group_index * 22
        draw.line((820, legend_y + 5, 845, legend_y + 5), fill=color, width=3)
        draw.text((850, legend_y), group[:32], fill="#334155", font=font)
    _save_chart(image, output)


def _bar_plot(labels, values, output, title):
    clean = [(str(label), float(value)) for label, value in zip(labels, values)
             if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if not clean:
        return
    image, draw, font = _chart_canvas(title, "Value")
    low, high = min(0.0, *(value for _, value in clean)), max(0.0, *(value for _, value in clean))
    if math.isclose(low, high):
        low, high = low - 1.0, high + 1.0
    top, bottom, left, right = 100, 570, 160, 1010
    zero_y = bottom - (0 - low) / (high - low) * (bottom - top)
    draw.line((left, zero_y, right, zero_y), fill="#64748b", width=2)
    slot = (right - left) / len(clean)
    for index, (label, value) in enumerate(clean):
        x = left + slot * index + slot * 0.18
        y_value = bottom - (value - low) / (high - low) * (bottom - top)
        draw.rectangle((x, min(zero_y, y_value), x + slot * 0.64, max(zero_y, y_value)),
                       fill=_PLOT_COLORS[index % len(_PLOT_COLORS)])
        draw.text((x, 580), label[:24], fill="#334155", font=font)
        draw.text((x, min(zero_y, y_value) - 18), f"{value:.3g}", fill="#111827", font=font)
    _save_chart(image, output)


def _scatter_plot(rows, x_key, y_key, series, output, title):
    points = [(row.get(x_key), row.get(y_key), str(row.get(series, "value"))) for row in rows
              if isinstance(row.get(x_key), (int, float)) and isinstance(row.get(y_key), (int, float))]
    if not points:
        return
    xs, ys = [float(point[0]) for point in points], [float(point[1]) for point in points]
    x_low, x_high = min(0.0, min(xs)), max(0.0, max(xs))
    y_low, y_high = min(0.0, min(ys)), max(0.0, max(ys))
    if math.isclose(x_low, x_high): x_low, x_high = x_low - 1.0, x_high + 1.0
    if math.isclose(y_low, y_high): y_low, y_high = y_low - 1.0, y_high + 1.0
    image, draw, font = _chart_canvas(title, y_key)
    left, top, width, height = 140, 90, 850, 460
    draw.line((left, top + height, left + width, top + height), fill="#64748b", width=2)
    draw.line((left, top, left, top + height), fill="#64748b", width=2)
    colors = {name: _PLOT_COLORS[index % len(_PLOT_COLORS)] for index, name in enumerate(sorted({p[2] for p in points}))}
    for x, y, label in points:
        px = left + (float(x) - x_low) / (x_high - x_low) * width
        py = top + height - (float(y) - y_low) / (y_high - y_low) * height
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=colors[label])
    draw.text((left, 590), f"{x_key}: {x_low:.3g} to {x_high:.3g}", fill="#475569", font=font)
    draw.text((left, 610), f"{y_key}: {y_low:.3g} to {y_high:.3g}", fill="#475569", font=font)
    for index, label in enumerate(colors):
        y = 80 + index * 22
        draw.ellipse((820, y, 830, y + 10), fill=colors[label])
        draw.text((838, y - 2), label[:28], fill="#334155", font=font)
    _save_chart(image, output)


def _write_figure(headline: Mapping[str, Any], output: Path) -> None:
    numeric = [(key, float(value)) for key, value in sorted(headline.items()) if isinstance(value, (int, float)) and math.isfinite(float(value))]
    width, row_height, margin = 1000, 42, 260
    height = max(260, 100 + row_height * len(numeric))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((24, 20), "Experiment headline metrics", fill="#111827", font=font)
    max_value = max((abs(value) for _, value in numeric), default=1.0) or 1.0
    for index, (name, value) in enumerate(numeric):
        y = 60 + index * row_height
        draw.text((24, y + 6), name[:38], fill="#334155", font=font)
        center = margin + 300
        extent = int(abs(value) / max_value * 300)
        color = "#2563eb" if value >= 0 else "#dc2626"
        left, right = (center, center + extent) if value >= 0 else (center - extent, center)
        draw.rectangle((left, y, right, y + 24), fill=color)
        draw.text((margin + 620, y + 6), f"{value:.4g}", fill="#111827", font=font)
    image.save(output)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _latex(value: Any) -> str:
    text = str(value)
    for character, replacement in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(character, replacement)
    return text
