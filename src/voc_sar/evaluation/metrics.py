"""Reproducible metrics for task return and communication valuation."""

import random
from collections.abc import Mapping, Sequence


def episode_return(rewards: Sequence[float], gamma: float) -> float:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1")
    return sum((gamma**step) * reward for step, reward in enumerate(rewards))


def communication_reduction(method_messages: int, baseline_messages: int) -> float:
    if method_messages < 0:
        raise ValueError("method message count cannot be negative")
    if baseline_messages <= 0:
        raise ValueError("baseline message count must be positive")
    return 1.0 - method_messages / baseline_messages


def receiver_regret(
    oracle_values: Mapping[int, float], selected_receiver: int | None
) -> float:
    """Regret against the best receiver for one fixed message, with no-send value 0."""
    best_value = max(0.0, max(oracle_values.values(), default=0.0))
    selected_value = 0.0 if selected_receiver is None else oracle_values.get(selected_receiver, 0.0)
    return max(0.0, best_value - selected_value)


def pair_regret(
    oracle_values: Mapping[tuple[object, int], float],
    selected_pair: tuple[object, int] | None,
) -> float:
    """Regret against the best message/receiver pair, including no-send."""
    best_value = max(0.0, max(oracle_values.values(), default=0.0))
    selected_value = 0.0 if selected_pair is None else oracle_values.get(selected_pair, 0.0)
    return max(0.0, best_value - selected_value)


def voc_mean_absolute_error(
    predictions: Sequence[float], oracle_values: Sequence[float]
) -> float:
    if len(predictions) != len(oracle_values):
        raise ValueError("predictions and oracle_values must have equal lengths")
    if not predictions:
        raise ValueError("at least one paired estimate is required")
    return sum(
        abs(prediction - oracle)
        for prediction, oracle in zip(predictions, oracle_values)
    ) / len(predictions)


def filter_recall(
    oracle_values: Mapping[tuple[object, int], float],
    shortlisted_pairs: Sequence[tuple[object, int]],
    threshold: float,
) -> float:
    """Recall of above-threshold oracle pairs in the supplied shortlist."""
    high_value = {pair for pair, value in oracle_values.items() if value > threshold}
    if not high_value:
        return 1.0
    return len(high_value.intersection(shortlisted_pairs)) / len(high_value)


def useful_message_rate(marginal_values: Sequence[float], threshold: float) -> float:
    if not marginal_values:
        return 0.0
    return sum(value > threshold for value in marginal_values) / len(marginal_values)


def action_agreement(first: Sequence[object], reference: Sequence[object]) -> float:
    if len(first) != len(reference):
        raise ValueError("action sequences must have equal lengths")
    if not first:
        return 0.0
    return sum(left == right for left, right in zip(first, reference)) / len(first)


def bootstrap_mean_interval(
    values: Sequence[float],
    confidence_level: float = 0.95,
    samples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one observation")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    )
    tail = (1.0 - confidence_level) / 2.0
    lower_index = int(tail * (samples - 1))
    upper_index = int((1.0 - tail) * (samples - 1))
    return means[lower_index], means[upper_index]


def bootstrap_cluster_mean_interval(
    clusters: Sequence[Sequence[float]],
    confidence_level: float = 0.95,
    samples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap cluster means so repeated contexts within a seed stay grouped."""
    nonempty_clusters = [cluster for cluster in clusters if cluster]
    if not nonempty_clusters:
        raise ValueError("cluster bootstrap requires at least one non-empty cluster")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        selected = [
            rng.choice(nonempty_clusters)
            for _ in range(len(nonempty_clusters))
        ]
        means.append(
            sum(sum(cluster) for cluster in selected)
            / sum(len(cluster) for cluster in selected)
        )
    means.sort()
    tail = (1.0 - confidence_level) / 2.0
    lower_index = int(tail * (samples - 1))
    upper_index = int((1.0 - tail) * (samples - 1))
    return means[lower_index], means[upper_index]


def positive_voc_rate(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(value > 0 for value in values) / len(values)


def bootstrap_paired_relative_reduction(
    rates: Sequence[tuple[float, float]],
    confidence_level: float,
    samples: int,
    seed: int,
) -> tuple[tuple[float, float] | None, int]:
    """Resample seeds jointly and take the same ratio of means as the point estimate."""
    if len(rates) < 2 or not any(low for low, _ in rates):
        return None, samples if rates and not any(low for low, _ in rates) else 0
    rng = random.Random(seed)
    effects = []
    undefined = 0
    for _ in range(samples):
        draw = rng.choices(rates, k=len(rates))
        low = sum(pair[0] for pair in draw)
        if low:
            effects.append(1 - sum(pair[1] for pair in draw) / low)
        else:
            undefined += 1
    if not effects:
        return None, undefined
    effects.sort()
    tail = (1 - confidence_level) / 2
    return (effects[int(tail * (len(effects) - 1))],
            effects[int((1 - tail) * (len(effects) - 1))]), undefined
