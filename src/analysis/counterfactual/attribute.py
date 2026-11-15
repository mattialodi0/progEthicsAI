"""What changes, besides the requested attribute, between matched images."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from statistics import mean, median
from typing import Any

from ..prompt_adherence import adherence_label, adherence_value
from ..records import detected_value, image_embedding
from ..stats_utils import (
    cohen_h,
    mean_summary,
    newcombe_difference,
    one_sample_t_p,
    rounded,
    two_proportion_p,
)
from .common import DETECTED_ATTRIBUTES, NUMERICAL_FEATURES, condition_value
from .pairs import attribute_pairs


# Numerical features that the swapped attribute is meant to change: their
# difference is adherence (chapter 4), not a side effect.
INTENDED_CHANGES = {"age": ("detected_age",)}
# Smallest number of images per side for a side-effect comparison.
MIN_SIDE_EFFECT_IMAGES = 5


def _summary(values: list[float]) -> dict[str, Any]:
    clean = [float(value) for value in values]
    return {
        "n": len(clean),
        "mean": round(mean(clean), 6) if clean else None,
        "median": round(median(clean), 6) if clean else None,
        "values": [round(value, 6) for value in clean],
    }


def _paired_summary(values: list[float]) -> dict[str, Any]:
    """``_summary`` plus the uncertainty of the mean paired difference."""
    result = _summary(values)
    stats = mean_summary(values)
    result.update(rounded({
        "sd": stats["sd"],
        "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"],
        # Standardized mean difference of paired data (Cohen's d_z).
        "dz": stats["mean"] / stats["sd"] if stats["sd"] else None,
        "p_value": one_sample_t_p(values),
    }))
    return result


def _side_effects(
    detected: dict[str, dict[str, tuple[list[str], list[str]]]],
) -> list[dict[str, Any]]:
    """Compare non-requested detected attributes between the two sides.

    ``detected[contrast][feature]`` holds the detected values of the first
    and second images of every matched pair. The two images are independent
    generations (pairs are matched by prompt template, not by seed), so the
    sides are compared as two independent samples.
    """
    rows = []
    for contrast, features in sorted(detected.items()):
        first_value, second_value = contrast.split(" vs ")
        for feature, (first, second) in sorted(features.items()):
            if min(len(first), len(second)) < MIN_SIDE_EFFECT_IMAGES:
                continue
            first_counts, second_counts = Counter(first), Counter(second)
            for category in sorted(set(first_counts) | set(second_counts)):
                k_first, k_second = first_counts[category], second_counts[category]
                difference, low, high = newcombe_difference(
                    k_second, len(second), k_first, len(first))
                rows.append(rounded({
                    "contrast": contrast,
                    "first": first_value,
                    "second": second_value,
                    "feature": feature,
                    "category": category,
                    "n_first": len(first),
                    "n_second": len(second),
                    "first_share": k_first / len(first),
                    "second_share": k_second / len(second),
                    "difference": difference,
                    "ci_low": low,
                    "ci_high": high,
                    "cohen_h": cohen_h(k_second / len(second), k_first / len(first)),
                    "p_value": two_proportion_p(k_second, len(second), k_first, len(first)),
                }))
    return rows


def _adherence(record: dict[str, Any], attribute: str) -> bool | None:
    features = record.get("prompt_feats")
    requested = features.get(attribute) if isinstance(features, dict) else None
    if requested is None:
        return None
    detected = adherence_value(record, attribute, requested)
    if detected is None:
        return False
    return adherence_label(attribute, requested) == adherence_label(attribute, detected)


def _distance(first: list[float], second: list[float]) -> float:
    length = min(len(first), len(second))
    return (sum((first[index] - second[index]) ** 2 for index in range(length)) ** 0.5
            if length else 0.0)


def _cosine_distance(first: list[float], second: list[float]) -> float | None:
    length = min(len(first), len(second))
    if not length:
        return None
    dot = sum(first[index] * second[index] for index in range(length))
    first_norm = sum(value * value for value in first[:length]) ** 0.5
    second_norm = sum(value * value for value in second[:length]) ** 0.5
    if not first_norm or not second_norm:
        return None
    return 1.0 - max(-1.0, min(1.0, dot / (first_norm * second_norm)))


def _detected_changes(change_counts: Counter[str]) -> dict[str, Any]:
    """How often the detected value of the swapped attribute differs in a pair."""
    total = sum(change_counts.values())
    changed = sum(
        count for label, count in change_counts.items()
        if label.split(" -> ")[0] != label.split(" -> ")[1]
    )
    return {
        "n": total,
        "changed": changed,
        "rate": changed / total if change_counts else None,
        "transitions": dict(sorted(change_counts.items())),
    }


def attribute_analysis(
    data: list[dict[str, Any]], attribute: str
) -> dict[str, Any]:
    pairs = attribute_pairs(data, attribute)
    change_counts: Counter[str] = Counter()
    adherence: defaultdict[str, list[bool]] = defaultdict(list)
    numerical: defaultdict[str, list[float]] = defaultdict(list)
    embedding_distances: list[float] = []
    within_distances: list[float] = []
    between_distances: list[float] = []
    embeddings_by_condition: defaultdict[str, dict[int, list[float]]] = defaultdict(dict)
    side_detected: defaultdict[str, defaultdict[str, tuple[list[str], list[str]]]] = (
        defaultdict(lambda: defaultdict(lambda: ([], [])))
    )
    for condition_pair, first, second in pairs:
        first_condition, second_condition = condition_pair.split(" vs ")
        for feature in DETECTED_ATTRIBUTES:
            if feature == attribute:
                continue
            for side, record in ((0, first), (1, second)):
                value = detected_value(record, feature)
                if value is not None:
                    side_detected[condition_pair][feature][side].append(value)
        first_detected = adherence_value(first, attribute, condition_value(first, attribute))
        second_detected = adherence_value(second, attribute, condition_value(second, attribute))
        first_detected = str(first_detected).lower() if first_detected is not None else "missing"
        second_detected = str(second_detected).lower() if second_detected is not None else "missing"
        change_counts[f"{first_detected} -> {second_detected}"] += 1
        for condition, record in ((first_condition, first), (second_condition, second)):
            result = _adherence(record, attribute)
            if result is not None:
                adherence[condition].append(result)
        for feature in NUMERICAL_FEATURES:
            if feature in INTENDED_CHANGES.get(attribute, ()):
                continue
            left, right = first.get(feature), second.get(feature)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                numerical[feature].append(float(right) - float(left))
        first_embedding, second_embedding = image_embedding(first), image_embedding(second)
        if first_embedding is not None and second_embedding is not None:
            distance = _cosine_distance(first_embedding, second_embedding)
            if distance is not None:
                embedding_distances.append(distance)
            between_distances.append(_distance(first_embedding, second_embedding))
        if first_embedding is not None:
            embeddings_by_condition[first_condition][first["_counterfactual_index"]] = first_embedding
        if second_embedding is not None:
            embeddings_by_condition[second_condition][second["_counterfactual_index"]] = second_embedding
    for vectors in embeddings_by_condition.values():
        within_distances.extend(
            _distance(left, right)
            for left, right in combinations(vectors.values(), 2)
        )
    return {
        "pairs": len(pairs),
        "detected_changes": _detected_changes(change_counts),
        "adherence": {
            condition: {
                "n": len(values),
                "rate": sum(values) / len(values) if values else None,
            }
            for condition, values in sorted(adherence.items())
        },
        "numerical_differences": {
            feature: _paired_summary(values) for feature, values in sorted(numerical.items())
        },
        "side_effects": _side_effects(side_detected),
        "embeddings": {
            "paired_cosine_distance": _summary(embedding_distances),
            "between_pair_euclidean_distance": _summary(between_distances),
            "within_condition_euclidean_distance": _summary(within_distances),
        },
    }
