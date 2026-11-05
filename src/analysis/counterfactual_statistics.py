"""Precomputed matched-pair statistics for chapter 6."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import re
from statistics import mean, median
from typing import Any

from .general_statistics import (
    MAIN_ATTRIBUTES,
    _canonical_demographic_value,
    _detected_value,
    _embedding,
)
from .prompt_adherence import _label as _adherence_label
from .prompt_adherence import _value as _adherence_value


COUNTERFACTUAL_ATTRIBUTES = ("gender", "race", "age", "emotion", "job", "workplace")
NUMERICAL_FEATURES = ("n_people", "person_area_ratio", "face_area_ratio")
PROMPT_VALUES = {
    "gender": ("male", "female", "man", "woman"),
    "race": ("white", "black", "asian", "hispanic", "middle eastern"),
    "age": ("young", "old", "middle-aged"),
    "emotion": ("happy", "sad", "angry", "calm", "bored", "nervous"),
    "job": (
        "doctor", "nurse", "engineer", "teacher", "artist", "scientist",
        "chef", "pilot", "writer", "musician", "athlete", "politician", "soldier",
    ),
    "workplace": (
        "office", "factory", "hospital", "school", "restaurant", "store",
        "laboratory", "construction site",
    ),
}
CONTROL_ATTRIBUTES = ("gender", "race", "age", "job")
DETECTED_ATTRIBUTES = tuple(dict.fromkeys((*MAIN_ATTRIBUTES, "emotion")))


def _prompt_key(prompt: str, attribute: str, value: str) -> str:
    pattern = r"(?<!\w)" + re.escape(value).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.sub(pattern, "__counterfactual__", prompt.lower(), count=1)


def _condition(record: dict[str, Any], attribute: str) -> str | None:
    features = record.get("prompt_feats")
    value = features.get(attribute) if isinstance(features, dict) else None
    if value is None:
        return None
    if attribute == "gender":
        return _canonical_demographic_value(attribute, value)
    return str(value).strip().lower() or None


def _pairs(
    data: list[dict[str, Any]], attribute: str
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    groups: defaultdict[str, defaultdict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, record in enumerate(data):
        condition = _condition(record, attribute)
        prompt = record.get("prompt")
        if condition is None or not isinstance(prompt, str):
            continue
        values = PROMPT_VALUES[attribute]
        matched = next(
            (
                value for value in values
                if re.search(
                    r"(?<!\w)" + re.escape(value).replace(r"\ ", r"\s+") + r"(?!\w)",
                    prompt,
                    re.IGNORECASE,
                )
            ),
            None,
        )
        if matched is None:
            matched = str(record["prompt_feats"][attribute]).lower()
        item = dict(record)
        item["_counterfactual_index"] = index
        groups[_prompt_key(prompt, attribute, matched)][condition].append(item)

    pairs = []
    for group in groups.values():
        for left, right in combinations(sorted(group), 2):
            for first, second in zip(
                sorted(group[left], key=lambda item: item["_counterfactual_index"]),
                sorted(group[right], key=lambda item: item["_counterfactual_index"]),
            ):
                pairs.append((f"{left} vs {right}", first, second))
    return pairs


def _summary(values: list[float]) -> dict[str, Any]:
    clean = [float(value) for value in values]
    return {
        "n": len(clean),
        "mean": round(mean(clean), 6) if clean else None,
        "median": round(median(clean), 6) if clean else None,
        "values": [round(value, 6) for value in clean],
    }


def _adherence(record: dict[str, Any], attribute: str) -> bool | None:
    features = record.get("prompt_feats")
    requested = features.get(attribute) if isinstance(features, dict) else None
    if requested is None:
        return None
    detected = _adherence_value(record, attribute, requested)
    if detected is None:
        return False
    return _adherence_label(attribute, requested) == _adherence_label(attribute, detected)


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


def _attribute_analysis(
    data: list[dict[str, Any]], attribute: str
) -> dict[str, Any]:
    pairs = _pairs(data, attribute)
    change_counts: Counter[str] = Counter()
    adherence: defaultdict[str, list[bool]] = defaultdict(list)
    numerical: defaultdict[str, list[float]] = defaultdict(list)
    embedding_distances: list[float] = []
    within_distances: list[float] = []
    between_distances: list[float] = []
    embeddings_by_condition: defaultdict[str, dict[int, list[float]]] = defaultdict(dict)
    for condition_pair, first, second in pairs:
        first_condition, second_condition = condition_pair.split(" vs ")
        first_detected = _adherence_value(first, attribute, _condition(first, attribute))
        second_detected = _adherence_value(second, attribute, _condition(second, attribute))
        first_detected = str(first_detected).lower() if first_detected is not None else "missing"
        second_detected = str(second_detected).lower() if second_detected is not None else "missing"
        change_counts[f"{first_detected} -> {second_detected}"] += 1
        for condition, record in ((first_condition, first), (second_condition, second)):
            result = _adherence(record, attribute)
            if result is not None:
                adherence[condition].append(result)
        for feature in NUMERICAL_FEATURES:
            left, right = first.get(feature), second.get(feature)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                numerical[feature].append(float(right) - float(left))
        first_embedding, second_embedding = _embedding(first), _embedding(second)
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
        "detected_changes": {
            "n": sum(change_counts.values()),
            "changed": sum(
                count for label, count in change_counts.items()
                if label.split(" -> ")[0] != label.split(" -> ")[1]
            ),
            "rate": (
                sum(count for label, count in change_counts.items()
                    if label.split(" -> ")[0] != label.split(" -> ")[1])
                / sum(change_counts.values())
                if change_counts else None
            ),
            "transitions": dict(sorted(change_counts.items())),
        },
        "adherence": {
            condition: {
                "n": len(values),
                "rate": sum(values) / len(values) if values else None,
            }
            for condition, values in sorted(adherence.items())
        },
        "numerical_differences": {
            feature: _summary(values) for feature, values in sorted(numerical.items())
        },
        "embeddings": {
            "paired_cosine_distance": _summary(embedding_distances),
            "between_pair_euclidean_distance": _summary(between_distances),
            "within_condition_euclidean_distance": _summary(within_distances),
        },
    }


def _matched_attribute_pairs(
    data: list[dict[str, Any]],
    *,
    mode: str,
    required_attributes: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Return pairs that vary one requested attribute and fix the rest."""
    pairs: list[dict[str, Any]] = []
    for variable in required_attributes:
        variable_groups: defaultdict[
            tuple[str, ...], defaultdict[str, list[dict[str, Any]]]
        ] = defaultdict(lambda: defaultdict(list))
        for index, record in enumerate(data):
            if str(record.get("mode")) != mode:
                continue
            values = tuple(_condition(record, attribute) for attribute in required_attributes)
            if any(value is None for value in values):
                continue
            variable_index = required_attributes.index(variable)
            fixed_values = values[:variable_index] + values[variable_index + 1:]
            item = dict(record)
            item["_counterfactual_index"] = index
            variable_groups[fixed_values][values[variable_index]].append(item)
        for fixed_values, by_value in variable_groups.items():
            for left, right in combinations(sorted(by_value), 2):
                for first, second in zip(
                    sorted(by_value[left], key=lambda item: item["_counterfactual_index"]),
                    sorted(by_value[right], key=lambda item: item["_counterfactual_index"]),
                ):
                    pairs.append({
                        "variable_attribute": variable,
                        "variable_values": [left, right],
                        "fixed_values": {
                            attribute: value
                            for attribute, value in zip(
                                (attribute for attribute in required_attributes if attribute != variable),
                                fixed_values,
                            )
                        },
                        "detected": {
                            side: {
                                attribute: _detected_value(record, attribute)
                                for attribute in DETECTED_ATTRIBUTES
                            }
                            for side, record in (("first", first), ("second", second))
                        },
                    })
    return pairs


def _mode_key(config: dict[str, Any] | None, label: str, default: str) -> str:
    """Return the raw ``mode`` value whose configured label is ``label``."""
    modes = (config or {}).get("prompt_modes")
    if isinstance(modes, dict):
        for key, value in modes.items():
            if str(value).strip().lower() == label:
                return str(key)
    return default


def _control_pairs(
    data: list[dict[str, Any]], mode: str = "0"
) -> list[dict[str, Any]]:
    """Return matched pairs from four-attribute control prompts."""
    return _matched_attribute_pairs(
        data, mode=mode, required_attributes=CONTROL_ATTRIBUTES
    )


def _double_pairs(
    data: list[dict[str, Any]], mode: str = "2"
) -> list[dict[str, Any]]:
    """Return matched pairs from prompts specifying exactly two attributes."""
    pairs: list[dict[str, Any]] = []
    for index, record in enumerate(data):
        if str(record.get("mode")) != mode:
            continue
        features = record.get("prompt_feats")
        if not isinstance(features, dict):
            continue
        specified = tuple(
            attribute for attribute in CONTROL_ATTRIBUTES
            if _condition(record, attribute) is not None
        )
        if len(specified) != 2:
            continue
        pairs.append((index, record, specified))

    result: list[dict[str, Any]] = []
    # Sorted: iterating a set of strings would make the pair order depend on
    # the per-process hash seed, so every run produced a different artifact.
    pair_specs = sorted({
        spec
        for _, _, specified in pairs
        for spec in ((specified[0], specified[1]), (specified[1], specified[0]))
    })
    for variable, fixed in pair_specs:
        groups: defaultdict[str, defaultdict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, record, specified in pairs:
            if specified != tuple(sorted((variable, fixed), key=CONTROL_ATTRIBUTES.index)):
                continue
            item = dict(record)
            item["_counterfactual_index"] = index
            groups[_condition(record, fixed)][_condition(record, variable)].append(item)
        for fixed_value, by_value in groups.items():
            for left, right in combinations(sorted(by_value), 2):
                for first, second in zip(
                    sorted(by_value[left], key=lambda item: item["_counterfactual_index"]),
                    sorted(by_value[right], key=lambda item: item["_counterfactual_index"]),
                ):
                    result.append({
                        "variable_attribute": variable,
                        "variable_values": [left, right],
                        "fixed_attribute": fixed,
                        "fixed_value": fixed_value,
                        "detected": {
                            side: {
                                attribute: _detected_value(record, attribute)
                                for attribute in DETECTED_ATTRIBUTES
                            }
                            for side, record in (("first", first), ("second", second))
                        },
                    })
    return result


def build_counterfactual_analysis(
    data: list[dict[str, Any]], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build all chapter-6 data without retaining raw records or embeddings."""
    control_mode = _mode_key(config, "control", "0")
    double_mode = _mode_key(config, "double variable", "2")
    results = {
        attribute: _attribute_analysis(data, attribute)
        for attribute in COUNTERFACTUAL_ATTRIBUTES
    }
    return {
        "attributes": list(COUNTERFACTUAL_ATTRIBUTES),
        "numerical_features": list(NUMERICAL_FEATURES),
        "by_attribute": results,
        "control_analysis": {
            "control_attributes": list(CONTROL_ATTRIBUTES),
            "detected_attributes": list(DETECTED_ATTRIBUTES),
            "pairs": _control_pairs(data, control_mode),
        },
        "double_analysis": {
            "control_attributes": list(CONTROL_ATTRIBUTES),
            "detected_attributes": list(DETECTED_ATTRIBUTES),
            "pairs": _double_pairs(data, double_mode),
        },
        "caveat": (
            "Embedding distance is not itself a bias score: the requested "
            "attribute should change some image content. Interpret distances "
            "alongside matched adherence and unrelated-feature differences."
        ),
    }
