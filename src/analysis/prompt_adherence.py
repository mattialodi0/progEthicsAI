"""Prompt-adherence measurements used by the general statistics facade."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from .general_statistics import (
    ADHERENCE_ATTRIBUTES,
    MAIN_ATTRIBUTES,
    _detected_prompt_value,
    _detected_value,
    _record_mode,
)


def _value(record: dict[str, Any], attribute: str, requested: Any) -> str | None:
    if requested is None:
        return None
    return (_detected_value(record, attribute) if attribute in MAIN_ATTRIBUTES
            else _detected_prompt_value(record, requested))


def _label(attribute: str, value: Any) -> str:
    label = str(value).strip().lower()
    return {"man": "male", "woman": "female"}.get(label, label) if attribute == "gender" else label


def _measurement(records: list[dict[str, Any]], attribute: str) -> dict[str, Any]:
    matrix: defaultdict[str, Counter[str]] = defaultdict(Counter)
    requested_count = total = correct = unmatched = 0
    for record in records:
        features = record.get("prompt_feats")
        requested = features.get(attribute) if isinstance(features, dict) else None
        if requested is None:
            continue
        requested_count += 1
        requested_label = _label(attribute, requested)
        detected = _value(record, attribute, requested)
        if detected is None:
            unmatched += 1
            continue
        detected_label = _label(attribute, detected)
        matrix[requested_label][detected_label] += 1
        total += 1
        correct += int(requested_label == detected_label or (
            attribute == "race" and requested_label == "hispanic"
            and detected_label == "latino hispanic"
        ))
    labels = sorted(set(matrix) | {item for row in matrix.values() for item in row})
    counts = {row: {column: matrix[row].get(column, 0) for column in labels} for row in labels}
    return {
        "labels": labels,
        "matrix": {
            row: {column: round(count / sum(counts[row].values()), 6)
                  if sum(counts[row].values()) else 0.0
                  for column, count in counts[row].items()}
            for row in labels
        },
        "counts": counts,
        "total": total,
        "accuracy": round(correct / total, 6) if total else None,
        "unmatched": unmatched,
        "requested": requested_count,
    }


def _accuracies(records: list[dict[str, Any]], attributes: list[str]) -> dict[str, Any]:
    return {attribute: _measurement(records, attribute)["accuracy"] for attribute in attributes}


def _by_category(data: list[dict[str, Any]], attributes: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category in MAIN_ATTRIBUTES:
        groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in data:
            features = record.get("prompt_feats")
            value = features.get(category) if isinstance(features, dict) else None
            if value is not None:
                groups[str(value).strip().lower()].append(record)
        result[category] = {
            value: {"n_images": len(records), "accuracy": _accuracies(records, attributes)}
            for value, records in sorted(groups.items())
        }
    return result


def _combinations(data: list[dict[str, Any]], attributes: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for first, second in combinations(attributes, 2):
        matrix: defaultdict[str, Counter[str]] = defaultdict(Counter)
        total = correct = unmatched = 0
        for record in data:
            features = record.get("prompt_feats")
            if not isinstance(features, dict):
                continue
            requested = (features.get(first), features.get(second))
            if None in requested:
                continue
            requested_label = " + ".join((_label(first, requested[0]), _label(second, requested[1])))
            detected = (_value(record, first, requested[0]), _value(record, second, requested[1]))
            if None in detected:
                unmatched += 1
                continue
            detected_label = " + ".join((_label(first, detected[0]), _label(second, detected[1])))
            matrix[requested_label][detected_label] += 1
            total += 1
            correct += int(requested_label == detected_label)
        labels = sorted(set(matrix) | {item for row in matrix.values() for item in row})
        counts = {row: {column: matrix[row].get(column, 0) for column in labels} for row in labels}
        result[f"{first} + {second}"] = {
            "attributes": [first, second], "labels": labels, "counts": counts,
            "matrix": {row: {column: round(count / sum(counts[row].values()), 6)
                             if sum(counts[row].values()) else 0.0
                             for column, count in counts[row].items()} for row in labels},
            "total": total, "accuracy": round(correct / total, 6) if total else None,
            "unmatched": unmatched,
        }
    return result


def prompt_adherence(data: list[dict[str, Any]], attributes: list[str], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Precompute adherence using the facial attributes supported by detectors."""
    del attributes
    config = config or {}
    detectable = list(ADHERENCE_ATTRIBUTES)
    modes: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in data:
        modes[_record_mode(record, config)].append(record)
    return {
        "attributes": detectable,
        "by_attribute": {attribute: _measurement(data, attribute) for attribute in detectable},
        "by_prompt_mode": {mode: {"n_images": len(records), "accuracy": _accuracies(records, detectable)}
                           for mode, records in sorted(modes.items())},
        "by_demographic_category": _by_category(data, detectable),
        "combinations": _combinations(data, detectable),
    }
