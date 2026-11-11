"""Prompt-adherence measurements used by the general statistics facade."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from .general_statistics import (
    ADHERENCE_ATTRIBUTES,
    AGE_CUTOFFS,
    AGE_GROUPS,
    CONFIDENT_PROBABILITY,
    MAIN_ATTRIBUTES,
    _detected_prompt_value,
    _detected_value,
    _record_mode,
)
from .preprocessing import PROBABILITY_ATTRIBUTES
from .stats_utils import finite, hedges_g, mann_whitney_p, mean_summary, rounded


def _value(record: dict[str, Any], attribute: str, requested: Any) -> str | None:
    if requested is None:
        return None
    return (_detected_value(record, attribute) if attribute in MAIN_ATTRIBUTES
            else _detected_prompt_value(record, requested))


LABEL_ALIASES = {
    "gender": {"man": "male", "woman": "female"},
    # DeepFace reports "latino hispanic" for the prompted "hispanic".
    "race": {"latino hispanic": "hispanic"},
}


def _label(attribute: str, value: Any) -> str:
    """Comparable label: prompt words and detector labels on one vocabulary."""
    label = str(value).strip().lower()
    return LABEL_ALIASES.get(attribute, {}).get(label, label)


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
        correct += int(requested_label == detected_label)
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


def _accuracy_counts(records: list[dict[str, Any]], attributes: list[str]) -> dict[str, Any]:
    """Correct and evaluated images per attribute, so intervals can be computed."""
    result = {}
    for attribute in attributes:
        measurement = _measurement(records, attribute)
        correct = sum(row.get(label, 0) for label, row in measurement["counts"].items())
        result[attribute] = {"correct": correct, "total": measurement["total"],
                             "unmatched": measurement["unmatched"]}
    return result


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
            value: {"n_images": len(records), "accuracy": _accuracies(records, attributes),
                    "counts": _accuracy_counts(records, attributes)}
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


def _age_numeric(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Detected age in years per requested age term, without the age cut-offs.

    The categorical measurement counts a hit only when the detected age falls
    in the requested bin. This one shows where each term actually lands and
    whether neighbouring terms are told apart at all.
    """
    groups: defaultdict[str, list[float]] = defaultdict(list)
    baseline: list[float] = []
    for record in data:
        age = finite(record.get("detected_age"))
        if age is None:
            continue
        features = record.get("prompt_feats")
        requested = features.get("age") if isinstance(features, dict) else None
        if requested is None:
            baseline.append(age)
        else:
            groups[_label("age", requested)].append(age)
    order = [term for term in AGE_GROUPS if term in groups]

    def summary(values: list[float]) -> dict[str, Any]:
        return rounded({**mean_summary(values), "values": values})

    separation = []
    for lower, higher in zip(order, order[1:]):
        effect = hedges_g(groups[higher], groups[lower])
        separation.append(rounded({
            "lower": lower,
            "higher": higher,
            "difference": mean_summary(groups[higher])["mean"] - mean_summary(groups[lower])["mean"],
            "hedges_g": effect["g"],
            "g_ci_low": effect["ci_low"],
            "g_ci_high": effect["ci_high"],
            "p_value": mann_whitney_p(groups[higher], groups[lower]),
        }))
    return {
        "cutoffs": list(AGE_CUTOFFS),
        "order": order,
        "groups": {term: summary(groups[term]) for term in order},
        "not_requested": summary(baseline),
        "separation": separation,
    }


def _probability_adherence(data: list[dict[str, Any]], attribute: str) -> dict[str, Any]:
    """Adherence that uses the class probabilities instead of only the top class.

    Per requested value: the usual accuracy, the mean probability given to the
    requested class (``soft_accuracy``) and the accuracy on images whose top
    class is confident. Values the detector has no class for are left out.
    """
    rows: defaultdict[str, list[tuple[bool, float, float]]] = defaultdict(list)
    for record in data:
        features = record.get("prompt_feats")
        requested = features.get(attribute) if isinstance(features, dict) else None
        probabilities = (record.get("face_probs") or {}).get(attribute)
        detected = _detected_value(record, attribute)
        if requested is None or detected is None or not probabilities:
            continue
        requested_label = _label(attribute, requested)
        if requested_label not in probabilities:
            continue
        rows[requested_label].append((
            _label(attribute, detected) == requested_label,
            probabilities[requested_label],
            max(probabilities.values()),
        ))

    def summary(items: list[tuple[bool, float, float]]) -> dict[str, Any]:
        confident = [correct for correct, _, top in items if top >= CONFIDENT_PROBABILITY]
        return rounded({
            "n": len(items),
            "accuracy": sum(correct for correct, _, _ in items) / len(items),
            "soft_accuracy": sum(probability for _, probability, _ in items) / len(items),
            "n_confident": len(confident),
            "confident_accuracy": sum(confident) / len(confident) if confident else None,
        })

    everything = [item for items in rows.values() for item in items]
    return {
        "threshold": CONFIDENT_PROBABILITY,
        "overall": summary(everything) if everything else None,
        "by_value": {value: summary(items) for value, items in sorted(rows.items())},
    }


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
        "by_prompt_mode": {mode: {"n_images": len(records),
                                  "accuracy": _accuracies(records, detectable),
                                  "counts": _accuracy_counts(records, detectable)}
                           for mode, records in sorted(modes.items())},
        "by_demographic_category": _by_category(data, detectable),
        "combinations": _combinations(data, detectable),
        "age_numeric": _age_numeric(data),
        "probabilities": {attribute: _probability_adherence(data, attribute)
                          for attribute in PROBABILITY_ATTRIBUTES},
    }
