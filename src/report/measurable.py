"""Requested values that a detector can never report.

Adherence compares a requested prompt value with a detected label. When the
detector's label set has no such value (DeepFace has no "bored" emotion, a job
name that never appears in any caption), adherence is 0% by construction. Such
a 0% says nothing about the model, so the report shows these values as
"n/a" instead of as a real result.
"""

from __future__ import annotations

from typing import Any, Iterable

from .plotly_helpers import numeric

NA_TEXT = "n/a: not in detector vocabulary"


def detected_labels(counts: dict[str, dict[str, Any]]) -> set[str]:
    """Labels that the detector produced at least once in a confusion matrix."""
    return {column for row in counts.values() for column, value in row.items()
            if numeric(value) > 0}


def unmeasurable_from_counts(counts: dict[str, dict[str, Any]]) -> set[str]:
    """Requested rows whose own label never appears among the detections."""
    detected = detected_labels(counts)
    return {row for row, values in counts.items()
            if sum(numeric(value) for value in values.values()) > 0 and row not in detected}


def unmeasurable_from_transitions(item: dict[str, Any]) -> set[str]:
    """Chapter 6: requested values with 0% adherence that no pair ever detected."""
    transitions = (item.get("detected_changes") or {}).get("transitions") or {}
    detected = {side.strip() for key in transitions for side in str(key).split("->")}
    detected.discard("missing")
    return {value for value, stats in (item.get("adherence") or {}).items()
            if numeric(stats.get("rate")) == 0 and value not in detected}


def measurable_accuracy(counts: dict[str, dict[str, Any]],
                        excluded: Iterable[str]) -> float | None:
    """Adherence over the requested rows that the detector can actually report."""
    excluded = set(excluded)
    correct = total = 0.0
    for row, values in counts.items():
        if row in excluded:
            continue
        total += sum(numeric(value) for value in values.values())
        correct += numeric(values.get(row))
    return correct / total if total else None


def unmeasurable_by_attribute(adherence: dict[str, Any]) -> dict[str, set[str]]:
    """Chapter 4 payload -> {attribute: requested values the detector cannot output}."""
    return {
        attribute: unmeasurable_from_counts(item.get("counts") or {})
        for attribute, item in (adherence.get("by_attribute") or {}).items()
    }
