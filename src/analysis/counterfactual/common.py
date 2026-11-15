"""Attributes and prompt values shared by the chapter-6 analyses."""

from __future__ import annotations

from typing import Any

from ..records import MAIN_ATTRIBUTES, canonical_demographic_value


NUMERICAL_FEATURES = ("n_people", "person_area_ratio", "face_area_ratio", "detected_age")
CONTROL_ATTRIBUTES = ("gender", "race", "age", "job")
DETECTED_ATTRIBUTES = MAIN_ATTRIBUTES


def condition_value(record: dict[str, Any], attribute: str) -> str | None:
    """The requested value of ``attribute`` (gender words mapped to male / female)."""
    features = record.get("prompt_feats")
    value = features.get(attribute) if isinstance(features, dict) else None
    if value is None:
        return None
    if attribute == "gender":
        return canonical_demographic_value(attribute, value)
    return str(value).strip().lower() or None
