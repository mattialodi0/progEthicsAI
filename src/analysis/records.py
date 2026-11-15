"""Accessors for the fields of one preprocessed image record.

Every chapter builder reads detected demographics, embeddings and prompt
modes through these helpers, so labels are normalized the same way
everywhere ("man" -> "male", ages in years -> age groups).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import isfinite, sqrt
import re
from typing import Any

from .matching import word_pattern


MAIN_ATTRIBUTES = ("gender", "race", "age", "emotion")
# Age groups in increasing order, and the detected ages (years) that separate them.
AGE_GROUPS = ("young", "middle-aged", "old")
AGE_CUTOFFS = (30, 50)
# A detected class is called confident when its probability reaches this value.
CONFIDENT_PROBABILITY = 0.6


def age_label(age: Any) -> str | None:
    """Age group of a detected age in years (or of an age-group word)."""
    if isinstance(age, str):
        value = age.strip().lower()
        if value in AGE_GROUPS:
            return value
    if isinstance(age, (int, float)) and not isinstance(age, bool):
        if age < AGE_CUTOFFS[0]:
            return "young"
        if age < AGE_CUTOFFS[1]:
            return "middle-aged"
        return "old"
    return None


def top_probability(record: dict[str, Any], attribute: str) -> float | None:
    """Probability of the most likely class of a face attribute (see preprocessing)."""
    probabilities = record.get("face_probs")
    scores = probabilities.get(attribute) if isinstance(probabilities, dict) else None
    return max(scores.values()) if isinstance(scores, dict) and scores else None


def canonical_demographic_value(attribute: str, value: Any) -> str | None:
    """Lower-cased label, with gender nouns mapped to male / female."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if attribute == "gender":
        return {"man": "male", "woman": "female"}.get(normalized, normalized)
    return normalized


def detected_value(record: dict[str, Any], attribute: str) -> str | None:
    """Most common detected value of ``attribute`` over the valid faces."""
    faces = record.get("facial_feats")
    if not isinstance(faces, (list, tuple)):
        return None

    values: list[str] = []
    for face in faces:
        if not isinstance(face, dict) or face.get("face_confidence", -1) == -1:
            continue
        if attribute == "age":
            value = age_label(face.get("age"))
        else:
            raw = face.get(f"dominant_{attribute}")
            value = canonical_demographic_value(attribute, raw)
        if value:
            values.append(value)
    return Counter(values).most_common(1)[0][0] if values else None


def detected_prompt_value(record: dict[str, Any], prompt_value: Any) -> str | None:
    """Find non-face attributes in Florence labels or the generated caption."""
    if prompt_value is None:
        return None
    candidate = str(prompt_value).strip().lower()
    florence = record.get("items_florence", {})
    labels = florence.get("labels", []) if isinstance(florence, dict) else []
    text = " ".join(str(label).lower() for label in labels)
    caption = record.get("caption")
    if isinstance(caption, str):
        text = f"{text} {caption.lower()}"
    return candidate if re.search(word_pattern(candidate), text) else None


def image_embedding(record: dict[str, Any]) -> list[float] | None:
    """The CLIP image embedding as a list of finite floats, or ``None``."""
    value = record.get("image_embedding")
    if value is None:
        return None
    # Supports numpy tensors/arrays without making numpy a required dependency.
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(isfinite(item) for item in result) else None


def euclidean_distance(first: list[float], second: list[float]) -> float:
    """Euclidean distance over the common dimensions of two vectors."""
    length = min(len(first), len(second))
    if not length:
        return 0.0
    return sqrt(sum((first[index] - second[index]) ** 2 for index in range(length)))


def record_mode(record: dict[str, Any], config: dict[str, Any]) -> str:
    """The prompt-mode label of a record (e.g. "single variable")."""
    labels = config.get("prompt_modes", {})
    raw = str(record.get("mode", "unknown"))
    return str(labels.get(raw, raw)) if isinstance(labels, dict) else raw


def object_labels(record: dict[str, Any]) -> list[str]:
    """Distinct lower-cased Florence-2 object labels of the image."""
    items = record.get("items_florence", {})
    labels = items.get("labels", []) if isinstance(items, dict) else []
    return sorted({str(label).strip().lower() for label in labels if str(label).strip()})


def group_records(
    data: list[dict[str, Any]], group_by: str, config: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Records grouped by prompt mode, detected gender or a prompt attribute."""
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in data:
        if group_by == "prompt_mode":
            value = record_mode(record, config)
        elif group_by == "demographic_group":
            value = detected_value(record, "gender")
        else:
            features = record.get("prompt_feats", {})
            value = features.get(group_by) if isinstance(features, dict) else None
        if value is not None:
            groups[str(value).lower()].append(record)
    return dict(sorted(groups.items()))
