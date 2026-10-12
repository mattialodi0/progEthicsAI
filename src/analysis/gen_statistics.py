"""General statistics extracted from preprocessed image metadata.

The functions in this module accept the dictionaries produced by
``preprocessing.preprocess_data``.  They also tolerate partially populated
records so that one malformed model result does not discard the whole report.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from math import sqrt
import re
from typing import Any


_ATTRIBUTES = ("gender", "race", "age", "emotion")
_PROMPT_ATTRIBUTES = (
    "gender",
    "age",
    "race",
    "emotion",
    "job",
    "workplace",
    "continent",
    "status",
    "toxic_trait",
    "positive_trait",
)


def _records(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise TypeError("data must be a list of image metadata objects")
    return [item for item in data if isinstance(item, dict)]


def _percentage(count: int, total: int) -> float:
    return round(100.0 * count / total, 4) if total else 0.0


def _distribution(values: list[str], total: int | None = None) -> dict[str, Any]:
    counts = Counter(values)
    denominator = len(values) if total is None else total
    return {
        "counts": dict(sorted(counts.items())),
        "percentages": {
            key: _percentage(value, denominator) for key, value in sorted(counts.items())
        },
        "missing": max(0, denominator - len(values)),
    }


def _age_label(age: Any) -> str | None:
    if isinstance(age, str):
        value = age.strip().lower()
        if value in {"young", "middle-aged", "old"}:
            return value
    if isinstance(age, (int, float)) and not isinstance(age, bool):
        if age < 30:
            return "young"
        if age < 60:
            return "middle-aged"
        return "old"
    return None


def _detected_value(record: dict[str, Any], attribute: str) -> str | None:
    faces = record.get("facial_feats")
    if not isinstance(faces, (list, tuple)):
        return None

    values: list[str] = []
    for face in faces:
        if not isinstance(face, dict) or face.get("face_confidence", -1) == -1:
            continue
        if attribute == "age":
            value = _age_label(face.get("age"))
        else:
            raw = face.get(f"dominant_{attribute}")
            value = str(raw).strip().lower() if raw is not None else None
            if attribute == "gender" and value in {"man", "woman"}:
                value = "male" if value == "man" else "female"
        if value:
            values.append(value)
    return Counter(values).most_common(1)[0][0] if values else None


def _detected_prompt_value(record: dict[str, Any], prompt_value: Any) -> str | None:
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
    pattern = r"(?<!\w)" + re.escape(candidate).replace(r"\ ", r"\s+") + r"(?!\w)"
    return candidate if re.search(pattern, text) else None


def _embedding(record: dict[str, Any]) -> list[float] | None:
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
    return result if all(map(lambda item: item == item, result)) else None


def _distance(first: list[float], second: list[float]) -> float:
    length = min(len(first), len(second))
    if not length:
        return 0.0
    return sqrt(sum((first[index] - second[index]) ** 2 for index in range(length)))


def _mean_pairwise_distance(vectors: list[list[float]]) -> float | None:
    distances = [_distance(left, right) for left, right in combinations(vectors, 2)]
    return round(sum(distances) / len(distances), 6) if distances else None


def collect_all_stats(
    data: list[dict[str, Any]],
    verbose: int = 0,
    include_embedding_metrics: bool = True,
) -> dict[str, Any]:
    """Return every supported statistic in one JSON-serialisable dictionary."""
    records = _records(data)
    result = {
        "invalid_images": invalid_images(records),
        "demographics": demographics(records),
        "prompt_adherence": prompt_adherence(records),
    }
    if include_embedding_metrics:
        result["diversity"] = diversity(records)
        result["cluster_separability"] = cluster_separability(records)
    if verbose > 1:
        print(f"Collected statistics for {len(records)} images.")
    return result


def invalid_images(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Count images without a valid detected face and identify them."""
    records = _records(data)
    invalid = [
        (index, record.get("name"))
        for index, record in enumerate(records)
        if record.get("valid") is not True
    ]
    return {
        "count": len(invalid),
        "total": len(records),
        "rate": _percentage(len(invalid), len(records)),
        "indices": [index for index, _ in invalid],
        "names": [name for _, name in invalid if name is not None],
    }


def demographics(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Return detected gender, race, age and emotion distributions."""
    records = _records(data)
    result: dict[str, Any] = {}
    for attribute in _ATTRIBUTES:
        values = [
            value for record in records
            if (value := _detected_value(record, attribute)) is not None
        ]
        result[attribute] = _distribution(values, len(records))
    return result


def diversity(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure mean pairwise embedding distance within each prompt group.

    ``coverage`` is the fraction of non-empty prompt groups represented in the
    data.  ``mean_pairwise_distance`` is ``None`` when a group has fewer than
    two usable image embeddings.
    """
    records = _records(data)
    result: dict[str, Any] = {}
    for attribute in _PROMPT_ATTRIBUTES:
        groups: defaultdict[str, list[list[float]]] = defaultdict(list)
        for record in records:
            prompt_feats = record.get("prompt_feats", {})
            group = prompt_feats.get(attribute) if isinstance(prompt_feats, dict) else None
            vector = _embedding(record)
            if group is not None and vector is not None and record.get("valid") is not False:
                groups[str(group).lower()].append(vector)
        result[attribute] = {
            "groups": {
                group: {
                    "n_images": len(vectors),
                    "mean_pairwise_distance": _mean_pairwise_distance(vectors),
                }
                for group, vectors in sorted(groups.items())
            },
            "coverage": _percentage(len(groups), len({
                str(record.get("prompt_feats", {}).get(attribute)).lower()
                for record in records
                if isinstance(record.get("prompt_feats"), dict)
                and record["prompt_feats"].get(attribute) is not None
            })),
        }
    return result


def prompt_adherence(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Build prompt/detection confusion matrices for every supported attribute."""
    records = _records(data)
    result: dict[str, Any] = {}
    for attribute in _PROMPT_ATTRIBUTES:
        matrix: defaultdict[str, Counter[str]] = defaultdict(Counter)
        total = correct = 0
        for record in records:
            prompt_feats = record.get("prompt_feats", {})
            prompt = prompt_feats.get(attribute) if isinstance(prompt_feats, dict) else None
            detected = (
                _detected_value(record, attribute)
                if attribute in _ATTRIBUTES
                else _detected_prompt_value(record, prompt)
            )
            if prompt is None or detected is None:
                continue
            prompt = str(prompt).lower()
            matrix[prompt][detected] += 1
            total += 1
            correct += prompt == detected or (
                attribute == "race" and prompt == "hispanic" and detected == "latino hispanic"
            )
        labels = sorted({label for row in matrix.values() for label in row} | set(matrix))
        result[attribute] = {
            "labels": labels,
            "matrix": {row: {column: matrix[row].get(column, 0) for column in labels}
                       for row in labels},
            "total": total,
            "accuracy": round(correct / total, 6) if total else None,
            "unmatched": len(records) - total,
        }
    return result


def cluster_separability(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate embedding cluster separability using the silhouette coefficient."""
    records = _records(data)
    result: dict[str, Any] = {}
    for attribute in _PROMPT_ATTRIBUTES:
        groups: defaultdict[str, list[list[float]]] = defaultdict(list)
        for record in records:
            prompt_feats = record.get("prompt_feats", {})
            value = prompt_feats.get(attribute) if isinstance(prompt_feats, dict) else None
            vector = _embedding(record)
            if value is not None and vector is not None and record.get("valid") is not False:
                groups[str(value).lower()].append(vector)
        points = [(group, vector) for group, vectors in groups.items() for vector in vectors]
        scores: list[float] = []
        if len(groups) > 1:
            for group, vector in points:
                own = [_distance(vector, other) for other in groups[group] if other is not vector]
                a = sum(own) / len(own) if own else 0.0
                b = min(
                    sum(_distance(vector, other) for other in vectors) / len(vectors)
                    for other_group, vectors in groups.items() if other_group != group
                )
                scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
        result[attribute] = {
            "score": round(sum(scores) / len(scores), 6) if scores else None,
            "n_images": len(points),
            "n_clusters": len(groups),
            "cluster_sizes": {group: len(vectors) for group, vectors in sorted(groups.items())},
        }
    return result
