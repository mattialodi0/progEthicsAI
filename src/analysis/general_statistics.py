from __future__ import annotations
from collections import Counter, defaultdict
from itertools import combinations
from math import isfinite, log, sqrt
import re
from typing import Any


MAIN_ATTRIBUTES = ("gender", "race", "age", "emotion")
ADHERENCE_ATTRIBUTES = MAIN_ATTRIBUTES


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
        if age < 50:
            return "middle-aged"
        return "old"
    return None


def _canonical_demographic_value(attribute: str, value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if attribute == "gender":
        return {"man": "male", "woman": "female"}.get(normalized, normalized)
    return normalized


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
            value = _canonical_demographic_value(attribute, raw)
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
    return result if all(isfinite(item) for item in result) else None

def _distance(first: list[float], second: list[float]) -> float:
    length = min(len(first), len(second))
    if not length:
        return 0.0
    return sqrt(sum((first[index] - second[index]) ** 2 for index in range(length)))

def _mean_pairwise_distance(vectors: list[list[float]]) -> float | None:
    distances = [_distance(left, right) for left, right in combinations(vectors, 2)]
    return round(sum(distances) / len(distances), 6) if distances else None


def _cosine_similarity(first: list[float], second: list[float]) -> float | None:
    length = min(len(first), len(second))
    if not length:
        return None
    dot = sum(first[index] * second[index] for index in range(length))
    first_norm = sqrt(sum(value * value for value in first[:length]))
    second_norm = sqrt(sum(value * value for value in second[:length]))
    if not first_norm or not second_norm:
        return None
    return round(max(-1.0, min(1.0, dot / (first_norm * second_norm))), 6)


def _entropy(values: list[str]) -> dict[str, float | int | None]:
    counts = Counter(values)
    total = len(values)
    categories = len(counts)
    entropy = (
        -sum((count / total) * log(count / total) for count in counts.values())
        if total
        else None
    )
    maximum = log(categories) if categories > 1 else 0.0
    return {
        "n": total,
        "categories": categories,
        "entropy": round(entropy, 6) if entropy is not None else None,
        "normalized_entropy": round(entropy / maximum, 6)
        if entropy is not None and maximum
        else (0.0 if entropy is not None else None),
        "simpson_diversity": round(
            1 - sum((count / total) ** 2 for count in counts.values()), 6
        )
        if total
        else None,
        "counts": dict(sorted(counts.items())),
    }


def _record_mode(record: dict[str, Any], config: dict[str, Any]) -> str:
    labels = config.get("prompt_modes", {})
    raw = str(record.get("mode", "unknown"))
    return str(labels.get(raw, raw)) if isinstance(labels, dict) else raw


def _caption_tokens(record: dict[str, Any]) -> list[str]:
    caption = record.get("caption")
    if not isinstance(caption, str):
        return []
    return sorted(set(re.findall(r"\b[a-z][a-z-]{2,}\b", caption.lower())))


def _object_labels(record: dict[str, Any]) -> list[str]:
    items = record.get("items_florence", {})
    labels = items.get("labels", []) if isinstance(items, dict) else []
    return sorted({str(label).strip().lower() for label in labels if str(label).strip()})


def _group_records(
    data: list[dict[str, Any]], group_by: str, config: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in data:
        if group_by == "prompt_mode":
            value = _record_mode(record, config)
        elif group_by == "demographic_group":
            value = _detected_value(record, "gender")
        else:
            features = record.get("prompt_feats", {})
            value = features.get(group_by) if isinstance(features, dict) else None
        if value is not None:
            groups[str(value).lower()].append(record)
    return dict(sorted(groups.items()))


def _embedding_group_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    vectors = [vector for record in records if (vector := _embedding(record)) is not None]
    similarities = [
        similarity
        for first, second in combinations(vectors, 2)
        if (similarity := _cosine_similarity(first, second)) is not None
    ]
    similarities.sort()
    median = (
        similarities[len(similarities) // 2]
        if similarities
        else None
    )
    return {
        "n_images": len(records),
        "n_embeddings": len(vectors),
        "embedding_dispersion": _mean_pairwise_distance(vectors),
        "pairwise_similarity": {
            "n_pairs": len(similarities),
            "mean": round(sum(similarities) / len(similarities), 6) if similarities else None,
            "minimum": min(similarities) if similarities else None,
            "median": median,
            "maximum": max(similarities) if similarities else None,
            "values": similarities,
        },
        "demographic_entropy": {
            attribute: _entropy([
                value for record in records
                if (value := _detected_value(record, attribute)) is not None
            ])
            for attribute in ("gender", "race", "age", "emotion")
        },
        "caption_unique": len({
            str(record["caption"]).strip().lower()
            for record in records
            if isinstance(record.get("caption"), str) and record["caption"].strip()
        }),
        "object_unique": len({
            label for record in records for label in _object_labels(record)
        }),
    }


def diversity_metrics(
    data: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Precompute categorical and semantic diversity for report presentation."""
    mode_records = _group_records(data, "prompt_mode", config)

    def group_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
        stats = _embedding_group_stats(records)
        stats.update({
            "caption_entropy": _entropy([
                caption for record in records
                if isinstance((caption := record.get("caption")), str) and caption.strip()
            ]),
            "object_entropy": _entropy([
                label for record in records for label in _object_labels(record)
            ]),
        })
        return stats

    groups: dict[str, dict[str, Any]] = {}
    for group_by in ("prompt_mode", "job", "workplace", "demographic_group"):
        groups[group_by] = {
            group: {
                **group_stats(records),
                "by_prompt_mode": {
                    mode: group_stats([
                        record for record in records
                        if _record_mode(record, config).lower() == mode
                    ])
                    for mode in mode_records
                    if any(
                        _record_mode(record, config).lower() == mode
                        for record in records
                    )
                },
            }
            for group, records in _group_records(data, group_by, config).items()
        }
    mode_labels = {_record_mode(record, config) for record in data}
    return {
        "groups": groups,
        "rare_categories": {
            attribute: [
                {"value": value, "count": count}
                for value, count in sorted(Counter(
                    str(record.get("prompt_feats", {}).get(attribute)).lower()
                    for record in data
                    if isinstance(record.get("prompt_feats"), dict)
                    and record["prompt_feats"].get(attribute) is not None
                ).items(), key=lambda item: (item[1], item[0]))
                if count == 1
            ]
            for attribute in ("job", "workplace", "gender", "race", "age")
        },
        "by_sample_count": [
            {
                "sample_count": count,
                "mean_embedding_dispersion": round(sum(
                    _embedding_group_stats(records)["embedding_dispersion"] or 0.0
                    for records in _group_records(data, "prompt_mode", config).values()
                    if len(records) >= count
                ) / max(1, sum(
                    len(records) >= count
                    for records in _group_records(data, "prompt_mode", config).values()
                )), 6),
            }
            for count in sorted({1, 2, 5, 10} & {
                len(records) for records in _group_records(data, "prompt_mode", config).values()
            })
        ],
        "available_prompt_modes": sorted(mode_labels),
    }


def collect_general_stats(
    data: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return every supported statistic in one JSON-serialisable dictionary."""

    verbose = config.get('verbose', 0)
    attributes = config.get('prompt_variables', {}).keys() if isinstance(config.get('prompt_variables'), dict) else []

    result = {
        "invalid_images": invalid_images(data),
        "demographics": demographics(data),
        "demographic_composition": demographic_composition(data, config),
        "prompt_adherence": prompt_adherence(data, list(attributes), config),
        "diversity": diversity(data, attributes),
        "diversity_metrics": diversity_metrics(data, config),
        "cluster_separability": cluster_separability(data, attributes),
    }

    if verbose > 1:
        print(f"Collected statistics for {len(data)} images.")
    return result


def demographic_composition(
    data: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Precompute requested/generated demographic composition by useful slices."""
    dimensions = ("gender", "race", "age", "emotion")
    slices: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for slice_name in ("prompt_mode", "job", "workplace"):
        slices[slice_name] = _group_records(data, slice_name, config)

    result: dict[str, Any] = {}
    for dimension in dimensions:
        detected = [
            value for record in data
            if (value := _detected_value(record, dimension)) is not None
        ]
        result[dimension] = {
            "detected": _distribution(detected),
            "by_slice": {
                slice_name: {
                    group: {
                        "detected": _distribution([
                            value for record in records
                            if (value := _detected_value(record, dimension)) is not None
                        ]),
                    }
                    for group, records in groups.items()
                }
                for slice_name, groups in slices.items()
            },
        }
    return result

def invalid_images(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Count images without a valid detected face and identify them."""
    data = data
    invalid = [
        (index, record.get("name"))
        for index, record in enumerate(data)
        if record.get("valid") is not True
    ]
    return {
        "count": len(invalid),
        "total": len(data),
        "indices": [index for index, _ in invalid],
    }

def demographics(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Return detected gender, race, age and emotion distributions."""
    data = data
    result: dict[str, Any] = {}
    for attribute in MAIN_ATTRIBUTES:
        values = [
            value for record in data
            if (value := _detected_value(record, attribute)) is not None
        ]
        result[attribute] = _distribution(values, len(data))
    return result

def diversity(data: list[dict[str, Any]], attributes: list[str]) -> dict[str, Any]:
    """Measure mean pairwise embedding distance within each prompt group.

    ``coverage`` is the fraction of non-empty prompt groups represented in the
    data.  ``mean_pairwise_distance`` is ``None`` when a group has fewer than
    two usable image embeddings.
    """
    result: dict[str, Any] = {}
    for attribute in attributes:
        groups: defaultdict[str, list[list[float]]] = defaultdict(list)
        for record in data:
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
                for record in data
                if isinstance(record.get("prompt_feats"), dict)
                and record["prompt_feats"].get(attribute) is not None
            })),
        }
    return result

from .prompt_adherence import _value as _adherence_value
from .prompt_adherence import prompt_adherence
from .embedding_metrics import cluster_separability
