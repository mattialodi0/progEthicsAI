"""Chapter 3 data: demographic composition and output diversity."""

from __future__ import annotations

from collections import Counter
from math import log
from typing import Any

from .records import detected_value, group_records, image_embedding, object_labels, record_mode


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


def _pairwise_embedding_stats(
    vectors: list[list[float]],
) -> tuple[list[float], float | None]:
    """Return (sorted pairwise cosine similarities, mean pairwise Euclidean distance).

    Computed with numpy over all pairs: similarities are clamped to [-1, 1]
    and rounded to 6 decimals, and pairs involving a zero-norm vector are
    skipped for similarity.
    """
    if len(vectors) < 2:
        return [], None
    import numpy as np

    dimensions = min(len(vector) for vector in vectors)
    matrix = np.asarray([vector[:dimensions] for vector in vectors], dtype=float)
    left, right = np.triu_indices(len(matrix), k=1)

    distance_sum = 0.0
    for index in range(len(matrix) - 1):
        distance_sum += float(np.sqrt(((matrix[index + 1:] - matrix[index]) ** 2).sum(axis=1)).sum())
    dispersion = round(distance_sum / len(left), 6)

    norms = np.sqrt((matrix * matrix).sum(axis=1))
    nonzero = norms > 0
    keep = nonzero[left] & nonzero[right]
    safe_norms = np.where(nonzero, norms, 1.0)
    cosine = (matrix @ matrix.T) / safe_norms[:, None] / safe_norms[None, :]
    similarities = np.round(np.clip(cosine[left[keep], right[keep]], -1.0, 1.0), 6)
    similarities.sort()
    return similarities.tolist(), dispersion


def _embedding_group_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    vectors = [vector for record in records if (vector := image_embedding(record)) is not None]
    similarities, dispersion = _pairwise_embedding_stats(vectors)
    median = (
        similarities[len(similarities) // 2]
        if similarities
        else None
    )
    return {
        "n_images": len(records),
        "n_embeddings": len(vectors),
        "embedding_dispersion": dispersion,
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
                if (value := detected_value(record, attribute)) is not None
            ])
            for attribute in ("gender", "race", "age", "emotion")
        },
        "caption_unique": len({
            str(record["caption"]).strip().lower()
            for record in records
            if isinstance(record.get("caption"), str) and record["caption"].strip()
        }),
        "object_unique": len({
            label for record in records for label in object_labels(record)
        }),
    }


def diversity_metrics(
    data: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Precompute categorical and semantic diversity for report presentation."""
    mode_records = group_records(data, "prompt_mode", config)

    def group_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
        stats = _embedding_group_stats(records)
        stats.update({
            "caption_entropy": _entropy([
                caption for record in records
                if isinstance((caption := record.get("caption")), str) and caption.strip()
            ]),
            "object_entropy": _entropy([
                label for record in records for label in object_labels(record)
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
                        if record_mode(record, config).lower() == mode
                    ])
                    for mode in mode_records
                    if any(
                        record_mode(record, config).lower() == mode
                        for record in records
                    )
                },
            }
            for group, records in group_records(data, group_by, config).items()
        }
    mode_labels = {record_mode(record, config) for record in data}
    # Reuse the per-mode statistics computed above instead of recomputing the
    # O(n^2) embedding statistics for every threshold. A threshold is shown
    # when at least one prompt-mode group is large enough to reach it.
    mode_sizes = {mode: len(records) for mode, records in mode_records.items()}
    mode_dispersion = {
        mode: groups["prompt_mode"][mode]["embedding_dispersion"] or 0.0
        for mode in mode_records
    }
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
                    mode_dispersion[mode]
                    for mode, size in mode_sizes.items() if size >= count
                ) / max(1, sum(size >= count for size in mode_sizes.values())), 6),
            }
            for count in (1, 2, 5, 10)
            if any(size >= count for size in mode_sizes.values())
        ],
        "available_prompt_modes": sorted(mode_labels),
    }


def demographic_composition(
    data: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Precompute requested/generated demographic composition by useful slices."""
    dimensions = ("gender", "race", "age", "emotion")
    slices: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for slice_name in ("prompt_mode", "job", "workplace"):
        slices[slice_name] = group_records(data, slice_name, config)

    def requested(record: dict[str, Any], dimension: str) -> bool:
        features = record.get("prompt_feats")
        return isinstance(features, dict) and features.get(dimension) is not None

    result: dict[str, Any] = {}
    for dimension in dimensions:
        detected = [
            value for record in data
            if (value := detected_value(record, dimension)) is not None
        ]
        # The model's default: only images whose prompt does not request this
        # dimension, so an explicit "female doctor" does not count as a choice.
        unrequested = [record for record in data if not requested(record, dimension)]
        result[dimension] = {
            "detected": _distribution(detected),
            "by_slice": {
                slice_name: {
                    group: {
                        "detected": _distribution([
                            value for record in records
                            if (value := detected_value(record, dimension)) is not None
                        ]),
                    }
                    for group, records in groups.items()
                }
                for slice_name, groups in slices.items()
            },
            "unrequested": {
                "n_images": len(unrequested),
                "detected": _distribution([
                    value for record in unrequested
                    if (value := detected_value(record, dimension)) is not None
                ]),
                "by_slice": {
                    slice_name: {
                        group: {
                            "n_images": len(subset),
                            "detected": _distribution([
                                value for record in subset
                                if (value := detected_value(record, dimension)) is not None
                            ]),
                        }
                        for group, records in groups.items()
                        if (subset := [record for record in records
                                       if not requested(record, dimension)])
                    }
                    for slice_name, groups in slices.items()
                },
            },
        }
    return result


def invalid_images(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Count images without a valid detected face and identify them."""
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
