"""Embedding-based diversity and separability measurements."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .general_statistics import _distance, _embedding


def cluster_separability(data: list[dict[str, Any]], attributes: list[str]) -> dict[str, Any]:
    """Estimate silhouette-style cluster separation for each prompt attribute."""
    result: dict[str, Any] = {}
    for attribute in attributes:
        groups: defaultdict[str, list[list[float]]] = defaultdict(list)
        for record in data:
            features = record.get("prompt_feats")
            value = features.get(attribute) if isinstance(features, dict) else None
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
