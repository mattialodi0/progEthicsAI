"""Chapter 7 data: image-feature distributions and embedding-space analysis."""

from __future__ import annotations

from typing import Any

from .general_statistics import _embedding, _record_mode
from .stats_utils import hedges_g, kruskal, mann_whitney_p, mean_summary, rounded

IMAGE_FEATURES = (
    "brightness", "saturation", "contrast", "n_people",
    "person_area_ratio", "face_area_ratio",
)
MAX_DISTANCE_POINTS = 600
# Label permutations behind the silhouette null distribution.
SILHOUETTE_PERMUTATIONS = 40
# Groups smaller than this are left out of the feature tests and the probe.
MIN_TEST_GROUP = 5
PROBE_FOLDS = 5


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _feature_value(record: dict[str, Any], feature: str) -> float | None:
    if feature in {"brightness", "saturation", "contrast"}:
        metrics = record.get("image_metrics")
        return _number(metrics.get(feature) if isinstance(metrics, dict) else None)
    return _number(record.get(feature))


def _group_value(record: dict[str, Any], group_by: str, config: dict[str, Any]) -> str | None:
    if group_by == "prompt_mode":
        return _record_mode(record, config)
    features = record.get("prompt_feats")
    value = features.get(group_by) if isinstance(features, dict) else None
    return str(value).strip().lower() if value is not None and str(value).strip() else None


def _pca(vectors: list[list[float]]) -> list[list[float]] | None:
    if len(vectors) < 2:
        return None
    dimensions = min(len(vector) for vector in vectors)
    if not dimensions:
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        matrix = np.asarray([vector[:dimensions] for vector in vectors], dtype=float)
        matrix -= matrix.mean(axis=0)
        _, _, components = np.linalg.svd(matrix, full_matrices=False)
        projection = matrix @ components[:2].T
    except (np.linalg.LinAlgError, ValueError):
        return None
    if projection.shape[1] == 1:
        projection = np.column_stack((projection[:, 0], np.zeros(len(vectors))))
    return [[round(float(x), 6), round(float(y), 6)] for x, y in projection]


def _umap(vectors: list[list[float]]) -> list[list[float]] | None:
    if len(vectors) < 3:
        return None
    try:
        import umap
        dimensions = min(len(vector) for vector in vectors)
        projection = umap.UMAP(
            n_components=2, n_neighbors=max(2, min(15, len(vectors) - 1)),
            random_state=42,
        ).fit_transform([vector[:dimensions] for vector in vectors])
    except (ImportError, ValueError):
        return None
    return [[round(float(x), 6), round(float(y), 6)] for x, y in projection]


def _sample_indices(size: int, max_points: int = MAX_DISTANCE_POINTS) -> list[int]:
    """Deterministic, evenly spaced subsample that bounds O(n^2) work."""
    if size <= max_points:
        return list(range(size))
    return [int(index * size / max_points) for index in range(max_points)]


def _distance_matrix(vectors: list[list[float]]) -> Any:
    """Exact pairwise Euclidean distances, row by row to bound memory."""
    import numpy as np

    dimensions = min(len(vector) for vector in vectors)
    matrix = np.asarray([vector[:dimensions] for vector in vectors], dtype=float)
    distances = np.empty((len(matrix), len(matrix)))
    for index, row in enumerate(matrix):
        distances[index] = np.sqrt(((matrix - row) ** 2).sum(axis=1))
    return distances


def _distance_distributions(distances: Any, labels: list[str | None]) -> dict[str, Any]:
    import numpy as np

    left, right = np.triu_indices(len(labels), k=1)
    values = np.round(distances[left, right], 6)
    label_array = np.asarray([str(label) if label is not None else "\0none" for label in labels])
    same = label_array[left] == label_array[right]
    return {"within": values[same].tolist(), "between": values[~same].tolist()}


def _silhouette(distances: Any, labels: list[str | None]) -> float | None:
    """Mean silhouette over labelled points, using a precomputed distance matrix."""
    import numpy as np

    groups = sorted({label for label in labels if label is not None})
    if len(groups) < 2:
        return None
    members = {
        group: np.asarray([index for index, label in enumerate(labels) if label == group])
        for group in groups
    }
    # Mean distance from every point to every group (self-distance is 0).
    sums = {group: distances[:, indices].sum(axis=1) for group, indices in members.items()}
    scores: list[float] = []
    for index, label in enumerate(labels):
        if label is None:
            continue
        own_size = len(members[label]) - 1
        a = float(sums[label][index] / own_size) if own_size else 0.0
        b = min(
            float(sums[group][index] / len(members[group]))
            for group in groups if group != label
        )
        scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return round(sum(scores) / len(scores), 6) if scores else None


def _feature_tests(values_by_group: dict[str, list[float]]) -> dict[str, Any]:
    """Kruskal–Wallis across groups and each group against the other groups."""
    groups = {group: values for group, values in values_by_group.items()
              if len(values) >= MIN_TEST_GROUP}
    if len(groups) < 2:
        return {}
    omnibus = kruskal(list(groups.values()))
    result_groups = {}
    for group, values in groups.items():
        rest = [value for other, items in groups.items() if other != group for value in items]
        summary = mean_summary(values)
        effect = hedges_g(values, rest)
        result_groups[group] = rounded({
            **summary,
            "rest_mean": mean_summary(rest)["mean"],
            "rest_n": len(rest),
            "hedges_g": effect["g"],
            "g_ci_low": effect["ci_low"],
            "g_ci_high": effect["ci_high"],
            "p_value": mann_whitney_p(values, rest),
        })
    return {**rounded(omnibus), "groups": result_groups}


def _silhouette_null(
    distances: Any, labels: list[str | None], observed: float | None
) -> dict[str, Any]:
    """Silhouette under randomly permuted labels (same group sizes)."""
    if observed is None:
        return {}
    import numpy as np

    generator = np.random.default_rng(0)
    positions = [index for index, label in enumerate(labels) if label is not None]
    values = [labels[index] for index in positions]
    scores = []
    for _ in range(SILHOUETTE_PERMUTATIONS):
        shuffled = list(labels)
        for index, label in zip(positions, generator.permutation(values)):
            shuffled[index] = str(label)
        score = _silhouette(distances, shuffled)
        if score is not None:
            scores.append(score)
    if not scores:
        return {}
    mean = sum(scores) / len(scores)
    spread = (sum((score - mean) ** 2 for score in scores) / max(1, len(scores) - 1)) ** 0.5
    return rounded({
        "null_mean": mean,
        "null_sd": spread,
        "permutations": len(scores),
        "p_value": (1 + sum(score >= observed for score in scores)) / (1 + len(scores)),
    })


def _probe(vectors: list[list[float]], labels: list[str | None]) -> dict[str, Any]:
    """Cross-validated linear probe: can the group be read off the embedding?"""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
    except ImportError:
        return {}
    counts: dict[str, int] = {}
    for label in labels:
        if label is not None:
            counts[label] = counts.get(label, 0) + 1
    keep = {label for label, count in counts.items() if count >= max(MIN_TEST_GROUP, PROBE_FOLDS)}
    indices = [index for index, label in enumerate(labels) if label in keep]
    if len(keep) < 2:
        return {}
    dimensions = min(len(vectors[index]) for index in indices)
    matrix = np.asarray([vectors[index][:dimensions] for index in indices], dtype=float)
    target = np.asarray([labels[index] for index in indices])
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    try:
        predicted = cross_val_predict(
            model, matrix, target,
            cv=StratifiedKFold(n_splits=PROBE_FOLDS, shuffle=True, random_state=0),
        )
    except ValueError:
        return {}
    accuracy = float(balanced_accuracy_score(target, predicted))
    chance = 1 / len(keep)
    return rounded({
        "balanced_accuracy": accuracy,
        "chance": chance,
        # 0 = chance, 1 = perfect: comparable across groupings with different k.
        "above_chance": (accuracy - chance) / (1 - chance),
        "n_images": len(indices),
        "n_classes": len(keep),
        "folds": PROBE_FOLDS,
    })


def build_visual_feature_analysis(
    data: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Build JSON-safe data for all chapter-7 plots."""
    prompt_feature_values: dict[str, set[str]] = {}
    modes: set[str] = set()
    for record in data:
        modes.add(_record_mode(record, config))
        prompt_feats = record.get("prompt_feats")
        if not isinstance(prompt_feats, dict):
            continue
        for key, value in prompt_feats.items():
            if value is not None and str(value).strip():
                prompt_feature_values.setdefault(key, set()).add(str(value).strip().lower())
    attributes = sorted(prompt_feature_values)
    groupings = ["prompt_mode", *attributes]
    groups_by_grouping = {
        "prompt_mode": sorted(modes),
        **{attribute: sorted(values) for attribute, values in prompt_feature_values.items()},
    }
    record_groups = [
        {
            grouping: _group_value(record, grouping, config)
            for grouping in groupings
        }
        for record in data
    ]
    feature_values = {
        feature: [_feature_value(record, feature) for record in data]
        for feature in IMAGE_FEATURES
    }
    features: dict[str, Any] = {}
    for feature in IMAGE_FEATURES:
        features[feature] = {
            "values": [value for value in feature_values[feature] if value is not None],
            "by_group": {
                grouping: {
                    group: [
                        value for value, record_group in zip(
                            feature_values[feature], record_groups
                        )
                        if record_group[grouping] == group and value is not None
                    ]
                    for group in groups_by_grouping[grouping]
                }
                for grouping in groupings
            },
        }
        features[feature]["tests"] = {
            grouping: tests
            for grouping in attributes
            if (tests := _feature_tests(features[feature]["by_group"][grouping]))
        }

    embedded = [
        (record_group, vector)
        for record_group, record in zip(record_groups, data)
        if (vector := _embedding(record)) is not None
    ]
    vectors = [vector for _, vector in embedded]
    pca = _pca(vectors)
    umap = _umap(vectors)
    points = [{
        "pca": pca[index] if pca else None,
        "umap": umap[index] if umap else None,
        "groups": dict(record_group),
    } for index, (record_group, _) in enumerate(embedded)]

    # The subsample is the same for every grouping, so the O(n^2) distance
    # matrix is computed once and reused for all groupings.
    sample_indices = _sample_indices(len(vectors))
    sample_vectors = [vectors[index] for index in sample_indices]
    distances = _distance_matrix(sample_vectors) if len(sample_vectors) > 1 else None
    embedding_groups = {}
    for grouping in groupings:
        labels = [embedded[index][0][grouping] for index in sample_indices]
        score = _silhouette(distances, labels) if distances is not None else None
        embedding_groups[grouping] = {
            "separability": {
                "score": score,
                "null": _silhouette_null(distances, labels, score) if distances is not None else {},
                "probe": _probe(vectors, [record_group[grouping] for record_group, _ in embedded]),
                "n_images": len(vectors),
                "n_clusters": len({
                    record_group[grouping] for record_group, _ in embedded
                    if record_group[grouping] is not None
                }),
            },
            "distances": (
                _distance_distributions(distances, labels)
                if distances is not None else {"within": [], "between": []}
            ),
        }
    return {
        "image_features": features,
        "embedding": {
            "n_embeddings": len(vectors),
            "group_by": groupings,
            "points": points,
            "projections": {"pca_available": pca is not None, "umap_available": umap is not None},
            "groups": embedding_groups,
        },
    }
