"""Chapter 7 data: image-feature distributions and embedding-space analysis."""

from __future__ import annotations

from typing import Any

from .embedding_space import MIN_TEST_GROUP, embedding_analysis
from .records import image_embedding, record_mode
from .stats_utils import finite, hedges_g, kruskal, mann_whitney_p, mean_summary, rounded


IMAGE_FEATURES = (
    "brightness", "saturation", "contrast", "n_people",
    "person_area_ratio", "face_area_ratio", "detected_age",
)
# Features that one prompt attribute requests directly. Images whose prompt
# sets that attribute are left out, so groups compare the model's default.
REQUESTED_BY = {"detected_age": "age"}


def _feature_value(record: dict[str, Any], feature: str) -> float | None:
    if feature in {"brightness", "saturation", "contrast"}:
        metrics = record.get("image_metrics")
        return finite(metrics.get(feature) if isinstance(metrics, dict) else None)
    requested_by = REQUESTED_BY.get(feature)
    prompt_feats = record.get("prompt_feats")
    if requested_by and isinstance(prompt_feats, dict) and prompt_feats.get(requested_by) is not None:
        return None
    return finite(record.get(feature))


def _group_value(record: dict[str, Any], group_by: str, config: dict[str, Any]) -> str | None:
    if group_by == "prompt_mode":
        return record_mode(record, config)
    features = record.get("prompt_feats")
    value = features.get(group_by) if isinstance(features, dict) else None
    return str(value).strip().lower() if value is not None and str(value).strip() else None


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


def build_visual_feature_analysis(
    data: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Build JSON-safe data for all chapter-7 plots."""
    prompt_feature_values: dict[str, set[str]] = {}
    modes: set[str] = set()
    for record in data:
        modes.add(record_mode(record, config))
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
        if (vector := image_embedding(record)) is not None
    ]
    return {
        "image_features": features,
        "embedding": embedding_analysis(embedded, groupings),
    }
