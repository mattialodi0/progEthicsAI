"""Chapter 5.3: objects over-represented in the images of a prompt group."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ..stats_utils import wilson


OBJECT_ASSOCIATION_ATTRIBUTES = (
    "job",
    "workplace",
    "continent",
    "toxic_trait",
    "positive_trait",
)
NON_OBJECT_LABELS = {
    "face",
    "human face",
    "man",
    "person",
    "woman",
}
MIN_OBJECT_COUNT = 3
ASSOCIATED_OBJECTS = 8


def _fisher_p(count: int, total: int, rest_count: int, rest_total: int) -> float | None:
    """Two-sided Fisher exact test: object share in the group vs the rest."""
    if not total or not rest_total:
        return None
    try:
        from scipy.stats import fisher_exact
    except ImportError:  # pragma: no cover - scipy ships with scikit-learn
        return None
    _, p_value = fisher_exact([[count, total - count], [rest_count, rest_total - rest_count]])
    return float(p_value)


def stereotypical_associations(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize detected object prevalence conditional on prompt contexts."""
    groups: dict[str, Counter[str]] = {
        attribute: Counter() for attribute in OBJECT_ASSOCIATION_ATTRIBUTES
    }
    group_objects: dict[str, defaultdict[str, Counter[str]]] = {
        attribute: defaultdict(Counter)
        for attribute in OBJECT_ASSOCIATION_ATTRIBUTES
    }
    object_totals: dict[str, Counter[str]] = {
        attribute: Counter() for attribute in OBJECT_ASSOCIATION_ATTRIBUTES
    }
    analyzable_totals: Counter[str] = Counter()

    for record in data:
        prompt_features = record.get("prompt_feats", {})
        if not isinstance(prompt_features, dict):
            continue
        florence = record.get("items_florence", {})
        labels = florence.get("labels", ()) if isinstance(florence, dict) else ()
        if not isinstance(labels, (list, tuple)):
            labels = ()
        objects = {
            str(label).strip().lower()
            for label in labels
            if isinstance(label, str)
            and str(label).strip()
            and str(label).strip().lower() not in NON_OBJECT_LABELS
        }
        for attribute in OBJECT_ASSOCIATION_ATTRIBUTES:
            group = prompt_features.get(attribute)
            if group is None:
                continue
            group_label = str(group).strip().lower()
            if not group_label:
                continue
            analyzable_totals[attribute] += 1
            groups[attribute][group_label] += 1
            for object_label in objects:
                object_totals[attribute][object_label] += 1
            for object_label in objects:
                group_objects[attribute][group_label][object_label] += 1

    result: dict[str, Any] = {}
    for attribute in OBJECT_ASSOCIATION_ATTRIBUTES:
        overall_total = analyzable_totals[attribute]
        if not overall_total:
            continue
        overall_prevalence = {
            label: count / overall_total
            for label, count in object_totals[attribute].items()
        }
        rows = []
        for group, group_total in sorted(groups[attribute].items()):
            candidates = []
            rest_total = overall_total - group_total
            for object_label, count in group_objects[attribute][group].items():
                conditional_probability = count / group_total
                baseline = overall_prevalence.get(object_label, 0.0)
                rest_count = object_totals[attribute][object_label] - count
                low, high = wilson(count, group_total)
                candidates.append({
                    "object": object_label,
                    "count": count,
                    "probability": round(conditional_probability, 6),
                    "lift": round(
                        conditional_probability / baseline, 6
                    ) if baseline else None,
                    # Wilson interval of P(object | group) divided by the
                    # (much more precise) pooled baseline P(object).
                    "lift_ci": [round(low / baseline, 6), round(high / baseline, 6)]
                    if baseline else [None, None],
                    # Same object in the other values of this attribute: the
                    # reference for the Fisher test below.
                    "rest_count": rest_count,
                    "rest_observations": rest_total,
                    "rest_probability": round(rest_count / rest_total, 6)
                    if rest_total else None,
                    "p_value": _fisher_p(count, group_total, rest_count, rest_total),
                })
            candidates.sort(
                key=lambda item: (
                    item["probability"], item["count"],
                    item["lift"] is not None, item["lift"] or 0, item["object"]
                ),
                reverse=True,
            )
            # Most over-represented objects (not the most frequent ones):
            # seen in at least MIN_OBJECT_COUNT images, ranked by lift.
            associated = sorted(
                (item for item in candidates
                 if item["count"] >= MIN_OBJECT_COUNT and item["lift"] is not None),
                key=lambda item: (-item["lift"], -item["count"], item["object"]),
            )[:ASSOCIATED_OBJECTS]
            rows.append({
                "group": group,
                "observations": group_total,
                "top_objects": candidates[:5],
                "associated_objects": associated,
                # Objects eligible for ``associated_objects`` (the test family).
                "objects_tested": sum(
                    1 for item in candidates
                    if item["count"] >= MIN_OBJECT_COUNT and item["lift"] is not None
                ),
            })
        result[attribute] = {
            "attribute": attribute,
            "baseline_observations": overall_total,
            "rows": rows,
        }
    return result
