from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .general_statistics import _detected_value


PROMPT_ATTRIBUTES = (
    "gender",
    "race",
    "age",
    "emotion",
    "job",
    "workplace",
    "continent",
    "status",
    "toxic_trait",
    "positive_trait",
)
OUTPUT_FEATURES = ("gender", "race", "age", "emotion")


def _association_matrix(
    data: list[dict[str, Any]],
    prompt_attribute: str,
    output_feature: str,
) -> dict[str, Any]:
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    prompt_categories: set[str] = set()
    output_categories: set[str] = set()

    for record in data:
        prompt_features = record.get("prompt_feats", {})
        prompt_value = (
            prompt_features.get(prompt_attribute)
            if isinstance(prompt_features, dict)
            else None
        )
        output_value = _detected_value(record, output_feature)
        if prompt_value is None or output_value is None:
            continue
        prompt_label = str(prompt_value).strip().lower()
        output_label = str(output_value).strip().lower()
        if not prompt_label or not output_label:
            continue
        prompt_categories.add(prompt_label)
        output_categories.add(output_label)
        counts[prompt_label][output_label] += 1

    rows = sorted(prompt_categories)
    columns = sorted(output_categories)
    matrix_counts = {
        row: {column: counts[row].get(column, 0) for column in columns}
        for row in rows
    }
    matrix_percentages = {}
    sample_counts = {}
    for row in rows:
        total = sum(matrix_counts[row].values())
        sample_counts[row] = total
        matrix_percentages[row] = {
            column: round(100.0 * matrix_counts[row][column] / total, 4)
            if total
            else 0.0
            for column in columns
        }

    return {
        "prompt_attribute": prompt_attribute,
        "output_feature": output_feature,
        "rows": rows,
        "columns": columns,
        "counts": matrix_counts,
        "percentages": matrix_percentages,
        "sample_counts": sample_counts,
        "total": sum(sample_counts.values()),
    }


def collect_bias_stats(
    data: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Precompute prompt-attribute/output-feature association matrices.

    Each matrix is conditioned on a prompt category: a row sums to 100% over
    the detected output categories when at least one paired observation exists.
    Images missing either side are excluded from that matrix and are represented
    by its sample counts and total.
    """
    if not isinstance(data, list):
        raise TypeError("data must be a list of image metadata objects")
    records = [record for record in data if isinstance(record, dict)]
    associations = {
        prompt_attribute: {
            output_feature: _association_matrix(
                records, prompt_attribute, output_feature
            )
            for output_feature in OUTPUT_FEATURES
        }
        for prompt_attribute in PROMPT_ATTRIBUTES
    }
    available = [
        prompt_attribute
        for prompt_attribute, features in associations.items()
        if any(matrix["total"] for matrix in features.values())
    ]
    available_features = [
        output_feature
        for output_feature in OUTPUT_FEATURES
        if any(
            associations[prompt_attribute][output_feature]["total"]
            for prompt_attribute in PROMPT_ATTRIBUTES
        )
    ]
    return {
        "prompt_attributes": available,
        "output_features": available_features,
        "associations": associations,
    }