from __future__ import annotations

import base64
from io import BytesIO
from collections import Counter, defaultdict
from itertools import combinations
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


def _heatmap_data_uri(
    values: dict[str, dict[str, float]],
    counts: dict[str, dict[str, int]],
    rows: list[str],
    columns: list[str],
    title: str,
) -> str:
    """Render one double-condition matrix with Seaborn outside the report."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as error:
        raise RuntimeError(
            "Chapter 5.2 requires matplotlib, pandas, and seaborn"
        ) from error

    frame = pd.DataFrame(
        [[values[row][column] for column in columns] for row in rows],
        index=rows,
        columns=columns,
    )
    annotations = pd.DataFrame(
        [
            [
                f"{values[row][column]:.1f}%\n(n={counts[row][column]})"
                for column in columns
            ]
            for row in rows
        ],
        index=rows,
        columns=columns,
    )
    width = max(6.0, 1.15 * len(columns) + 2.5)
    height = max(4.0, 0.65 * len(rows) + 2.0)
    figure, axis = plt.subplots(figsize=(width, height))
    sns.heatmap(
        frame.T,
        annot=annotations.T,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Detected percentage (%)"},
        ax=axis,
    )
    axis.set_title(title)
    axis.set_ylabel("Prompt attribute pair")
    axis.set_xlabel("Detected feature category")
    figure.tight_layout()
    output = BytesIO()
    figure.savefig(output, format="png", dpi=130)
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _double_association(
    data: list[dict[str, Any]],
    first_attribute: str,
    second_attribute: str,
) -> dict[str, Any]:
    counts_by_output: dict[str, Counter[str]] = {
        output: Counter() for output in OUTPUT_FEATURES
    }

    for record in data:
        prompt_features = record.get("prompt_feats", {})
        if not isinstance(prompt_features, dict):
            continue
        first_value = prompt_features.get(first_attribute)
        second_value = prompt_features.get(second_attribute)
        if first_value is None or second_value is None:
            continue
        first_label = str(first_value).strip().lower()
        second_label = str(second_value).strip().lower()
        if not first_label or not second_label:
            continue
        detected = {
            output: _detected_value(record, output)
            for output in OUTPUT_FEATURES
        }
        for output, detected_value in detected.items():
            if detected_value is None:
                continue
            pair_label = f"{first_label} × {second_label}"
            counts_by_output[output][pair_label, str(detected_value).strip().lower()] += 1

    outputs: dict[str, Any] = {}
    for output in OUTPUT_FEATURES:
        pair_counts = counts_by_output[output]
        columns = sorted({pair for pair, _ in pair_counts})
        rows = sorted({detected for _, detected in pair_counts})
        counts = {
            row: {
                column: pair_counts.get((column, row), 0)
                for column in columns
            }
            for row in rows
        }
        total = sum(sum(row.values()) for row in counts.values())
        if not total:
            continue
        pair_percentages = {
            row: {
                column: round(
                    100.0 * counts[row][column]
                    / max(1, sum(counts[detected][column] for detected in rows)),
                    4,
                )
                for column in columns
            }
            for row in rows
        }
        title = (
            f"Prompt {first_attribute} x {second_attribute} | "
            f"detected {output}"
        )
        outputs[output] = {
            "output_feature": output,
            "rows": rows,
            "columns": columns,
            "counts": counts,
            "percentages": pair_percentages,
            "image": _heatmap_data_uri(
                pair_percentages, counts, rows, columns, title
            ),
            "total": total,
        }

    return {
        "first_attribute": first_attribute,
        "second_attribute": second_attribute,
        "outputs": outputs,
    }


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
    double_associations = {}
    for first_attribute, second_attribute in combinations(PROMPT_ATTRIBUTES, 2):
        association = _double_association(
            records, first_attribute, second_attribute
        )
        if association["outputs"]:
            key = f"{first_attribute}__{second_attribute}"
            double_associations[key] = association
    return {
        "prompt_attributes": available,
        "output_features": available_features,
        "associations": associations,
        "double_associations": double_associations,
    }