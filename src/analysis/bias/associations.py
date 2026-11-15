"""Chapter 5.1: prompt-attribute x detected-feature contingency tables."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any

from ..records import detected_value
from ..stats_utils import contingency_statistics, regularized_gamma_q, rounded


# Bootstrap resamples behind the Cramér's V interval of every association matrix.
V_BOOTSTRAP = 300


def _association_statistics(
    counts: dict[str, dict[str, int]],
    rows: list[str],
    columns: list[str],
) -> dict[str, float | int]:
    """Calculate Cramér's V and Pearson chi-square uncertainty statistics."""
    total = sum(counts[row][column] for row in rows for column in columns)
    degrees_of_freedom = max(0, (len(rows) - 1) * (len(columns) - 1))
    if total == 0 or degrees_of_freedom == 0:
        return {
            "chi_square": 0.0,
            "degrees_of_freedom": degrees_of_freedom,
            "p_value": 1.0,
            "cramers_v": 0.0,
        }

    row_totals = {row: sum(counts[row].values()) for row in rows}
    column_totals = {
        column: sum(counts[row][column] for row in rows)
        for column in columns
    }
    chi_square = 0.0
    for row in rows:
        for column in columns:
            expected = row_totals[row] * column_totals[column] / total
            if expected:
                observed = counts[row][column]
                chi_square += (observed - expected) ** 2 / expected
    p_value = regularized_gamma_q(degrees_of_freedom / 2, chi_square / 2)
    cramers_v = math.sqrt(
        chi_square / (total * min(len(rows) - 1, len(columns) - 1))
    )
    extended = contingency_statistics(
        [[counts[row][column] for column in columns] for row in rows],
        bootstrap=V_BOOTSTRAP,
    )
    return {
        "chi_square": round(chi_square, 6),
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": round(p_value, 6),
        # Unrounded p-value: tiny values matter for multiple-testing correction.
        "p_value_exact": p_value,
        "cramers_v": round(cramers_v, 6),
        # Bergsma's bias correction removes the upward bias of V in small or
        # sparse tables; the interval is a multinomial percentile bootstrap.
        "cramers_v_corrected": round(extended["cramers_v_corrected"], 6),
        "cramers_v_ci": rounded(extended["cramers_v_ci"]),
    }


def _adjusted_residuals(
    counts: dict[str, dict[str, int]], rows: list[str], columns: list[str]
) -> dict[str, dict[str, float]]:
    """Standardized Pearson residuals; |r| > 1.96 marks a deviating cell."""
    residuals = contingency_statistics(
        [[counts[row][column] for column in columns] for row in rows]
    )["adjusted_residuals"]
    return {
        row: {column: round(residuals[i][j], 4) for j, column in enumerate(columns)}
        for i, row in enumerate(rows)
    }


def association_matrix(
    data: list[dict[str, Any]],
    prompt_attribute: str,
    output_feature: str,
    *,
    unrequested_only: bool = False,
) -> dict[str, Any]:
    """Prompt-attribute × detected-feature contingency table.

    With ``unrequested_only`` the table only uses images whose prompt does
    not itself request the output feature (e.g. job × detected gender
    without the "female doctor" prompts). That isolates the model's default
    choice from adherence to an explicit request.
    """
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    prompt_categories: set[str] = set()
    output_categories: set[str] = set()

    for record in data:
        prompt_features = record.get("prompt_feats", {})
        if (unrequested_only and isinstance(prompt_features, dict)
                and prompt_features.get(output_feature) is not None):
            continue
        prompt_value = (
            prompt_features.get(prompt_attribute)
            if isinstance(prompt_features, dict)
            else None
        )
        output_value = detected_value(record, output_feature)
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
        "statistics": _association_statistics(
            matrix_counts, rows, columns
        ),
        "adjusted_residuals": _adjusted_residuals(matrix_counts, rows, columns),
        "unrequested_only": unrequested_only,
        "total": sum(sample_counts.values()),
    }
