"""Chapter 5.4: linear probability model with an interaction term.

Fits ``Y = b0 + b1 G + b2 R + b3 (G x R)`` for a binary outcome by ordinary
least squares and tests the interaction terms with a likelihood-ratio-style
statistic. Pure Python: the systems are tiny.
"""

from __future__ import annotations

import math
from typing import Any

from ..stats_utils import regularized_gamma_q


def _solve_linear_system(
    matrix: list[list[float]], vector: list[float]
) -> list[float] | None:
    """Solve a small normal-equation system with Gaussian elimination."""
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    size = len(vector)
    for pivot in range(size):
        index = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
        if abs(augmented[index][pivot]) < 1e-10:
            return None
        augmented[pivot], augmented[index] = augmented[index], augmented[pivot]
        divisor = augmented[pivot][pivot]
        augmented[pivot] = [value / divisor for value in augmented[pivot]]
        for row in range(size):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[pivot])
            ]
    return [augmented[index][-1] for index in range(size)]


def interaction_model(
    observations: list[tuple[str, str, int]],
    first_attribute: str,
    second_attribute: str,
    output_category: str,
) -> dict[str, Any] | None:
    """Fit a categorical main-effects plus interaction model when estimable."""
    if len(observations) < 20:
        return None
    first_values = sorted({first for first, _, _ in observations})
    second_values = sorted({second for _, second, _ in observations})
    if len(first_values) < 2 or len(second_values) < 2:
        return None
    columns = [("intercept",)]
    columns.extend(("first", value) for value in first_values[1:])
    columns.extend(("second", value) for value in second_values[1:])
    columns.extend(
        ("interaction", first, second)
        for first in first_values[1:]
        for second in second_values[1:]
    )

    def design(first: str, second: str, include_interactions: bool) -> list[float]:
        values = [1.0]
        values.extend(float(first == value) for value in first_values[1:])
        values.extend(float(second == value) for value in second_values[1:])
        if include_interactions:
            values.extend(
                float(first == value and second == other)
                for value in first_values[1:]
                for other in second_values[1:]
            )
        return values

    def fit(include_interactions: bool) -> tuple[list[float], float, int] | None:
        rows = [design(first, second, include_interactions)
                for first, second, _ in observations]
        targets = [float(value) for _, _, value in observations]
        width = len(rows[0])
        normal = [[sum(row[i] * row[j] for row in rows) for j in range(width)]
                  for i in range(width)]
        right = [sum(row[i] * target for row, target in zip(rows, targets))
                 for i in range(width)]
        coefficients = _solve_linear_system(normal, right)
        if coefficients is None:
            return None
        residual = sum(
            (target - sum(value * coefficient for value, coefficient in zip(row, coefficients))) ** 2
            for row, target in zip(rows, targets)
        )
        return coefficients, residual, len(targets) - width

    reduced = fit(False)
    full = fit(True)
    if reduced is None or full is None or full[2] <= 0:
        return None
    interaction_df = (len(first_values) - 1) * (len(second_values) - 1)
    if reduced[1] <= 1e-12 and full[1] <= 1e-12:
        return None
    full_error = max(full[1], 1e-12)
    reduced_error = max(reduced[1], full_error)
    statistic = max(0.0, len(observations) * math.log(reduced_error / full_error))
    return {
        "output_category": output_category,
        "n": len(observations),
        "first_levels": first_values,
        "second_levels": second_values,
        "interaction_df": interaction_df,
        "interaction_statistic": round(statistic, 6),
        "interaction_p_value": round(
            regularized_gamma_q(interaction_df / 2, statistic / 2), 6
        ),
        "interaction_terms": [
            {
                "first": first,
                "second": second,
                "estimate": round(full[0][index], 6),
                "effect_percentage_points": round(100 * full[0][index], 4),
            }
            for index, column in enumerate(columns)
            if len(column) == 3 and column[0] == "interaction"
            for _, first, second in [column]
        ],
    }
