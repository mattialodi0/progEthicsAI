from __future__ import annotations

import base64
import math
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
OBJECT_ASSOCIATION_ATTRIBUTES = ("job", "workplace")
NON_OBJECT_LABELS = {
    "face",
    "human face",
    "man",
    "person",
    "woman",
}


def _regularized_gamma_q(shape: float, value: float) -> float:
    """Return the regularized upper incomplete gamma function Q(a, x)."""
    if value <= 0:
        return 1.0
    if value < shape + 1.0:
        term = 1.0 / shape
        series = term
        for index in range(1, 1000):
            term *= value / (shape + index)
            series += term
            if abs(term) <= abs(series) * 3e-14:
                break
        return max(
            0.0,
            min(1.0, 1.0 - series * math.exp(-value + shape * math.log(value)
                                                - math.lgamma(shape))),
        )

    tiny = 1e-300
    numerator = 1.0
    denominator = value + 1.0 - shape
    if abs(denominator) < tiny:
        denominator = tiny
    fraction = numerator / denominator
    for index in range(1, 1000):
        an = -index * (index - shape)
        if abs(denominator) < tiny:
            denominator = tiny
        if abs(numerator) < tiny:
            numerator = tiny
        denominator = an / denominator + value + 1.0 - shape
        if abs(denominator) < tiny:
            denominator = tiny
        numerator = (value + index - shape) + an / numerator
        if abs(numerator) < tiny:
            numerator = tiny
        denominator = 1.0 / denominator
        delta = numerator * denominator
        fraction *= delta
        if abs(delta - 1.0) <= 3e-14:
            break
    return max(
        0.0,
        min(1.0, math.exp(-value + shape * math.log(value)
                          - math.lgamma(shape)) * fraction),
    )


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
    p_value = _regularized_gamma_q(degrees_of_freedom / 2, chi_square / 2)
    cramers_v = math.sqrt(
        chi_square / (total * min(len(rows) - 1, len(columns) - 1))
    )
    return {
        "chi_square": round(chi_square, 6),
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": round(p_value, 6),
        "cramers_v": round(cramers_v, 6),
    }


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
        "statistics": _association_statistics(
            matrix_counts, rows, columns
        ),
        "total": sum(sample_counts.values()),
    }


def _stereotypical_associations(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize detected object prevalence conditional on jobs and workplaces."""
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
            for object_label, count in group_objects[attribute][group].items():
                conditional_probability = count / group_total
                baseline = overall_prevalence.get(object_label, 0.0)
                candidates.append({
                    "object": object_label,
                    "count": count,
                    "probability": round(conditional_probability, 6),
                    "lift": round(
                        conditional_probability / baseline, 6
                    ) if baseline else None,
                })
            candidates.sort(
                key=lambda item: (
                    item["probability"], item["count"],
                    item["lift"] is not None, item["lift"] or 0, item["object"]
                ),
                reverse=True,
            )
            rows.append({
                "group": group,
                "observations": group_total,
                "top_objects": candidates[:5],
            })
        result[attribute] = {
            "attribute": attribute,
            "baseline_observations": overall_total,
            "rows": rows,
        }
    return result


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


def _interaction_model(
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
    first_reference, second_reference = first_values[0], second_values[0]
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
            _regularized_gamma_q(interaction_df / 2, statistic / 2), 6
        ),
        "interaction_terms": [
            {
                "first": first,
                "second": second,
                "estimate": round(full[0][index], 6),
            }
            for index, column in enumerate(columns)
            if len(column) == 3 and column[0] == "interaction"
            for _, first, second in [column]
        ],
    }


def _intersectional_analysis(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare joint prompt groups with their constituent group expectations."""
    result: dict[str, Any] = {}
    for first_attribute, second_attribute in combinations(PROMPT_ATTRIBUTES, 2):
        key = f"{first_attribute}__{second_attribute}"
        outputs: dict[str, Any] = {}
        for output_feature in OUTPUT_FEATURES:
            cells: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
            first_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
            second_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
            for record in data:
                prompt = record.get("prompt_feats", {})
                if not isinstance(prompt, dict):
                    continue
                first = prompt.get(first_attribute)
                second = prompt.get(second_attribute)
                detected = _detected_value(record, output_feature)
                if first is None or second is None or detected is None:
                    continue
                first_label = str(first).strip().lower()
                second_label = str(second).strip().lower()
                detected_label = str(detected).strip().lower()
                if not first_label or not second_label or not detected_label:
                    continue
                cells[(first_label, second_label)][detected_label] += 1
                first_counts[first_label][detected_label] += 1
                second_counts[second_label][detected_label] += 1
            categories = sorted({
                category for counts in cells.values() for category in counts
            })
            if not cells or not categories:
                continue
            rows = []
            forest = []
            for (first, second), counts in sorted(cells.items()):
                sample_size = sum(counts.values())
                first_total = sum(first_counts[first].values())
                second_total = sum(second_counts[second].values())
                percentages = {}
                for category in categories:
                    joint = counts.get(category, 0) / sample_size
                    first_probability = first_counts[first].get(category, 0) / first_total
                    second_probability = second_counts[second].get(category, 0) / second_total
                    expected = (first_probability + second_probability) / 2
                    difference = joint - expected
                    standard_error = math.sqrt(max(
                        1e-12,
                        joint * (1 - joint) / sample_size
                        + first_probability * (1 - first_probability) / (4 * first_total)
                        + second_probability * (1 - second_probability) / (4 * second_total),
                    ))
                    percentages[category] = {
                        "percentage": round(100 * joint, 4),
                        "count": counts.get(category, 0),
                    }
                    forest.append({
                        "group": f"{first} × {second}",
                        "category": category,
                        "difference": round(100 * difference, 4),
                        "lower": round(100 * (difference - 1.96 * standard_error), 4),
                        "upper": round(100 * (difference + 1.96 * standard_error), 4),
                        "n": sample_size,
                    })
                rows.append({
                    "group": f"{first} × {second}",
                    "first": first,
                    "second": second,
                    "n": sample_size,
                    "values": percentages,
                })
            models = []
            for category in categories:
                category_observations = []
                for record in data:
                    prompt = record.get("prompt_feats", {})
                    detected = _detected_value(record, output_feature)
                    if not isinstance(prompt, dict) or detected is None:
                        continue
                    first = prompt.get(first_attribute)
                    second = prompt.get(second_attribute)
                    if first is None or second is None:
                        continue
                    category_observations.append((
                        str(first).strip().lower(), str(second).strip().lower(),
                        int(str(detected).strip().lower() == category),
                    ))
                model = _interaction_model(
                    category_observations, first_attribute, second_attribute, category
                )
                if model is not None:
                    models.append(model)
            outputs[output_feature] = {
                "output_feature": output_feature,
                "categories": categories,
                "rows": rows,
                "forest": forest,
                "models": models,
                "total": sum(row["n"] for row in rows),
            }
        if outputs:
            result[key] = {
                "first_attribute": first_attribute,
                "second_attribute": second_attribute,
                "outputs": outputs,
            }
    return result


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
        "intersectional": _intersectional_analysis(records),
        "stereotypical_associations": _stereotypical_associations(records),
    }
