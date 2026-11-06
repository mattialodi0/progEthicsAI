from __future__ import annotations

import base64
import math
from io import BytesIO
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from .general_statistics import _detected_value
from .stats_utils import (
    contingency_statistics,
    regularized_gamma_q,
    rounded,
    wilson,
)


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


_regularized_gamma_q = regularized_gamma_q
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
    p_value = _regularized_gamma_q(degrees_of_freedom / 2, chi_square / 2)
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
        "adjusted_residuals": _adjusted_residuals(matrix_counts, rows, columns),
        "unrequested_only": unrequested_only,
        "total": sum(sample_counts.values()),
    }


def _stereotypical_associations(data: list[dict[str, Any]]) -> dict[str, Any]:
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
                "effect_percentage_points": round(100 * full[0][index], 4),
            }
            for index, column in enumerate(columns)
            if len(column) == 3 and column[0] == "interaction"
            for _, first, second in [column]
        ],
    }


def _requests_output(
    prompt: dict[str, Any], output_feature: str, first_attribute: str, second_attribute: str
) -> bool:
    """True when the prompt itself asks for the detected feature.

    E.g. a control prompt "young white female nurse" requests gender, so it
    must not shape the race × job → detected-gender expectation. When the
    output is one of the pair (race × job → detected race) every record
    requests it by construction and nothing is excluded.
    """
    return (output_feature not in (first_attribute, second_attribute)
            and prompt.get(output_feature) is not None)


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
                if _requests_output(prompt, output_feature, first_attribute, second_attribute):
                    continue
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
            first_levels = sorted({row["first"] for row in rows})
            second_levels = sorted({row["second"] for row in rows})
            difference_in_differences: dict[str, Any] = {}
            if len(first_levels) >= 2 and len(second_levels) >= 2:
                did_first_levels = first_levels[:2]
                # Report the change from the last sorted level to the first,
                # matching the conventional baseline-then-comparison reading
                # used in the report (e.g. White gap versus Black gap).
                did_second_levels = [second_levels[-1], second_levels[0]]
                for category in categories:
                    points = [
                        {
                            "first": row["first"],
                            "second": row["second"],
                            "percentage": row["values"][category]["percentage"],
                            "n": row["n"],
                        }
                        for row in rows
                        if row["first"] in did_first_levels
                    ]
                    gaps = []
                    for second in second_levels:
                        values = {
                            row["first"]: row["values"][category]["percentage"]
                            for row in rows
                            if row["second"] == second
                            and row["first"] in did_first_levels
                        }
                        if len(values) == 2:
                            gaps.append({
                                "second": second,
                                "difference_percentage_points": round(
                                    values[did_first_levels[1]]
                                    - values[did_first_levels[0]],
                                    4,
                                ),
                            })
                    comparison_gaps = {
                        gap["second"]: gap["difference_percentage_points"]
                        for gap in gaps
                    }
                    if all(second in comparison_gaps for second in did_second_levels):
                        difference_in_differences[category] = {
                            "available": True,
                            "first_levels": did_first_levels,
                            "second_levels": second_levels,
                            "comparison_levels": did_second_levels,
                            "points": points,
                            "gaps": gaps,
                            "difference_in_differences_percentage_points": round(
                                comparison_gaps[did_second_levels[1]]
                                - comparison_gaps[did_second_levels[0]],
                                4,
                            ),
                        }
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
                    if _requests_output(prompt, output_feature, first_attribute,
                                        second_attribute):
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
                "difference_in_differences": difference_in_differences,
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
    # Same matrices restricted to images that do not request the detected
    # feature: the common comparison standard used by the synthesis chapter.
    associations_unrequested = {
        prompt_attribute: {
            output_feature: _association_matrix(
                records, prompt_attribute, output_feature, unrequested_only=True
            )
            for output_feature in OUTPUT_FEATURES
            if output_feature != prompt_attribute
        }
        for prompt_attribute in PROMPT_ATTRIBUTES
    }
    return {
        "prompt_attributes": available,
        "output_features": available_features,
        "associations": associations,
        "associations_unrequested": associations_unrequested,
        "double_associations": double_associations,
        "intersectional": _intersectional_analysis(records),
        "stereotypical_associations": _stereotypical_associations(records),
    }
