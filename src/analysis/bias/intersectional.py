"""Chapter 5.4: do joint prompt groups deviate from their constituent groups?"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import math
from typing import Any

from ..records import detected_value
from .common import OUTPUT_FEATURES, PROMPT_ATTRIBUTES
from .interaction_model import interaction_model


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


Observation = tuple[str, str, str]


def _observations(
    data: list[dict[str, Any]], first_attribute: str, second_attribute: str, output_feature: str
) -> list[Observation]:
    """(first value, second value, detected category) of every usable record.

    Records whose prompt itself requests the detected feature are left out.
    """
    observations = []
    for record in data:
        prompt = record.get("prompt_feats", {})
        if not isinstance(prompt, dict):
            continue
        first = prompt.get(first_attribute)
        second = prompt.get(second_attribute)
        if _requests_output(prompt, output_feature, first_attribute, second_attribute):
            continue
        detected = detected_value(record, output_feature)
        if first is None or second is None or detected is None:
            continue
        observations.append((str(first).strip().lower(), str(second).strip().lower(),
                             str(detected).strip().lower()))
    return observations


def _count_cells(observations: list[Observation]) -> tuple[
    defaultdict[tuple[str, str], Counter[str]],
    defaultdict[str, Counter[str]],
    defaultdict[str, Counter[str]],
]:
    """Detected-category counts per joint group and per single group."""
    cells: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    first_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    second_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for first_label, second_label, detected_label in observations:
        if not first_label or not second_label or not detected_label:
            continue
        cells[(first_label, second_label)][detected_label] += 1
        first_counts[first_label][detected_label] += 1
        second_counts[second_label][detected_label] += 1
    return cells, first_counts, second_counts


def _rows_and_forest(
    cells: dict[tuple[str, str], Counter[str]],
    first_counts: dict[str, Counter[str]],
    second_counts: dict[str, Counter[str]],
    categories: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Joint-group shares, and their deviation from the mean of the two
    single-group shares with a 95% interval (the forest plot)."""
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
    return rows, forest


def _difference_in_differences(
    rows: list[dict[str, Any]], categories: list[str]
) -> dict[str, Any]:
    """How the gap between two first-attribute levels changes across the second."""
    first_levels = sorted({row["first"] for row in rows})
    second_levels = sorted({row["second"] for row in rows})
    result: dict[str, Any] = {}
    if len(first_levels) < 2 or len(second_levels) < 2:
        return result
    did_first_levels = first_levels[:2]
    # Report the change from the last sorted level to the first, matching the
    # conventional baseline-then-comparison reading used in the report (e.g.
    # White gap versus Black gap).
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
                if row["second"] == second and row["first"] in did_first_levels
            }
            if len(values) == 2:
                gaps.append({
                    "second": second,
                    "difference_percentage_points": round(
                        values[did_first_levels[1]] - values[did_first_levels[0]], 4,
                    ),
                })
        comparison_gaps = {gap["second"]: gap["difference_percentage_points"] for gap in gaps}
        if all(second in comparison_gaps for second in did_second_levels):
            result[category] = {
                "available": True,
                "first_levels": did_first_levels,
                "second_levels": second_levels,
                "comparison_levels": did_second_levels,
                "points": points,
                "gaps": gaps,
                "difference_in_differences_percentage_points": round(
                    comparison_gaps[did_second_levels[1]] - comparison_gaps[did_second_levels[0]],
                    4,
                ),
            }
    return result


def _interaction_models(
    observations: list[Observation], first_attribute: str, second_attribute: str,
    categories: list[str],
) -> list[dict[str, Any]]:
    """One interaction model per detected category (binary outcome)."""
    models = []
    for category in categories:
        model = interaction_model(
            [(first, second, int(detected == category)) for first, second, detected in observations],
            first_attribute, second_attribute, category,
        )
        if model is not None:
            models.append(model)
    return models


def intersectional_analysis(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare joint prompt groups with their constituent group expectations."""
    result: dict[str, Any] = {}
    for first_attribute, second_attribute in combinations(PROMPT_ATTRIBUTES, 2):
        outputs: dict[str, Any] = {}
        for output_feature in OUTPUT_FEATURES:
            observations = _observations(data, first_attribute, second_attribute, output_feature)
            cells, first_counts, second_counts = _count_cells(observations)
            categories = sorted({category for counts in cells.values() for category in counts})
            if not cells or not categories:
                continue
            rows, forest = _rows_and_forest(cells, first_counts, second_counts, categories)
            outputs[output_feature] = {
                "output_feature": output_feature,
                "categories": categories,
                "rows": rows,
                "forest": forest,
                "models": _interaction_models(observations, first_attribute, second_attribute,
                                              categories),
                "difference_in_differences": _difference_in_differences(rows, categories),
                "total": sum(row["n"] for row in rows),
            }
        if outputs:
            result[f"{first_attribute}__{second_attribute}"] = {
                "first_attribute": first_attribute,
                "second_attribute": second_attribute,
                "outputs": outputs,
            }
    return result
