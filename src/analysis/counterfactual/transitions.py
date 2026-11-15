"""Detected values on both sides of matched pairs, for the chapter-6.1 view."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

from ..records import detected_value
from .common import CONTROL_ATTRIBUTES, DETECTED_ATTRIBUTES, condition_value


def _matched_attribute_pairs(
    data: list[dict[str, Any]],
    *,
    mode: str,
    required_attributes: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Return pairs that vary one requested attribute and fix the rest."""
    pairs: list[dict[str, Any]] = []
    for variable in required_attributes:
        variable_groups: defaultdict[
            tuple[str, ...], defaultdict[str, list[dict[str, Any]]]
        ] = defaultdict(lambda: defaultdict(list))
        for index, record in enumerate(data):
            if str(record.get("mode")) != mode:
                continue
            values = tuple(condition_value(record, attribute) for attribute in required_attributes)
            if any(value is None for value in values):
                continue
            variable_index = required_attributes.index(variable)
            fixed_values = values[:variable_index] + values[variable_index + 1:]
            item = dict(record)
            item["_counterfactual_index"] = index
            variable_groups[fixed_values][values[variable_index]].append(item)
        for fixed_values, by_value in variable_groups.items():
            for left, right in combinations(sorted(by_value), 2):
                for first, second in zip(
                    sorted(by_value[left], key=lambda item: item["_counterfactual_index"]),
                    sorted(by_value[right], key=lambda item: item["_counterfactual_index"]),
                ):
                    pairs.append({
                        "variable_attribute": variable,
                        "variable_values": [left, right],
                        "fixed_values": {
                            attribute: value
                            for attribute, value in zip(
                                (attribute for attribute in required_attributes if attribute != variable),
                                fixed_values,
                            )
                        },
                        "detected": {
                            side: {
                                attribute: detected_value(record, attribute)
                                for attribute in DETECTED_ATTRIBUTES
                            }
                            for side, record in (("first", first), ("second", second))
                        },
                    })
    return pairs


def control_pairs(
    data: list[dict[str, Any]], mode: str = "0"
) -> list[dict[str, Any]]:
    """Return matched pairs from four-attribute control prompts."""
    return _matched_attribute_pairs(
        data, mode=mode, required_attributes=CONTROL_ATTRIBUTES
    )


def double_pairs(
    data: list[dict[str, Any]], mode: str = "2"
) -> list[dict[str, Any]]:
    """Return matched pairs from prompts specifying exactly two attributes."""
    pairs: list[dict[str, Any]] = []
    for index, record in enumerate(data):
        if str(record.get("mode")) != mode:
            continue
        features = record.get("prompt_feats")
        if not isinstance(features, dict):
            continue
        specified = tuple(
            attribute for attribute in CONTROL_ATTRIBUTES
            if condition_value(record, attribute) is not None
        )
        if len(specified) != 2:
            continue
        pairs.append((index, record, specified))

    result: list[dict[str, Any]] = []
    # Sorted: iterating a set of strings would make the pair order depend on
    # the per-process hash seed, so every run produced a different artifact.
    pair_specs = sorted({
        spec
        for _, _, specified in pairs
        for spec in ((specified[0], specified[1]), (specified[1], specified[0]))
    })
    for variable, fixed in pair_specs:
        groups: defaultdict[str, defaultdict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, record, specified in pairs:
            if specified != tuple(sorted((variable, fixed), key=CONTROL_ATTRIBUTES.index)):
                continue
            item = dict(record)
            item["_counterfactual_index"] = index
            groups[condition_value(record, fixed)][condition_value(record, variable)].append(item)
        for fixed_value, by_value in groups.items():
            for left, right in combinations(sorted(by_value), 2):
                for first, second in zip(
                    sorted(by_value[left], key=lambda item: item["_counterfactual_index"]),
                    sorted(by_value[right], key=lambda item: item["_counterfactual_index"]),
                ):
                    result.append({
                        "variable_attribute": variable,
                        "variable_values": [left, right],
                        "fixed_attribute": fixed,
                        "fixed_value": fixed_value,
                        "detected": {
                            side: {
                                attribute: detected_value(record, attribute)
                                for attribute in DETECTED_ATTRIBUTES
                            }
                            for side, record in (("first", first), ("second", second))
                        },
                    })
    return result
