"""Matched prompt pairs that differ only in one attribute."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import re
from typing import Any

from ..matching import word_pattern
from .common import condition_value


PROMPT_VALUES = {
    "gender": ("male", "female", "man", "woman"),
    "race": ("white", "black", "asian", "hispanic", "middle eastern"),
    "age": ("young", "old", "middle-aged"),
    "emotion": ("happy", "sad", "angry", "calm", "bored", "nervous"),
    "job": (
        "doctor", "nurse", "engineer", "teacher", "artist", "scientist",
        "chef", "pilot", "writer", "musician", "athlete", "politician", "soldier",
    ),
    "workplace": (
        "office", "factory", "hospital", "school", "restaurant", "store",
        "laboratory", "construction site",
    ),
}


def _prompt_key(prompt: str, value: str) -> str:
    """The prompt with the varied value blanked out: equal keys form a pair."""
    return re.sub(word_pattern(value), "__counterfactual__", prompt.lower(), count=1)


def attribute_pairs(
    data: list[dict[str, Any]], attribute: str
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """(contrast, first record, second record) for prompts identical except for
    ``attribute``; records are paired in generation order."""
    groups: defaultdict[str, defaultdict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, record in enumerate(data):
        condition = condition_value(record, attribute)
        prompt = record.get("prompt")
        if condition is None or not isinstance(prompt, str):
            continue
        values = PROMPT_VALUES[attribute]
        matched = next(
            (value for value in values if re.search(word_pattern(value), prompt, re.IGNORECASE)),
            None,
        )
        if matched is None:
            matched = str(record["prompt_feats"][attribute]).lower()
        item = dict(record)
        item["_counterfactual_index"] = index
        groups[_prompt_key(prompt, matched)][condition].append(item)

    pairs = []
    for group in groups.values():
        for left, right in combinations(sorted(group), 2):
            for first, second in zip(
                sorted(group[left], key=lambda item: item["_counterfactual_index"]),
                sorted(group[right], key=lambda item: item["_counterfactual_index"]),
            ):
                pairs.append((f"{left} vs {right}", first, second))
    return pairs
