"""Whole-word matching of prompt vocabulary in free text."""

from __future__ import annotations

import re


def word_pattern(value: str) -> str:
    """Regex that matches ``value`` as a whole word; its spaces match any whitespace."""
    return r"(?<!\w)" + re.escape(value).replace(r"\ ", r"\s+") + r"(?!\w)"
