"""The common results table of chapter 9 and its effect-strength scale."""

from __future__ import annotations

from typing import Any

from ..stats_utils import Z95, cohen_h, newcombe_difference, normal_p
from .constants import BANDS, FAMILIES, LENSES, MAX_STRENGTH, SOCIAL_ATTRIBUTES


def pct(value: float | None, digits: int = 0) -> str:
    """Format a 0-1 share as a whole-number percentage."""
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def human(value: Any) -> str:
    """Turn an artifact key such as ``toxic_trait`` into readable text."""
    return str(value).replace("__", " × ").replace("_", " ")


def strength_score(magnitude: float | None, family: str | None) -> float | None:
    """Map |effect| onto 0–6: 1 = small, 2 = medium, 3 = large threshold."""
    if magnitude is None or family not in FAMILIES:
        return None
    small, medium, large = FAMILIES[family]["thresholds"]
    value = abs(magnitude)
    if value < small:
        return value / small
    if value < medium:
        return 1 + (value - small) / (medium - small)
    if value < large:
        return 2 + (value - medium) / (large - medium)
    return min(MAX_STRENGTH, 3 + (value - large) / (large - medium))


def rank_key(strength_low: float | None, evidence: int, n: int | None,
             strength: float | None) -> tuple:
    """Sort key: conservative strength first, but every effect that is large
    even at its lower CI bound counts as equally large, and is then ordered by
    evidence level and sample size (better-replicated first)."""
    return (-min(strength_low or 0.0, 3.0), evidence, -(n or 0), -(strength or 0.0))


def band(score: float | None) -> str | None:
    """Effect band (negligible / small / medium / large) of a strength score."""
    if score is None:
        return None
    return BANDS[min(3, int(score))]


def reliable_band(row: dict[str, Any]) -> str | None:
    """Band of the smallest effect compatible with the data (lower CI bound).

    Used for verdicts. A supported result (q < α, point estimate at least
    small) is never reported below "small", even when its interval dips
    under the small threshold.
    """
    value = band(row.get("strength_low"))
    return "small" if value == "negligible" else value


def scene_question(attribute: str | None) -> str:
    """Question for a non-demographic image property (brightness, framing,
    caption tone). Only differences between social groups answer the
    representational-quality question; "doctor images are brighter than
    other jobs" describes the scene the prompt asks for, so it is kept as a
    result but filed under context and does not drive a fairness verdict."""
    return "quality" if attribute in SOCIAL_ATTRIBUTES else "context"


def p_from_ci(estimate: float, low: float, high: float) -> float | None:
    """Two-sided p-value of a normal-approximation 95% interval."""
    se = (high - low) / (2 * Z95)
    return normal_p(estimate / se) if se > 0 else None


def h_interval(reference: float, low: float | None, high: float | None) -> list[float | None]:
    """Cohen's h for the two ends of a CI on (p − reference)."""
    if low is None or high is None:
        return [None, None]
    return [cohen_h(min(1.0, max(0.0, reference + low)), reference),
            cohen_h(min(1.0, max(0.0, reference + high)), reference)]


def proportion_gap(k: int, n: int, rest_k: int, rest_n: int) -> dict[str, Any]:
    """Result fields comparing a group's share k / n with the rest's share.

    The effect is the difference in percentage points with Newcombe's 95%
    interval; the magnitude is Cohen's h with the matching interval.
    """
    difference, low, high = newcombe_difference(k, n, rest_k, rest_n)
    return {
        "effect": difference, "effect_unit": "pp", "ci": [low, high], "family": "h",
        "magnitude": cohen_h(k / n, rest_k / rest_n),
        "magnitude_ci": h_interval(rest_k / rest_n, low, high),
    }


class ResultTable:
    """Accumulates result rows and the extra (unstored) tests of each family."""

    def __init__(self, settings: dict[str, Any]):
        self.rows: list[dict[str, Any]] = []
        self.hidden_tests = 0
        self.settings = settings
        self.warnings: list[str] = []

    def add(self, **fields: Any) -> dict[str, Any]:
        lens = fields["lens"]
        row = {
            "id": f"r{len(self.rows) + 1}",
            "lens": lens,
            "source": LENSES[lens]["source"],
            "link": fields.pop("link", LENSES[lens]["link"]),
            "level": "specific",
            "attribute": None,
            "subgroup": None,
            "reference": None,
            "feature": None,
            "category": None,
            "scope": None,
            "metric": None,
            "effect": None,
            "effect_unit": None,
            "ci": [None, None],
            "family": None,
            "magnitude": None,
            "magnitude_ci": [None, None],
            "n": None,
            "n_total": None,
            "p_value": None,
            "context": False,
            "stereotype_candidate": False,
            "question": None,
            "keys": [],
            "flags": [],
            "details": {},
            "statement": "",
        }
        row.update(fields)
        row["strength"] = strength_score(row["magnitude"], row["family"])
        row["strength_ci"] = [
            strength_score(value, row["family"]) if value is not None else None
            for value in row["magnitude_ci"]
        ]
        if (row["magnitude_ci"][0] is not None and row["magnitude_ci"][1] is not None
                and row["magnitude_ci"][0] * row["magnitude_ci"][1] < 0):
            # The interval crosses zero: the smallest compatible strength is 0.
            row["strength_ci"] = [0.0, max(row["strength_ci"])]
        elif all(value is not None for value in row["strength_ci"]):
            row["strength_ci"] = sorted(row["strength_ci"])
        row["band"] = band(row["strength"])
        # Conservative strength: the smallest effect compatible with the data
        # (lower end of the interval). Used for ranking, so a large effect
        # seen in 8 images does not outrank a slightly smaller one in 200.
        row["strength_low"] = (row["strength_ci"][0] if row["strength_ci"][0] is not None
                               else row["strength"])
        row["band_low"] = band(row["strength_low"])
        self.rows.append(row)
        return row


def add_flag(row: dict[str, Any], code: str, text: str) -> None:
    """Attach a caveat flag to a result row, once per code."""
    if all(flag["code"] != code for flag in row["flags"]):
        row["flags"].append({"code": code, "text": text})
