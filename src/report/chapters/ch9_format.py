"""Chapter 9: formatting of synthesis results (effects, q-values, badges)."""

from __future__ import annotations

from typing import Any

from ..explain import badge, esc, number, pct


BAND_TONE = {"negligible": "neutral", "small": "low", "medium": "mid", "large": "high"}
# Verdict of a fairness question: the band of its top-ranked supported result
# at the lower end of the confidence interval (see
# ``synthesis.summaries.answer_questions``).
BAND_TEXT = {None: "No supported disparity", "negligible": "Negligible",
             "small": "Largest reliable disparity: small",
             "medium": "Largest reliable disparity: moderate",
             "large": "Largest reliable disparity: large"}
EVIDENCE_TONE = {"confirmed": "high", "convergent": "mid", "supported": "low"}
TIER_TONE = {"supported": "high", "descriptive": "mid", "negligible": "neutral",
             "context": "neutral", "measurement": "low"}
FLAG_TEXT = {
    "small_n": "small n",
    "low_detection": "low face detection",
    "low_adherence": "request often not followed",
    "label_mismatch": "label mismatch",
    "mode_inconsistent": "differs by prompt mode",
}


def format_effect(row: dict[str, Any]) -> str:
    """Effect in its native unit with its 95% interval."""
    value = number(row.get("effect"))
    if value is None:
        return "–"
    low, high = (number(item) for item in (row.get("ci") or [None, None]))
    unit = row.get("effect_unit")
    if unit == "pp":
        text = f"{100 * value:+.0f} pp"
        interval = f" [{100 * low:+.0f}, {100 * high:+.0f}]" if low is not None and high is not None else ""
    elif unit == "share":
        text = pct(value, 0)
        interval = f" [{pct(low, 0)}, {pct(high, 0)}]" if low is not None and high is not None else ""
    elif unit == "ratio":
        return f"{value:.2f}×"
    elif unit == "value":
        text = f"{value:+.3g}"
        interval = f" [{low:+.3g}, {high:+.3g}]" if low is not None and high is not None else ""
    else:
        text = f"{value:.2f}"
        interval = f" [{low:.2f}, {high:.2f}]" if low is not None and high is not None else ""
    return text + interval


def format_q(value: Any) -> str:
    value = number(value)
    if value is None:
        return "–"
    if value < 0.001:
        return "q &lt; 0.001"
    return f"q = {value:.3f}"


def flag_badges(row: dict[str, Any]) -> str:
    return "".join(
        f'<span title="{esc(flag.get("text", ""))}">'
        f'{badge("⚠ " + FLAG_TEXT.get(flag.get("code"), flag.get("code", "")), "mid")}</span> '
        for flag in row.get("flags") or []
    )


def band_badge(band: str | None, prefix: str = "") -> str:
    if not band:
        return badge(prefix + "n/a")
    return badge(prefix + band, BAND_TONE.get(band, "neutral"))


def stat_line(row: dict[str, Any]) -> str:
    strength = number(row.get("strength"))
    strength_text = "–" if strength is None else f"{strength:.1f}"
    if strength is not None and row.get("band"):
        strength_text += f" ({row['band']})"
    low = number(row.get("strength_low"))
    if strength is not None and low is not None and row.get("band_low"):
        strength_text += f", at least {low:.1f} ({row['band_low']})"
    parts = [
        f"<span><b>Effect</b> {esc(format_effect(row))}</span>",
        f"<span><b>Strength</b> {esc(strength_text)}</span>",
        f"<span><b>n</b> {esc(row.get('n'))}</span>",
        f"<span>{format_q(row.get('q_value'))}</span>",
        f'<span><a href="{esc(row.get("link", "#"))}">→ {esc(row.get("source", ""))}</a></span>',
    ]
    return f'<div class="stat-line">{"".join(parts)}</div>'


def reliable_counts(answer: dict[str, Any]) -> str:
    """How many supported results of a question stay large / moderate at the
    lower end of their interval, e.g. "89 reliably large, 81 reliably moderate"."""
    reliable = answer.get("reliable") or {}
    return ", ".join(f"{reliable.get(key, 0):,} reliably {word}"
                     for key, word in (("large", "large"), ("medium", "moderate")))
