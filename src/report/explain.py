"""Reusable explanatory blocks for the fairness report.

Every chapter uses the same vocabulary of blocks so readers learn the layout
once:

* ``chapter_header`` – chapter number, title, the question the chapter answers
  and a short introduction.
* ``guide`` – up to three call-outs: *How to read* (what the chart encodes),
  *What to look for* (which patterns would indicate a fairness problem) and
  *Caveats* (why a pattern might not be a model problem).
* ``metric_defs`` – a compact table of metric definitions with their range
  and the direction that matters.
* ``findings`` – observations computed from this dataset (see ``insights``).

All arguments are trusted HTML written in this package; values that come from
the analysis artifact must be escaped with ``esc`` before being passed in.
"""

from __future__ import annotations

import html
from typing import Any, Iterable

_TITLES = {
    "read": "How to read",
    "look": "What to look for",
    "caution": "Caveats",
    "finding": "Key observations in this dataset",
}


def esc(value: Any) -> str:
    """Escape a value for HTML text and attributes."""
    return html.escape(str(value))


def pct(value: Any, digits: int = 1) -> str:
    """Format a 0-1 proportion as a percentage."""
    try:
        return f"{100 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def label(value: Any) -> str:
    """Turn an artifact key such as ``toxic_trait`` into ``Toxic trait``."""
    text = str(value).replace("__", " × ").replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def _body(items: str | Iterable[str]) -> str:
    if isinstance(items, str):
        return f"<p>{items}</p>"
    return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def callout(kind: str, body: str | Iterable[str], title: str | None = None) -> str:
    """Return one coloured call-out box."""
    heading = title or _TITLES.get(kind, kind)
    return (
        f'<aside class="callout callout-{esc(kind)}">'
        f'<div class="callout-title">{esc(heading)}</div>{_body(body)}</aside>'
    )


def guide(
    *,
    read: str | Iterable[str] | None = None,
    look: str | Iterable[str] | None = None,
    caution: str | Iterable[str] | None = None,
) -> str:
    """Return the standard reading guide for a section."""
    parts = [
        callout(kind, body)
        for kind, body in (("read", read), ("look", look), ("caution", caution))
        if body
    ]
    return f'<div class="guide">{"".join(parts)}</div>' if parts else ""


def chapter_header(number: int, title: str, question: str, intro: str) -> str:
    """Return the chapter eyebrow, title, key question and introduction."""
    return (
        f'<div class="eyebrow">Chapter {number}</div>'
        f'<h2 id="chapter-{number}">{esc(title)}</h2>'
        f'<p class="question"><span>Key question</span>{question}</p>'
        f'<div class="intro">{intro}</div>'
    )


def metric_defs(
    entries: Iterable[tuple[str, str, str]],
    *,
    title: str = "Metrics used in this section",
    open_: bool = False,
) -> str:
    """Return a collapsible table of (metric, definition, interpretation)."""
    rows = "".join(
        f"<tr><th scope='row'>{name}</th><td>{definition}</td><td>{reading}</td></tr>"
        for name, definition, reading in entries
    )
    return (
        f'<details class="defs"{" open" if open_ else ""}><summary>{esc(title)}</summary>'
        "<table><thead><tr><th>Metric</th><th>Definition</th>"
        "<th>How to interpret it</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></details>"
    )


def findings(items: Iterable[str], *, title: str | None = None) -> str:
    """Return the auto-generated observations box, or nothing if empty."""
    items = [item for item in items if item]
    if not items:
        return ""
    return callout("finding", items, title).replace(
        "</aside>",
        '<p class="callout-foot">Computed automatically from the analysis '
        "artifact. These are descriptive patterns in this sample, not verdicts: "
        "read them together with the caveats.</p></aside>",
    )


def badge(text: str, tone: str = "neutral") -> str:
    """Return a small inline pill (tones: neutral, low, mid, high)."""
    return f'<span class="badge badge-{esc(tone)}">{esc(text)}</span>'


def cramers_band(value: float | None) -> tuple[str, str]:
    """Return a conventional strength label and badge tone for Cramér's V."""
    if value is None:
        return "n/a", "neutral"
    if value < 0.1:
        return "negligible", "neutral"
    if value < 0.3:
        return "weak", "low"
    if value < 0.5:
        return "moderate", "mid"
    return "strong", "high"
