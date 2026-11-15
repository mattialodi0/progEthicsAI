"""Chapter 2: HTML tables for detection rates, classifier confidence and
prompt-attribute counts."""

from __future__ import annotations

import html
from typing import Any

from ..explain import esc, guide, label
from ..plotly_helpers import numeric


def _bar_class(rate: float) -> str:
    return "bar bad" if rate < 70 else "bar warn" if rate < 90 else "bar"


def _rate_bar(percent: float, what: str) -> str:
    return (f"<div class='{_bar_class(percent)}' role='img' aria-label='{percent:.1f}% "
            f"{what}'><span style='width:{percent:.1f}%'></span></div>")


def detection_table(detection: dict[str, Any]) -> str:
    """Valid / total extractions per feature, with a rate bar."""

    def row(name: str) -> str:
        item = detection[name]
        percent = 100 * numeric(item["valid"]) / item["total"] if item["total"] else 0
        return (
            f"<tr><th scope='row'>{html.escape(label(name))}</th>"
            f"<td>{item['valid']} / {item['total']}</td>"
            f"<td>{_rate_bar(percent, 'valid')}</td>"
            f"<td>{percent:.1f}%</td><td>{item['total'] - item['valid']}</td></tr>"
        )

    return (
        "<table><thead><tr><th>Feature</th><th>Valid / total</th>"
        "<th>Rate</th><th>Percent</th><th>Missing</th></tr></thead><tbody>"
        + "".join(row(name) for name in detection)
        + "</tbody></table>"
    )


def missing_outputs_table(missing: dict[str, Any]) -> str:
    if not missing:
        return ""
    return (
        "<h4>Missing outputs</h4><table><thead><tr><th>Output</th><th>Missing values</th>"
        "</tr></thead><tbody>" + "".join(
            f"<tr><th scope='row'>{esc(label(name))}</th><td>{esc(value)}</td></tr>"
            for name, value in missing.items())
        + "</tbody></table><p class='note'>Missing demographic detections are counted per "
        "feature (gender, race, age and emotion each), so one image without a face adds 4."
        "</p>"
    )


def _confidence_table(first_column: str, rows: dict[str, Any], threshold: str) -> str:

    def row(name: str, item: dict[str, Any]) -> str:
        percent = 100 * numeric(item.get("confident")) / item["n"] if item.get("n") else 0
        return (
            f"<tr><th scope='row'>{html.escape(label(name))}</th><td>{item.get('n', 0)}</td>"
            f"<td>{100 * numeric(item.get('median')):.0f}%</td>"
            f"<td>{_rate_bar(percent, 'confident')}</td>"
            f"<td>{percent:.1f}%</td></tr>"
        )

    return (
        f"<table><thead><tr><th>{first_column}</th><th>Images with a face</th>"
        f"<th>Median top probability</th><th>Confident labels (≥ {threshold})</th>"
        "<th>Percent</th></tr></thead><tbody>"
        + "".join(row(name, item) for name, item in rows.items())
        + "</tbody></table>"
    )


def confidence_section(confidence: dict[str, Any]) -> str:
    """How often the face model's top class is a clear winner, per attribute."""
    attributes = {
        name: item for name, item in (confidence.get("attributes") or {}).items()
        if item.get("n")
    }
    if not attributes:
        return ""
    threshold = f"{100 * numeric(confidence.get('threshold'), 0.6):.0f}%"
    # Break down the attribute whose labels are least often confident.
    weakest = min(attributes, key=lambda name: (
        numeric(attributes[name].get("confident")) / attributes[name]["n"]))
    breakdown = {name: item for name, item in
                 (attributes[weakest].get("by_requested") or {}).items()
                 if item.get("n")}
    return (
        "<h4>Classifier confidence</h4>"
        "<p>A detected value can be available and still uncertain. For gender, race and "
        "emotion the face model returns a probability per class and the report uses the "
        "most likely one; the table shows how often that class is a clear winner.</p>"
        + guide(
            look="An attribute or a requested group with few confident labels: its "
                 "detected values are close to a guess between two classes, so adherence "
                 "and bias results about it deserve less weight (chapter 4 repeats the "
                 "adherence check without the uncertain labels).",
            caution="The probabilities are not calibrated: a 60% label is not right "
                    "exactly 60% of the time. Use the shares to compare attributes and "
                    "groups, not as error rates.",
        )
        + _confidence_table("Feature", attributes, threshold)
        + (f"<h4>{html.escape(label(weakest))} confidence by requested "
           f"{html.escape(weakest)}</h4>"
           + _confidence_table(f"Requested {html.escape(weakest)}", breakdown, threshold)
           if len(breakdown) > 1 else "")
    )


def attribute_sections(attribute_counts: dict[str, dict[str, int]], total: float) -> str:
    """One card per prompt variable with the image count of each value."""

    def section(attribute: str, values: dict[str, int]) -> str:
        top = max((numeric(count) for count in values.values()), default=0)
        rows = ""
        for value, count in sorted(values.items(), key=lambda item: -numeric(item[1])):
            width = 100 * numeric(count) / top if top else 0
            share = 100 * numeric(count) / total if total else 0
            rows += (
                f"<tr><th scope='row'>{html.escape(str(value))}</th><td>{count}</td>"
                f"<td><div class='bar' title='{share:.1f}% of all images'>"
                f"<span style='width:{width:.1f}%'></span></div></td></tr>"
            )
        return (
            f"<article class='plot-card'><h4>{html.escape(attribute)}</h4>"
            "<table><thead><tr><th>Value</th><th>Images</th><th>Relative count</th></tr>"
            f"</thead><tbody>{rows}</tbody></table></article>"
        )

    return "".join(
        section(label(attribute), values) for attribute, values in attribute_counts.items()
    )
