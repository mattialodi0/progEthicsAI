"""Renderer for chapter 4: Prompt adherence."""

from __future__ import annotations

import html
from typing import Any

import plotly.graph_objects as go

from .plotly_helpers import chart, numeric


def _value(value: Any) -> str:
    return "-" if value is None else f"{100 * float(value):.1f}%"


def _matrix_value(value: Any) -> str:
    return "-" if value is None else f"{100 * float(value):.1f}%"


def _race_measurement(measurement: dict[str, Any]) -> dict[str, Any]:
    """Merge equivalent Hispanic labels before rendering the race matrix."""
    if not measurement or not measurement.get("counts"):
        return measurement
    counts: dict[str, dict[str, int]] = {}
    normalize = lambda value: (
        "hispanic" if str(value).strip().lower() in {"latino hispanic", "hispanic"}
        else str(value)
    )
    for row, columns in measurement["counts"].items():
        normalized_row = normalize(row)
        target = counts.setdefault(normalized_row, {})
        for column, value in columns.items():
            normalized_column = normalize(column)
            target[normalized_column] = target.get(normalized_column, 0) + int(value)
    labels = sorted(set(counts) | {column for row in counts.values() for column in row})
    return {
        **measurement,
        "labels": labels,
        "counts": {
            row: {column: counts.get(row, {}).get(column, 0) for column in labels}
            for row in labels
        },
    }


def _matrix(title: str, measurement: dict[str, Any]) -> str:
    labels = measurement.get("labels", [])
    counts = measurement.get("counts") or measurement.get("matrix", {})
    if not labels or not counts or measurement.get("total", 0) <= 0:
        return ""
    matrix = {
        row: {
            column: numeric(value) / max(1, sum(numeric(item) for item in counts.get(row, {}).values()))
            for column, value in counts.get(row, {}).items()
        }
        for row in labels
    }
    figure = go.Figure(
        go.Heatmap(
            x=labels,
            y=labels,
            z=[
                [100 * numeric(matrix.get(row, {}).get(column)) for column in labels]
                for row in labels
            ],
            text=[
                [
                    str(measurement.get("counts", {}).get(row, {}).get(column, 0))
                    for column in labels
                ]
                for row in labels
            ],
            texttemplate="%{z:.1f}%<br>(%{text})",
            colorscale="Blues",
            zmin=0,
            zmax=100,
            colorbar={"title": "%"},
            hovertemplate="Requested: %{y}<br>Detected: %{x}<br>%{z:.1f}%<br>Count: %{text}<extra></extra>",
        )
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
    matrix_height = max(360, 120 + 60 * len(labels))
    return (
        f"<article class=\"plot-card\"><h4>{html.escape(title)}</h4>"
        f"<p><strong>Accuracy:</strong> {_value(measurement.get('accuracy'))} "
        f"({measurement.get('total', 0)} matched; {measurement.get('unmatched', 0)} unmatched)</p>"
        f"{chart(figure, title='Requested vs detected attributes', height=matrix_height)}</article>"
    )


def render_chapter_four(adherence: dict[str, Any]) -> str:
    """Render measurements computed by the analysis pipeline."""
    attributes = adherence.get("by_attribute", {})
    modes = adherence.get("by_prompt_mode", {})
    combinations = adherence.get("combinations", {})
    attribute_names = adherence.get("attributes", [])
    attribute_html = "".join(
        _matrix(
            attribute,
            _race_measurement(measurement) if attribute == "race" else measurement,
        )
        for attribute, measurement in attributes.items()
    ) or '<p class="note">No adherence observations are available.</p>'
    mode_figure = go.Figure(
        [
            go.Bar(
                name=str(attribute),
                x=list(modes),
                y=[
                    100 * numeric(measurement.get("accuracy", {}).get(attribute))
                    for measurement in modes.values()
                ],
            )
            for attribute in attribute_names
        ]
    )
    mode_figure.update_layout(
        barmode="group",
        yaxis={"range": [0, 100], "title": "Accuracy (%)"},
    )
    mode_chart = chart(mode_figure, title="Adherence accuracy by prompt mode (%)")
    combination_html = "".join(
        _matrix(name, measurement)
        for name, measurement in combinations.items()
        if measurement.get("total", 0) > 0
    ) or '<p class="note">No two-attribute prompt combinations are available.</p>'

    return f"""
  <div class="eyebrow">Chapter 4</div>
  <h2 id="chapter-4">Prompt adherence</h2>
  <p class="subtitle">Whether explicitly requested prompt attributes are present in
  the detected image features. All measurements below are precomputed by the
  analysis pipeline.</p>
  <section id="chapter-4-attributes">
    <h3>4.1 Overall adherence by attribute</h3>
    <div class="">{attribute_html}</div>
  </section>
  <section id="chapter-4-modes">
    <h3>4.2 Adherence by prompt mode</h3>
    {mode_chart}
  </section>
  <section id="chapter-4-combinations">
    <h3>4.3 Adherence to combinations of two attributes</h3>
    <div class="">{combination_html}</div>
  </section>
"""