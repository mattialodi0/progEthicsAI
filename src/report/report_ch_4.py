"""Renderer for chapter 4: Prompt adherence."""

from __future__ import annotations

import html
from typing import Any

import plotly.graph_objects as go

from .plotly_helpers import chart, numeric


def _value(value: Any) -> str:
    return "-" if value is None else f"{100 * float(value):.1f}%"


def _matrix(title: str, measurement: dict[str, Any]) -> str:
    labels = measurement.get("labels", [])
    counts = measurement.get("counts") or measurement.get("matrix", {})
    if not labels or not counts or measurement.get("total", 0) <= 0:
        return ""
    matrix = {
        row: {
            column: numeric(value) / max(1, sum(numeric(item)
                                                for item in counts.get(row, {}).values()))
            for column, value in counts.get(row, {}).items()
        }
        for row in labels
    }
    figure = go.Figure(
        go.Heatmap(
            x=labels,
            y=labels,
            z=[
                [100 * numeric(matrix.get(row, {}).get(column))
                 for column in labels]
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


def _demographic_chart(category: str, groups: dict[str, Any], attributes: list[str]) -> str:
    """Adherence to each attribute, split by the requested value of ``category``."""
    values = list(groups)
    figure = go.Figure([
        go.Bar(
            name=str(attribute),
            x=values,
            y=[100 * numeric(groups[value].get("accuracy", {}).get(attribute)) for value in values],
            customdata=[groups[value].get("n_images", 0) for value in values],
            hovertemplate=(
                f"Requested {html.escape(category)}: %{{x}}<br>{html.escape(str(attribute))} "
                "adherence: %{y:.1f}%<br>%{customdata} images<extra></extra>"
            ),
        )
        for attribute in attributes
    ])
    figure.update_layout(barmode="group", yaxis={"range": [0, 100], "title": "Accuracy (%)"},
                         xaxis={"title": f"Requested {category}"})
    return (
        f'<article class="plot-card"><h4>By requested {html.escape(category)}</h4>'
        f"{chart(figure, title='')}</article>"
    )


def render_chapter_four(adherence: dict[str, Any]) -> str:
    """Render measurements computed by the analysis pipeline."""
    attributes = adherence.get("by_attribute", {})
    modes = adherence.get("by_prompt_mode", {})
    combinations = adherence.get("combinations", {})
    attribute_names = adherence.get("attributes", [])
    attribute_html = "".join(
        _matrix(attribute, measurement)
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
    demographic_html = "".join(
        _demographic_chart(category, groups, attribute_names)
        for category, groups in adherence.get("by_demographic_category", {}).items()
        if groups
    ) or '<p class="note">No demographic adherence breakdown is available.</p>'
    combination_html = "".join(
        _matrix(name, measurement)
        for name, measurement in combinations.items()
        if measurement.get("total", 0) > 0
    ) or '<p class="note">No two-attribute prompt combinations are available.</p>'

    return f"""
  <div class="eyebrow">Chapter 4</div>
  <h2 id="chapter-4">Prompt adherence</h2>
  <p class="subtitle">This chapter measures whether the image-generation model follows the attributes explicitly requested in the prompt.</p>
  <p class="subtitle">Section 4.1 shows the overall adherence by attribute, that is, how many of the attributes requested in the prompt are present in the generated images.
    We then examine the adherence by prompt mode and by requested demographic category, expressed as accuracy, and for combinations of two attributes.</p>
  <section id="chapter-4-attributes">
    <h3>4.1 Overall adherence by attribute</h3>
    <div>{attribute_html}</div>
  </section>
  <section id="chapter-4-modes">
    <h3>4.2 Adherence by prompt mode</h3>
    {mode_chart}
  </section>
  <section id="chapter-4-demographics">
    <h3>4.3 Adherence by demographic category</h3>
    <p>For each requested value of a demographic attribute, the bars show how
    often each requested attribute was detected as requested. A lower bar for one
    group means the model follows prompts less reliably for that group (or the
    detector is less accurate for it).</p>
    <div>{demographic_html}</div>
  </section>
  <section id="chapter-4-combinations">
    <h3>4.4 Adherence to combinations of two attributes</h3>
    <div>{combination_html}</div>
  </section>
"""
