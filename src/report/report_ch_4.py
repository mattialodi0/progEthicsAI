"""Renderer for chapter 4: Prompt adherence."""

from __future__ import annotations

import html
from typing import Any


def _value(value: Any) -> str:
    return "-" if value is None else f"{100 * float(value):.1f}%"


def _matrix_value(value: Any) -> str:
    return "-" if value is None else f"{100 * float(value):.1f}%"


def _matrix(title: str, measurement: dict[str, Any]) -> str:
    labels = measurement.get("labels", [])
    header = (
        "<tr><th>Requested / detected</th>"
        + "".join(f"<th>{html.escape(str(label))}</th>" for label in labels)
        + "</tr>"
    )
    body = "".join(
        "<tr><th>"
        + html.escape(str(row))
        + "</th>"
        + "".join(
            f"<td>{_matrix_value(measurement.get('matrix', {}).get(row, {}).get(column, 0))}"
            f"<small> ({measurement.get('counts', {}).get(row, {}).get(column, 0)})</small></td>"
            for column in labels
        )
        + "</tr>"
        for row in labels
    )
    return (
        f"<article class=\"plot-card\"><h4>{html.escape(title)}</h4>"
        f"<p><strong>Accuracy:</strong> {_value(measurement.get('accuracy'))} "
        f"({measurement.get('total', 0)} matched; {measurement.get('unmatched', 0)} unmatched)</p>"
        f"<table><thead>{header}</thead><tbody>{body}</tbody></table></article>"
    )


def render_chapter_four(analysis_results: dict[str, Any]) -> str:
    """Render measurements computed by the analysis pipeline."""
    adherence = analysis_results.get("general_stats", {}).get("prompt_adherence", {})
    attributes = adherence.get("by_attribute", {})
    modes = adherence.get("by_prompt_mode", {})
    categories = adherence.get("by_demographic_category", {})
    combinations = adherence.get("combinations", {})
    attribute_names = adherence.get("attributes", [])
    mode_headers = "".join(f"<th>{html.escape(str(name))}</th>" for name in attribute_names)

    attribute_html = "".join(
        _matrix(attribute, measurement) for attribute, measurement in attributes.items()
    ) or '<p class="note">No adherence observations are available.</p>'
    mode_rows = "".join(
        "<tr><td>"
        + html.escape(str(mode))
        + f"</td><td>{measurement.get('n_images', 0)}</td>"
        + "".join(
            f"<td>{_value(measurement.get('accuracy', {}).get(attribute))}</td>"
            for attribute in attribute_names
        )
        + "</tr>"
        for mode, measurement in modes.items()
    )
    category_html = "".join(
        "<article class=\"plot-card\"><h4>"
        + html.escape(str(category))
        + "</h4><table><thead><tr><th>Requested category</th><th>Images</th>"
        + mode_headers
        + "</tr></thead><tbody>"
        + "".join(
            "<tr><td>"
            + html.escape(str(value))
            + f"</td><td>{measurement.get('n_images', 0)}</td>"
            + "".join(
                f"<td>{_value(measurement.get('accuracy', {}).get(attribute))}</td>"
                for attribute in attribute_names
            )
            + "</tr>"
            for value, measurement in groups.items()
        )
        + "</tbody></table></article>"
        for category, groups in categories.items()
    )
    combination_html = "".join(
        _matrix(name, measurement) for name, measurement in combinations.items()
    ) or '<p class="note">No two-attribute prompt combinations are available.</p>'

    return f"""
  <div class="eyebrow">Chapter 4</div>
  <h2 id="chapter-4">Prompt adherence</h2>
  <p class="subtitle">Whether explicitly requested prompt attributes are present in
  the detected image features. All measurements below are precomputed by the
  analysis pipeline.</p>
  <section id="chapter-4-attributes">
    <h3>4.1 Overall adherence by attribute</h3>
    <div class="plot-grid">{attribute_html}</div>
  </section>
  <section id="chapter-4-modes">
    <h3>4.2 Adherence by prompt mode</h3>
    <table><thead><tr><th>Prompt mode</th><th>Images</th>{mode_headers}</tr></thead>
    <tbody>{mode_rows}</tbody></table>
  </section>
  <section id="chapter-4-demographics">
    <h3>4.3 Adherence by demographic category</h3>
    <div class="plot-grid">{category_html}</div>
  </section>
  <section id="chapter-4-combinations">
    <h3>4.4 Adherence to combinations of two attributes</h3>
    <div class="plot-grid">{combination_html}</div>
  </section>
"""