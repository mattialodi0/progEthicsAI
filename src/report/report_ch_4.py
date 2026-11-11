"""Renderer for chapter 4: Prompt adherence."""

from __future__ import annotations

import html
from typing import Any

import plotly.graph_objects as go

from .explain import chapter_header, findings, guide, metric_defs
from .insights import chapter_four
from .plotly_helpers import chart, numeric


def _value(value: Any) -> str:
    return "-" if value is None else f"{100 * float(value):.1f}%"


def _heatmap(measurement: dict[str, Any], title: str) -> str:
    labels = measurement.get("labels", [])
    counts = measurement.get("counts") or measurement.get("matrix", {})
    matrix = {
        row: {
            column: numeric(value) / max(1, sum(numeric(item)
                                                for item in counts.get(row, {}).values()))
            for column, value in counts.get(row, {}).items()
        }
        for row in labels
    }
    requested = [row for row in labels if sum(numeric(value) for value in
                                              counts.get(row, {}).values()) > 0]
    figure = go.Figure(
        go.Heatmap(
            x=labels,
            y=requested,
            z=[[100 * numeric(matrix.get(row, {}).get(column)) for column in labels]
               for row in requested],
            text=[[str(measurement.get("counts", {}).get(row, {}).get(column, 0))
                   for column in labels] for row in requested],
            texttemplate="%{z:.0f}%<br>(%{text})",
            colorscale="Blues",
            zmin=0,
            zmax=100,
            colorbar={"title": "% of row"},
            hovertemplate="Requested: %{y}<br>Detected: %{x}<br>%{z:.1f}% of requested"
                          "<br>Count: %{text}<extra></extra>",
        )
    )
    figure.update_xaxes(title=f"Detected {title}", tickangle=-30)
    figure.update_yaxes(title=f"Requested {title}", autorange="reversed")
    figure.update_xaxes(constrain="domain")
    figure.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
    return chart(figure, title="", height=max(340, 170 + 95 * len(requested)))


def _matrix(title: str, measurement: dict[str, Any], *, collapsible: bool = False) -> str:
    labels = measurement.get("labels", [])
    counts = measurement.get("counts") or measurement.get("matrix", {})
    if not labels or not counts or measurement.get("total", 0) <= 0:
        return ""
    stats = (f"<strong>Accuracy:</strong> {_value(measurement.get('accuracy'))} · "
             f"{measurement.get('total', 0)} images with a detection · "
             f"{measurement.get('unmatched', 0)} without")
    body = _heatmap(measurement, title)
    if collapsible:
        return (f"<details><summary>{html.escape(title)} — accuracy "
                f"{_value(measurement.get('accuracy'))} (n={measurement.get('total', 0)})"
                f"</summary>{body}</details>")
    return (f'<article class="plot-card"><h4>Requested vs detected {html.escape(title)}</h4>'
            f'<p class="note">{stats}</p>{body}</article>')


def _accuracy_chart(attributes: dict[str, Any]) -> str:
    names = [name for name, item in attributes.items() if item.get("accuracy") is not None]
    if not names:
        return ""
    values = [100 * numeric(attributes[name].get("accuracy")) for name in names]
    figure = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        text=[f"{value:.1f}%" for value in values], textposition="outside", cliponaxis=False,
        customdata=[[attributes[name].get("total", 0), attributes[name].get("unmatched", 0)]
                    for name in names],
        marker={"color": ["#0072B2" if value >= 80 else "#E69F00" if value >= 60 else "#D55E00"
                          for value in values]},
        hovertemplate="%{y}: %{x:.1f}% adherence<br>%{customdata[0]} images with a detection"
                      "<br>%{customdata[1]} without<extra></extra>",
    ))
    figure.update_layout(xaxis={"title": "Adherence (%)", "range": [0, 105]},
                         yaxis={"autorange": "reversed"})
    return chart(figure, title="", height=60 + 52 * len(names))


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
        if any(groups[value].get("accuracy", {}).get(attribute) is not None for value in values)
    ])
    figure.update_layout(barmode="group", yaxis={"range": [0, 100], "title": "Adherence (%)"},
                         xaxis={"title": f"Requested {category}"},
                         legend={"title": {"text": "Attribute"}})
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
                    None if measurement.get("accuracy", {}).get(attribute) is None
                    else 100 * numeric(measurement.get("accuracy", {}).get(attribute))
                    for measurement in modes.values()
                ],
                customdata=[measurement.get("n_images", 0) for measurement in modes.values()],
                hovertemplate=f"{html.escape(str(attribute))}: %{{y:.1f}}%<br>"
                              "%{customdata} images in mode<extra></extra>",
            )
            for attribute in attribute_names
        ]
    )
    mode_figure.update_layout(
        barmode="group",
        yaxis={"range": [0, 100], "title": "Adherence (%)"},
        xaxis={"title": "Prompt mode"},
        legend={"title": {"text": "Attribute"}},
    )
    mode_chart = chart(mode_figure, title="")
    demographic_html = "".join(
        _demographic_chart(category, groups, attribute_names)
        for category, groups in adherence.get("by_demographic_category", {}).items()
        if groups
    ) or '<p class="note">No demographic adherence breakdown is available.</p>'
    combination_html = "".join(
        _matrix(name, measurement, collapsible=True)
        for name, measurement in combinations.items()
        if measurement.get("total", 0) > 0
    ) or '<p class="note">No two-attribute prompt combinations are available.</p>'

    return f"""
  {chapter_header(4, "Prompt adherence",
      "When a prompt asks for a specific gender, race, age or emotion, does the image "
      "show it — and equally well for every group?",
      "<p>Adherence connects the input prompt to the measured image. It is a "
      "<strong>quality-of-service</strong> question: a model that reliably draws "
      "“a male doctor” but often turns “a female doctor” into a man serves the two groups "
      "unequally, even if it never produces offensive content. Low adherence for one group "
      "also tells you that the model's default (chapter 3) overrides explicit "
      "instructions.</p>")}
  {findings(chapter_four(adherence))}
  {metric_defs([
      ("Adherence (accuracy)", "Images whose detected value equals the requested value ÷ "
       "images that request the value and have a detection.", "100% = always followed. "
       "Compare across groups of the same attribute: the gap is the disparity."),
      ("Unmatched / without detection", "Images that request the attribute but where the "
       "detector returned nothing (usually no face found).", "Reported separately so that "
       "excluding failed detections cannot hide a group-specific failure."),
      ("Confusion matrix cell", "Share of images requesting the row value that were "
       "detected as the column value.", "The diagonal is adherence; off-diagonal cells show "
       "<em>what the model drew instead</em>."),
  ], open_=False)}
  <section id="chapter-4-attributes">
    <h3>4.1 Overall adherence by attribute</h3>
    {guide(
        read=["The bar chart summarises adherence per attribute (blue ≥ 80%, amber 60–80%, "
              "red &lt; 60%).",
              "In each confusion matrix, rows are the requested value and columns the "
              "detected value; each row sums to 100%. Cell text: row percentage and (count)."],
        look=["Rows with a weak diagonal: the requested value is often not drawn.",
              "Asymmetric errors: if “female” is often detected as “male” but never the "
              "reverse, the model drifts towards a default group.",
              "Systematic substitutions (e.g. “old” → “middle-aged”), which show what the "
              "model considers the typical version of a group."],
        caution=["Errors can come from the detector rather than the model, especially for "
                 "age and race. Detector error rates can themselves differ across groups.",
                 "Emotion adherence is limited by the label sets: DeepFace detects angry, "
                 "disgust, fear, happy, sad, surprise and neutral, so requested emotions such "
                 "as calm, bored or nervous can never match exactly.",
                 "Requested and detected label names must match for a hit; a requested value "
                 "with 0% adherence may simply use a different label than the detector "
                 "(e.g. “hispanic” vs “latino hispanic”)."],
    )}
    <article class="plot-card"><h4>Adherence per attribute</h4>{_accuracy_chart(attributes)}</article>
    {attribute_html}
  </section>
  <section id="chapter-4-modes">
    <h3>4.2 Adherence by prompt mode</h3>
    <p>Does adding more attributes to a prompt make the model follow each of them less?</p>
    {guide(
        read="Grouped bars: one colour per attribute, one group per prompt mode. Missing bars "
             "mean the attribute is never requested in that mode.",
        look="Adherence that falls from single-variable to double-variable to control prompts "
             "means attributes compete: the model drops some requested attributes when "
             "several are combined. Compare which attribute is sacrificed.",
    )}
    <article class="plot-card">{mode_chart}</article>
  </section>
  <section id="chapter-4-demographics">
    <h3>4.3 Adherence by demographic category</h3>
    <p>For each requested value of a demographic attribute, the bars show how often each
    requested attribute was detected as requested.</p>
    {guide(
        read="In “By requested gender”, the <em>gender</em> bars give the per-group adherence "
             "to gender itself; the other bars show whether, e.g., race or age is followed "
             "less reliably when the prompt also asks for a woman.",
        look=["A lower bar for one group means prompts mentioning that group are followed "
              "less reliably — an unequal quality of service.",
              "Gaps of more than ~10 percentage points with n ≥ 30 per group deserve "
              "attention; smaller groups need larger gaps to be convincing."],
        caution="A lower bar can also mean the detector is less accurate for that group.",
    )}
    <div class="plot-grid">{demographic_html}</div>
  </section>
  <section id="chapter-4-combinations">
    <h3>4.4 Adherence to combinations of two attributes</h3>
    <p>Joint adherence: both requested attributes must be detected correctly. Expand a
    combination to see its confusion matrix.</p>
    {guide(
        read="Rows are requested pairs (e.g. “female + black”), columns are detected pairs.",
        look="Pairs whose joint adherence is much lower than the adherence of their two "
             "attributes separately: the combination itself is hard for the model (e.g. it "
             "can draw women and can draw Black people, but often fails at Black women). "
             "This is an intersectional quality-of-service gap.",
    )}
    {combination_html}
  </section>
"""
