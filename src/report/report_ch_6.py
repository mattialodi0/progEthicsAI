"""Renderer for chapter 6: matched counterfactual analysis."""

from __future__ import annotations

import html
import json
from typing import Any

import plotly.graph_objects as go

from .plotly_helpers import chart, numeric


def _bar_chart(data: dict[str, Any]) -> str:
    attributes = data.get("attributes", [])
    values = [
        100 * numeric(data.get("by_attribute", {}).get(attribute, {})
                      .get("detected_changes", {}).get("rate"))
        for attribute in attributes
    ]
    figure = go.Figure(go.Bar(x=attributes, y=values, text=[f"{value:.1f}%" for value in values]))
    figure.update_yaxes(title="Changed detected category (%)", range=[0, 100])
    return chart(figure, title="Detected attribute change rate", height=360)


def _transition_heatmap(data: dict[str, Any]) -> str:
    transitions = data.get("by_attribute", {})
    labels = sorted({
        label for item in transitions.values()
        for label in item.get("detected_changes", {}).get("transitions", {})
    })
    if not labels:
        return '<p class="note">No matched categorical transitions are available.</p>'
    figure = go.Figure(go.Heatmap(
        x=labels,
        y=list(transitions),
        z=[
            [numeric(transitions[attribute].get("detected_changes", {})
                     .get("transitions", {}).get(label))
             for label in labels]
            for attribute in transitions
        ],
        colorscale="Blues",
        colorbar={"title": "Pairs"},
    ))
    return chart(figure, title="Heatmap of detected category changes", height=460)


def _numerical_plots(data: dict[str, Any]) -> str:
    figures = []
    for feature in data.get("numerical_features", []):
        figure = go.Figure()
        for attribute in data.get("attributes", []):
            values = data.get("by_attribute", {}).get(attribute, {}).get(
                "numerical_differences", {}).get(feature, {}).get("values", [])
            if values:
                figure.add_trace(go.Box(y=values, name=attribute, boxmean=True))
        if figure.data:
            figure.update_yaxes(title="Counterfactual difference")
            figures.append(chart(figure, title=f"Paired differences: {feature}", height=360))
    return "".join(f'<article class="plot-card">{figure}</article>' for figure in figures) or (
        '<p class="note">No paired numerical features are available.</p>'
    )


def _embedding_plots(data: dict[str, Any]) -> str:
    attributes = data.get("attributes", [])
    series = {}
    for label, key in (
        ("Paired cosine distance", "paired_cosine_distance"),
        ("Within-condition Euclidean distance", "within_condition_euclidean_distance"),
        ("Between-condition Euclidean distance", "between_pair_euclidean_distance"),
    ):
        series[label] = [
            numeric(data.get("by_attribute", {}).get(attribute, {}).get("embeddings", {})
                    .get(key, {}).get("mean"))
            for attribute in attributes
        ]
    figure = go.Figure([
        go.Bar(name=label, x=attributes, y=values)
        for label, values in series.items()
    ])
    figure.update_layout(barmode="group", yaxis={"title": "Mean distance"})
    return chart(figure, title="Embedding distances within and between conditions", height=390)


def _metrics_table(data: dict[str, Any]) -> str:
    rows = []
    for attribute in data.get("attributes", []):
        item = data.get("by_attribute", {}).get(attribute, {})
        adherence = item.get("adherence", {})
        adherence_text = ", ".join(
            f"{condition}: {100 * numeric(stats.get('rate')):.1f}%"
            for condition, stats in adherence.items()
        ) or "-"
        embedding_mean = item.get("embeddings", {}).get(
            "paired_cosine_distance", {}).get("mean")
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(attribute), item.get("pairs", 0),
                f"{100 * numeric(item.get('detected_changes', {}).get('rate')):.1f}%",
                html.escape(adherence_text),
                "-" if embedding_mean is None else f"{embedding_mean:.4f}",
            )
        )
    return (
        "<table><thead><tr><th>Attribute</th><th>Matched pairs</th>"
        "<th>Detected change</th><th>Example adherence</th>"
        "<th>Mean paired cosine distance</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def render_chapter_six(data: dict[str, Any]) -> str:
    """Render metrics and plots from the precomputed chapter-6 artifact."""
    serialized = json.dumps(data, ensure_ascii=True).replace("</", "<\\/")
    return f"""
    <div id="chapter-container">
  <div class="eyebrow">Chapter 6</div>
  <h2 id="chapter-6">Counterfactual analysis</h2>
  <p class="subtitle">Matched prompts differ only in the investigated attribute.
  Metrics and plots below are computed before report rendering.</p>
  <section id="chapter-6-overview">
    <h3>6.1 and 6.2 Overview</h3>
    {_metrics_table(data)}
    <p class="note">{html.escape(str(data.get("caveat", "")))}</p>
  </section>
  <section id="chapter-6-features">
    <h3>6.1 Counterfactual analysis from extracted features</h3>
    <p>These results compare detected categories, adherence, and paired numerical
    feature differences for prompt-matched counterfactuals.</p>
    <div class="plot-grid"><article class="plot-card">{_bar_chart(data)}</article>
    <article class="plot-card">{_transition_heatmap(data)}</article></div>
    <div class="plot-grid">{_numerical_plots(data)}</div>
  </section>
  <section id="chapter-6-embeddings">
    <h3>6.2 Counterfactual analysis from embeddings</h3>
    <p>Cosine distance measures semantic change between paired image embeddings.
    Distances are descriptive and should not be interpreted as bias without
    considering the intended attribute change.</p>
    <article class="plot-card">{_embedding_plots(data)}</article>
  </section>
  <script>window.chapterSixData = {serialized};</script>
</div>
"""