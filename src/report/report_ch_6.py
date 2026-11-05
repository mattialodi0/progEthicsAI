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
    return chart(figure, title="", height=360)


def _transition_selector(
    data: dict[str, Any],
    *,
    prefix: str,
    description: str,
    double: bool = False,
) -> str:
    serialized = json.dumps(data, ensure_ascii=True).replace("</", "<\\/")
    return f"""
            <p>{description}</p>
            <div class="checkbox-grid" aria-label="{prefix} selectors">
              <label>Changing attribute
                <select id="{prefix}-variable"></select>
              </label>
              <label>Before
                <select id="{prefix}-start"></select>
              </label>
              <label>After
                <select id="{prefix}-end"></select>
              </label>
              <div id="{prefix}-fixed-selectors"></div>
            </div>
            <div id="{prefix}-plots" aria-live="polite"></div>
            <script>
              const {prefix}Data = {serialized};
              const {prefix}Title = (value) => String(value).replaceAll("_", " ")
                .replace(/\b\w/g, (letter) => letter.toUpperCase());
              const {prefix}Variable = document.getElementById("{prefix}-variable");
              const {prefix}Start = document.getElementById("{prefix}-start");
              const {prefix}End = document.getElementById("{prefix}-end");
              const {prefix}Fixed = document.getElementById("{prefix}-fixed-selectors");
              const {prefix}Plots = document.getElementById("{prefix}-plots");
              const {prefix}Pairs = {prefix}Data.pairs || [];
              const {prefix}Attributes = {prefix}Data.control_attributes || [];
              const {prefix}Detected = Array.from(new Set(
                {prefix}Data.detected_attributes || []));
              const {prefix}Option = (value) =>
                '<option value="' + value + '">' + {prefix}Title(value) + "</option>";
              const {prefix}Missing = (value) => value === null || value === undefined
                ? "missing" : String(value);
              const {prefix}Render = () => {{
                const variable = {prefix}Variable.value;
                const available = {{}};
                {prefix}Pairs.filter((pair) => pair.variable_attribute === variable)
                  .forEach((pair) => {{
                    {(" (available[pair.fixed_attribute] ||= new Set()).add(pair.fixed_value);" if double else
                       "Object.entries(pair.fixed_values || {}).forEach(([attribute, value]) => " +
                       "(available[attribute] ||= new Set()).add(value));")}
                  }});
                const fixedAttributes = {(
                    "Object.keys(available).sort()"
                    if double else
                    f"{prefix}Attributes.filter((attribute) => attribute !== variable)"
                )};
                {(
                    f'''{prefix}Fixed.innerHTML =
                  '<label>Fixed attribute<select id="{prefix}-fixed-attribute">' +
                  fixedAttributes.map({prefix}Option).join("") +
                  '</select></label><label>Fixed value<select id="{prefix}-fixed-value"></select></label>';
                const fixedAttribute = document.getElementById("{prefix}-fixed-attribute");
                const fixedValue = document.getElementById("{prefix}-fixed-value");
                const updateFixedValues = () => {{
                  fixedValue.innerHTML = Array.from(available[fixedAttribute.value] || [])
                    .sort().map({prefix}Option).join("");
                }};
                fixedAttribute.addEventListener("change", () => {{
                  updateFixedValues();
                  draw();
                }});
                updateFixedValues();'''
                    if double else
                    f'''{prefix}Fixed.innerHTML = fixedAttributes.map((attribute) => {{
                  const values = Array.from(available[attribute] || []).sort();
                  return "<label>Fix " + {prefix}Title(attribute) +
                    '<select data-attribute="' + attribute + '">' +
                    values.map({prefix}Option).join("") + "</select></label>";
                }}).join("");'''
                )}
                const transitions = Array.from(new Set(
                  {prefix}Pairs.filter((pair) => pair.variable_attribute === variable)
                    .flatMap((pair) => pair.variable_values || [])
                )).sort();
                const previousStart = {prefix}Start.value;
                const previousEnd = {prefix}End.value;
                {prefix}Start.innerHTML = transitions.map({prefix}Option).join("");
                {prefix}End.innerHTML = transitions.map({prefix}Option).join("");
                {prefix}Start.value = transitions.includes(previousStart)
                  ? previousStart : (transitions[0] || "");
                {prefix}End.value = transitions.includes(previousEnd)
                  ? previousEnd : (transitions[transitions.length - 1] || "");
                const selected = () => {(
                    f'''({{
                  attribute: document.getElementById("{prefix}-fixed-attribute").value,
                  value: document.getElementById("{prefix}-fixed-value").value
                }})'''
                    if double else
                    f'''Object.fromEntries(
                  Array.from({prefix}Fixed.querySelectorAll("select")).map((select) =>
                    [select.dataset.attribute, select.value]))'''
                )};
                // Pairs are stored with their two values in sorted order; accept
                // either order and orient each pair as before -> after.
                const matching = () => {prefix}Pairs.flatMap((pair) => {{
                  if (pair.variable_attribute !== variable) return [];
                  const chosen = selected();
                  const fixedMatches = {(
                      "pair.fixed_attribute === chosen.attribute && pair.fixed_value === chosen.value"
                      if double else
                      "Object.entries(chosen).every(([attribute, value]) => " +
                      "pair.fixed_values && pair.fixed_values[attribute] === value)"
                  )};
                  if (!fixedMatches) return [];
                  const [left, right] = pair.variable_values || [];
                  if (left === {prefix}Start.value && right === {prefix}End.value) return [pair];
                  if (left === {prefix}End.value && right === {prefix}Start.value) {{
                    return [{{...pair, detected: {{
                      first: pair.detected.second, second: pair.detected.first
                    }}}}];
                  }}
                  return [];
                }});
                const draw = () => {{
                  const pairs = matching();
                  if ({prefix}Start.value === {prefix}End.value) {{
                    {prefix}Plots.innerHTML =
                      '<p class="note">Choose two different values for Before and After.</p>';
                    return;
                  }}
                  if (!pairs.length) {{
                    {prefix}Plots.innerHTML =
                      '<p class="note">No matched pairs fit these selections.</p>';
                    return;
                  }}
                  const attributes = {prefix}Detected.filter(
                    (attribute) => attribute !== variable);
                  const labels = attributes.flatMap((attribute) =>
                    [{prefix}Title(attribute) + ' (before)', {prefix}Title(attribute) + ' (after)']);
                  const categories = Array.from(new Set(attributes.flatMap((attribute) =>
                    pairs.flatMap((pair) => [
                      {prefix}Missing(pair.detected.first[attribute]),
                      {prefix}Missing(pair.detected.second[attribute])
                    ])))).sort();
                  const counts = (attribute, side, category) => pairs.filter((pair) =>
                    {prefix}Missing(pair.detected[side][attribute]) === category).length;
                  {prefix}Plots.innerHTML = '<div id="{prefix}-plot"></div>';
                  const traces = categories.map((category) => ({{
                    type: "bar",
                    name: {prefix}Title(category),
                    x: labels,
                    y: attributes.flatMap((attribute) => [
                      counts(attribute, "first", category),
                      counts(attribute, "second", category)
                    ]),
                    text: attributes.flatMap((attribute) => [
                      counts(attribute, "first", category) ? {prefix}Title(category) : "",
                      counts(attribute, "second", category) ? {prefix}Title(category) : ""
                    ]),
                    textposition: "inside",
                    insidetextanchor: "middle",
                    hovertemplate: "%{{x}}<br>Detected: %{{text}}<br>Pairs: %{{y}}<extra></extra>"
                  }}));
                  Plotly.react(document.getElementById("{prefix}-plot"), traces, {{
                    barmode: "stack",
                    height: Math.max(420, attributes.length * 80),
                    margin: {{l: 65, r: 25, t: 20, b: 100}},
                    yaxis: {{title: "Matched pairs"}},
                    template: "plotly_white"
                  }}, {{responsive: true, displaylogo: false}});
                }};
                {(
                    f'''document.getElementById("{prefix}-fixed-value")
                  .addEventListener("change", draw);'''
                    if double else
                    f'''{prefix}Fixed.querySelectorAll("select").forEach((select) =>
                  select.addEventListener("change", draw));'''
                )}
                // Assigned, not added: Render runs on every attribute change and
                // addEventListener would stack handlers bound to stale closures.
                {prefix}Start.onchange = draw;
                {prefix}End.onchange = draw;
                draw();
              }};
              {prefix}Variable.innerHTML = {prefix}Attributes.map({prefix}Option).join("");
              {prefix}Variable.addEventListener("change", {prefix}Render);
              if ({prefix}Attributes.length) {prefix}Render();
              else {prefix}Plots.innerHTML =
                '<p class="note">No matching analysis is available.</p>';
            </script>
"""


def _numerical_plots(data: dict[str, Any]) -> str:
    figures = []
    features = data.get("numerical_features", [])
    for feature in features:
        figure = go.Figure()
        for attribute in data.get("attributes", []):
            values = data.get("by_attribute", {}).get(attribute, {}).get(
                "numerical_differences", {}).get(feature, {}).get("values", [])
            if values:
                figure.add_trace(go.Box(y=values, name=attribute, boxmean=True))
        if figure.data:
            figure.update_yaxes(title="Counterfactual difference")
            figures.append(chart(figure, title="", height=360))
    return "".join(
        f'<article class="plot-card"><h4>6.1.{4 + index} Paired differences: '
        f'{html.escape(features[index].replace("_", " "))}</h4>{figure}</article>'
        for index, figure in enumerate(figures)
    ) or '<p class="note">No paired numerical features are available.</p>'


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
    return f"""
  <div class="eyebrow">Chapter 6</div>
  <h2 id="chapter-6">Counterfactual analysis</h2>
  <p class="subtitle">Counterfactual analysis investigates what changes
    when one prompt attribute changes while the rest of the prompt remains fixed.
    The features and metrics we will use are:</p>
    <ul>
      <li>Distribution of detected features: the distribution of attributes detected by the model in the generated images.</li>
      <li>Embeddings: the image embeddings extracted from the generated images.</li>
      <li>Detected change: the percentage of matched pairs where the detected categorical features changed.</li>
      <li>Example adherence: the percentage of matched pairs where the generated images adhered to the intended attribute change.</li>
      <li>Mean paired cosine distance: the mean cosine distance between the embeddings of matched pairs.</li>
    </ul>
  <section id="chapter-6-overview">
    <h3>Overview</h3>
    {_metrics_table(data)}
    <p class="note">{html.escape(str(data.get("caveat", "")))}</p>
  </section>
  <section id="chapter-6-features">
    <h3>6.1 Counterfactual analysis from extracted features</h3>
    <p>Each matched pair holds two images whose prompts differ only in the changing
    attribute. The stacked bars show, for every other detected attribute, the detected
    categories before and after the change: if the bars differ, changing one attribute
    also shifted attributes that were not asked to change.</p>
    <article class="plot-card">
      <h4>6.1.1 Detected value transitions in control prompts</h4>
      {_transition_selector(data.get("control_analysis", {}), prefix="chapterSixControl",
                            description="Select the changing attribute, its before and after values, and the three fixed attribute values.")
      }
    </article>
    <article class="plot-card">
      <h4>6.1.2 Detected value transitions in double-attribute prompts</h4>
      {_transition_selector(data.get("double_analysis", {}), prefix="chapterSixDouble",
                            description="Select the changing attribute, its before and after values, and the fixed attribute value.",
                            double=True)
      }
    </article>
    <article class="plot-card">
      <h4>6.1.3 Aggregate detected category changes</h4>
      {_bar_chart(data)}
    </article>
    <p>The plots below show, for each numerical image feature, the distribution of
    the paired difference (after minus before) for each changing attribute. A box
    centred on zero means the change did not systematically affect that feature.</p>
    {_numerical_plots(data)}
  </section>
  <section id="chapter-6-embeddings">
    <h3>6.2 Counterfactual analysis from embeddings</h3>
    <p>Cosine distance measures semantic change between paired image embeddings.
    Distances are descriptive and should not be interpreted as bias without
    considering the intended attribute change.</p>
    <article class="plot-card">{_embedding_plots(data)}</article>
  </section>
"""
