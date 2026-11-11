"""Renderer for chapter 6: matched counterfactual analysis."""

from __future__ import annotations

import json
from typing import Any

import plotly.graph_objects as go

from .explain import chapter_header, esc, findings, guide, label, metric_defs
from .insights import chapter_six
from .plotly_helpers import chart, numeric


def _bar_chart(data: dict[str, Any]) -> str:
    attributes = data.get("attributes", [])
    by_attribute = data.get("by_attribute", {})
    values = [
        100 * numeric(by_attribute.get(attribute, {}).get("detected_changes", {}).get("rate"))
        for attribute in attributes
    ]
    figure = go.Figure(go.Bar(
        x=attributes, y=values, text=[f"{value:.0f}%" for value in values],
        textposition="outside", cliponaxis=False,
        customdata=[by_attribute.get(attribute, {}).get("pairs", 0) for attribute in attributes],
        hovertemplate="%{x}: detected value differs in %{y:.1f}% of pairs"
                      "<br>%{customdata} matched pairs<extra></extra>",
    ))
    figure.update_layout(yaxis={"title": "Pairs whose detected value changed (%)",
                                "range": [0, 110]},
                         xaxis={"title": "Attribute changed in the prompt"})
    return chart(figure, title="", height=340)


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
                .replace(/\\b\\w/g, (letter) => letter.toUpperCase());
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
                  // Two-level (multicategory) x axis: each attribute groups its
                  // "before" and "after" bars side by side under one label.
                  // Blank spacer slots (unique whitespace names) separate the groups.
                  const slots = attributes.flatMap((attribute, i) => [
                    ...(i ? [{{spacer: true, group: " ".repeat(i), side: " "}}] : []),
                    {{attribute, side: "first", group: {prefix}Title(attribute), label: "before"}},
                    {{attribute, side: "second", group: {prefix}Title(attribute), label: "after"}}
                  ]);
                  const groups = slots.map((slot) => slot.group);
                  const sides = slots.map((slot) => slot.spacer ? " " : slot.label);
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
                    x: [groups, sides],
                    customdata: groups.map((group, i) => group.trim() + " (" + sides[i] + ")"),
                    y: slots.map((slot) => slot.spacer ? null
                      : counts(slot.attribute, slot.side, category)),
                    text: slots.map((slot) => !slot.spacer
                      && counts(slot.attribute, slot.side, category)
                      ? {prefix}Title(category) : ""),
                    textposition: "inside",
                    insidetextanchor: "middle",
                    hovertemplate: "%{{customdata}}<br>Detected: %{{text}}<br>Pairs: %{{y}}<extra></extra>"
                  }}));
                  Plotly.react(document.getElementById("{prefix}-plot"), traces, {{
                    barmode: "stack",
                    bargap: 0.05,
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


_FEATURE_HELP = {
    "n_people": "number of people detected in the image",
    "person_area_ratio": "share of the image covered by people (framing / zoom)",
    "face_area_ratio": "share of the image covered by the face (close-up vs full body)",
    "detected_age": "age in years estimated by the face model; not shown for age swaps, "
                    "where a change is the intended effect",
}


def _numerical_plots(data: dict[str, Any]) -> str:
    cards = []
    for feature in data.get("numerical_features", []):
        figure = go.Figure()
        for attribute in data.get("attributes", []):
            values = data.get("by_attribute", {}).get(attribute, {}).get(
                "numerical_differences", {}).get(feature, {}).get("values", [])
            if values:
                figure.add_trace(go.Box(y=[abs(value) for value in values], name=attribute,
                                        boxmean=True, boxpoints="outliers", showlegend=False))
        if figure.data:
            figure.update_yaxes(title="|After − before|", rangemode="tozero")
            cards.append(
                f'<article class="plot-card"><h4>{esc(label(feature))}</h4>'
                f'<p class="note">{esc(_FEATURE_HELP.get(feature, ""))}</p>'
                f"{chart(figure, title='', height=340)}</article>"
            )
    return (f'<div class="plot-grid">{"".join(cards)}</div>' if cards else
            '<p class="note">No paired numerical features are available.</p>')


def _embedding_plots(data: dict[str, Any]) -> str:
    attributes = data.get("attributes", [])
    by_attribute = data.get("by_attribute", {})
    series = {}
    for name, key in (
        ("Within-condition distance (same prompt value)", "within_condition_euclidean_distance"),
        ("Between-pair distance (matched counterfactual)", "between_pair_euclidean_distance"),
    ):
        series[name] = [
            numeric(by_attribute.get(attribute, {}).get("embeddings", {}).get(key, {}).get("mean"))
            for attribute in attributes
        ]
    distance_figure = go.Figure([
        go.Bar(name=name, x=attributes, y=values,
               hovertemplate=f"{name}<br>%{{x}}: %{{y:.3f}}<extra></extra>")
        for name, values in series.items()
    ])
    distance_figure.update_layout(barmode="group", yaxis={"title": "Mean Euclidean distance"},
                                  xaxis={"title": "Attribute changed in the prompt"})
    cosine_figure = go.Figure()
    for attribute in attributes:
        values = by_attribute.get(attribute, {}).get("embeddings", {}).get(
            "paired_cosine_distance", {}).get("values", [])
        if values:
            cosine_figure.add_trace(go.Box(y=values, name=attribute, boxmean=True,
                                           boxpoints=False, showlegend=False))
    cosine_figure.update_layout(yaxis={"title": "Paired cosine distance", "rangemode": "tozero"},
                                xaxis={"title": "Attribute changed in the prompt"})
    return (
        '<div class="plot-grid">'
        '<article class="plot-card"><h4>6.2.1 Paired cosine distance</h4>'
        f'{chart(cosine_figure, title="", height=360)}</article>'
        '<article class="plot-card"><h4>6.2.2 Counterfactual vs random variation</h4>'
        f'{chart(distance_figure, title="", height=360)}</article></div>'
    )


def _metrics_table(data: dict[str, Any]) -> str:
    rows = []
    for attribute in data.get("attributes", []):
        item = data.get("by_attribute", {}).get(attribute, {})
        adherence = item.get("adherence", {})
        adherence_text = "<br>".join(
            f"{esc(condition)}: {100 * numeric(stats.get('rate')):.0f}%"
            for condition, stats in adherence.items()
        ) or "-"
        embeddings = item.get("embeddings", {})
        cosine = embeddings.get("paired_cosine_distance", {}).get("mean")
        within = embeddings.get("within_condition_euclidean_distance", {}).get("mean")
        between = embeddings.get("between_pair_euclidean_distance", {}).get("mean")
        ratio = (f"{between / within:.2f}" if isinstance(between, (int, float))
                 and isinstance(within, (int, float)) and within else "-")
        rows.append(
            "<tr><th scope='row'>{}</th><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td></tr>".format(
                esc(attribute), item.get("pairs", 0),
                f"{100 * numeric(item.get('detected_changes', {}).get('rate')):.0f}%",
                adherence_text,
                "-" if cosine is None else f"{cosine:.3f}",
                ratio,
            )
        )
    return (
        "<div class='table-wrap'><table><thead><tr><th>Changed attribute</th>"
        "<th>Matched pairs</th><th>Detected value changed</th>"
        "<th>Adherence per requested value</th><th>Mean paired cosine distance</th>"
        "<th>Between ÷ within distance</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )


def render_chapter_six(data: dict[str, Any]) -> str:
    """Render metrics and plots from the precomputed chapter-6 artifact."""
    return f"""
  {chapter_header(6, "Counterfactual analysis",
      "If only one word of the prompt changes, what else in the image changes?",
      "<p>Chapter 5 compares groups of prompts that may differ in several ways. "
      "Counterfactual analysis is more controlled: it pairs prompts that are identical "
      "except for one attribute — e.g. “young white <strong>female</strong> doctor” vs "
      "“young white <strong>male</strong> doctor” — and compares the two images. The "
      "requested attribute <em>should</em> change. A fair model should leave everything "
      "else (emotion, age, framing, objects, setting) unchanged; systematic changes in "
      "unrelated properties show what the model bundles together with the attribute.</p>"
      "<p>Pairs come from control prompts (gender, race, age and job all fixed) and from "
      "double-variable prompts (two attributes fixed). Images in a pair are matched by "
      "generation order, not by seed, so part of every difference is ordinary random "
      "variation between two generations.</p>")}
  {findings(chapter_six(data))}
  {metric_defs([
      ("Detected value changed", "Share of matched pairs in which the detector reports a "
       "different value of the <em>changed</em> attribute for the two images.",
       "Should be close to 100%: the prompt asked for a different value. Low values mean the "
       "model ignored the change for at least one side."),
      ("Adherence per requested value", "For each requested value, the share of paired "
       "images where it was detected as requested.", "Unequal rates (e.g. one gender "
       "followed far more often than the other) are an adherence disparity under matched "
       "prompts."),
      ("Paired cosine distance", "1 − cosine similarity between the CLIP embeddings of the "
       "two images in a pair.", "How much the image changed overall. Compare attributes, "
       "keeping in mind that some changes (job, workplace) should alter the scene a lot."),
      ("Between ÷ within distance", "Mean Euclidean distance within matched pairs divided "
       "by the mean distance between images that share the same requested value.",
       "&lt; 1: changing the word moves the image less than the typical difference between "
       "two images with the same value; &gt; 1: the attribute change dominates."),
  ], open_=False)}
  <section id="chapter-6-overview">
    <h3>Overview</h3>
    {guide(
        read="One row per attribute that is changed between the two prompts of a pair.",
        look=["Large adherence gaps between the values of one attribute: the model follows "
              "the change in one direction more reliably than in the other.",
              "Demographic attributes (gender, race, age) with a cosine distance close to "
              "that of job or workplace: changing the person changes the whole scene, "
              "not just the person."],
        caution=[esc(data.get("caveat", "")),
                 "An adherence of 0% for one value usually means that the requested label "
                 "never matches the detector's label set (e.g. “hispanic” vs “latino "
                 "hispanic”, or job names that the captions do not contain)."],
    )}
    {_metrics_table(data)}
  </section>
  <section id="chapter-6-features">
    <h3>6.1 Counterfactual analysis from extracted features</h3>
    <p>Each matched pair holds two images whose prompts differ only in the changing
    attribute. The stacked bars show, for every <em>other</em> detected attribute, the
    detected categories before and after the change.</p>
    {guide(
        read="Select the attribute that changes, its before and after values, and the "
             "values that stay fixed. Each pair of bars (before / after) counts the matched "
             "pairs per detected category.",
        look=["Bars that differ between before and after for an attribute that was "
              "<em>not</em> changed: e.g. switching “female” to “male” also shifts detected "
              "age or emotion. That is a bundled association, measured under matched "
              "conditions.",
              "The same shift recurring across many fixed-value combinations — a single "
              "combination of 10 pairs is weak evidence on its own."],
        caution="Detected categories also change between two generations of the same "
                "prompt; small differences of one or two pairs are within random variation.",
    )}
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
      <h4>6.1.3 How often the requested change is detected</h4>
      <p class="note">Share of matched pairs in which the changed attribute is detected
      with a different value in the two images. Ideally close to 100%.</p>
      {_bar_chart(data)}
    </article>
    <h4>6.1.4 Paired differences in numerical image features</h4>
    {guide(
        read="Each box is the distribution of the absolute difference |after − before| over "
             "all matched pairs for one changed attribute; the dashed line inside each box is "
             "its mean. It shows how much the feature changes, not in which direction.",
        look="Demographic attributes whose boxes sit clearly higher than the others: e.g. if "
             "switching gender changes the face area much more than switching emotion, the "
             "framing (portrait vs full body) depends on the gender. Large changes for job or "
             "workplace are expected, since they alter the scene.",
        caution="Two generations of the same prompt already differ, so absolute differences "
                "are never zero: compare attributes with each other rather than with zero. "
                "The plot does not show which value is framed closer or farther.",
    )}
    {_numerical_plots(data)}
  </section>
  <section id="chapter-6-embeddings">
    <h3>6.2 Counterfactual analysis from embeddings</h3>
    <p>CLIP embeddings summarise the whole image, so they capture changes the detectors
    miss (clothing, background, style, lighting).</p>
    {guide(
        read=["6.2.1: distribution of the cosine distance between the two images of each "
              "pair, per changed attribute.",
              "6.2.2: mean distance within matched pairs vs mean distance between images "
              "that share the same requested value (a baseline for random variation)."],
        look=["A demographic change whose between-pair distance approaches or exceeds the "
              "within-condition baseline: the change moves the image as much as an entirely "
              "new generation, i.e. much more than just the person.",
              "Large differences between demographic attributes (e.g. changing race moves "
              "images much more than changing gender)."],
        caution="A larger distance is not automatically more bias: the requested attribute "
                "itself must change part of the image. Embeddings cannot tell which part "
                "changed; use 6.1 and chapter 8 to see what it was.",
    )}
    {_embedding_plots(data)}
  </section>
"""
