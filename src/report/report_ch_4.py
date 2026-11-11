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


def _separation_reading(g: Any) -> str:
    if g is None:
        return "too few images to compare"
    if g <= -0.2:
        return "reversed: the older term is drawn younger"
    if g < 0.2:
        return "not separated"
    return "partly separated" if g < 0.8 else "clearly separated"


def _age_years_card(age: dict[str, Any]) -> str:
    """Detected age in years for each requested age term (no bins)."""
    groups = age.get("groups") or {}
    if not groups:
        return ""
    entries = [(term, groups[term], "#0072B2") for term in age.get("order", []) if term in groups]
    baseline = age.get("not_requested") or {}
    if baseline.get("n"):
        entries.append(("no age requested", baseline, "#7b8794"))
    figure = go.Figure([
        go.Box(y=item.get("values", []), name=f"{name} (n={item.get('n', 0)})", boxmean=True,
               boxpoints="outliers", marker={"color": color}, showlegend=False)
        for name, item, color in entries
    ])
    for cutoff, name in zip(age.get("cutoffs", []), ("young | middle-aged", "middle-aged | old")):
        figure.add_hline(y=cutoff, line_dash="dot", line_color="#a35f00",
                         annotation_text=f"{name} cut-off ({cutoff})",
                         annotation_position="top left",
                         annotation_font={"size": 11, "color": "#a35f00"})
    figure.update_yaxes(title="Detected age (years)")
    figure.update_xaxes(title="Requested age")
    rows = "".join(
        f"<tr><th scope='row'>{html.escape(item['lower'])} → {html.escape(item['higher'])}</th>"
        f"<td>{numeric(item.get('difference')):+.1f} years</td>"
        + ("<td>-</td>" if item.get("hedges_g") is None else
           f"<td>{numeric(item.get('hedges_g')):+.2f} "
           f"[{numeric(item.get('g_ci_low')):+.2f}, {numeric(item.get('g_ci_high')):+.2f}]</td>")
        + f"<td>{_separation_reading(item.get('hedges_g'))}</td></tr>"
        for item in age.get("separation", [])
    )
    table = (
        "<table><thead><tr><th>Neighbouring terms</th><th>Difference of means</th>"
        "<th>Hedges' g [95% CI]</th><th>Reading</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>" if rows else ""
    )
    return f"""
    <article class="plot-card"><h4>Detected age in years by requested age</h4>
    {guide(
        read=["One box per requested age term, plus the images whose prompt requests no age "
              "(grey): the line is the median, the dashed line the mean. Dotted lines are the "
              "cut-offs that turn years into young / middle-aged / old in the matrix above.",
              "The table compares neighbouring terms: a positive difference means the older "
              "term is drawn older."],
        look=["Terms whose boxes overlap: the model (as seen by the face model) does not "
              "tell them apart, whatever the adherence percentage says.",
              "A box that sits just below a cut-off: adherence is low because of where the "
              "cut-off is, not because the images ignore the request.",
              "Requested terms compared with the grey box: how far a request moves the age "
              "away from the model's default."],
        caution="The face model compresses ages towards 30–40 and rarely returns values "
                "above 65, so read the years as a relative scale. Differences between boxes "
                "are meaningful; “48 years” is not a literal age.",
    )}
    {chart(figure, title="", height=400)}
    {table}</article>"""


def _probability_card(probabilities: dict[str, Any]) -> str:
    """Adherence recomputed with the class probabilities of the face model."""
    usable = {attribute: item for attribute, item in probabilities.items()
              if (item or {}).get("overall")}
    if not usable:
        return ""
    threshold = next(iter(usable.values())).get("threshold", 0.6)
    series = (("accuracy", "Top label (as above)", "#0072B2"),
              ("soft_accuracy", "Probability-weighted", "#E69F00"),
              ("confident_accuracy", "Confident labels only", "#009E73"))

    def cell(item: dict[str, Any], key: str) -> str:
        return "-" if item.get(key) is None else f"{100 * numeric(item[key]):.1f}%"

    rows = "".join(
        f"<tr><th scope='row'>{html.escape(attribute)}</th><td>{item['overall'].get('n', 0)}</td>"
        + "".join(f"<td>{cell(item['overall'], key)}</td>" for key, _, _ in series)
        + f"<td>{item['overall'].get('n_confident', 0)}</td></tr>"
        for attribute, item in usable.items()
    )
    table = (
        "<table><thead><tr><th>Attribute</th><th>Images</th>"
        + "".join(f"<th>{name}</th>" for _, name, _ in series)
        + f"<th>Images with a confident label</th></tr></thead><tbody>{rows}</tbody></table>"
    )
    # The breakdown is shown for the attribute where the three measures disagree most.
    large = {name: item for name, item in usable.items() if item["overall"].get("n", 0) >= 30}
    attribute = max(large or usable, key=lambda name: abs(
        numeric(usable[name]["overall"].get("accuracy"))
        - numeric(usable[name]["overall"].get("soft_accuracy"))))
    values = usable[attribute].get("by_value") or {}
    figure = go.Figure([
        go.Bar(
            name=name, x=list(values), marker={"color": color},
            y=[None if item.get(key) is None else 100 * numeric(item[key])
               for item in values.values()],
            customdata=[[item.get("n", 0), item.get("n_confident", 0)]
                        for item in values.values()],
            hovertemplate=f"Requested {html.escape(attribute)}: %{{x}}<br>{name}: %{{y:.1f}}%"
                          "<br>%{customdata[0]} images, %{customdata[1]} with a confident "
                          "label<extra></extra>",
        )
        for key, name, color in series
    ])
    figure.update_layout(barmode="group", yaxis={"range": [0, 100], "title": "Adherence (%)"},
                         xaxis={"title": f"Requested {attribute}"})
    return f"""
    <article class="plot-card"><h4>Does adherence depend on uncertain labels?</h4>
    <p class="note">The face model returns a probability for every class. The matrices above
    keep only the most likely class; this check uses the probabilities to see how much of a
    result rests on labels the face model itself is unsure about.</p>
    {guide(
        read=[f"<strong>Top label</strong>: the adherence reported above. "
              f"<strong>Probability-weighted</strong>: mean probability given to the requested "
              f"class. <strong>Confident labels only</strong>: adherence on images whose top "
              f"class has a probability of at least {100 * numeric(threshold):.0f}%.",
              f"The chart shows the attribute where the measures differ most "
              f"({html.escape(attribute)}); the table gives every attribute."],
        look=["Values where the three bars agree: the result does not depend on how "
              "uncertain labels are counted.",
              "Values where the probability-weighted bar is far below the top-label bar: "
              "many hits are narrow wins between two classes, so the adherence is fragile.",
              "Values with few confident images (hover): the face model cannot tell this "
              "group from another one, and a low adherence may be a measurement limit "
              "rather than a model failure."],
        caution="The probabilities are not calibrated, so the probability-weighted value is "
                "a sensitivity check, not a better estimate. Use it to decide how much to "
                "trust a gap, not to replace the adherence above.",
    )}
    {chart(figure, title="", height=380)}
    {table}</article>"""


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
      ("Detected age (years)", "Age estimated by the face model, before it is cut into "
       "young (&lt; 30), middle-aged (30–49) and old (50+).", "Shows where each requested "
       "age term lands and whether two terms are told apart, without depending on the "
       "cut-offs. A relative scale: compare terms, do not read the years literally."),
      ("Probability-weighted adherence", "Mean probability that the face model gives to the "
       "requested class.", "Close to the adherence = labels are clear-cut. Much lower = "
       "many hits are narrow wins between two classes."),
      ("Confident-label adherence", "Adherence on the images whose most likely class has a "
       "probability of at least 60%.", "Check how many images remain: a group with few "
       "confident labels is one the face model cannot measure well."),
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
    {_age_years_card(adherence.get("age_numeric") or {})}
    {_probability_card(adherence.get("probabilities") or {})}
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
