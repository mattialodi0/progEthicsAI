"""Chapter 4: detected age in years and class-probability adherence."""

from __future__ import annotations

import html
from typing import Any

import plotly.graph_objects as go

from ..explain import guide
from ..plotly_helpers import chart, numeric


def _separation_reading(g: Any) -> str:
    if g is None:
        return "too few images to compare"
    if g <= -0.2:
        return "reversed: the older term is drawn younger"
    if g < 0.2:
        return "not separated"
    return "partly separated" if g < 0.8 else "clearly separated"


def age_years_card(age: dict[str, Any]) -> str:
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


def probability_card(probabilities: dict[str, Any]) -> str:
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
