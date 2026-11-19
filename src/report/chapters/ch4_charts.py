"""Chapter 4: requested-versus-detected matrices and adherence charts."""

from __future__ import annotations

import html
from typing import Any

import plotly.graph_objects as go

from ..measurable import NA_TEXT, measurable_accuracy, unmeasurable_from_counts
from ..plotly_helpers import chart, numeric


def _value(value: Any) -> str:
    return "-" if value is None else f"{100 * float(value):.1f}%"


def _heatmap(measurement: dict[str, Any], title: str) -> str:
    counts = measurement.get("counts") or measurement.get("matrix", {})
    row_total = {row: sum(numeric(value) for value in counts.get(row, {}).values())
                 for row in measurement.get("labels", [])}
    requested = [row for row, total in row_total.items() if total > 0]
    # Columns that were never detected carry no information; dropping them keeps
    # the matrix readable (e.g. "bored", which the emotion detector cannot output).
    columns = [column for column in measurement.get("labels", [])
               if any(numeric(counts.get(row, {}).get(column)) > 0 for row in requested)]
    unmeasurable = unmeasurable_from_counts(counts)
    row_names = [f"{row} (n/a)" if row in unmeasurable else row for row in requested]
    figure = go.Figure(
        go.Heatmap(
            x=columns,
            y=row_names,
            z=[[100 * numeric(counts.get(row, {}).get(column)) / row_total[row]
                for column in columns] for row in requested],
            text=[[str(int(numeric(counts.get(row, {}).get(column)))) for column in columns]
                  for row in requested],
            texttemplate="%{z:.0f}%<br>(%{text})",
            textfont={"size": 11},
            colorscale="Blues",
            zmin=0,
            zmax=100,
            showscale=False,
            xgap=1,
            ygap=1,
            hovertemplate="Requested: %{y}<br>Detected: %{x}<br>%{z:.1f}% of requested"
                          "<br>Count: %{text}<extra></extra>",
        )
    )
    # Outline the diagonal: those cells are the adherence.
    figure.update_layout(shapes=[
        {"type": "rect", "xref": "x", "yref": "y",
         "x0": columns.index(row) - 0.5, "x1": columns.index(row) + 0.5,
         "y0": position - 0.5, "y1": position + 0.5,
         "line": {"color": "#102a43", "width": 2}}
        for position, row in enumerate(requested) if row in columns
    ])
    figure.update_xaxes(title=f"Detected {title}", tickangle=-30, showgrid=False,
                        constrain="domain")
    # Lock the y scale to the x scale so every cell is square.
    figure.update_yaxes(title=f"Requested {title}", autorange="reversed", showgrid=False,
                        scaleanchor="x", scaleratio=1, constrain="domain")
    html_chart = chart(figure, title="", height=max(340, 170 + 95 * len(requested)))
    if unmeasurable:
        html_chart += (
            '<p class="note">n/a: the detector never outputs '
            + ", ".join(f"“{html.escape(row)}”" for row in sorted(unmeasurable))
            + ", so these rows can never match.</p>"
        )
    return html_chart


def adherence_matrix(title: str, measurement: dict[str, Any], *, collapsible: bool = False) -> str:
    labels = measurement.get("labels", [])
    counts = measurement.get("counts") or measurement.get("matrix", {})
    if not labels or not counts or measurement.get("total", 0) <= 0:
        return ""
    unmeasurable = unmeasurable_from_counts(counts)
    measurable = measurable_accuracy(counts, unmeasurable) if unmeasurable else None
    measurable_text = (f" ({_value(measurable)} on values the detector can output)"
                       if measurable is not None else "")
    stats = (f"<strong>Accuracy:</strong> {_value(measurement.get('accuracy'))}"
             f"{measurable_text} · "
             f"{measurement.get('total', 0)} images with a detection · "
             f"{measurement.get('unmatched', 0)} without")
    body = _heatmap(measurement, title)
    if collapsible:
        return (f"<details><summary>{html.escape(title)} — accuracy "
                f"{_value(measurement.get('accuracy'))} (n={measurement.get('total', 0)})"
                f"</summary>{body}</details>")
    return (f'<article class="plot-card"><h4>Requested vs detected {html.escape(title)}</h4>'
            f'<p class="note">{stats}</p>{body}</article>')


def accuracy_chart(attributes: dict[str, Any]) -> str:
    names = [name for name, item in attributes.items() if item.get("accuracy") is not None]
    if not names:
        return ""
    values = [100 * numeric(attributes[name].get("accuracy")) for name in names]
    measurable = {}
    for name in names:
        counts = attributes[name].get("counts") or {}
        excluded = unmeasurable_from_counts(counts)
        if excluded:
            measurable[name] = (measurable_accuracy(counts, excluded), sorted(excluded))
    labels = []
    for name, value in zip(names, values):
        text = f"{value:.1f}%"
        if name in measurable and measurable[name][0] is not None:
            text += f" · {100 * measurable[name][0]:.1f}% excl. n/a values"
        labels.append(text)
    figure = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        text=labels, textposition="outside", cliponaxis=False,
        customdata=[[attributes[name].get("total", 0), attributes[name].get("unmatched", 0),
                     ", ".join(measurable.get(name, (None, ["none"]))[1])]
                    for name in names],
        marker={"color": ["#0072B2" if value >= 80 else "#E69F00" if value >= 60 else "#D55E00"
                          for value in values]},
        hovertemplate="%{y}: %{x:.1f}% adherence<br>%{customdata[0]} images with a detection"
                      "<br>%{customdata[1]} without<br>Values the detector cannot output: "
                      "%{customdata[2]}<extra></extra>",
    ))
    figure.update_layout(xaxis={"title": "Adherence (%)", "range": [0, 105]},
                         yaxis={"autorange": "reversed"})
    figure.update_layout(margin={"r": 190})
    return chart(figure, title="", height=60 + 52 * len(names))


def demographic_chart(category: str, groups: dict[str, Any], attributes: list[str],
                      unmeasurable: set[str] | None = None) -> str:
    """Adherence to each attribute, split by the requested value of ``category``.

    ``unmeasurable`` holds values of ``category`` the detector cannot output:
    their own-attribute bar is shown as "n/a" rather than as 0%.
    """
    unmeasurable = unmeasurable or set()
    values = list(groups)

    def adherence(value: str, attribute: str) -> float | None:
        raw = groups[value].get("accuracy", {}).get(attribute)
        if raw is None or (attribute == category and value in unmeasurable):
            return None
        return 100 * numeric(raw)

    figure = go.Figure([
        go.Bar(
            name=str(attribute),
            x=values,
            y=[adherence(value, attribute) for value in values],
            customdata=[groups[value].get("n_images", 0) for value in values],
            hovertemplate=(
                f"Requested {html.escape(category)}: %{{x}}<br>{html.escape(str(attribute))} "
                "adherence: %{y:.1f}%<br>%{customdata} images<extra></extra>"
            ),
        )
        for attribute in attributes
        if any(groups[value].get("accuracy", {}).get(attribute) is not None for value in values)
    ])
    figure.update_layout(
        barmode="group", yaxis={"range": [0, 100], "title": "Adherence (%)"},
        xaxis={"title": f"Requested {category}"},
        legend={"title": {"text": "Attribute"}},
        annotations=[
            {"x": value, "y": 4, "yanchor": "bottom", "text": "n/a", "showarrow": False,
             "font": {"color": "#7b8794", "size": 11}, "hovertext": NA_TEXT}
            for value in values if value in unmeasurable
        ],
    )
    return (
        f'<article class="plot-card"><h4>By requested {html.escape(category)}</h4>'
        f"{chart(figure, title='')}</article>"
    )


def mode_chart(modes: dict[str, Any], attribute_names: list[str]) -> str:
    """Grouped bars: adherence per attribute in each prompt mode."""
    figure = go.Figure(
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
    figure.update_layout(
        barmode="group",
        yaxis={"range": [0, 100], "title": "Adherence (%)"},
        xaxis={"title": "Prompt mode"},
        legend={"title": {"text": "Attribute"}},
    )
    return chart(figure, title="")
