"""Chapter 4: requested-versus-detected matrices and adherence charts."""

from __future__ import annotations

import html
from typing import Any

import plotly.graph_objects as go

from ..plotly_helpers import chart, numeric


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


def adherence_matrix(title: str, measurement: dict[str, Any], *, collapsible: bool = False) -> str:
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


def accuracy_chart(attributes: dict[str, Any]) -> str:
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


def demographic_chart(category: str, groups: dict[str, Any], attributes: list[str]) -> str:
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
