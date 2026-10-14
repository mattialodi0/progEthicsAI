"""Small helpers shared by the HTML report chapter renderers."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.io import to_html


def chart(
    figure: go.Figure,
    *,
    title: str,
    height: int = 360,
) -> str:
    """Return a responsive Plotly chart without duplicating the Plotly bundle."""
    figure.update_layout(
        title=title,
        height=height,
        margin={"l": 55, "r": 25, "t": 65, "b": 55},
        template="plotly_white",
        legend={"orientation": "h", "y": -0.2},
    )
    return to_html(
        figure,
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
    )


def numeric(value: Any, default: float = 0.0) -> float:
    """Convert a metric to a finite chart value."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default
