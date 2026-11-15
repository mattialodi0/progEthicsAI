"""Helpers shared by the observation builders."""

from __future__ import annotations

from typing import Any

from ..explain import number


# Groups smaller than this are never highlighted.
MIN_GROUP = 10


def category_shares(counts: dict[str, Any]) -> dict[str, float]:
    total = sum(number(value) or 0 for value in counts.values())
    if not total:
        return {}
    return {str(key): (number(value) or 0) / total for key, value in counts.items()}


def strongest_associations(bias: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    """Rank prompt-attribute × detected-feature pairs by Cramér's V.

    Pairs where the prompt attribute and the detected feature are the same
    (e.g. prompt gender → detected gender) measure adherence, not bias, and
    are excluded.
    """
    rows = []
    for prompt, outputs in (bias.get("associations") or {}).items():
        for output, matrix in (outputs or {}).items():
            if prompt == output or not isinstance(matrix, dict):
                continue
            stats = matrix.get("statistics") or {}
            value, p_value = number(stats.get("cramers_v")), number(stats.get("p_value"))
            if value is None or (matrix.get("total") or 0) < 30:
                continue
            rows.append({"prompt": prompt, "output": output, "v": value,
                         "p": p_value, "n": matrix.get("total", 0), "matrix": matrix})
    rows.sort(key=lambda row: -row["v"])
    return rows[:limit]


def most_overrepresented(matrix: dict[str, Any]) -> tuple[str, str, float, float, int] | None:
    """Return the cell whose conditional share most exceeds the column average."""
    counts = matrix.get("counts") or {}
    columns = matrix.get("columns") or []
    total = sum(sum(number(value) or 0 for value in row.values()) for row in counts.values())
    if not total:
        return None
    overall = {column: sum(number(row.get(column)) or 0 for row in counts.values()) / total
               for column in columns}
    best = None
    for row_name, row in counts.items():
        n = sum(number(value) or 0 for value in row.values())
        if n < MIN_GROUP:
            continue
        for column in columns:
            share = (number(row.get(column)) or 0) / n
            gap = share - overall[column]
            if best is None or gap > best[3] - best[4]:
                best = (row_name, column, n, share, overall[column])
    if best is None:
        return None
    row_name, column, n, share, base = best
    return row_name, column, share, base, int(n)
