"""Observations for chapter 5: associations, objects and intersections."""

from __future__ import annotations

from typing import Any

from ..explain import cramers_band, esc, label, number, pct
from .common import MIN_GROUP, most_overrepresented, strongest_associations


def chapter_five(bias: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    strongest = strongest_associations(bias)
    if strongest:
        notes.append("Strongest associations between a prompt attribute and a different "
                     "detected feature: " + "; ".join(
                         f"{esc(label(row['prompt']))} → {esc(label(row['output']))} "
                         f"(V={row['v']:.2f}, {cramers_band(row['v'])[0]}, n={row['n']})"
                         for row in strongest[:5]) + ".")
        for row in strongest[:3]:
            cell = most_overrepresented(row["matrix"])
            if cell and cell[2] - cell[3] >= 0.15:
                group, category, share, base, n = cell
                notes.append(
                    f"Prompt {esc(label(row['prompt']))} = <strong>{esc(group)}</strong>: "
                    f"detected {esc(row['output'])} is <strong>{esc(category)}</strong> in "
                    f"{pct(share, 0)} of faces vs {pct(base, 0)} across all "
                    f"{esc(row['prompt'])} values (n={n})."
                )
    return notes


def stereotypes(associations: dict[str, Any], *, min_count: int = 3) -> list[str]:
    items = []
    for attribute, data in (associations or {}).items():
        for row in (data or {}).get("rows", []):
            for item in row.get("top_objects", []):
                lift = number(item.get("lift"))
                if lift and item.get("count", 0) >= min_count and lift >= 2:
                    items.append((lift, attribute, row.get("group"), item))
    items.sort(key=lambda entry: -entry[0])
    if not items:
        return []
    return ["Strongest over-represented objects (at least "
            f"{min_count} images): " + "; ".join(
                f"{esc(group)} → <strong>{esc(item['object'])}</strong> "
                f"(lift {lift:.1f}, in {pct(item['probability'], 0)} of images, "
                f"n={item['count']})"
                for lift, _, group, item in items[:6]) + "."]


def intersectional(data: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    forest = []
    models = []
    for pair, item in (data or {}).items():
        for output, matrix in (item.get("outputs") or {}).items():
            for row in matrix.get("forest", []):
                lower, upper = number(row.get("lower")), number(row.get("upper"))
                if lower is None or upper is None or row.get("n", 0) < MIN_GROUP:
                    continue
                if lower > 0 or upper < 0:
                    forest.append((abs(row["difference"]), pair, output, row))
            for model in matrix.get("models", []) or []:
                p_value = number(model.get("interaction_p_value"))
                if p_value is not None and p_value < 0.05:
                    models.append((p_value, pair, output, model))
    forest.sort(key=lambda entry: -entry[0])
    seen = set()
    picked = []
    for _, pair, output, row in forest:
        key = (pair, output, row["group"])
        if key in seen:
            continue
        seen.add(key)
        picked.append((pair, output, row))
        if len(picked) == 5:
            break
    if picked:
        notes.append("Largest joint-group deviations whose 95% CI excludes zero: " + "; ".join(
            f"{esc(row['group'])} → detected {esc(output)} = {esc(row['category'])}: "
            f"{row['difference']:+.1f} pp (n={row['n']})"
            for pair, output, row in picked) + ".")
    models.sort(key=lambda entry: entry[0])
    if models:
        notes.append(f"{len(models)} interaction model(s) have a joint interaction p-value "
                     "below 0.05, e.g. " + "; ".join(
                         f"{esc(label(pair))} → {esc(output)} = "
                         f"{esc(model['output_category'])} (p={model['interaction_p_value']:.3f})"
                         for _, pair, output, model in models[:3])
                     + ". With many models tested, expect some false positives.")
    return notes
