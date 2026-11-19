"""Chapter 9: data and filters of the interactive plots and results explorer.

The ranked-effects plot, the effect-versus-reliability plot and the results
explorer share one set of filters and are drawn in the browser
(``static/chapter_9.js``) from an embedded JSON copy of the results table.
"""

from __future__ import annotations

from typing import Any

from ...analysis.synthesis.constants import MAX_STRENGTH
from ..explain import esc
from ..static_files import script_json


# Fields of a result row shipped to the browser (the rest stays in the artifact).
ROW_FIELDS = (
    "id", "lens", "source", "link", "level", "attribute", "subgroup", "reference",
    "feature", "category", "scope", "metric", "effect", "effect_unit", "ci", "family",
    "strength", "strength_ci", "strength_low", "band", "band_low", "n", "n_total", "p_value",
    "q_value", "tier", "context", "stereotype_candidate", "question", "cluster",
    "statement",
)


def results_payload(data: dict[str, Any]) -> str:
    rows = []
    for row in data.get("results") or []:
        item = {field: row.get(field) for field in ROW_FIELDS}
        item["flags_text"] = "; ".join(flag.get("text", "").rstrip(".")
                                       for flag in row.get("flags") or [])
        rows.append(item)
    settings = data.get("settings") or {}
    payload = {
        "rows": rows,
        "fields": list(ROW_FIELDS),
        "lenses": settings.get("lenses") or {},
        "tiers": settings.get("tiers") or {},
        "small_n": settings.get("small_n", 30),
        "alpha": settings.get("alpha", 0.05),
        "max_strength": MAX_STRENGTH,
    }
    return script_json(payload)


def explorer_controls(data: dict[str, Any]) -> str:
    questions = "".join(
        f'<option value="{esc(answer["key"])}">{esc(answer["title"])}</option>'
        for answer in data.get("answers") or []
    )
    tiers = (data.get("settings") or {}).get("tiers") or {}
    tier_options = "".join(
        f'<option value="{esc(key)}">{esc(value)} only</option>'
        for key, value in tiers.items() if key != "supported"
    )
    return f"""
    <div class="checkbox-grid ch9-filters" aria-label="Chapter 9 filters">
      <label>Lens <select id="ch9-lens"><option value="">All lenses</option></select></label>
      <label>Prompt attribute <select id="ch9-attribute"><option value="">All</option></select></label>
      <label>Fairness question <select id="ch9-question"><option value="">All</option>{questions}
        <option value="context">Context</option><option value="measurement">Measurement</option></select></label>
      <label>Tier <select id="ch9-tier">
        <option value="supported">Statistically supported</option>
        <option value="substantive">All substantive (no context / measurement)</option>
        <option value="all">Everything</option>{tier_options}</select></label>
      <label>Minimum n <input id="ch9-min-n" type="number" min="0" step="1" value="0"
        style="width:5rem"></label>
      <label><input id="ch9-specific" type="checkbox" checked> Specific results only</label>
      <label>Search <input id="ch9-search" type="search" placeholder="e.g. nurse, f12"
        style="width:11rem"></label>
    </div>"""
