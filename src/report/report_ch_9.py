"""Renderer for chapter 9: cross-analysis synthesis and key findings.

All content comes from ``report_data["chapter_9"]`` (built by
``analysis.synthesis``): nothing here is specific to one model or dataset.
Static parts (answers, finding cards, evidence matrix, measurement caveats)
are rendered in Python; the ranked-effects plot, the effect-versus-reliability
plot and the results explorer share one set of filters and are drawn in the
browser from an embedded JSON copy of the results table.
"""

from __future__ import annotations

import json
from typing import Any

from .explain import badge, callout, chapter_header, esc, guide, label, metric_defs, pct

_BAND_TONE = {"negligible": "neutral", "small": "low", "medium": "mid", "large": "high"}
_BAND_TEXT = {None: "No supported disparity", "negligible": "Negligible",
              "small": "Small disparity", "medium": "Moderate disparity",
              "large": "Large disparity"}
_EVIDENCE_TONE = {"confirmed": "high", "convergent": "mid", "supported": "low"}
_TIER_TONE = {"supported": "high", "descriptive": "mid", "negligible": "neutral",
              "context": "neutral", "measurement": "low"}
_FLAG_TEXT = {
    "small_n": "small n",
    "low_detection": "low face detection",
    "low_adherence": "request often not followed",
    "label_mismatch": "label mismatch",
    "mode_inconsistent": "differs by prompt mode",
}
# Fields of a result row shipped to the browser (the rest stays in the artifact).
_ROW_FIELDS = (
    "id", "lens", "source", "link", "level", "attribute", "subgroup", "reference",
    "feature", "category", "scope", "metric", "effect", "effect_unit", "ci", "family",
    "strength", "strength_ci", "strength_low", "band", "n", "n_total", "p_value",
    "q_value", "tier", "context", "stereotype_candidate", "question", "cluster",
    "statement",
)


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _effect(row: dict[str, Any]) -> str:
    """Effect in its native unit with its 95% interval."""
    value = _number(row.get("effect"))
    if value is None:
        return "–"
    low, high = (_number(item) for item in (row.get("ci") or [None, None]))
    unit = row.get("effect_unit")
    if unit == "pp":
        text = f"{100 * value:+.0f} pp"
        interval = f" [{100 * low:+.0f}, {100 * high:+.0f}]" if low is not None and high is not None else ""
    elif unit == "share":
        text = pct(value, 0)
        interval = f" [{pct(low, 0)}, {pct(high, 0)}]" if low is not None and high is not None else ""
    elif unit == "ratio":
        return f"{value:.2f}×"
    elif unit == "value":
        text = f"{value:+.3g}"
        interval = f" [{low:+.3g}, {high:+.3g}]" if low is not None and high is not None else ""
    else:
        text = f"{value:.2f}"
        interval = f" [{low:.2f}, {high:.2f}]" if low is not None and high is not None else ""
    return text + interval


def _q(value: Any) -> str:
    value = _number(value)
    if value is None:
        return "–"
    if value < 0.001:
        return "q &lt; 0.001"
    return f"q = {value:.3f}"


def _flags(row: dict[str, Any]) -> str:
    return "".join(
        f'<span title="{esc(flag.get("text", ""))}">'
        f'{badge("⚠ " + _FLAG_TEXT.get(flag.get("code"), flag.get("code", "")), "mid")}</span> '
        for flag in row.get("flags") or []
    )


def _band_badge(band: str | None, prefix: str = "") -> str:
    if not band:
        return badge(prefix + "n/a")
    return badge(prefix + band, _BAND_TONE.get(band, "neutral"))


def _stat_line(row: dict[str, Any]) -> str:
    strength = _number(row.get("strength"))
    strength_text = "–" if strength is None else f"{strength:.1f}"
    if strength is not None and row.get("band"):
        strength_text += f" ({row['band']})"
    parts = [
        f"<span><b>Effect</b> {esc(_effect(row))}</span>",
        f"<span><b>Strength</b> {esc(strength_text)}</span>",
        f"<span><b>n</b> {esc(row.get('n'))}</span>",
        f"<span>{_q(row.get('q_value'))}</span>",
        f'<span><a href="{esc(row.get("link", "#"))}">→ {esc(row.get("source", ""))}</a></span>',
    ]
    return f'<div class="stat-line">{"".join(parts)}</div>'


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def _overview(data: dict[str, Any], rows: dict[str, dict[str, Any]],
              clusters: dict[str, dict[str, Any]]) -> str:
    summary = data.get("summary") or {}
    correction = data.get("correction") or {}
    evidence = summary.get("evidence") or {}
    alpha = (data.get("settings") or {}).get("alpha", 0.05)
    multi = evidence.get("confirmed", 0) + evidence.get("convergent", 0)
    tiles = [
        ("Statistical tests", f"{correction.get('tests', 0):,}",
         f"{summary.get('tests', 0):,} stored results + {correction.get('hidden_tests', 0):,} "
         "screened candidates, corrected together."),
        ("Supported after FDR", f"{summary.get('supported', 0):,}",
         f"q &lt; {alpha} and at least a small effect."),
        ("Distinct findings", f"{summary.get('findings', 0):,}",
         "Supported results grouped by the claim they make."),
        ("Multi-source findings", f"{multi:,}",
         f"{evidence.get('confirmed', 0)} counterfactual-confirmed, "
         f"{evidence.get('convergent', 0)} convergent."),
        ("Stereotype candidates", f"{summary.get('stereotype_candidates', 0):,}",
         "Social group ↔ content links, for human review."),
        ("Large effect, small n", f"{summary.get('large_small_n', 0):,}",
         "Supported large effects whose smallest group has n &lt; "
         f"{(data.get('settings') or {}).get('small_n', 30)}."),
    ]
    cards = "".join(
        f'<div class="metric"><span class="metric-label">{esc(name)}</span>'
        f'<strong class="metric-value">{value}</strong>'
        f'<span class="metric-help">{help_text}</span></div>'
        for name, value, help_text in tiles
    )
    # Overall assessment: one line per fairness question plus the top finding.
    lines = []
    for answer in data.get("answers") or []:
        top = rows.get((answer.get("top") or [None])[0])
        verdict = _BAND_TEXT.get(answer.get("band"), "No supported disparity")
        line = (f"<strong>{esc(answer['title'])}</strong>: {esc(verdict.lower())} — "
                f"{answer.get('supported', 0)} of {answer.get('tests', 0)} tests supported")
        if top:
            line += f". Strongest: {esc(top['statement'].rstrip('.'))}"
        lines.append(line + ".")
    key = [clusters[item] for item in data.get("key_findings") or [] if item in clusters]
    if key:
        lead = key[0]
        lines.insert(0, f"<strong>Most robust finding</strong>: {esc(lead['statement'].rstrip('.'))} "
                        f"({esc((data.get('settings') or {}).get('evidence_levels', {}).get(lead['evidence'], lead['evidence']))}; "
                        f"sources: {esc(', '.join(lead['sources']))}).")
    assessment = callout("finding", lines, "Overall assessment") if lines else ""
    return f'<div class="metrics">{cards}</div>{assessment}'


def _answers(data: dict[str, Any], rows: dict[str, dict[str, Any]]) -> str:
    cards = []
    for answer in data.get("answers") or []:
        tone = _BAND_TONE.get(answer.get("band"), "neutral")
        verdict = _BAND_TEXT.get(answer.get("band"), "No supported disparity")
        items = []
        for row_id in answer.get("top") or []:
            row = rows.get(row_id)
            if not row:
                continue
            items.append(
                f"<li>{esc(row['statement'])} {_flags(row)}{_stat_line(row)}</li>"
            )
        if not items:
            items.append("<li>No result reaches q &lt; α with at least a small effect.</li>")
        context = [rows[row_id] for row_id in answer.get("context") or [] if row_id in rows]
        context_html = ("<p class='note'>Context: " + " ".join(
            esc(row["statement"]) for row in context[:2]) + "</p>") if context else ""
        extra = []
        if answer.get("stereotype_candidates"):
            extra.append(f"{answer['stereotype_candidates']} candidate stereotype finding(s)")
        if answer.get("convergent"):
            extra.append(f"{answer['convergent']} confirmed by a second evidence source")
        extra_html = f"<p class='note'>{esc('; '.join(extra))}.</p>" if extra else ""
        cards.append(f"""
      <article class="plot-card answer-card">
        <div class="answer-head"><h4>{esc(answer['title'])}</h4>
          {badge(verdict, tone)}</div>
        <p class="note">{esc(answer['question'])}</p>
        <p><strong>{answer.get('supported', 0)}</strong> of {answer.get('tests', 0)} specific
        tests supported{f", {answer['omnibus_supported']} omnibus test(s) significant"
                        if answer.get('omnibus_supported') else ""}.
        <a href="{esc(answer['link'])}">Go to the chapter →</a></p>
        <ol class="answer-list">{''.join(items)}</ol>{extra_html}{context_html}
      </article>""")
    return f'<div class="plot-grid answer-grid">{"".join(cards)}</div>'


def _finding_cards(data: dict[str, Any], rows: dict[str, dict[str, Any]],
                   clusters: dict[str, dict[str, Any]]) -> str:
    levels = (data.get("settings") or {}).get("evidence_levels") or {}
    cards = []
    for position, cluster_id in enumerate(data.get("key_findings") or [], start=1):
        cluster = clusters.get(cluster_id)
        if not cluster:
            continue
        lead = rows.get(cluster["lead"], {})
        evidence_rows = [rows[row_id] for row_id in cluster.get("rows", []) if row_id in rows]
        supported = [row for row in evidence_rows if row.get("tier") == "supported"]
        others = [row for row in evidence_rows if row.get("tier") != "supported"]
        table_rows = "".join(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                esc(row.get("source")), esc(row.get("statement")), esc(_effect(row)),
                esc(row.get("n")), _q(row.get("q_value")),
                badge(row.get("tier", ""), _TIER_TONE.get(row.get("tier"), "neutral")),
            )
            for row in (supported[:6] + others[:2])
        )
        hidden = max(0, len(supported) - 6) + max(0, len(others) - 2)
        more = (f"<p class='note'>{hidden} further related result(s): "
                f'<a href="#chapter-9-explorer" data-ch9-filter="search:{esc(cluster_id)}">'
                f"show finding {esc(cluster_id)} in the explorer</a>.</p>" if hidden else
                f'<p class="note"><a href="#chapter-9-explorer" '
                f'data-ch9-filter="search:{esc(cluster_id)}">Show in the explorer</a></p>')
        badges = (
            badge(levels.get(cluster["evidence"], cluster["evidence"]),
                  _EVIDENCE_TONE.get(cluster["evidence"], "neutral"))
            + " " + _band_badge(cluster.get("band"), "effect: ")
            + (" " + badge("stereotype candidate", "mid") if cluster.get("stereotype_candidate") else "")
            + " " + badge(f"{len(cluster.get('sources', []))} source(s)")
        )
        cards.append(f"""
      <article class="plot-card finding-card">
        <div class="finding-rank">{position}</div>
        <div class="finding-body">
          <p class="finding-statement">{esc(cluster['statement'])}</p>
          <p>{badges} {_flags(lead)}</p>
          {_stat_line(lead)}
          <details class="defs"><summary>Evidence behind this finding
            ({len(supported)} supported result(s) · {esc(', '.join(cluster.get('sources', [])))})</summary>
            <div class="table-wrap"><table><thead><tr><th>Source</th><th>Result</th>
            <th>Effect [95% CI]</th><th>n</th><th>q</th><th>Tier</th></tr></thead>
            <tbody>{table_rows}</tbody></table></div>{more}
          </details>
        </div>
      </article>""")
    if not cards:
        return ('<p class="placeholder">No finding is supported with at least a small effect at '
                "the lower end of its confidence interval.</p>")
    return "".join(cards)


def _matrix(data: dict[str, Any], rows: dict[str, dict[str, Any]]) -> str:
    matrix = data.get("matrix") or {}
    lenses = (data.get("settings") or {}).get("lenses") or {}
    columns = matrix.get("columns") or []
    if not matrix.get("rows"):
        return '<p class="placeholder">No results are available for the evidence matrix.</p>'
    head = "".join(
        f'<th scope="col"><a href="{esc(lenses.get(column, {}).get("link", "#"))}">'
        f'{esc(lenses.get(column, {}).get("label", column))}</a>'
        f'<span class="matrix-source">{esc(lenses.get(column, {}).get("source", ""))}</span></th>'
        for column in columns
    )
    body = []
    for item in matrix["rows"]:
        cells = []
        for column in columns:
            cell = (item.get("cells") or {}).get(column)
            if not cell:
                cells.append('<td class="matrix-cell matrix-na">–</td>')
                continue
            best = rows.get(cell.get("best"))
            band = cell.get("band") if cell.get("supported") else None
            tone = _BAND_TONE.get(band, "none") if band else "none"
            title = esc(best["statement"]) if best else ""
            marker = "✓" if cell.get("supported") else "○"
            context = column in ("embedding", "detection")
            cells.append(
                f'<td class="matrix-cell tone-{tone}{" matrix-context" if context else ""}" '
                f'title="{title}"><span class="matrix-mark">{marker}</span> '
                f'{cell.get("supported", 0)}/{cell.get("tests", 0)}'
                f'<span class="matrix-band">{esc(band or "")}</span></td>'
            )
        body.append(
            f'<tr><th scope="row">{esc(label(item["attribute"]))}</th>{"".join(cells)}'
            f'<td class="matrix-convergence">{item.get("convergence", 0)}</td></tr>'
        )
    null_cells = [
        (item["attribute"], column)
        for item in matrix["rows"] for column in columns
        if column not in ("embedding", "detection")
        and (item.get("cells") or {}).get(column)
        and item["cells"][column].get("tests") and not item["cells"][column].get("supported")
    ]
    null_html = ""
    if null_cells:
        null_html = callout("read", [
            "Tested but <strong>nothing supported</strong> (absence of evidence at this sample "
            "size, not evidence of absence): " + "; ".join(
                f"{esc(label(attribute))} × {esc(lenses.get(column, {}).get('label', column))}"
                for attribute, column in null_cells) + "."
        ], "Null results", suggestion=False)
    return f"""
    <div class="table-wrap"><table class="evidence-matrix">
      <thead><tr><th scope="col">Prompt attribute</th>{head}
        <th scope="col">Lenses with support</th></tr></thead>
      <tbody>{''.join(body)}</tbody></table></div>
    <p class="note">Cell = supported / tested results for that attribute and lens; colour = effect
    band of the strongest supported result (hover for it); ✓ = at least one supported result,
    ○ = tested, none supported; – = not measured. Embedding and face-detection columns are
    context and measurement checks, so they are not counted as bias lenses.</p>
    {null_html}"""


def _measurement(data: dict[str, Any], rows: dict[str, dict[str, Any]]) -> str:
    measurement = data.get("measurement") or {}
    settings = data.get("settings") or {}
    items = []
    detection = [rows[row_id] for row_id in measurement.get("low_detection") or []
                 if row_id in rows]
    if detection:
        items.append(
            f"<strong>Face detection gaps</strong> (≥ {100 * settings.get('detection_gap', 0.1):.0f} pp "
            "below the other values, supported): " + "; ".join(
                esc(row["statement"].rstrip(".")) for row in detection[:8]) + ". Face-based results about "
            "these subgroups rest on fewer images, and the missing images may differ "
            "systematically from the detected ones."
        )
    mismatch = measurement.get("label_mismatch_values") or []
    if mismatch:
        items.append(
            "<strong>Label vocabulary mismatches</strong>: the detector never outputs "
            + ", ".join(f"{esc(item['attribute'])} = “{esc(item['value'])}”" for item in mismatch)
            + ". Adherence for these values is 0% by construction; they are excluded from "
            "the quality-of-service ranking."
        )
    low = measurement.get("low_adherence") or []
    if low:
        items.append(
            f"<strong>Requests rarely followed</strong> (&lt; {pct(settings.get('low_adherence', 0.5), 0)} "
            "adherence): " + ", ".join(
                f"{esc(item['attribute'])} = {esc(item['value'])} ({pct(item['rate'], 0)})"
                for item in low) + ". Results conditioned on these prompt values describe "
            "images that often do not show the requested group."
        )
    affected = measurement.get("affected_supported") or {}
    if affected:
        items.append("Supported results carrying a caveat: " + ", ".join(
            f"{esc(_FLAG_TEXT.get(code, code))}: {count}" for code, count in sorted(affected.items())
        ) + ". Filter the explorer by tier to inspect them.")
    if not items:
        items.append("No measurement caveat was triggered.")
    warnings = data.get("warnings") or []
    warning_html = callout("caution", [esc(item) for item in warnings],
                           "Synthesis warnings", suggestion=False) if warnings else ""
    return callout("caution", items, "Measurement reliability", suggestion=False) + warning_html


# --------------------------------------------------------------------------
# Interactive part
# --------------------------------------------------------------------------

_SCRIPT = r"""
(() => {
  const payload = JSON.parse(document.getElementById("chapter-9-data").textContent);
  const rows = payload.rows;
  const lenses = payload.lenses;
  const smallN = payload.small_n;
  const alpha = payload.alpha;
  const lensColor = {};
  const palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
                   "#8C6D31", "#7F7F7F", "#332288", "#B8336A", "#44AA99"];
  Object.keys(lenses).forEach((lens, index) => { lensColor[lens] = palette[index % palette.length]; });
  const tierColor = {supported: "#c0392b", descriptive: "#c97a00", negligible: "#9aa5b1",
                     context: "#829ab1", measurement: "#2f6fb0"};
  const tierLabel = payload.tiers;
  const el = (id) => document.getElementById(id);
  const controls = {
    lens: el("ch9-lens"), attribute: el("ch9-attribute"), tier: el("ch9-tier"),
    minN: el("ch9-min-n"), question: el("ch9-question"), search: el("ch9-search"),
    specific: el("ch9-specific")
  };
  const options = (select, values, labels) => {
    select.innerHTML += values.map((value) =>
      '<option value="' + value + '">' + (labels ? labels[value] || value : value) + "</option>").join("");
  };
  options(controls.lens, Object.keys(lenses), Object.fromEntries(
    Object.entries(lenses).map(([key, value]) => [key, value.label])));
  options(controls.attribute, Array.from(new Set(rows.map((row) => row.attribute)
    .filter(Boolean))).sort());
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[c]);
  const fmtEffect = (row) => {
    const v = row.effect;
    if (v === null || v === undefined) return "–";
    const [lo, hi] = row.ci || [null, null];
    const both = lo !== null && hi !== null && lo !== undefined && hi !== undefined;
    if (row.effect_unit === "pp") return (100 * v >= 0 ? "+" : "") + (100 * v).toFixed(0) + " pp" +
      (both ? " [" + (100 * lo).toFixed(0) + ", " + (100 * hi).toFixed(0) + "]" : "");
    if (row.effect_unit === "share") return (100 * v).toFixed(0) + "%" +
      (both ? " [" + (100 * lo).toFixed(0) + "%, " + (100 * hi).toFixed(0) + "%]" : "");
    if (row.effect_unit === "ratio") return v.toFixed(2) + "×";
    if (row.effect_unit === "value") return (v >= 0 ? "+" : "") + v.toPrecision(3) +
      (both ? " [" + lo.toPrecision(3) + ", " + hi.toPrecision(3) + "]" : "");
    return v.toFixed(2) + (both ? " [" + lo.toFixed(2) + ", " + hi.toFixed(2) + "]" : "");
  };
  const fmtQ = (q) => q === null || q === undefined ? "–" : (q < 0.001 ? "<0.001" : q.toFixed(3));
  const rankKey = (row) => [-Math.min(row.strength_low ?? 0, 3), -(row.n ?? 0), -(row.strength ?? 0)];
  const compare = (a, b) => {
    const ka = rankKey(a), kb = rankKey(b);
    for (let i = 0; i < ka.length; i += 1) if (ka[i] !== kb[i]) return ka[i] - kb[i];
    return 0;
  };
  let sortField = null, sortAsc = false;
  // ``anyTier``: the reliability plot compares tiers, so it ignores the tier
  // filter and shows every substantive result.
  const filtered = (anyTier) => {
    const lens = controls.lens.value, attribute = controls.attribute.value;
    const tier = anyTier ? "substantive" : controls.tier.value;
    const question = controls.question.value;
    const minN = Number(controls.minN.value) || 0;
    const search = controls.search.value.trim().toLowerCase();
    const specific = controls.specific.checked;
    return rows.filter((row) =>
      (!lens || row.lens === lens) &&
      (!attribute || row.attribute === attribute) &&
      (!question || row.question === question) &&
      (tier === "all" || (tier === "supported" ? row.tier === "supported"
        : tier === "substantive" ? !["context", "measurement"].includes(row.tier)
        : row.tier === tier)) &&
      (row.n === null || row.n === undefined || row.n >= minN) &&
      (!specific || row.level === "specific") &&
      (!search || (/^f\d+$/.test(search) ? row.cluster === search
        : (row.statement + " " + row.id).toLowerCase().includes(search))));
  };
  const short = (text, size) => text.length > size ? text.slice(0, size - 1) + "…" : text;
  const drawForest = (data) => {
    const ranked = data.filter((row) => row.strength !== null && row.strength !== undefined)
      .sort(compare).slice(0, 30).reverse();
    const target = el("ch9-forest");
    if (!ranked.length) { target.innerHTML = '<p class="note">No result matches the filters.</p>'; return; }
    target.innerHTML = "";
    const traces = Object.keys(lenses).map((lens) => {
      const items = ranked.filter((row) => row.lens === lens);
      return {
        type: "scatter", mode: "markers", name: lenses[lens].label,
        x: items.map((row) => row.strength),
        y: items.map((row) => row.id),
        marker: {color: lensColor[lens], size: 10,
                 symbol: items.map((row) => row.tier === "supported" ? "circle" : "circle-open")},
        error_x: {type: "data", symmetric: false, color: lensColor[lens], thickness: 1.5,
                  array: items.map((row) => Math.max(0, (row.strength_ci?.[1] ?? row.strength) - row.strength)),
                  arrayminus: items.map((row) => Math.max(0, row.strength - (row.strength_ci?.[0] ?? row.strength)))},
        customdata: items.map((row) => [escapeHtml(row.statement), fmtEffect(row), row.n, fmtQ(row.q_value),
                                        tierLabel[row.tier] || row.tier]),
        hovertemplate: "%{customdata[0]}<br>Effect %{customdata[1]} · n = %{customdata[2]} · q %{customdata[3]}" +
                       "<br>Strength %{x:.2f} · %{customdata[4]}<extra></extra>"
      };
    }).filter((trace) => trace.x.length);
    const labels = Object.fromEntries(ranked.map((row) => [row.id, short(row.statement, 70)]));
    Plotly.react(target, traces, {
      height: Math.max(420, 26 * ranked.length + 120),
      margin: {l: 430, r: 20, t: 20, b: 50},
      xaxis: {title: "Strength (1 = small, 2 = medium, 3 = large); bars = 95% CI", range: [0, 6.3], zeroline: false},
      yaxis: {type: "category", categoryorder: "array", categoryarray: ranked.map((row) => row.id),
              tickmode: "array", tickvals: ranked.map((row) => row.id),
              ticktext: ranked.map((row) => labels[row.id]), automargin: false, tickfont: {size: 11}},
      shapes: [1, 2, 3].map((x) => ({type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1,
                                     line: {dash: "dot", color: "#9aa5b1", width: 1}})),
      legend: {orientation: "h", y: -0.12}
    });
  };
  const drawScatter = (data) => {
    const target = el("ch9-reliability");
    const usable = data.filter((row) => row.strength !== null && row.strength !== undefined && row.n);
    if (!usable.length) { target.innerHTML = '<p class="note">No result matches the filters.</p>'; return; }
    target.innerHTML = "";
    const tiers = Array.from(new Set(usable.map((row) => row.tier)));
    const traces = tiers.map((tier) => {
      const items = usable.filter((row) => row.tier === tier);
      return {
        type: "scattergl", mode: "markers", name: tierLabel[tier] || tier,
        x: items.map((row) => row.n), y: items.map((row) => row.strength),
        marker: {color: tierColor[tier] || "#7f7f7f", size: 7, opacity: 0.65},
        customdata: items.map((row) => [escapeHtml(short(row.statement, 140)), fmtEffect(row),
                                        fmtQ(row.q_value), lenses[row.lens].label]),
        hovertemplate: "%{customdata[0]}<br>%{customdata[3]} · effect %{customdata[1]} · q %{customdata[2]}" +
                       "<br>n = %{x} · strength %{y:.2f}<extra></extra>"
      };
    });
    Plotly.react(target, traces, {
      height: 460, margin: {l: 60, r: 20, t: 30, b: 60},
      xaxis: {title: "n of the smallest compared group (log scale)", type: "log"},
      yaxis: {title: "Strength", range: [0, 6.3]},
      shapes: [
        {type: "rect", xref: "x", yref: "y", x0: 1, x1: smallN, y0: 3, y1: 6.3,
         fillcolor: "#c0392b", opacity: 0.07, line: {width: 0}},
        ...[1, 2, 3].map((y) => ({type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y,
                                  line: {dash: "dot", color: "#9aa5b1", width: 1}}))
      ],
      annotations: [{x: 0, y: 1.0, xref: "paper", yref: "paper", xanchor: "left", yanchor: "bottom",
                     text: "shaded: large effect, n < " + smallN, showarrow: false,
                     font: {size: 11, color: "#9b2c1f"}}],
      legend: {orientation: "h", y: -0.18}
    });
  };
  const columns = [
    ["statement", "Result"], ["lens", "Lens"], ["effect", "Effect [95% CI]"],
    ["strength", "Strength"], ["n", "n"], ["q_value", "q"], ["tier", "Tier"]
  ];
  const drawTable = (data) => {
    let items = data.slice();
    if (sortField) {
      items.sort((a, b) => {
        const x = a[sortField], y = b[sortField];
        const nx = x === null || x === undefined, ny = y === null || y === undefined;
        if (nx || ny) return nx === ny ? 0 : (nx ? 1 : -1);
        return (x > y ? 1 : x < y ? -1 : 0) * (sortAsc ? 1 : -1);
      });
    } else {
      items.sort(compare);
    }
    const shown = items.slice(0, 250);
    el("ch9-count").textContent = data.length + " result(s) match" +
      (data.length > shown.length ? "; showing the first " + shown.length + "." : ".");
    const head = columns.map(([key, name]) =>
      '<th scope="col"><button type="button" class="sort" data-field="' + key + '">' + name +
      (sortField === key ? (sortAsc ? " ▲" : " ▼") : "") + "</button></th>").join("");
    const body = shown.map((row) => "<tr><td>" + escapeHtml(row.statement) +
      (row.flags_text ? '<br><span class="note">⚠ ' + escapeHtml(row.flags_text) + "</span>" : "") +
      "</td><td>" + '<a href="' + row.link + '">' + escapeHtml(lenses[row.lens].label) + "</a></td><td>" +
      fmtEffect(row) + "</td><td>" + (row.strength === null || row.strength === undefined ? "–" : row.strength.toFixed(2)) +
      "</td><td>" + (row.n ?? "–") + "</td><td>" + fmtQ(row.q_value) + "</td><td>" +
      escapeHtml(tierLabel[row.tier] || row.tier) + (row.cluster ? " · " + row.cluster : "") + "</td></tr>").join("");
    el("ch9-table").innerHTML = "<table><thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody></table>";
    el("ch9-table").querySelectorAll("button.sort").forEach((button) => {
      button.onclick = () => {
        const field = button.dataset.field;
        if (sortField === field) sortAsc = !sortAsc; else { sortField = field; sortAsc = field === "statement"; }
        drawTable(filtered(false));
      };
    });
  };
  const csvField = (value) => {
    if (value === null || value === undefined) return "";
    const text = Array.isArray(value) ? value.join(";") : String(value);
    return /[",\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  };
  el("ch9-download").onclick = () => {
    const data = filtered(false);
    const fields = payload.fields.concat(["flags_text"]);
    const csv = [fields.join(",")].concat(data.map((row) => fields.map((field) => csvField(row[field])).join(","))).join("\n");
    const blob = new Blob([csv], {type: "text/csv"});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "fairness_results.csv";
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  };
  const draw = () => {
    const data = filtered(false);
    drawForest(data); drawScatter(filtered(true)); drawTable(data);
  };
  Object.values(controls).forEach((control) => control.addEventListener(
    control.tagName === "INPUT" && control.type !== "checkbox" ? "input" : "change", draw));
  document.querySelectorAll("[data-ch9-filter]").forEach((link) => link.addEventListener("click", () => {
    const [field, value] = link.dataset.ch9Filter.split(":");
    if (field === "search") controls.search.value = value;
    if (field === "lens") controls.lens.value = value;
    controls.tier.value = "substantive";
    el("ch9-explorer-details").open = true;
    draw();
  }));
  draw();
})();
"""


def _payload(data: dict[str, Any]) -> str:
    rows = []
    for row in data.get("results") or []:
        item = {field: row.get(field) for field in _ROW_FIELDS}
        item["flags_text"] = "; ".join(flag.get("text", "").rstrip(".")
                                       for flag in row.get("flags") or [])
        rows.append(item)
    settings = data.get("settings") or {}
    payload = {
        "rows": rows,
        "fields": list(_ROW_FIELDS),
        "lenses": settings.get("lenses") or {},
        "tiers": settings.get("tiers") or {},
        "small_n": settings.get("small_n", 30),
        "alpha": settings.get("alpha", 0.05),
    }
    return json.dumps(payload, ensure_ascii=True).replace("</", "<\\/")


def _explorer_controls(data: dict[str, Any]) -> str:
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


_STYLE = """
  <style>
    .answer-grid { grid-template-columns: repeat(auto-fit, minmax(min(420px, 100%), 1fr)); }
    .answer-head { align-items: center; display: flex; flex-wrap: wrap; gap: .5rem;
      justify-content: space-between; }
    .answer-head h4 { margin: 0; }
    .answer-list { padding-left: 1.2rem; } .answer-list li { margin-bottom: .6rem; }
    .answer-list .stat-line { color: var(--muted); font-size: .82rem; margin-top: .15rem; }
    .finding-card { display: grid; gap: 1rem; grid-template-columns: 2.4rem minmax(0, 1fr); }
    .finding-rank { align-items: center; background: var(--finding-bg); border-radius: 50%;
      color: var(--finding); display: flex; font-size: 1.1rem; font-weight: 800; height: 2.4rem;
      justify-content: center; width: 2.4rem; }
    .finding-statement { font-size: 1.05rem; font-weight: 600; margin: 0 0 .4rem; }
    .finding-body .stat-line { color: var(--muted); }
    .evidence-matrix th, .evidence-matrix td { text-align: center; white-space: nowrap; }
    .evidence-matrix th[scope=row] { text-align: left; }
    .evidence-matrix thead th { font-size: .78rem; vertical-align: bottom; }
    .matrix-source { color: var(--muted); display: block; font-size: .68rem; font-weight: 400; }
    .matrix-cell { border-left: 1px solid #fff; font-size: .85rem; }
    .matrix-band { display: block; font-size: .7rem; text-transform: uppercase; letter-spacing: .04em; }
    .matrix-mark { font-weight: 700; }
    .tone-high { background: #f9d7d3; color: #9b2c1f; }
    .tone-mid { background: #fdebd0; color: #8a4b00; }
    .tone-low { background: #e0ecf9; color: #1f4f82; }
    .tone-neutral, .tone-none { background: #f4f6f8; color: var(--muted); }
    .matrix-context { opacity: .75; font-style: italic; }
    .matrix-na { color: #bcccdc; }
    .matrix-convergence { font-weight: 800; }
    .ch9-filters { position: sticky; top: 0; z-index: 5; }
    #ch9-table table { font-size: .86rem; }
    #ch9-table button.sort { background: none; border: 0; color: var(--muted); cursor: pointer;
      font: inherit; font-weight: 700; padding: 0; }
    .tier-list dt { font-weight: 700; } .tier-list dd { margin: 0 0 .4rem; }
  </style>"""


def render_chapter_nine(data: dict[str, Any]) -> str:
    """Render the cross-analysis chapter from the synthesis payload."""
    if not data or not data.get("results"):
        return (chapter_header(9, "Cross-analysis and key findings",
                               "Which disparities are robust across the audit?",
                               "<p>The synthesis data is not available.</p>")
                + '<p class="placeholder">Rerun <code>python -m src.analysis.analyzer</code> '
                  "to build chapter 9.</p>")
    rows = {row["id"]: row for row in data.get("results") or []}
    clusters = {cluster["id"]: cluster for cluster in data.get("clusters") or []}
    settings = data.get("settings") or {}
    alpha = settings.get("alpha", 0.05)
    families = settings.get("families") or {}
    family_rows = "".join(
        f"<tr><th scope='row'>{esc(value['label'])}</th><td>"
        + " / ".join(f"{threshold:g}" for threshold in value["thresholds"])
        + "</td></tr>"
        for value in families.values()
    )
    tiers = settings.get("tiers") or {}
    evidence_levels = settings.get("evidence_levels") or {}
    correction = data.get("correction") or {}
    return f"""
  {_STYLE}
  {chapter_header(9, "Cross-analysis and key findings",
      "Which disparities are large, statistically reliable and confirmed by more than one "
      "kind of evidence — and which are not?",
      "<p>Chapters 3–8 each look at the model through one lens. This chapter puts every "
      "effect they measured into <strong>one results table</strong> with a common format: "
      "the subgroup, what it is compared with, the effect in its own unit with a 95% "
      "confidence interval, a standardized <em>strength</em>, the sample size and a p-value. "
      "All p-values of the audit are then corrected <em>together</em> for multiple testing, "
      "and results that make the same claim are linked across independent evidence "
      "sources — the face detector, the object detector, the captions, pixel statistics and "
      "matched counterfactual prompts. A pattern seen by several instruments, or one that "
      "persists when only a single prompt word is changed, is much stronger evidence than a "
      "single chart.</p>"
      "<p>Everything below is computed from the analysis artifact; a different model or "
      "dataset produces different findings. Findings are classified by the evidence behind "
      "them, not labelled “fair” or “unfair”: whether a disparity is harmful is a judgement "
      "the evidence informs.</p>")}
  {_overview(data, rows, clusters)}
  {metric_defs([
      ("Strength", "The effect size mapped onto a common scale using the conventional "
       "small / medium / large thresholds of its own family (table below): 1 = small, "
       "2 = medium, 3 = large; above 3 it keeps growing linearly (capped at 6).",
       "Makes a gap in percentage points, a mean difference and a Cramér's V comparable. "
       "Bands: &lt; 1 negligible, 1–2 small, 2–3 medium, ≥ 3 large."),
      ("Conservative strength", "The strength at the lower end of the effect's 95% "
       "confidence interval.", "Used for ranking: the smallest effect still compatible with "
       "the data, so a huge gap in 8 images does not outrank a solid one in 200."),
      ("q-value", f"Benjamini–Hochberg adjusted p-value over all {correction.get('tests', 0):,} "
       "tests of the audit, including the candidate objects and caption words screened "
       "before the top ones were stored.", f"A result is <em>supported</em> when q &lt; {alpha} "
       "and its strength is at least small. The expected share of false discoveries among "
       f"supported results is at most {pct(alpha, 0)}."),
      ("Comparison standard", "Every specific result compares a subgroup with the "
       "<em>other values of the same prompt attribute</em>, using only images whose prompt "
       "does not request the measured feature (e.g. job × detected gender without “female "
       "doctor” prompts).", "Removes the most common confound: a difference that is just the "
       "prompt asking for it."),
      ("Evidence level", "<em>" + esc(evidence_levels.get("confirmed", "")) + "</em>: supported "
       "in a pooled comparison and in matched prompt pairs that change only this attribute. "
       "<em>" + esc(evidence_levels.get("convergent", "")) + "</em>: supported by two or more "
       "measurement instruments. <em>" + esc(evidence_levels.get("supported", "")) + "</em>: "
       "one instrument.", "Matched pairs hold the rest of the prompt fixed, so they rule out "
       "template differences; several instruments rule out a single detector's error."),
      ("Stereotype candidate", "A supported link between a social-group attribute "
       "(gender, race, age, continent, status, traits) and non-requested content, or between "
       "any prompt attribute and a non-requested demographic.", "A pattern that matches the "
       "shape of a stereotype and deserves human review. It is not a verdict."),
  ], open_=False)}
  <details class="defs"><summary>Effect-size families and thresholds (small / medium / large)</summary>
    <table><thead><tr><th>Family</th><th>Thresholds</th></tr></thead><tbody>{family_rows}</tbody></table>
    <p class="note">Proportion gaps use Cohen's h, mean differences Hedges' g (or d_z for paired
    differences), contingency tables Cohen's w = V·√(min(r, c) − 1), group differences on
    continuous features ε², embedding structure the silhouette above its permutation null.</p>
  </details>

  <section id="chapter-9-answers">
    <h3>9.1 Answers to the fairness questions</h3>
    <p>The five questions of the reading guide, answered from the results table. The verdict
    is the effect band of the strongest supported result; the three listed results are the
    strongest by conservative strength (ties broken by sample size).</p>
    {guide(
        read="Each card counts the specific tests that address the question and how many are "
             "supported. Below each result: its effect with 95% CI, the strength, the smallest "
             "group size, the q-value and a link to the chapter that measured it.",
        look=["A large verdict backed by results with large n and no ⚠ caveats.",
              "Questions where nothing is supported despite many tests: an informative null, "
              "if the samples are large enough to detect a small effect."],
        caution="The verdict summarises the strongest result only. Read the listed results and "
                "the evidence matrix before generalising to the whole question.",
    )}
    {_answers(data, rows)}
  </section>

  <section id="chapter-9-findings">
    <h3>9.2 Key findings</h3>
    <p>The strongest supported findings, grouped by the claim they make and ranked by
    conservative strength, then evidence level and sample size. At most two findings per
    attribute × feature theme and three per lens are listed, so one dominant pattern does not
    hide the others. Expand a finding to see every result behind it.</p>
    {guide(
        read="Badges give the evidence level, the effect band and the number of independent "
             "sources; ⚠ badges are caveats from 9.6 (hover for details).",
        look=["Counterfactual-confirmed and convergent findings first: they survive a change "
              "of instrument or of prompt template.",
              "Stereotype candidates linking a social group to roles, emotions, objects or "
              "words."],
        caution=["Every demographic label is a detector prediction. A finding about "
                 "“detected female” is a finding about what the face model reports.",
                 "Images in matched pairs are partly the same images as in pooled comparisons; "
                 "confirmation shows the pattern is not a template artefact, it is not an "
                 "independent replication."],
    )}
    {_finding_cards(data, rows, clusters)}
  </section>

  <section id="chapter-9-matrix">
    <h3>9.3 Evidence matrix</h3>
    <p>One row per prompt attribute, one column per lens. A disparity that appears in several
    columns of the same row is a robust property of how the model treats that attribute.</p>
    {_matrix(data, rows)}
  </section>

  <section id="chapter-9-ranked">
    <h3>9.4 Ranked effect sizes</h3>
    <p>The filters below drive 9.4, 9.5 and the explorer in 9.7. The plot shows the 30
    strongest results that match, with the 95% CI of their strength.</p>
    {_explorer_controls(data)}
    {guide(
        read="Each row is one result; colour = lens; filled marker = supported, open = not "
             "supported. Dotted lines mark the small, medium and large thresholds. Hover for "
             "the full statement, effect, n and q.",
        look="Long bars reaching down towards 0: a large point estimate with little "
             "certainty. Narrow bars far to the right are the most reliable large effects.",
        caution="Strength puts different metrics on one scale using conventions, not a "
                "physical unit; compare bands, not decimals.",
    )}
    <article class="plot-card"><div id="ch9-forest"></div></article>
  </section>

  <section id="chapter-9-reliability">
    <h3>9.5 Effect versus reliability</h3>
    <p>Strength against the size of the smallest compared group. Points in the shaded corner
    are large effects measured on fewer than {esc(settings.get("small_n", 30))} images:
    candidates for re-testing with more generations rather than conclusions.</p>
    <article class="plot-card"><div id="ch9-reliability"></div></article>
  </section>

  <section id="chapter-9-measurement">
    <h3>9.6 Measurement reliability and caveats</h3>
    <p>Problems in the measurement pipeline can create or hide disparities. These checks are
    applied automatically and attached as ⚠ flags to the affected results.</p>
    {_measurement(data, rows)}
  </section>

  <section id="chapter-9-explorer">
    <h3>9.7 Results explorer</h3>
    <p>The full results table (filtered as above). Click a column to sort; the default order is
    the ranking used in 9.2. <button type="button" id="ch9-download">Download filtered results
    (CSV)</button></p>
    <details class="defs" id="ch9-explorer-details">
      <summary>Show the results table (<span id="ch9-count" aria-live="polite"></span>)</summary>
      <div id="ch9-table" class="table-wrap"></div>
    </details>
  </section>

  <details class="defs" id="chapter-9-method"><summary>Method and limitations of the synthesis</summary>
    <dl class="tier-list">
      {''.join(f"<dt>{esc(name)}</dt><dd>{esc(_TIER_HELP.get(key, ''))}</dd>" for key, name in tiers.items())}
    </dl>
    <ul>
      <li>Intervals: Wilson score intervals for proportions and Newcombe's method for their
      differences; t intervals for means; approximate intervals for Hedges' g; a multinomial
      bootstrap for Cramér's V; Wald intervals for log odds ratios of caption words.</li>
      <li>Tests: two-proportion z-tests, Fisher's exact test (objects), chi-square tests
      (contingency tables, adherence by requested value, parity), Mann–Whitney and
      Kruskal–Wallis tests (image features), one-sample t-tests (paired differences), a
      permutation null (silhouette).</li>
      <li>Multiple testing: one Benjamini–Hochberg family over the whole audit
      ({correction.get('tests', 0):,} tests). Screened candidates that were not stored are
      counted with p = 1, which is conservative. Tests are not independent (the same images
      appear in several comparisons); BH remains valid under the positive dependence typical
      here, but treat q-values near α as borderline.</li>
      <li>Default representation is compared with parity over the detector's categories.
      Parity is a reference point, not a claim about the correct distribution.</li>
      <li>Linking results into findings uses exact matches of (attribute, subgroup, feature,
      category, direction); caption words are linked to demographic categories through a small
      lexicon (configurable with <code>synthesis_concept_lexicon</code>). Links can be missed
      when two instruments use different words for the same thing.</li>
      <li>Pairs in chapter 6 are matched by prompt template and generation order, not by
      seed. Each side is therefore treated as an independent sample.</li>
      <li>The synthesis cannot establish causes (training data, architecture) and inherits
      every limitation of the detectors, the captioner and CLIP.</li>
    </ul>
  </details>
  <script type="application/json" id="chapter-9-data">{_payload(data)}</script>
  <script>{_SCRIPT}</script>
"""


_TIER_HELP = {
    "supported": "q below α after the audit-wide correction and at least a small effect.",
    "descriptive": "At least a small effect, but not statistically supported after correction "
                   "(often a small group).",
    "negligible": "Effect below the small threshold, whatever its p-value.",
    "context": "Expected or descriptive information (e.g. a caption naming the requested job, "
               "objects of the requested workplace, embedding separability of a requested "
               "attribute). Not counted as a disparity.",
    "measurement": "A property of the measurement pipeline (face-detection gaps, label "
                   "vocabulary mismatches) rather than of the model.",
}
