"""Chapter 9: the statically rendered sections (overview, answers, findings,
evidence matrix and measurement caveats)."""

from __future__ import annotations

from typing import Any

from ..explain import badge, callout, esc, label, pct
from .ch9_format import (
    BAND_TEXT,
    BAND_TONE,
    EVIDENCE_TONE,
    FLAG_TEXT,
    TIER_TONE,
    band_badge,
    flag_badges,
    format_effect,
    format_q,
    reliable_counts,
    stat_line,
)


def render_overview(data: dict[str, Any], rows: dict[str, dict[str, Any]],
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
        verdict = BAND_TEXT.get(answer.get("band"), "No supported disparity")
        line = (f"<strong>{esc(answer['title'])}</strong>: {esc(verdict.lower())} — "
                f"{answer.get('supported', 0)} of {answer.get('tests', 0)} tests supported")
        if answer.get("supported"):
            line += f" ({reliable_counts(answer)})"
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


def render_answers(data: dict[str, Any], rows: dict[str, dict[str, Any]]) -> str:
    cards = []
    for answer in data.get("answers") or []:
        tone = BAND_TONE.get(answer.get("band"), "neutral")
        verdict = BAND_TEXT.get(answer.get("band"), "No supported disparity")
        items = []
        for row_id in answer.get("top") or []:
            row = rows.get(row_id)
            if not row:
                continue
            items.append(
                f"<li>{esc(row['statement'])} {flag_badges(row)}{stat_line(row)}</li>"
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
        {f"Of the supported results, {reliable_counts(answer)} at the lower end of the 95% CI."
         if answer.get('supported') else ""}
        <a href="{esc(answer['link'])}">Go to the chapter →</a></p>
        <ol class="answer-list">{''.join(items)}</ol>{extra_html}{context_html}
      </article>""")
    return f'<div class="plot-grid answer-grid">{"".join(cards)}</div>'


def render_finding_cards(data: dict[str, Any], rows: dict[str, dict[str, Any]],
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
                esc(row.get("source")), esc(row.get("statement")), esc(format_effect(row)),
                esc(row.get("n")), format_q(row.get("q_value")),
                badge(row.get("tier", ""), TIER_TONE.get(row.get("tier"), "neutral")),
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
                  EVIDENCE_TONE.get(cluster["evidence"], "neutral"))
            + " " + band_badge(cluster.get("band"), "effect: ")
            + (" " + badge("stereotype candidate", "mid") if cluster.get("stereotype_candidate") else "")
            + " " + badge(f"{len(cluster.get('sources', []))} source(s)")
        )
        cards.append(f"""
      <article class="plot-card finding-card">
        <div class="finding-rank">{position}</div>
        <div class="finding-body">
          <p class="finding-statement">{esc(cluster['statement'])}</p>
          <p>{badges} {flag_badges(lead)}</p>
          {stat_line(lead)}
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


def render_matrix(data: dict[str, Any], rows: dict[str, dict[str, Any]]) -> str:
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
            tone = BAND_TONE.get(band, "none") if band else "none"
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


def render_measurement(data: dict[str, Any], rows: dict[str, dict[str, Any]]) -> str:
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
            f"{esc(FLAG_TEXT.get(code, code))}: {count}" for code, count in sorted(affected.items())
        ) + ". Filter the explorer by tier to inspect them.")
    if not items:
        items.append("No measurement caveat was triggered.")
    warnings = data.get("warnings") or []
    warning_html = callout("caution", [esc(item) for item in warnings],
                           "Synthesis warnings", suggestion=False) if warnings else ""
    return callout("caution", items, "Measurement reliability", suggestion=False) + warning_html
