"""Chapter 1: headline indicators and conclusions of the executive summary."""

from __future__ import annotations

from typing import Any

from ..measurable import measurable_accuracy, unmeasurable_from_counts
from ..explain import esc, label, number, pct
from .common import MIN_GROUP, category_shares, strongest_associations


def headline_indicators(report_data: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """Return (label, value, explanation, link) tiles for the executive summary."""
    tiles: list[tuple[str, str, str, str]] = []
    summary = report_data.get("chapter_1", {})
    generated = number(summary.get("generated_images"))
    valid = number(summary.get("valid_images"))
    if generated:
        tiles.append((
            "Face detection rate", pct((valid or 0) / generated),
            "Images where a face was found.",
            "#chapter-2-detections",
        ))
    composition = report_data.get("chapter_3", {}).get("composition", {})
    unrequested = composition.get("gender", {}).get("unrequested")
    if isinstance(unrequested, dict):
        shares = category_shares(unrequested.get("detected", {}).get("counts", {}))
        help_text = "Images whose prompt doesn't request a gender."
    else:  # artifacts built before the unrequested slice existed
        gender = composition.get("gender", {}).get("by_slice", {}).get("prompt_mode", {})
        single = next((value for key, value in gender.items()
                       if str(key).lower() == "single variable"), None)
        shares = category_shares((single or {}).get("detected", {}).get("counts", {}))
        help_text = "Single-variable prompts."
    if shares:
        top, share = max(shares.items(), key=lambda item: item[1])
        tiles.append((
            "Default gender", f"{pct(share, 0)} {esc(top)}", help_text,
            "#chapter-3-distributions",
        ))
    adherence = report_data.get("chapter_4", {}).get("by_attribute", {})
    # Requests the detector cannot output would score 0% by construction, so the
    # tile uses the adherence on the values it can report.
    accuracies = {}
    for name, item in adherence.items():
        if number(item.get("accuracy")) is None or item.get("total", 0) < MIN_GROUP:
            continue
        counts = item.get("counts") or {}
        excluded = unmeasurable_from_counts(counts)
        accuracies[name] = (measurable_accuracy(counts, excluded) if excluded
                            else number(item.get("accuracy")))
    accuracies = {name: value for name, value in accuracies.items() if value is not None}
    if accuracies:
        worst = min(accuracies, key=accuracies.get)
        best = max(accuracies, key=accuracies.get)
        tiles.append((
            "Prompt adherence", f"{pct(accuracies[best], 0)} → {pct(accuracies[worst], 0)}",
            f"Best ({esc(best)}) to worst ({esc(worst)}) attribute, on values the detector "
            "can output.",
            "#chapter-4-attributes",
        ))
    strongest = strongest_associations(report_data.get("chapter_5", {}), limit=1)
    if strongest:
        item = strongest[0]
        tiles.append((
            "Strongest association", f"V = {item['v']:.2f}",
            f"{esc(label(item['prompt']))} → {esc(label(item['output']))} "
            f"(n={item['n']}).",
            "#chapter-5-overview",
        ))
    synthesis = report_data.get("chapter_9") or {}
    summary = synthesis.get("summary") or {}
    if summary.get("tests"):
        evidence = summary.get("evidence") or {}
        multi = evidence.get("confirmed", 0) + evidence.get("convergent", 0)
        tiles.append((
            "Supported findings", f"{summary.get('findings', 0)}",
            f"{summary.get('supported', 0)} of {summary.get('tests', 0)} results survive the "
            f"audit-wide FDR correction; {multi} confirmed by a second evidence source.",
            "#chapter-9",
        ))
    return tiles


def conclusive_insights(report_data: dict[str, Any]) -> list[str]:
    """Return short HTML conclusions shown under the executive-summary scorecard.

    Everything is drawn from the chapter-9 synthesis: how many results survive
    the audit-wide correction, the most robust finding, and how much the
    measurements can be trusted. The per-question verdicts are in the scorecard
    (``question_scorecard``). The wording states what was measured and never
    calls a model fair or unfair.
    """
    synthesis = report_data.get("chapter_9") or {}
    summary = synthesis.get("summary") or {}
    if not summary.get("tests"):
        return []
    clusters = {item.get("id"): item for item in synthesis.get("clusters") or []}
    answers = synthesis.get("answers") or []
    settings = synthesis.get("settings") or {}
    levels = settings.get("evidence_levels") or {}
    alpha = settings.get("alpha", 0.05)
    evidence = summary.get("evidence") or {}
    multi = evidence.get("confirmed", 0) + evidence.get("convergent", 0)
    items: list[str] = []

    found = [a for a in answers if a.get("band") and a.get("supported")]
    if found:
        items.append(
            f"Disparities were found in <strong>{len(found)} of {len(answers)}</strong> fairness "
            f"questions (table above). Out of {summary.get('tests', 0):,} tests, "
            f"{summary.get('supported', 0):,} "
            f"remain after correcting for multiple testing (q &lt; {alpha}) with at least a "
            "small effect, and they merge into "
            f"{summary.get('findings', 0):,} distinct findings."
        )
    else:
        items.append(
            f"None of the {summary.get('tests', 0):,} tests supports a disparity after "
            "correcting for multiple testing. That is not proof of fairness: it means this "
            "sample could not tell the groups apart from noise."
        )

    key = [clusters[item] for item in synthesis.get("key_findings") or [] if item in clusters]
    if key:
        lead = key[0]
        level = levels.get(lead.get("evidence"), lead.get("evidence", ""))
        items.append(
            f"<strong>Most robust finding</strong> ({esc(str(level).lower())}; "
            f"{esc(', '.join(lead.get('sources', [])))}): "
            f"{esc(lead['statement'].rstrip('.'))}."
        )
    if multi:
        items.append(
            f"{multi} finding(s) are corroborated by a second kind of evidence "
            f"({evidence.get('confirmed', 0)} also hold when only one prompt word changes); "
            f"the other {evidence.get('supported', 0):,} rest on a single measurement and "
            "should be read as leads."
        )
    measurement = synthesis.get("measurement") or {}
    affected = measurement.get("affected_supported") or {}
    caveats = []
    if affected.get("small_n"):
        caveats.append(f"{affected['small_n']} supported results rest on groups of fewer "
                       f"than {settings.get('small_n', 30)} images")
    if measurement.get("low_detection"):
        caveats.append("face detection is notably worse for some subgroups")
    if measurement.get("label_mismatch_values"):
        caveats.append("some requested values cannot be output by the detector")
    if caveats:
        text = "; ".join(caveats)
        items.append("<strong>Reliability.</strong> " + text[:1].upper() + text[1:]
                     + ". All demographic labels are detector predictions "
                     "(see chapter 10).")
    return items


def question_scorecard(report_data: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per fairness question of the chapter-9 synthesis, for chapter 1."""
    synthesis = report_data.get("chapter_9") or {}
    rows = {row.get("id"): row for row in synthesis.get("results") or []}
    card = []
    for answer in synthesis.get("answers") or []:
        top = rows.get((answer.get("top") or [None])[0])
        card.append({
            "title": answer.get("title", ""),
            "question": answer.get("question", ""),
            "band": answer.get("band") if answer.get("supported") else None,
            "supported": answer.get("supported", 0),
            "tests": answer.get("tests", 0),
            "reliable": answer.get("reliable") or {},
            "example": top.get("statement", "") if top else "",
            "example_flags": [flag.get("code") for flag in (top or {}).get("flags") or []],
            "link": answer.get("link", "#chapter-9"),
        })
    return card
