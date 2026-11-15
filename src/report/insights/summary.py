"""Chapter 1: headline indicators and conclusions of the executive summary."""

from __future__ import annotations

from typing import Any

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
    accuracies = {
        name: number(item.get("accuracy")) for name, item in adherence.items()
        if number(item.get("accuracy")) is not None and item.get("total", 0) >= MIN_GROUP
    }
    if accuracies:
        worst = min(accuracies, key=accuracies.get)
        best = max(accuracies, key=accuracies.get)
        tiles.append((
            "Prompt adherence", f"{pct(accuracies[best], 0)} → {pct(accuracies[worst], 0)}",
            f"Best ({esc(best)}) to worst ({esc(worst)}) attribute.",
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


_BAND_WORDS = {"negligible": "negligible", "small": "small", "medium": "moderate",
               "large": "large"}


def conclusive_insights(report_data: dict[str, Any]) -> list[str]:
    """Return short HTML conclusions for the executive summary.

    Everything is drawn from the chapter-9 synthesis: how many results survive
    the audit-wide correction, which fairness questions show a disparity, the
    most robust finding, and how much the measurements can be trusted. The
    wording states what was measured and never calls a model fair or unfair.
    """
    synthesis = report_data.get("chapter_9") or {}
    summary = synthesis.get("summary") or {}
    if not summary.get("tests"):
        return []
    rows = {row.get("id"): row for row in synthesis.get("results") or []}
    clusters = {item.get("id"): item for item in synthesis.get("clusters") or []}
    answers = synthesis.get("answers") or []
    settings = synthesis.get("settings") or {}
    levels = settings.get("evidence_levels") or {}
    alpha = settings.get("alpha", 0.05)
    evidence = summary.get("evidence") or {}
    multi = evidence.get("confirmed", 0) + evidence.get("convergent", 0)
    items: list[str] = []

    found = [a for a in answers if a.get("band") and a.get("supported")]
    clear = [a for a in answers if not a.get("supported")]
    if found:
        names = ", ".join(f"<strong>{esc(a['title'].lower())}</strong>" for a in found)
        items.append(
            f"Disparities were found in {len(found)} of {len(answers)} fairness questions: "
            f"{names}. Out of {summary.get('tests', 0):,} tests, {summary.get('supported', 0):,} "
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

    # One line per question with a supported disparity, largest effect first.
    order = {"large": 0, "medium": 1, "small": 2, "negligible": 3}
    for answer in sorted(found, key=lambda a: order.get(a["band"], 4)):
        top = rows.get((answer.get("top") or [None])[0])
        if not top:
            continue
        reliable = answer.get("reliable") or {}
        items.append(
            f"<strong>{esc(answer['title'])}</strong> "
            f"(largest reliable disparity: {esc(_BAND_WORDS.get(answer['band'], answer['band']))}; "
            f"{answer['supported']} of {answer['tests']} tests supported, "
            f"{reliable.get('large', 0)} reliably large): "
            f"{esc(top['statement'].rstrip('.'))}. "
            f'<a href="{esc(answer.get("link", "#chapter-9"))}">Details &rarr;</a>'
        )
    if clear:
        items.append(
            "No supported disparity for " + ", ".join(esc(a["title"].lower()) for a in clear)
            + "."
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
    stereotypes = summary.get("stereotype_candidates", 0)
    if stereotypes:
        items.append(
            f"{stereotypes:,} supported results link a social group (gender, race, age, "
            "origin, status or trait) to content nobody asked for. They are candidates for "
            "human review as possible stereotypes, not verdicts."
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
