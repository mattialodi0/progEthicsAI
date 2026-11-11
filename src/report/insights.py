"""Data-driven observations shown in each chapter's "Key observations" box.

Each function reads one chapter payload of the analysis artifact and returns a
list of short HTML sentences. They only describe the sample (largest gaps,
strongest associations, weakest coverage) and always quote the sample size, so
the reader can judge how much weight each observation deserves. Thresholds
are deliberately conservative to avoid highlighting noise from tiny groups.
"""

from __future__ import annotations

from typing import Any

from .explain import cramers_band, esc, label, pct

# Groups smaller than this are never highlighted.
MIN_GROUP = 10


def _num(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _shares(counts: dict[str, Any]) -> dict[str, float]:
    total = sum(_num(value) or 0 for value in counts.values())
    if not total:
        return {}
    return {str(key): (_num(value) or 0) / total for key, value in counts.items()}


# --------------------------------------------------------------------------- 1
def headline_indicators(report_data: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """Return (label, value, explanation, link) tiles for the executive summary."""
    tiles: list[tuple[str, str, str, str]] = []
    summary = report_data.get("chapter_1", {})
    generated = _num(summary.get("generated_images"))
    valid = _num(summary.get("valid_images"))
    if generated:
        tiles.append((
            "Face detection rate", pct((valid or 0) / generated),
            "Images where a face was found.",
            "#chapter-2-detections",
        ))
    composition = report_data.get("chapter_3", {}).get("composition", {})
    unrequested = composition.get("gender", {}).get("unrequested")
    if isinstance(unrequested, dict):
        shares = _shares(unrequested.get("detected", {}).get("counts", {}))
        help_text = "Images whose prompt doesn't request a gender."
    else:  # artifacts built before the unrequested slice existed
        gender = composition.get("gender", {}).get("by_slice", {}).get("prompt_mode", {})
        single = next((value for key, value in gender.items()
                       if str(key).lower() == "single variable"), None)
        shares = _shares((single or {}).get("detected", {}).get("counts", {}))
        help_text = "Single-variable prompts."
    if shares:
        top, share = max(shares.items(), key=lambda item: item[1])
        tiles.append((
            "Default gender", f"{pct(share, 0)} {esc(top)}", help_text,
            "#chapter-3-distributions",
        ))
    adherence = report_data.get("chapter_4", {}).get("by_attribute", {})
    accuracies = {
        name: _num(item.get("accuracy")) for name, item in adherence.items()
        if _num(item.get("accuracy")) is not None and item.get("total", 0) >= MIN_GROUP
    }
    if accuracies:
        worst = min(accuracies, key=accuracies.get)
        best = max(accuracies, key=accuracies.get)
        tiles.append((
            "Prompt adherence", f"{pct(accuracies[best], 0)} → {pct(accuracies[worst], 0)}",
            f"Best ({esc(best)}) to worst ({esc(worst)}) attribute.",
            "#chapter-4-attributes",
        ))
    strongest = _strongest_associations(report_data.get("chapter_5", {}), limit=1)
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
        items.append(
            f"<strong>{esc(answer['title'])}</strong> "
            f"({esc(_BAND_WORDS.get(answer['band'], answer['band']))} disparity, "
            f"{answer['supported']} of {answer['tests']} tests supported): "
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


# --------------------------------------------------------------------------- 2
def chapter_two(coverage: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    conditioned = {key: _num(value) or 0 for key, value in
                   (coverage.get("conditioned_images") or {}).items()}
    if len(conditioned) >= 2:
        most = max(conditioned, key=conditioned.get)
        least = min(conditioned, key=conditioned.get)
        notes.append(
            f"Testing effort is uneven: <strong>{esc(most)}</strong> is specified in "
            f"{int(conditioned[most])} images, <strong>{esc(least)}</strong> in only "
            f"{int(conditioned[least])}. Results for sparsely tested variables are "
            "less precise."
        )
    detection = coverage.get("valid_detection") or {}
    rates = {name: item["valid"] / item["total"] for name, item in detection.items()
             if isinstance(item, dict) and item.get("total")}
    if rates:
        worst = min(rates, key=rates.get)
        if rates[worst] < 0.999:
            missing = detection[worst]["total"] - detection[worst]["valid"]
            notes.append(
                f"<strong>{esc(worst.capitalize())}</strong> is the least available feature "
                f"({pct(rates[worst])} valid, {missing} images missing). Every face-based "
                "result in later chapters is computed on the images where a face was found."
            )
    ratios = []
    for attribute, values in (coverage.get("attribute_counts") or {}).items():
        counts = [_num(value) or 0 for value in values.values()]
        if len(counts) >= 2 and min(counts) > 0:
            ratios.append((max(counts) / min(counts), attribute, values))
    if ratios:
        ratio, attribute, values = max(ratios)
        if ratio >= 2:
            top = max(values, key=lambda key: values[key])
            low = min(values, key=lambda key: values[key])
            notes.append(
                f"Within <strong>{esc(label(attribute))}</strong>, values are tested unevenly: "
                f"{esc(top)} appears in {values[top]} images vs {esc(low)} in {values[low]} "
                f"({ratio:.1f}× difference). Compare percentages, not raw counts."
            )
    confidence = coverage.get("classifier_confidence") or {}
    shares = {name: item["confident"] / item["n"]
              for name, item in (confidence.get("attributes") or {}).items() if item.get("n")}
    if len(shares) >= 2:
        low = min(shares, key=shares.get)
        high = max(shares, key=shares.get)
        if shares[low] < 0.9:
            groups = {name: item["confident"] / item["n"] for name, item in
                      (confidence["attributes"][low].get("by_requested") or {}).items()
                      if item.get("n", 0) >= MIN_GROUP and name != "not requested"}
            worst = min(groups, key=groups.get) if groups else None
            notes.append(
                f"The face model is confident (top class ≥ {pct(confidence.get('threshold'), 0)}) "
                f"about <strong>{esc(low)}</strong> for only {pct(shares[low], 0)} of faces, "
                f"against {pct(shares[high], 0)} for {esc(high)}"
                + (f"; the least certain requested group is <strong>{esc(worst)}</strong> "
                   f"({pct(groups[worst], 0)} confident)" if worst else "")
                + f". Results about detected {esc(low)} rest on many near-ties between classes."
            )
    return notes


# --------------------------------------------------------------------------- 3
def chapter_three(chapter: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    composition = chapter.get("composition", {})
    for dimension in ("gender", "race"):
        slices = composition.get(dimension, {}).get("by_slice", {}).get("prompt_mode", {})
        parts = []
        for mode, value in slices.items():
            shares = _shares((value or {}).get("detected", {}).get("counts", {}))
            if shares:
                top, share = max(shares.items(), key=lambda item: item[1])
                parts.append(f"{esc(mode)}: {pct(share, 0)} {esc(top)}")
        if parts:
            notes.append(
                f"Most frequent detected <strong>{dimension}</strong> by prompt mode — "
                + "; ".join(parts) + ". Prompt modes that leave the attribute "
                "unspecified reveal the model's default."
            )
    jobs = composition.get("gender", {}).get("by_slice", {}).get("job", {})
    skewed = []
    for job, value in jobs.items():
        counts = (value or {}).get("detected", {}).get("counts", {})
        n = sum(_num(count) or 0 for count in counts.values())
        shares = _shares(counts)
        if n >= MIN_GROUP and shares:
            top, share = max(shares.items(), key=lambda item: item[1])
            skewed.append((share, job, top, int(n)))
    skewed.sort(reverse=True)
    one_sided = [item for item in skewed if item[0] >= 0.85]
    if one_sided:
        notes.append(
            "Jobs where one detected gender makes up at least 85% of faces "
            "(all prompt modes): " + ", ".join(
                f"{esc(job)} ({pct(share, 0)} {esc(top)}, n={n})"
                for share, job, top, n in one_sided[:6]) + "."
        )
    female_majority = [item for item in skewed if item[2] == "female"]
    if female_majority:
        notes.append(
            "Jobs where female is the majority detected gender: " + ", ".join(
                f"{esc(job)} ({pct(share, 0)}, n={n})"
                for share, job, _, n in female_majority) + "."
        )
    groups = (chapter.get("diversity") or {}).get("groups", {}).get("job", {})
    dispersion = {name: _num(stats.get("embedding_dispersion"))
                  for name, stats in groups.items() if isinstance(stats, dict)}
    dispersion = {name: value for name, value in dispersion.items() if value is not None}
    if len(dispersion) >= 3:
        low = min(dispersion, key=dispersion.get)
        high = max(dispersion, key=dispersion.get)
        notes.append(
            f"Least visually varied job: <strong>{esc(low)}</strong> (embedding dispersion "
            f"{dispersion[low]:.3f}); most varied: <strong>{esc(high)}</strong> "
            f"({dispersion[high]:.3f})."
        )
    return notes


# --------------------------------------------------------------------------- 4
def chapter_four(adherence: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    by_attribute = adherence.get("by_attribute", {})
    accuracies = {
        name: _num(item.get("accuracy")) for name, item in by_attribute.items()
        if _num(item.get("accuracy")) is not None
    }
    if accuracies:
        ordered = sorted(accuracies.items(), key=lambda item: -item[1])
        notes.append("Adherence by attribute, best to worst: " + ", ".join(
            f"{esc(name)} {pct(value)}" for name, value in ordered) + ".")
    for category, groups in (adherence.get("by_demographic_category") or {}).items():
        values = {
            group: _num((stats.get("accuracy") or {}).get(category))
            for group, stats in groups.items()
            if (stats.get("n_images") or 0) >= MIN_GROUP
        }
        values = {group: value for group, value in values.items() if value is not None}
        if len(values) >= 2:
            low = min(values, key=values.get)
            high = max(values, key=values.get)
            gap = values[high] - values[low]
            if gap >= 0.1:
                notes.append(
                    f"Requested <strong>{esc(category)}</strong> is followed less often for "
                    f"<strong>{esc(low)}</strong> ({pct(values[low])}) than for "
                    f"{esc(high)} ({pct(values[high])}): a gap of "
                    f"{100 * gap:.0f} percentage points."
                )
    confusions = []
    for attribute, item in by_attribute.items():
        for requested, row in (item.get("counts") or {}).items():
            total = sum(_num(value) or 0 for value in row.values())
            if total < MIN_GROUP:
                continue
            for detected, count in row.items():
                if detected != requested and (_num(count) or 0) / total >= 0.2:
                    confusions.append(((_num(count) or 0) / total, attribute, requested,
                                       detected, int(count), int(total)))
    confusions.sort(reverse=True)
    if confusions:
        notes.append("Most frequent substitutions (requested → detected): " + "; ".join(
            f"{esc(requested)} → {esc(detected)} in {pct(share, 0)} of {esc(attribute)} "
            f"prompts (n={total})"
            for share, attribute, requested, detected, _, total in confusions[:4]) + ".")
    age = adherence.get("age_numeric") or {}
    age_groups = {term: item for term, item in (age.get("groups") or {}).items()
                  if (item.get("n") or 0) >= MIN_GROUP and _num(item.get("mean")) is not None}
    if len(age_groups) >= 2:
        text = ("Mean detected age by requested term: " + ", ".join(
            f"{esc(term)} {item['mean']:.0f}" for term, item in age_groups.items()) + " years")
        baseline = _num((age.get("not_requested") or {}).get("mean"))
        if baseline is not None:
            text += f" (no age requested: {baseline:.0f})"
        merged = [item for item in age.get("separation", [])
                  if _num(item.get("hedges_g")) is not None and item["hedges_g"] < 0.2
                  and item["lower"] in age_groups and item["higher"] in age_groups]
        if merged:
            text += (". " + " and ".join(
                f"<strong>“{esc(item['higher'])}” is not drawn older than "
                f"“{esc(item['lower'])}”</strong> (g = {item['hedges_g']:+.2f})"
                for item in merged)
                + ", so adherence gaps between these terms reflect the age cut-offs more "
                  "than the images")
        notes.append(text + ".")
    fragile = []
    for attribute, item in (adherence.get("probabilities") or {}).items():
        for value, stats in (item.get("by_value") or {}).items():
            hard, soft = _num(stats.get("accuracy")), _num(stats.get("soft_accuracy"))
            if hard is not None and soft is not None and (stats.get("n") or 0) >= MIN_GROUP:
                fragile.append((hard - soft, attribute, value, hard, soft, stats))
    fragile.sort(key=lambda entry: -entry[0])
    if fragile and fragile[0][0] >= 0.1:
        notes.append("Adherence that rests on uncertain labels (top label vs "
                     "probability-weighted): " + "; ".join(
                         f"{esc(attribute)} = {esc(value)} {pct(hard, 0)} vs {pct(soft, 0)}, "
                         f"{stats.get('n_confident', 0)} of {stats.get('n')} labels confident"
                         for gap, attribute, value, hard, soft, stats in fragile[:3]
                         if gap >= 0.1)
                     + ". For these groups the face model is a weak instrument: treat their "
                       "adherence as approximate.")
    return notes


# --------------------------------------------------------------------------- 5
def _strongest_associations(bias: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
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
            value, p_value = _num(stats.get("cramers_v")), _num(stats.get("p_value"))
            if value is None or (matrix.get("total") or 0) < 30:
                continue
            rows.append({"prompt": prompt, "output": output, "v": value,
                         "p": p_value, "n": matrix.get("total", 0), "matrix": matrix})
    rows.sort(key=lambda row: -row["v"])
    return rows[:limit]


def _most_overrepresented(matrix: dict[str, Any]) -> tuple[str, str, float, float, int] | None:
    """Return the cell whose conditional share most exceeds the column average."""
    counts = matrix.get("counts") or {}
    columns = matrix.get("columns") or []
    total = sum(sum(_num(value) or 0 for value in row.values()) for row in counts.values())
    if not total:
        return None
    overall = {column: sum(_num(row.get(column)) or 0 for row in counts.values()) / total
               for column in columns}
    best = None
    for row_name, row in counts.items():
        n = sum(_num(value) or 0 for value in row.values())
        if n < MIN_GROUP:
            continue
        for column in columns:
            share = (_num(row.get(column)) or 0) / n
            gap = share - overall[column]
            if best is None or gap > best[3] - best[4]:
                best = (row_name, column, n, share, overall[column])
    if best is None:
        return None
    row_name, column, n, share, base = best
    return row_name, column, share, base, int(n)


def chapter_five(bias: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    strongest = _strongest_associations(bias)
    if strongest:
        notes.append("Strongest associations between a prompt attribute and a different "
                     "detected feature: " + "; ".join(
                         f"{esc(label(row['prompt']))} → {esc(label(row['output']))} "
                         f"(V={row['v']:.2f}, {cramers_band(row['v'])[0]}, n={row['n']})"
                         for row in strongest[:5]) + ".")
        for row in strongest[:3]:
            cell = _most_overrepresented(row["matrix"])
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
                lift = _num(item.get("lift"))
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
                lower, upper = _num(row.get("lower")), _num(row.get("upper"))
                if lower is None or upper is None or row.get("n", 0) < MIN_GROUP:
                    continue
                if lower > 0 or upper < 0:
                    forest.append((abs(row["difference"]), pair, output, row))
            for model in matrix.get("models", []) or []:
                p_value = _num(model.get("interaction_p_value"))
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


# --------------------------------------------------------------------------- 6
def chapter_six(data: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    by_attribute = data.get("by_attribute", {})
    gaps = []
    for attribute, item in by_attribute.items():
        rates = {condition: _num(stats.get("rate"))
                 for condition, stats in (item.get("adherence") or {}).items()
                 if (stats.get("n") or 0) >= MIN_GROUP}
        rates = {key: value for key, value in rates.items() if value is not None}
        if len(rates) >= 2:
            low = min(rates, key=rates.get)
            high = max(rates, key=rates.get)
            gaps.append((rates[high] - rates[low], attribute, low, high, rates))
    gaps.sort(reverse=True)
    for gap, attribute, low, high, rates in gaps[:3]:
        if gap >= 0.1:
            zero_note = (" A 0% rate usually means the requested label never matches a "
                         "detector label; check the label mapping before reading it as bias."
                         if rates[low] == 0 else "")
            notes.append(
                f"In matched <strong>{esc(attribute)}</strong> pairs, the requested value is "
                f"realised for {esc(high)} {pct(rates[high], 0)} of the time but for "
                f"{esc(low)} only {pct(rates[low], 0)}.{zero_note}"
            )
    distances = {attribute: _num((item.get("embeddings") or {})
                                 .get("paired_cosine_distance", {}).get("mean"))
                 for attribute, item in by_attribute.items()}
    distances = {key: value for key, value in distances.items() if value is not None}
    if distances:
        ordered = sorted(distances.items(), key=lambda item: -item[1])
        notes.append("Mean paired cosine distance (how much the image changes), largest "
                     "first: " + ", ".join(f"{esc(name)} {value:.3f}"
                                           for name, value in ordered) + ".")
    shifts = []
    for attribute, item in by_attribute.items():
        for feature, stats in (item.get("numerical_differences") or {}).items():
            mean = _num(stats.get("mean"))
            if mean is None or (stats.get("n") or 0) < MIN_GROUP:
                continue
            if feature != "detected_age":
                shifts.append((abs(mean), attribute, feature, mean, stats.get("n")))
                continue
            # Years, not a ratio: reported on its own, and only for two-value
            # swaps, where "second − first" has a single meaning.
            conditions = sorted((item.get("adherence") or {}).keys())
            low, high = _num(stats.get("ci_low")), _num(stats.get("ci_high"))
            if len(conditions) == 2 and low is not None and high is not None and low * high > 0:
                notes.append(
                    f"With the rest of the prompt fixed, changing <strong>{esc(attribute)}"
                    f"</strong> from {esc(conditions[0])} to {esc(conditions[1])} changes the "
                    f"detected age by {mean:+.1f} years on average (95% CI {low:+.1f} to "
                    f"{high:+.1f}, n={stats.get('n')} pairs)."
                )
    shifts = [shift for shift in sorted(shifts, reverse=True) if shift[0] > 0.05]
    if shifts:
        notes.append("Largest systematic shifts in non-requested numerical features: " + "; ".join(
            f"changing {esc(attribute)} moves {esc(label(feature).lower())} by {mean:+.3f} "
            f"on average (n={n})" for _, attribute, feature, mean, n in shifts[:3]) + ".")
    return notes


# --------------------------------------------------------------------------- 7
def chapter_seven(chapter: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    groups = (chapter.get("embedding") or {}).get("groups", {})
    scores = {name: _num((item.get("separability") or {}).get("score"))
              for name, item in groups.items()}
    scores = {key: value for key, value in scores.items() if value is not None}
    if scores:
        ordered = sorted(scores.items(), key=lambda item: -item[1])
        notes.append("Silhouette score by grouping, most separated first: " + ", ".join(
            f"{esc(label(name))} {value:.2f}" for name, value in ordered[:5])
            + ". All values below 0.25 indicate heavily overlapping groups."
            if ordered[0][1] < 0.25 else
            "Silhouette score by grouping, most separated first: " + ", ".join(
                f"{esc(label(name))} {value:.2f}" for name, value in ordered[:5]) + ".")
    features = chapter.get("image_features") or {}
    gaps = []
    for feature, item in features.items():
        overall = [value for value in item.get("values", []) if _num(value) is not None]
        if len(overall) < 2:
            continue
        mean_all = sum(overall) / len(overall)
        spread = (sum((value - mean_all) ** 2 for value in overall) / (len(overall) - 1)) ** 0.5
        if not spread:
            continue
        for grouping in ("gender", "race", "age"):
            means = {}
            for name, values in (item.get("by_group", {}).get(grouping) or {}).items():
                clean = [value for value in values if _num(value) is not None]
                if len(clean) >= MIN_GROUP:
                    means[name] = sum(clean) / len(clean)
            if len(means) >= 2:
                low = min(means, key=means.get)
                high = max(means, key=means.get)
                gaps.append(((means[high] - means[low]) / spread, feature, grouping, low, high))
    gaps.sort(reverse=True)
    if gaps and gaps[0][0] >= 0.2:
        notes.append("Largest demographic differences in image features (difference of means "
                     "in standard deviations): " + "; ".join(
                         f"{esc(label(feature).lower())} by prompt {esc(grouping)}: "
                         f"{esc(high)} vs {esc(low)}, {gap:.2f} SD"
                         for gap, feature, grouping, low, high in gaps[:3] if gap >= 0.2) + ".")
    return notes


# --------------------------------------------------------------------------- 8
def chapter_eight(chapter: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    overall = chapter.get("overall_sentiment") or {}
    if _num(overall.get("mean")) is not None:
        notes.append(
            f"Average caption sentiment is {overall['mean']:+.2f} "
            f"({pct(overall.get('positive_share'), 0)} positive, "
            f"{pct(overall.get('negative_share'), 0)} negative captions)."
        )
    gaps = []
    for attribute, profile in ((chapter.get("lexical_profiles") or {}).get("prompt") or {}).items():
        stats = {group: item.get("sentiment", {}).get("all", {})
                 for group, item in (profile.get("groups") or {}).items()}
        stats = {group: value for group, value in stats.items()
                 if _num(value.get("mean")) is not None and (value.get("n") or 0) >= MIN_GROUP}
        if len(stats) < 2:
            continue
        low = min(stats, key=lambda key: stats[key]["mean"])
        high = max(stats, key=lambda key: stats[key]["mean"])
        separated = (_num(stats[low].get("ci_high")) or 0) < (_num(stats[high].get("ci_low")) or 0)
        gaps.append((stats[high]["mean"] - stats[low]["mean"], attribute, low, high,
                     stats, separated))
    gaps.sort(key=lambda entry: -entry[0])
    clear = [gap for gap in gaps if gap[5]]
    if clear:
        notes.append("Prompt attributes with clearly separated sentiment (non-overlapping 95% "
                     "CIs): " + "; ".join(
                         f"{esc(label(attribute))}: {esc(low)} "
                         f"{stats[low]['mean']:+.2f} vs {esc(high)} {stats[high]['mean']:+.2f}"
                         for _, attribute, low, high, stats, _ in clear[:4]) + ".")
    return notes
