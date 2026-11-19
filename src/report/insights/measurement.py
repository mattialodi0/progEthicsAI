"""Observations for chapters 2-4: coverage, representation and adherence."""

from __future__ import annotations

from typing import Any

from ..measurable import measurable_accuracy, unmeasurable_by_attribute
from ..explain import esc, label, number, pct
from .common import MIN_GROUP, category_shares


def chapter_two(coverage: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    conditioned = {key: number(value) or 0 for key, value in
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
        counts = [number(value) or 0 for value in values.values()]
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


def chapter_three(chapter: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    composition = chapter.get("composition", {})
    for dimension in ("gender", "race"):
        slices = composition.get(dimension, {}).get("by_slice", {}).get("prompt_mode", {})
        parts = []
        for mode, value in slices.items():
            shares = category_shares((value or {}).get("detected", {}).get("counts", {}))
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
        n = sum(number(count) or 0 for count in counts.values())
        shares = category_shares(counts)
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
    dispersion = {name: number(stats.get("embedding_dispersion"))
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


def chapter_four(adherence: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    by_attribute = adherence.get("by_attribute", {})
    accuracies = {
        name: number(item.get("accuracy")) for name, item in by_attribute.items()
        if number(item.get("accuracy")) is not None
    }
    unmeasurable = unmeasurable_by_attribute(adherence)
    if accuracies:
        ordered = sorted(accuracies.items(), key=lambda item: -item[1])
        notes.append("Adherence by attribute, best to worst: " + ", ".join(
            f"{esc(name)} {pct(value)}" for name, value in ordered) + ".")
    for attribute, values in unmeasurable.items():
        if not values:
            continue
        measurable = measurable_accuracy(
            (by_attribute.get(attribute) or {}).get("counts") or {}, values)
        notes.append(
            f"The {esc(attribute)} detector cannot output "
            + ", ".join(f"“{esc(value)}”" for value in sorted(values))
            + ": these requests can never match and are shown as n/a"
            + (f". On the values it can output, {esc(attribute)} adherence is "
               f"{pct(measurable)}." if measurable is not None else ".")
        )
    for category, groups in (adherence.get("by_demographic_category") or {}).items():
        values = {
            group: number((stats.get("accuracy") or {}).get(category))
            for group, stats in groups.items()
            if (stats.get("n_images") or 0) >= MIN_GROUP
            and group not in unmeasurable.get(category, set())
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
            total = sum(number(value) or 0 for value in row.values())
            if total < MIN_GROUP or requested in unmeasurable.get(attribute, set()):
                continue
            for detected, count in row.items():
                if detected != requested and (number(count) or 0) / total >= 0.2:
                    confusions.append(((number(count) or 0) / total, attribute, requested,
                                       detected, int(count), int(total)))
    confusions.sort(reverse=True)
    if confusions:
        notes.append("Most frequent substitutions (requested → detected): " + "; ".join(
            f"{esc(requested)} → {esc(detected)} in {pct(share, 0)} of {esc(attribute)} "
            f"prompts (n={total})"
            for share, attribute, requested, detected, _, total in confusions[:4]) + ".")
    age = adherence.get("age_numeric") or {}
    age_groups = {term: item for term, item in (age.get("groups") or {}).items()
                  if (item.get("n") or 0) >= MIN_GROUP and number(item.get("mean")) is not None}
    if len(age_groups) >= 2:
        text = ("Mean detected age by requested term: " + ", ".join(
            f"{esc(term)} {item['mean']:.0f}" for term, item in age_groups.items()) + " years")
        baseline = number((age.get("not_requested") or {}).get("mean"))
        if baseline is not None:
            text += f" (no age requested: {baseline:.0f})"
        merged = [item for item in age.get("separation", [])
                  if number(item.get("hedges_g")) is not None and item["hedges_g"] < 0.2
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
            hard, soft = number(stats.get("accuracy")), number(stats.get("soft_accuracy"))
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
