"""Observations for chapters 6-8: counterfactuals, image features, captions."""

from __future__ import annotations

from typing import Any

from ..explain import esc, label, number, pct
from .common import MIN_GROUP


def chapter_six(data: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    by_attribute = data.get("by_attribute", {})
    gaps = []
    for attribute, item in by_attribute.items():
        rates = {condition: number(stats.get("rate"))
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
    distances = {attribute: number((item.get("embeddings") or {})
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
            mean = number(stats.get("mean"))
            if mean is None or (stats.get("n") or 0) < MIN_GROUP:
                continue
            if feature != "detected_age":
                shifts.append((abs(mean), attribute, feature, mean, stats.get("n")))
                continue
            # Years, not a ratio: reported on its own, and only for two-value
            # swaps, where "second − first" has a single meaning.
            conditions = sorted((item.get("adherence") or {}).keys())
            low, high = number(stats.get("ci_low")), number(stats.get("ci_high"))
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


def chapter_seven(chapter: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    groups = (chapter.get("embedding") or {}).get("groups", {})
    scores = {name: number((item.get("separability") or {}).get("score"))
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
        overall = [value for value in item.get("values", []) if number(value) is not None]
        if len(overall) < 2:
            continue
        mean_all = sum(overall) / len(overall)
        spread = (sum((value - mean_all) ** 2 for value in overall) / (len(overall) - 1)) ** 0.5
        if not spread:
            continue
        for grouping in ("gender", "race", "age"):
            means = {}
            for name, values in (item.get("by_group", {}).get(grouping) or {}).items():
                clean = [value for value in values if number(value) is not None]
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


def chapter_eight(chapter: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    overall = chapter.get("overall_sentiment") or {}
    if number(overall.get("mean")) is not None:
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
                 if number(value.get("mean")) is not None and (value.get("n") or 0) >= MIN_GROUP}
        if len(stats) < 2:
            continue
        low = min(stats, key=lambda key: stats[key]["mean"])
        high = max(stats, key=lambda key: stats[key]["mean"])
        separated = (number(stats[low].get("ci_high")) or 0) < (number(stats[high].get("ci_low")) or 0)
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
