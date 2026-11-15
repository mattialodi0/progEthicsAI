"""Chapter 9 lens read from chapter 8: caption words and caption sentiment."""

from __future__ import annotations

import math
from typing import Any

from ..stats_utils import Z95, finite, normal_p
from .constants import DEFAULT_CONCEPTS, DEMOGRAPHIC, SOCIAL_ATTRIBUTES
from .table import ResultTable, human, pct, proportion_gap


def _concepts(config: dict[str, Any]) -> dict[str, dict[str, set[str]]]:
    lexicon = config.get("synthesis_concept_lexicon") or DEFAULT_CONCEPTS
    return {dimension: {category: {str(word).lower() for word in words}
                        for category, words in categories.items()}
            for dimension, categories in lexicon.items()}


def caption_results(table: ResultTable, report_data: dict[str, Any], config: dict[str, Any]) -> None:
    chapter = report_data.get("chapter_8") or {}
    concepts = _concepts(config)
    associations = chapter.get("term_associations")
    if associations is None:
        table.warnings.append("Chapter 8 has no tested term associations; rebuild chapter 8.")
        associations = {}
    for attribute, groups in associations.items():
        if attribute == "prompt_mode":
            continue
        for group, entry in groups.items():
            terms = entry.get("terms", []) if isinstance(entry, dict) else entry
            tested = entry.get("tested", len(terms)) if isinstance(entry, dict) else len(terms)
            table.hidden_tests += max(0, tested - len(terms))
            group_words = [word for word in str(group).lower().replace("-", " ").split()
                           if len(word) >= 3]
            for term in terms:
                word = str(term.get("term", "")).lower()
                k, n = term.get("group_documents", 0), term.get("group_n", 0)
                rest_k, rest_n = term.get("rest_documents", 0), term.get("rest_n", 0)
                if not n or not rest_n:
                    continue
                share, rest_share = k / n, rest_k / rest_n
                gap = proportion_gap(k, n, rest_k, rest_n)
                mapped = [(dimension, category) for dimension, categories in concepts.items()
                          for category, words in categories.items() if word in words]
                # The caption naming what the prompt asked for is adherence,
                # not an association ("female" prompts described as "woman").
                # Same word or same stem ("africa" → "african", "scientist" →
                # "science") counts as an echo of the prompt.
                echo = any(word == item or (len(item) >= 5 and word[:5] == item[:5])
                           for item in group_words) or any(dimension == attribute
                                                           for dimension, _ in mapped)
                demographic_concept = any(dimension in DEMOGRAPHIC for dimension, _ in mapped)
                social = (attribute in SOCIAL_ATTRIBUTES or demographic_concept) and not echo
                keys = [(attribute, group, "term", word, "+"),
                        (attribute, group, "object", word, "+")]
                keys += [(attribute, group, dimension, category, "+")
                         for dimension, category in mapped]
                log_odds = finite(term.get("log_odds"))
                table.add(
                    lens="caption", attribute=attribute, subgroup=group,
                    reference=f"other {human(attribute)} values", feature="caption term",
                    category=word, metric="share of captions with term",
                    **gap,
                    n=min(n, rest_n), n_total=n + rest_n,
                    p_value=finite(term.get("p_value")),
                    question="stereotyping" if social else "context",
                    stereotype_candidate=social, context=echo or not social,
                    keys=keys,
                    details={"share": share, "rest_share": rest_share,
                             "odds_ratio": math.exp(log_odds) if log_odds is not None else None,
                             "concepts": mapped, "echo": echo},
                    statement=(f"Captions for prompt {human(attribute)} = {group} contain "
                               f"“{word}” in {pct(share)} of {n} images vs {pct(rest_share)} "
                               f"for other {human(attribute)} values."),
                )

    profiles = ((chapter.get("lexical_profiles") or {}).get("prompt")) or {}
    for attribute, profile in profiles.items():
        if attribute in ("prompt_mode", "emotion"):
            continue  # requested emotion *should* change sentiment
        stats = {
            group: (item.get("sentiment") or {}).get("all") or {}
            for group, item in (profile.get("groups") or {}).items()
        }
        stats = {group: value for group, value in stats.items()
                 if finite(value.get("mean")) is not None and (value.get("n") or 0) >= 2}
        if len(stats) < 2:
            continue
        for group, value in stats.items():
            n, mean = value["n"], value["mean"]
            sd = finite(value.get("sd"))
            if sd is None:  # older artifacts: recover the SD from the CI
                sd = ((value.get("ci_high", mean) - value.get("ci_low", mean))
                      / (2 * 1.96) * math.sqrt(n))
            rest = [(item["n"], item["mean"], finite(item.get("sd")) or
                     (item.get("ci_high", item["mean"]) - item.get("ci_low", item["mean"]))
                     / (2 * 1.96) * math.sqrt(item["n"]))
                    for other, item in stats.items() if other != group]
            rest_n = sum(count for count, _, _ in rest)
            if n < table.settings["min_cell"] or rest_n < table.settings["min_cell"]:
                continue
            rest_mean = sum(count * value_mean for count, value_mean, _ in rest) / rest_n
            rest_variance = (sum((count - 1) * spread ** 2 + count * (value_mean - rest_mean) ** 2
                                 for count, value_mean, spread in rest) / max(1, rest_n - 1))
            pooled = math.sqrt(((n - 1) * sd ** 2 + (rest_n - 1) * rest_variance)
                               / max(1, n + rest_n - 2))
            if not pooled:
                continue
            g = (mean - rest_mean) / pooled
            se_g = math.sqrt((n + rest_n) / (n * rest_n) + g * g / (2 * (n + rest_n)))
            se_diff = math.sqrt(sd ** 2 / n + rest_variance / rest_n)
            difference = mean - rest_mean
            if g >= 0:
                direction, sign = "more positive", "+"
            else:
                direction, sign = "more negative", "-"
            social = attribute in SOCIAL_ATTRIBUTES
            table.add(
                lens="caption", attribute=attribute, subgroup=group,
                reference=f"other {human(attribute)} values", feature="caption sentiment",
                category=direction, metric="mean caption sentiment difference",
                effect=difference, effect_unit="value",
                ci=[difference - Z95 * se_diff, difference + Z95 * se_diff],
                family="g", magnitude=g, magnitude_ci=[g - Z95 * se_g, g + Z95 * se_g],
                n=min(n, rest_n), n_total=n + rest_n,
                p_value=normal_p(difference / se_diff) if se_diff else None,
                # Tone towards a social group is stereotyping; tone of a job or
                # place is a property of the scene, not of a group.
                question="stereotyping" if social else "context",
                stereotype_candidate=social,
                keys=[(attribute, group, "sentiment", "", sign)],
                details={"mean": mean, "rest_mean": rest_mean},
                statement=(f"Captions for prompt {human(attribute)} = {group} are {direction} "
                           f"than for other {human(attribute)} values (VADER {mean:+.2f} vs "
                           f"{rest_mean:+.2f})."),
            )
