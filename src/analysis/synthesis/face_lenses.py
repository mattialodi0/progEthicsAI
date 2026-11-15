"""Chapter 9 lenses read from chapters 2-4: default representation, prompt
adherence and face detection."""

from __future__ import annotations

import math
from typing import Any

from ..stats_utils import chi2_sf, cohen_h, one_proportion_p, two_proportion_p, wilson
from .constants import DEFAULT_MIN_CELL, DEMOGRAPHIC, FACE_FEATURES
from .table import ResultTable, add_flag, human, pct, proportion_gap


def representation_results(table: ResultTable, report_data: dict[str, Any]) -> None:
    composition = (report_data.get("chapter_3") or {}).get("composition") or {}
    for dimension in DEMOGRAPHIC:
        item = composition.get(dimension) or {}
        unrequested = item.get("unrequested")
        if not isinstance(unrequested, dict):
            table.warnings.append(f"Chapter 3 has no unrequested composition for {dimension}; "
                                  "rebuild chapter 3.")
            continue
        counts = (unrequested.get("detected") or {}).get("counts") or {}
        # Parity reference: every category the detector produced in this audit.
        categories = sorted(set((item.get("detected") or {}).get("counts") or {}) | set(counts))
        n = sum(counts.get(category, 0) for category in categories)
        if n < 2 or len(categories) < 2:
            continue
        reference = 1 / len(categories)
        statistic = sum((counts.get(category, 0) - n * reference) ** 2 / (n * reference)
                        for category in categories)
        w = math.sqrt(statistic / n)
        top = max(categories, key=lambda category: counts.get(category, 0))
        by_mode = {
            mode: value.get("detected", {}).get("counts", {})
            for mode, value in (unrequested.get("by_slice", {}).get("prompt_mode") or {}).items()
        }
        table.add(
            lens="representation", level="omnibus", attribute="default", feature=dimension,
            scope=f"images whose prompt does not request {dimension}",
            metric="Cohen's w vs parity", effect=w, family="w", magnitude=w, n=n, n_total=n,
            p_value=chi2_sf(statistic, len(categories) - 1), question="representation",
            details={"counts": counts, "reference": reference},
            statement=(f"When {dimension} is not requested, detected {dimension} is far from "
                       f"parity across {len(categories)} detector categories (most common: "
                       f"{top}, {pct(counts.get(top, 0) / n)}).")
            if w >= 0.1 else
            f"When {dimension} is not requested, detected {dimension} is close to parity.",
        )
        for category in categories:
            count = counts.get(category, 0)
            share = count / n
            if len(categories) == 2 and share <= reference:
                continue  # the mirror of the other category
            low, high = wilson(count, n)
            consistent = [
                mode for mode, mode_counts in by_mode.items()
                if sum(mode_counts.values()) >= DEFAULT_MIN_CELL
            ]
            same_direction = all(
                (mode_counts.get(category, 0) / sum(mode_counts.values()) - reference)
                * (share - reference) > 0
                for mode, mode_counts in by_mode.items() if mode in consistent
            )
            row = table.add(
                lens="representation", attribute="default", subgroup="default output",
                feature=dimension, category=category,
                reference=f"parity ({pct(reference)})",
                scope=f"images whose prompt does not request {dimension}",
                metric="share of detected faces", effect=share, effect_unit="share",
                ci=[low, high], family="h", magnitude=cohen_h(share, reference),
                magnitude_ci=[cohen_h(low, reference), cohen_h(high, reference)],
                n=n, n_total=n, p_value=one_proportion_p(count, n, reference),
                question="representation",
                keys=[("default", "default output", dimension, category,
                       "+" if share > reference else "-")],
                details={"by_prompt_mode": {
                    mode: {"share": mode_counts.get(category, 0) / sum(mode_counts.values()),
                           "n": sum(mode_counts.values())}
                    for mode, mode_counts in by_mode.items() if mode in consistent
                }, "consistent_across_modes": same_direction and len(consistent) > 1},
                statement=(f"When {dimension} is not requested, {pct(share)} of detected "
                           f"faces are {category} (parity reference {pct(reference)})."),
            )
            if len(consistent) > 1 and not same_direction:
                add_flag(row, "mode_inconsistent",
                         "The direction of this gap differs between prompt modes.")


def adherence_results(table: ResultTable, report_data: dict[str, Any],
                      state: dict[str, Any]) -> None:
    chapter = report_data.get("chapter_4") or {}
    detector_labels: dict[str, set[str]] = {}
    for attribute, item in (chapter.get("by_attribute") or {}).items():
        counts = item.get("counts") or {}
        detected = {column for row in counts.values() for column, value in row.items() if value}
        detector_labels[attribute] = detected
        rates = {}
        for requested, row in counts.items():
            total = sum(row.values())
            if total:
                rates[requested] = (row.get(requested, 0), total)
        state["adherence"][attribute] = {key: k / n for key, (k, n) in rates.items()}
        if len(rates) < 2:
            continue
        # Omnibus: does adherence depend on the requested value?
        table_counts = [[k, n - k] for k, n in rates.values()]
        total_n = sum(n for _, n in rates.values())
        correct = sum(k for k, _ in rates.values())
        statistic = 0.0
        for k, n in rates.values():
            for observed, column_total in ((k, correct), (n - k, total_n - correct)):
                expected = n * column_total / total_n
                if expected:
                    statistic += (observed - expected) ** 2 / expected
        best = max(rates, key=lambda key: rates[key][0] / rates[key][1])
        worst = min(rates, key=lambda key: rates[key][0] / rates[key][1])
        w = math.sqrt(statistic / total_n) if total_n else 0.0
        missing = sorted(requested for requested in rates if requested not in detected)
        state["label_mismatch"].extend((attribute, requested) for requested in missing)
        omnibus = table.add(
            lens="adherence", level="omnibus", attribute=attribute, feature=attribute,
            metric="Cohen's w (adherence × requested value)", effect=w, family="w",
            magnitude=w, n=min(n for _, n in rates.values()), n_total=total_n,
            p_value=chi2_sf(statistic, len(table_counts) - 1), question="adherence",
            statement=(f"Adherence to requested {attribute} depends on the requested value: "
                       f"{best} {pct(rates[best][0] / rates[best][1])} vs {worst} "
                       f"{pct(rates[worst][0] / rates[worst][1])}."),
        )
        if missing:
            which = "these values" if len(missing) > 1 else "this value"
            add_flag(omnibus, "label_mismatch",
                     "The detector never outputs " + ", ".join(f"“{value}”" for value in missing)
                     + f": adherence for {which} is 0% by construction, so the gap reflects "
                     "the label vocabulary.")
        for requested, (k, n) in rates.items():
            rest_k = correct - k
            rest_n = total_n - n
            if n < DEFAULT_MIN_CELL or rest_n < DEFAULT_MIN_CELL or k / n >= rest_k / rest_n:
                continue  # report the under-served side; the other is its mirror
            gap = proportion_gap(k, n, rest_k, rest_n)
            row = table.add(
                lens="adherence", attribute=attribute, subgroup=requested,
                reference=f"other requested {attribute} values", feature=attribute,
                category="as requested", metric="adherence rate gap",
                **gap,
                n=min(n, rest_n), n_total=total_n,
                p_value=two_proportion_p(k, n, rest_k, rest_n), question="adherence",
                keys=[(attribute, requested, "adherence", attribute,
                       "+" if gap["effect"] > 0 else "-")],
                details={"rate": k / n, "rest_rate": rest_k / rest_n, "count": k},
                statement=(f"Requested {attribute} = {requested} is rendered as requested in "
                           f"{pct(k / n)} of {n} images vs {pct(rest_k / rest_n)} for the "
                           f"other {attribute} values."),
            )
            if requested not in detected:
                add_flag(row, "label_mismatch",
                         f"The detector never outputs “{requested}”: the gap reflects the label "
                         "vocabulary, not the model.")
    state["detector_labels"] = detector_labels

    # Adherence of attribute X conditioned on a demographic category Y, and on
    # the prompt mode: is the request followed equally well for every group?
    contexts = dict(chapter.get("by_demographic_category") or {})
    contexts = {**{category: groups for category, groups in contexts.items()},
                "prompt_mode": chapter.get("by_prompt_mode") or {}}
    for context, groups in contexts.items():
        for attribute in FACE_FEATURES:
            if attribute == context:
                continue
            values = {
                group: (stats.get("counts", {}).get(attribute, {}).get("correct"),
                        stats.get("counts", {}).get(attribute, {}).get("total"))
                for group, stats in groups.items()
            }
            values = {group: (k, n) for group, (k, n) in values.items() if n}
            if len(values) < 2:
                continue
            total_k = sum(k for k, _ in values.values())
            total_n = sum(n for _, n in values.values())
            for group, (k, n) in values.items():
                rest_k, rest_n = total_k - k, total_n - n
                if (n < DEFAULT_MIN_CELL or rest_n < DEFAULT_MIN_CELL
                        or k / n >= rest_k / rest_n):
                    continue
                gap = proportion_gap(k, n, rest_k, rest_n)
                scope = ("prompt mode" if context == "prompt_mode" else
                         f"prompts that request both {context} and {attribute}")
                table.add(
                    lens="adherence", attribute=context, subgroup=group,
                    reference=f"other {human(context)} values", feature=attribute,
                    category="as requested", scope=scope,
                    metric=f"{attribute} adherence gap", **gap,
                    n=min(n, rest_n), n_total=total_n,
                    p_value=two_proportion_p(k, n, rest_k, rest_n), question="adherence",
                    keys=[(context, group, "adherence", attribute,
                           "+" if gap["effect"] > 0 else "-")],
                    details={"rate": k / n, "rest_rate": rest_k / rest_n},
                    statement=(f"Requested {attribute} is followed in {pct(k / n)} of images "
                               f"when {human(context)} = {group} vs {pct(rest_k / rest_n)} "
                               f"for other {human(context)} values."),
                )


def detection_results(table: ResultTable, report_data: dict[str, Any], state: dict[str, Any]) -> None:
    groups = (report_data.get("chapter_2") or {}).get("detection_by_group")
    if not isinstance(groups, dict):
        table.warnings.append("Chapter 2 has no detection rate by subgroup; rebuild chapter 2.")
        return
    for attribute, values in groups.items():
        total_k = sum(item.get("face_detected", 0) for item in values.values())
        total_n = sum(item.get("total", 0) for item in values.values())
        for value, item in values.items():
            k, n = item.get("face_detected", 0), item.get("total", 0)
            rest_k, rest_n = total_k - k, total_n - n
            if not n or rest_n < DEFAULT_MIN_CELL or k / n >= rest_k / rest_n:
                continue
            gap = proportion_gap(k, n, rest_k, rest_n)
            row = table.add(
                lens="detection", attribute=attribute, subgroup=value,
                reference=f"other {human(attribute)} values", feature="face",
                category="detected", metric="face-detection rate gap", **gap,
                n=min(n, rest_n), n_total=total_n,
                p_value=two_proportion_p(k, n, rest_k, rest_n), question="measurement",
                details={"rate": k / n, "rest_rate": rest_k / rest_n, "images": n},
                statement=(f"A face is detected in {pct(k / n)} of images for "
                           + ("" if attribute == "prompt_mode" else "prompt ")
                           + f"{human(attribute)} = {value} vs {pct(rest_k / rest_n)} for the "
                           f"other values ({n - k} of {n} images excluded from face-based "
                           "results)."),
            )
            state["detection_rows"].append(row)
