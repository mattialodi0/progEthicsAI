"""Chapter 9 data: cross-analysis synthesis of chapters 2–8.

Unlike the other chapter builders this one does not read image records. It
reads the ``report_data`` payloads of the other chapters and turns every
effect they measured into one row of a common **results table**:

    lens, attribute, subgroup, reference, feature, category, metric,
    effect (native units) + 95% CI, standardized magnitude + CI,
    strength score, n, p-value, BH q-value, flags, tier, statement

On top of that table it computes

* a Benjamini–Hochberg false-discovery-rate correction over *all* tests of
  the audit (one family, so a result is never "significant" only because it
  was looked at in isolation);
* **evidence clusters**: results from different evidence sources (face
  detector, object detector, captions, pixel statistics, matched
  counterfactual pairs) that make the same claim about the same subgroup;
* an **evidence matrix** (prompt attribute × lens);
* data-driven answers to the five fairness questions of the reading guide;
* measurement-reliability caveats (face-detection gaps, label mismatches,
  poorly followed requests) that are attached to the affected results.

Everything is derived from the artifact, so a different model or dataset
produces different findings. The function is cheap and deterministic, so the
analyzer rebuilds it on every run.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Iterable

from .stats_utils import (
    Z95,
    chi2_sf,
    cohen_h,
    finite,
    newcombe_difference,
    normal_p,
    one_proportion_p,
    rounded,
    two_proportion_p,
    wilson,
)

SCHEMA = 1

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_ALPHA = 0.05
# Results whose smallest compared group is below this are flagged "small n".
DEFAULT_SMALL_N = 30
# Cells/groups smaller than this are not turned into specific results.
DEFAULT_MIN_CELL = 5
# A subgroup whose face-detection rate is this much below the rest (and the
# gap is supported) gets a measurement caveat on every face-based result.
DETECTION_GAP = 0.10
# Requested values followed less often than this weaken results about them.
LOW_ADHERENCE = 0.5

DEMOGRAPHIC = ("gender", "race", "age")
FACE_FEATURES = ("gender", "race", "age", "emotion")
# Numerical features (chapters 6 and 7) that estimate a demographic.
DEMOGRAPHIC_FEATURES = ("detected_age",)
# Prompt attributes that name a social group or a social judgement. An
# association that links one of them to non-requested content is a
# *candidate* stereotype (a pattern for a human to review, not a verdict).
SOCIAL_ATTRIBUTES = ("gender", "race", "age", "continent", "status",
                     "toxic_trait", "positive_trait")
ATTRIBUTE_ORDER = ("gender", "race", "age", "emotion", "job", "workplace", "continent",
                   "status", "toxic_trait", "positive_trait", "prompt_mode")

LENSES: dict[str, dict[str, str]] = {
    "representation": {"label": "Default representation", "source": "Face detector",
                       "link": "#chapter-3-distributions", "chapter": "3.1"},
    "adherence": {"label": "Prompt adherence", "source": "Face detector",
                  "link": "#chapter-4", "chapter": "4"},
    "association": {"label": "Demographic association", "source": "Face detector",
                    "link": "#chapter-5-single-variable", "chapter": "5.1"},
    "intersectional": {"label": "Intersectional deviation", "source": "Face detector",
                       "link": "#chapter-5-intersectional", "chapter": "5.4"},
    "objects": {"label": "Object association", "source": "Object detector",
                "link": "#chapter-5-stereotypical", "chapter": "5.3"},
    "counterfactual": {"label": "Matched counterfactual", "source": "Matched pairs",
                       "link": "#chapter-6", "chapter": "6"},
    "visual": {"label": "Image features", "source": "Pixel statistics",
               "link": "#chapter-7-features", "chapter": "7.1"},
    "caption": {"label": "Caption language", "source": "Caption model",
                "link": "#chapter-8", "chapter": "8"},
    "embedding": {"label": "Embedding separability", "source": "CLIP embedding",
                  "link": "#chapter-7-embeddings", "chapter": "7.2"},
    "diversity": {"label": "Output diversity", "source": "CLIP embedding",
                  "link": "#chapter-3-diversity", "chapter": "3.2"},
    "detection": {"label": "Face detection", "source": "Face detector",
                  "link": "#chapter-2-detections", "chapter": "2.4"},
}

# Conventional small / medium / large thresholds per effect-size family.
FAMILIES: dict[str, dict[str, Any]] = {
    "h": {"label": "Cohen's h", "thresholds": (0.2, 0.5, 0.8)},
    "g": {"label": "Hedges' g", "thresholds": (0.2, 0.5, 0.8)},
    "dz": {"label": "Cohen's d_z", "thresholds": (0.2, 0.5, 0.8)},
    "w": {"label": "Cohen's w", "thresholds": (0.1, 0.3, 0.5)},
    "epsilon2": {"label": "ε² (Kruskal–Wallis)", "thresholds": (0.01, 0.06, 0.14)},
    "silhouette": {"label": "Silhouette", "thresholds": (0.1, 0.25, 0.5)},
}
BANDS = ("negligible", "small", "medium", "large")
# Above the "large" threshold the score keeps growing linearly (so very large
# effects can still be ranked against each other) up to this cap.
MAX_STRENGTH = 6.0

# Caption words that express a value of a detected dimension. Used to link
# caption evidence to face-detector evidence ("nurse" captions saying
# "woman" corroborate "nurse" faces detected as female). Override with
# ``synthesis_concept_lexicon`` in the config.
DEFAULT_CONCEPTS: dict[str, dict[str, tuple[str, ...]]] = {
    "gender": {
        "female": ("woman", "women", "female", "girl", "lady", "dress", "skirt", "lipstick"),
        "male": ("man", "men", "male", "boy", "guy", "gentleman", "beard", "mustache",
                 "moustache"),
    },
    "age": {
        "old": ("elderly", "old", "older", "aged", "senior", "wrinkle", "wrinkled"),
        "young": ("young", "youthful", "teenage", "teenager", "child", "kid"),
    },
    "emotion": {
        "happy": ("smile", "smiling", "happy", "cheerful", "joyful", "grin", "laugh",
                  "laughing", "joy"),
        "sad": ("sad", "somber", "sombre", "sorrow", "melancholy", "tear", "cry", "crying",
                "gloomy"),
        "angry": ("angry", "anger", "furious", "frown", "scowl", "rage"),
        "neutral": ("serious", "neutral", "stern", "expressionless"),
        "fear": ("afraid", "fear", "scared", "anxious", "worried", "nervous"),
    },
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _pct(value: float | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def _human(value: Any) -> str:
    return str(value).replace("__", " × ").replace("_", " ")


def strength_score(magnitude: float | None, family: str | None) -> float | None:
    """Map |effect| onto 0–6: 1 = small, 2 = medium, 3 = large threshold."""
    if magnitude is None or family not in FAMILIES:
        return None
    small, medium, large = FAMILIES[family]["thresholds"]
    value = abs(magnitude)
    if value < small:
        return value / small
    if value < medium:
        return 1 + (value - small) / (medium - small)
    if value < large:
        return 2 + (value - medium) / (large - medium)
    return min(MAX_STRENGTH, 3 + (value - large) / (large - medium))


def rank_key(strength_low: float | None, evidence: int, n: int | None,
             strength: float | None) -> tuple:
    """Sort key: conservative strength first, but every effect that is large
    even at its lower CI bound counts as equally large, and is then ordered by
    evidence level and sample size (better-replicated first)."""
    return (-min(strength_low or 0.0, 3.0), evidence, -(n or 0), -(strength or 0.0))


def band(score: float | None) -> str | None:
    if score is None:
        return None
    return BANDS[min(3, int(score))]


def _p_from_ci(estimate: float, low: float, high: float) -> float | None:
    """Two-sided p-value of a normal-approximation 95% interval."""
    se = (high - low) / (2 * Z95)
    return normal_p(estimate / se) if se > 0 else None


def _h_interval(reference: float, low: float | None, high: float | None) -> list[float | None]:
    """Cohen's h for the two ends of a CI on (p − reference)."""
    if low is None or high is None:
        return [None, None]
    return [cohen_h(min(1.0, max(0.0, reference + low)), reference),
            cohen_h(min(1.0, max(0.0, reference + high)), reference)]


class _Table:
    """Accumulates result rows and the extra (unstored) tests of each family."""

    def __init__(self, settings: dict[str, Any]):
        self.rows: list[dict[str, Any]] = []
        self.hidden_tests = 0
        self.settings = settings
        self.warnings: list[str] = []

    def add(self, **fields: Any) -> dict[str, Any]:
        lens = fields["lens"]
        row = {
            "id": f"r{len(self.rows) + 1}",
            "lens": lens,
            "source": LENSES[lens]["source"],
            "link": fields.pop("link", LENSES[lens]["link"]),
            "level": "specific",
            "attribute": None,
            "subgroup": None,
            "reference": None,
            "feature": None,
            "category": None,
            "scope": None,
            "metric": None,
            "effect": None,
            "effect_unit": None,
            "ci": [None, None],
            "family": None,
            "magnitude": None,
            "magnitude_ci": [None, None],
            "n": None,
            "n_total": None,
            "p_value": None,
            "context": False,
            "stereotype_candidate": False,
            "question": None,
            "keys": [],
            "flags": [],
            "details": {},
            "statement": "",
        }
        row.update(fields)
        row["strength"] = strength_score(row["magnitude"], row["family"])
        row["strength_ci"] = [
            strength_score(value, row["family"]) if value is not None else None
            for value in row["magnitude_ci"]
        ]
        if (row["magnitude_ci"][0] is not None and row["magnitude_ci"][1] is not None
                and row["magnitude_ci"][0] * row["magnitude_ci"][1] < 0):
            # The interval crosses zero: the smallest compatible strength is 0.
            row["strength_ci"] = [0.0, max(row["strength_ci"])]
        elif all(value is not None for value in row["strength_ci"]):
            row["strength_ci"] = sorted(row["strength_ci"])
        row["band"] = band(row["strength"])
        # Conservative strength: the smallest effect compatible with the data
        # (lower end of the interval). Used for ranking, so a large effect
        # seen in 8 images does not outrank a slightly smaller one in 200.
        row["strength_low"] = (row["strength_ci"][0] if row["strength_ci"][0] is not None
                               else row["strength"])
        self.rows.append(row)
        return row


def _flag(row: dict[str, Any], code: str, text: str) -> None:
    if all(flag["code"] != code for flag in row["flags"]):
        row["flags"].append({"code": code, "text": text})


# --------------------------------------------------------------------------
# Extractors: one per lens
# --------------------------------------------------------------------------

def _representation(table: _Table, report_data: dict[str, Any]) -> None:
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
                       f"{top}, {_pct(counts.get(top, 0) / n)}).")
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
                reference=f"parity ({_pct(reference)})",
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
                statement=(f"When {dimension} is not requested, {_pct(share)} of detected "
                           f"faces are {category} (parity reference {_pct(reference)})."),
            )
            if len(consistent) > 1 and not same_direction:
                _flag(row, "mode_inconsistent",
                      "The direction of this gap differs between prompt modes.")


def _adherence(table: _Table, report_data: dict[str, Any],
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
                       f"{best} {_pct(rates[best][0] / rates[best][1])} vs {worst} "
                       f"{_pct(rates[worst][0] / rates[worst][1])}."),
        )
        if missing:
            which = "these values" if len(missing) > 1 else "this value"
            _flag(omnibus, "label_mismatch",
                  "The detector never outputs " + ", ".join(f"“{value}”" for value in missing)
                  + f": adherence for {which} is 0% by construction, so the gap reflects "
                  "the label vocabulary.")
        for requested, (k, n) in rates.items():
            rest_k = correct - k
            rest_n = total_n - n
            if n < DEFAULT_MIN_CELL or rest_n < DEFAULT_MIN_CELL or k / n >= rest_k / rest_n:
                continue  # report the under-served side; the other is its mirror
            difference, low, high = newcombe_difference(k, n, rest_k, rest_n)
            row = table.add(
                lens="adherence", attribute=attribute, subgroup=requested,
                reference=f"other requested {attribute} values", feature=attribute,
                category="as requested", metric="adherence rate gap",
                effect=difference, effect_unit="pp", ci=[low, high], family="h",
                magnitude=cohen_h(k / n, rest_k / rest_n),
                magnitude_ci=_h_interval(rest_k / rest_n, low, high),
                n=min(n, rest_n), n_total=total_n,
                p_value=two_proportion_p(k, n, rest_k, rest_n), question="adherence",
                keys=[(attribute, requested, "adherence", attribute,
                       "+" if difference > 0 else "-")],
                details={"rate": k / n, "rest_rate": rest_k / rest_n, "count": k},
                statement=(f"Requested {attribute} = {requested} is rendered as requested in "
                           f"{_pct(k / n)} of {n} images vs {_pct(rest_k / rest_n)} for the "
                           f"other {attribute} values."),
            )
            if requested not in detected:
                _flag(row, "label_mismatch",
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
                difference, low, high = newcombe_difference(k, n, rest_k, rest_n)
                scope = ("prompt mode" if context == "prompt_mode" else
                         f"prompts that request both {context} and {attribute}")
                table.add(
                    lens="adherence", attribute=context, subgroup=group,
                    reference=f"other {_human(context)} values", feature=attribute,
                    category="as requested", scope=scope,
                    metric=f"{attribute} adherence gap", effect=difference, effect_unit="pp",
                    ci=[low, high], family="h", magnitude=cohen_h(k / n, rest_k / rest_n),
                    magnitude_ci=_h_interval(rest_k / rest_n, low, high),
                    n=min(n, rest_n), n_total=total_n,
                    p_value=two_proportion_p(k, n, rest_k, rest_n), question="adherence",
                    keys=[(context, group, "adherence", attribute,
                           "+" if difference > 0 else "-")],
                    details={"rate": k / n, "rest_rate": rest_k / rest_n},
                    statement=(f"Requested {attribute} is followed in {_pct(k / n)} of images "
                               f"when {_human(context)} = {group} vs {_pct(rest_k / rest_n)} "
                               f"for other {_human(context)} values."),
                )


def _detection(table: _Table, report_data: dict[str, Any], state: dict[str, Any]) -> None:
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
            difference, low, high = newcombe_difference(k, n, rest_k, rest_n)
            row = table.add(
                lens="detection", attribute=attribute, subgroup=value,
                reference=f"other {_human(attribute)} values", feature="face",
                category="detected", metric="face-detection rate gap", effect=difference,
                effect_unit="pp", ci=[low, high], family="h",
                magnitude=cohen_h(k / n, rest_k / rest_n),
                magnitude_ci=_h_interval(rest_k / rest_n, low, high),
                n=min(n, rest_n), n_total=total_n,
                p_value=two_proportion_p(k, n, rest_k, rest_n), question="measurement",
                details={"rate": k / n, "rest_rate": rest_k / rest_n, "images": n},
                statement=(f"A face is detected in {_pct(k / n)} of images for "
                           + ("" if attribute == "prompt_mode" else "prompt ")
                           + f"{_human(attribute)} = {value} vs {_pct(rest_k / rest_n)} for the "
                           f"other values ({n - k} of {n} images excluded from face-based "
                           "results)."),
            )
            state["detection_rows"].append(row)


def _association(table: _Table, report_data: dict[str, Any]) -> None:
    chapter = report_data.get("chapter_5") or {}
    source = chapter.get("associations_unrequested")
    fallback = not isinstance(source, dict)
    if fallback:
        table.warnings.append("Chapter 5 has no unrequested-only associations; using all "
                              "images (rebuild chapter 5).")
        source = chapter.get("associations") or {}
    min_cell = table.settings["min_cell"]
    for attribute, outputs in source.items():
        for feature, matrix in (outputs or {}).items():
            if attribute == feature or not isinstance(matrix, dict):
                continue
            counts = matrix.get("counts") or {}
            columns = matrix.get("columns") or []
            total = matrix.get("total") or 0
            stats = matrix.get("statistics") or {}
            if total < 2 or len(counts) < 2 or len(columns) < 2:
                continue
            scope = (f"images whose prompt does not request {feature}" if not fallback
                     else "all images")
            v = finite(stats.get("cramers_v")) or 0.0
            factor = math.sqrt(max(1, min(len(counts), len(columns)) - 1))
            interval = stats.get("cramers_v_ci") or [None, None]
            p_value = finite(stats.get("p_value_exact"))
            if p_value is None:
                p_value = finite(stats.get("p_value"))
            table.add(
                lens="association", level="omnibus", attribute=attribute, feature=feature,
                scope=scope, metric="Cramér's V", effect=v,
                ci=[finite(interval[0]), finite(interval[1])], family="w",
                magnitude=v * factor,
                magnitude_ci=[(finite(interval[0]) or 0) * factor if interval[0] is not None
                              else None,
                              (finite(interval[1]) or 0) * factor if interval[1] is not None
                              else None],
                n=min((sum(row.values()) for row in counts.values()), default=0),
                n_total=total, p_value=p_value, question="stereotyping",
                stereotype_candidate=attribute in SOCIAL_ATTRIBUTES or feature in DEMOGRAPHIC,
                details={"cramers_v_corrected": stats.get("cramers_v_corrected")},
                statement=(f"Prompt {_human(attribute)} is associated with detected {feature} "
                           f"(Cramér's V = {v:.2f}; {scope})."),
            )
            column_totals = {column: sum(row.get(column, 0) for row in counts.values())
                             for column in columns}
            for group, row_counts in counts.items():
                n = sum(row_counts.values())
                rest_n = total - n
                if n < min_cell or rest_n < min_cell:
                    continue
                for category in columns:
                    k = row_counts.get(category, 0)
                    rest_k = column_totals[category] - k
                    share, rest_share = k / n, rest_k / rest_n
                    if share <= rest_share:
                        continue  # under-representation is the mirror of another cell
                    difference, low, high = newcombe_difference(k, n, rest_k, rest_n)
                    table.add(
                        lens="association", attribute=attribute, subgroup=group,
                        reference=f"other {_human(attribute)} values", feature=feature,
                        category=category, scope=scope, metric=f"share detected as {category}",
                        effect=difference, effect_unit="pp", ci=[low, high], family="h",
                        magnitude=cohen_h(share, rest_share),
                        magnitude_ci=_h_interval(rest_share, low, high),
                        n=min(n, rest_n), n_total=total,
                        p_value=two_proportion_p(k, n, rest_k, rest_n),
                        question="stereotyping",
                        stereotype_candidate=(attribute in SOCIAL_ATTRIBUTES
                                              or feature in DEMOGRAPHIC),
                        keys=[(attribute, group, feature, category, "+")],
                        details={"share": share, "rest_share": rest_share, "count": k},
                        statement=(f"Prompt {_human(attribute)} = {group}: detected {feature} "
                                   f"is {category} in {_pct(share)} of {n} faces vs "
                                   f"{_pct(rest_share)} for other {_human(attribute)} values."),
                    )


def _intersectional(table: _Table, report_data: dict[str, Any]) -> None:
    data = (report_data.get("chapter_5") or {}).get("intersectional") or {}
    min_cell = table.settings["min_cell"]
    for pair, item in data.items():
        first = item.get("first_attribute")
        second = item.get("second_attribute")
        for feature, output in (item.get("outputs") or {}).items():
            if feature in (first, second):
                # The chapter's expectation averages the two single-attribute
                # groups; when one of them *requests* the detected feature that
                # average is not a meaningful baseline (female × asian → race
                # is ~100% asian by request). Joint-group adherence is covered
                # by the adherence-by-category results instead.
                continue
            question = "stereotyping"
            joint = {row.get("group"): row for row in output.get("rows", [])}
            for forest in output.get("forest", []):
                difference = finite(forest.get("difference"))
                low, high = finite(forest.get("lower")), finite(forest.get("upper"))
                n = forest.get("n") or 0
                if difference is None or low is None or high is None or n < min_cell:
                    continue
                if difference <= 0:
                    continue  # mirrored by the over-represented category
                group = forest.get("group")
                category = forest.get("category")
                observed = ((joint.get(group) or {}).get("values", {}).get(category, {})
                            .get("percentage"))
                observed = (finite(observed) or 0) / 100
                expected = observed - difference / 100
                table.add(
                    lens="intersectional", attribute=pair, subgroup=group,
                    reference=f"average of the {first} and {second} groups alone",
                    feature=feature, category=category,
                    metric="joint − expected share", effect=difference / 100,
                    effect_unit="pp", ci=[low / 100, high / 100], family="h",
                    magnitude=cohen_h(observed, expected),
                    magnitude_ci=_h_interval(expected, low / 100, high / 100),
                    n=n, n_total=output.get("total"),
                    p_value=_p_from_ci(difference, low, high), question=question,
                    stereotype_candidate=question == "stereotyping",
                    keys=[(pair, group, feature, category, "+")],
                    details={"observed": observed, "expected": expected,
                             "attributes": [first, second]},
                    statement=(f"{group}: detected {feature} = {category} in "
                               f"{_pct(observed)} of {n} faces, {abs(difference):.0f} pp more "
                               f"than expected from {first} and {second} separately."),
                )
            for model in output.get("models", []) or []:
                p_value = finite(model.get("interaction_p_value"))
                terms = model.get("interaction_terms") or []
                largest = max((abs(finite(term.get("effect_percentage_points")) or 0)
                               for term in terms), default=None)
                if p_value is None:
                    continue
                table.add(
                    lens="intersectional", level="omnibus", attribute=pair, feature=feature,
                    category=model.get("output_category"), metric="interaction test",
                    effect=(largest / 100) if largest is not None else None,
                    effect_unit="pp", n=model.get("n"), n_total=model.get("n"),
                    p_value=p_value, question=question,
                    details={"attributes": [first, second], "df": model.get("interaction_df")},
                    statement=(f"The {first} × {second} interaction changes the share of "
                               f"detected {feature} = {model.get('output_category')} "
                               f"(largest interaction term {largest or 0:.0f} pp)."),
                )


def _objects(table: _Table, report_data: dict[str, Any]) -> None:
    data = (report_data.get("chapter_5") or {}).get("stereotypical_associations") or {}
    for attribute, item in data.items():
        for row in item.get("rows", []):
            group = row.get("group")
            n = row.get("observations") or 0
            listed = row.get("associated_objects")
            if listed is None:
                table.warnings.append("Chapter 5.3 has no tested object associations; "
                                      "rebuild chapter 5.")
                return
            tested = row.get("objects_tested", len(listed))
            table.hidden_tests += max(0, tested - len(listed))
            for entry in listed:
                k = entry.get("count", 0)
                rest_k = entry.get("rest_count")
                rest_n = entry.get("rest_observations")
                if rest_k is None or not rest_n or not n:
                    continue
                share, rest_share = k / n, rest_k / rest_n
                difference, low, high = newcombe_difference(k, n, rest_k, rest_n)
                social = attribute in SOCIAL_ATTRIBUTES
                lift = finite(entry.get("lift"))
                table.add(
                    lens="objects", attribute=attribute, subgroup=group,
                    reference=f"other {_human(attribute)} values", feature="object",
                    category=entry.get("object"), metric="share of images with object",
                    effect=difference, effect_unit="pp", ci=[low, high], family="h",
                    magnitude=cohen_h(share, rest_share),
                    magnitude_ci=_h_interval(rest_share, low, high),
                    n=min(n, rest_n), n_total=n + rest_n,
                    p_value=finite(entry.get("p_value")),
                    question="stereotyping" if social else "context",
                    stereotype_candidate=social,
                    # Job and workplace objects are mostly what the prompt asks
                    # for (a chef with a cutting board): context, not bias.
                    context=not social,
                    keys=[(attribute, group, "object", entry.get("object"), "+")],
                    details={"share": share, "rest_share": rest_share, "lift": lift,
                             "count": k},
                    statement=(f"Prompt {_human(attribute)} = {group}: “{entry.get('object')}” "
                               f"appears in {_pct(share)} of {n} images vs "
                               f"{_pct(rest_share, 1)} for other {_human(attribute)} values"
                               + (f" (lift {lift:.1f})." if lift else ".")),
                )


def _counterfactual(table: _Table, report_data: dict[str, Any]) -> None:
    chapter = report_data.get("chapter_6") or {}
    min_cell = table.settings["min_cell"]
    for attribute, item in (chapter.get("by_attribute") or {}).items():
        # Changing a social-group word is the counterfactual-consistency
        # question; changing the job or workplace and seeing the detected
        # demographics move is occupational / contextual stereotyping.
        social_change = attribute in SOCIAL_ATTRIBUTES
        effects = item.get("side_effects")
        if effects is None:
            table.warnings.append("Chapter 6 has no side-effect tests; rebuild chapter 6.")
            effects = []
        for effect in effects:
            difference = finite(effect.get("difference"))
            n_first, n_second = effect.get("n_first", 0), effect.get("n_second", 0)
            if difference is None or difference <= 0 or min(n_first, n_second) < min_cell:
                continue
            first, second = effect.get("first"), effect.get("second")
            feature, category = effect.get("feature"), effect.get("category")
            first_share = finite(effect.get("first_share")) or 0.0
            second_share = finite(effect.get("second_share")) or 0.0
            low, high = finite(effect.get("ci_low")), finite(effect.get("ci_high"))
            social = attribute in SOCIAL_ATTRIBUTES or feature in DEMOGRAPHIC
            table.add(
                lens="counterfactual", attribute=attribute, subgroup=second,
                reference=f"matched {first} prompts", feature=feature, category=category,
                scope="prompt pairs identical except for this attribute",
                metric=f"share detected as {category}", effect=difference, effect_unit="pp",
                ci=[low, high], family="h", magnitude=cohen_h(second_share, first_share),
                magnitude_ci=_h_interval(first_share, low, high),
                n=min(n_first, n_second), n_total=n_first + n_second,
                p_value=finite(effect.get("p_value")),
                question=("counterfactual" if social_change else
                          "stereotyping" if feature in FACE_FEATURES else "quality"),
                stereotype_candidate=social,
                keys=[(attribute, second, feature, category, "+")],
                details={"first": first, "second": second, "first_share": first_share,
                         "second_share": second_share},
                statement=(f"Same prompt, only {_human(attribute)} changed {first} → {second}: "
                           f"detected {feature} = {category} goes from {_pct(first_share)} to "
                           f"{_pct(second_share)} ({n_first} vs {n_second} images)."),
            )
        conditions = sorted((item.get("adherence") or {}).keys())
        if len(conditions) == 2:
            # Paired numerical differences are (second − first) of the
            # alphabetically ordered values; only meaningful for 2 values.
            first, second = conditions
            for feature, stats in (item.get("numerical_differences") or {}).items():
                mean = finite(stats.get("mean"))
                dz = finite(stats.get("dz"))
                if mean is None or dz is None or (stats.get("n") or 0) < min_cell:
                    continue
                higher = second if mean > 0 else first
                low, high = finite(stats.get("ci_low")), finite(stats.get("ci_high"))
                n = stats.get("n")
                sd = finite(stats.get("sd")) or 0.0
                demographic = feature in DEMOGRAPHIC_FEATURES
                table.add(
                    lens="counterfactual", attribute=attribute, subgroup=second,
                    reference=f"matched {first} prompts", feature=f"visual:{feature}",
                    scope="prompt pairs identical except for this attribute",
                    metric=f"mean paired change in {_human(feature)}", effect=mean,
                    effect_unit="value", ci=[low, high], family="dz", magnitude=dz,
                    magnitude_ci=[low / sd if sd and low is not None else None,
                                  high / sd if sd and high is not None else None],
                    n=n, n_total=n, p_value=finite(stats.get("p_value")),
                    question=("counterfactual" if social_change else
                              "stereotyping" if demographic else "quality"),
                    stereotype_candidate=demographic,
                    keys=[(attribute, higher, f"visual:{feature}", "", "+")],
                    statement=(f"Same prompt, only {_human(attribute)} changed {first} → "
                               f"{second}: {_human(feature)} changes by {mean:+.3f} on average "
                               f"(d_z = {dz:+.2f}, {n} pairs)."),
                )
        embeddings = item.get("embeddings") or {}
        between = finite((embeddings.get("between_pair_euclidean_distance") or {}).get("mean"))
        within = finite((embeddings.get("within_condition_euclidean_distance") or {}).get("mean"))
        if between is not None and within:
            table.add(
                lens="counterfactual", level="omnibus", attribute=attribute,
                feature="embedding", metric="between ÷ within distance",
                effect=between / within, effect_unit="ratio", n=item.get("pairs"),
                n_total=item.get("pairs"), context=True, question="counterfactual",
                statement=(f"Changing only {_human(attribute)} moves the CLIP embedding "
                           f"{between / within:.2f}× as far as two generations with the same "
                           f"{_human(attribute)} value differ."),
            )


def _visual(table: _Table, report_data: dict[str, Any]) -> None:
    features = (report_data.get("chapter_7") or {}).get("image_features") or {}
    min_cell = table.settings["min_cell"]
    for feature, item in features.items():
        tests = item.get("tests")
        if tests is None:
            table.warnings.append("Chapter 7 has no image-feature tests; rebuild chapter 7.")
            return
        # Detected age is a face-model estimate of a demographic, not a pixel
        # statistic: a group drawn older or younger is a stereotyping result.
        demographic = feature in DEMOGRAPHIC_FEATURES
        extra = ({"source": "Face detector", "question": "stereotyping",
                  "stereotype_candidate": True} if demographic else {"question": "quality"})
        for attribute, result in tests.items():
            if attribute == "prompt_mode":
                continue
            epsilon = finite(result.get("epsilon_squared"))
            groups = result.get("groups") or {}
            if epsilon is not None:
                table.add(
                    lens="visual", level="omnibus", attribute=attribute,
                    feature=f"visual:{feature}", metric="ε² (Kruskal–Wallis)", effect=epsilon,
                    family="epsilon2", magnitude=epsilon,
                    n=min((group.get("n") or 0 for group in groups.values()), default=0),
                    n_total=result.get("n"), p_value=finite(result.get("p_value")),
                    **extra,
                    statement=(f"{_human(feature).capitalize()} differs between prompt "
                               f"{_human(attribute)} values (ε² = {epsilon:.3f})."),
                )
            for group, stats in groups.items():
                g = finite(stats.get("hedges_g"))
                n = stats.get("n") or 0
                rest_n = stats.get("rest_n") or 0
                if g is None or g <= 0 or min(n, rest_n) < min_cell:
                    continue  # lower values are the mirror of a higher group
                mean, rest = finite(stats.get("mean")), finite(stats.get("rest_mean"))
                table.add(
                    lens="visual", attribute=attribute, subgroup=group,
                    reference=f"other {_human(attribute)} values", feature=f"visual:{feature}",
                    metric=f"difference in {_human(feature)}",
                    effect=(mean - rest) if mean is not None and rest is not None else None,
                    effect_unit="value", family="g", magnitude=g,
                    magnitude_ci=[finite(stats.get("g_ci_low")), finite(stats.get("g_ci_high"))],
                    ci=[None, None], n=min(n, rest_n), n_total=n + rest_n,
                    p_value=finite(stats.get("p_value")), **extra,
                    keys=[(attribute, group, f"visual:{feature}", "", "+")],
                    details={"mean": mean, "rest_mean": rest},
                    statement=(f"Images for prompt {_human(attribute)} = {group} have higher "
                               f"{_human(feature)} than other {_human(attribute)} values "
                               f"(mean {mean:.3g} vs {rest:.3g}, g = {g:+.2f})."),
                )


def _embedding(table: _Table, report_data: dict[str, Any]) -> None:
    groups = ((report_data.get("chapter_7") or {}).get("embedding") or {}).get("groups") or {}
    for attribute, item in groups.items():
        if attribute == "prompt_mode":
            continue
        separability = item.get("separability") or {}
        score = finite(separability.get("score"))
        if score is None:
            continue
        null = separability.get("null") or {}
        probe = separability.get("probe") or {}
        null_mean, null_sd = finite(null.get("null_mean")), finite(null.get("null_sd"))
        p_value = (normal_p((score - null_mean) / null_sd)
                   if null_mean is not None and null_sd else finite(null.get("p_value")))
        accuracy = finite(probe.get("balanced_accuracy"))
        probe_text = (f"; a linear probe recovers it with {_pct(accuracy)} balanced accuracy "
                      f"(chance {_pct(finite(probe.get('chance')))})" if accuracy is not None
                      else "")
        table.add(
            lens="embedding", level="omnibus", attribute=attribute, feature="embedding",
            metric="silhouette vs permutation null", effect=score, family="silhouette",
            magnitude=score - (null_mean or 0.0), n=probe.get("n_images"),
            n_total=separability.get("n_images"), p_value=p_value, context=True,
            question="quality",
            details={"null_mean": null_mean, "null_sd": null_sd, "probe": probe},
            statement=(f"CLIP embeddings group images by prompt {_human(attribute)}: "
                       f"silhouette {score:.2f} vs {null_mean if null_mean is not None else 0:.2f} "
                       f"under shuffled labels{probe_text}."),
        )


def _concepts(config: dict[str, Any]) -> dict[str, dict[str, set[str]]]:
    lexicon = config.get("synthesis_concept_lexicon") or DEFAULT_CONCEPTS
    return {dimension: {category: {str(word).lower() for word in words}
                        for category, words in categories.items()}
            for dimension, categories in lexicon.items()}


def _caption(table: _Table, report_data: dict[str, Any], config: dict[str, Any]) -> None:
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
                difference, low, high = newcombe_difference(k, n, rest_k, rest_n)
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
                    reference=f"other {_human(attribute)} values", feature="caption term",
                    category=word, metric="share of captions with term",
                    effect=difference, effect_unit="pp", ci=[low, high], family="h",
                    magnitude=cohen_h(share, rest_share),
                    magnitude_ci=_h_interval(rest_share, low, high),
                    n=min(n, rest_n), n_total=n + rest_n,
                    p_value=finite(term.get("p_value")),
                    question="stereotyping" if social else "context",
                    stereotype_candidate=social, context=echo or not social,
                    keys=keys,
                    details={"share": share, "rest_share": rest_share,
                             "odds_ratio": math.exp(log_odds) if log_odds is not None else None,
                             "concepts": mapped, "echo": echo},
                    statement=(f"Captions for prompt {_human(attribute)} = {group} contain "
                               f"“{word}” in {_pct(share)} of {n} images vs {_pct(rest_share)} "
                               f"for other {_human(attribute)} values."),
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
                reference=f"other {_human(attribute)} values", feature="caption sentiment",
                category=direction, metric="mean caption sentiment difference",
                effect=difference, effect_unit="value",
                ci=[difference - Z95 * se_diff, difference + Z95 * se_diff],
                family="g", magnitude=g, magnitude_ci=[g - Z95 * se_g, g + Z95 * se_g],
                n=min(n, rest_n), n_total=n + rest_n,
                p_value=normal_p(difference / se_diff) if se_diff else None,
                question="quality" if not social else "stereotyping",
                stereotype_candidate=social,
                keys=[(attribute, group, "sentiment", "", sign)],
                details={"mean": mean, "rest_mean": rest_mean},
                statement=(f"Captions for prompt {_human(attribute)} = {group} are {direction} "
                           f"than for other {_human(attribute)} values (VADER {mean:+.2f} vs "
                           f"{rest_mean:+.2f})."),
            )


def _diversity(table: _Table, report_data: dict[str, Any]) -> None:
    groups = (((report_data.get("chapter_3") or {}).get("diversity") or {})
              .get("groups") or {}).get("demographic_group") or {}
    values = {group: (finite(stats.get("embedding_dispersion")), stats.get("n_images") or 0)
              for group, stats in groups.items()}
    values = {group: value for group, value in values.items() if value[0] is not None}
    if len(values) < 2:
        return
    low = min(values, key=lambda group: values[group][0])
    high = max(values, key=lambda group: values[group][0])
    table.add(
        lens="diversity", level="omnibus", attribute="detected gender", subgroup=low,
        reference=high, feature="embedding dispersion",
        metric="ratio of embedding dispersion", effect=values[low][0] / values[high][0],
        effect_unit="ratio", n=min(values[low][1], values[high][1]),
        n_total=sum(value[1] for value in values.values()), context=True, question="quality",
        details={group: {"dispersion": value[0], "n": value[1]}
                 for group, value in values.items()},
        statement=(f"Images detected as {low} are less varied than images detected as {high} "
                   f"(mean pairwise embedding distance {values[low][0]:.3f} vs "
                   f"{values[high][0]:.3f}). Group sizes differ and the groups come from "
                   "different prompts, so this is descriptive only."),
    )


# --------------------------------------------------------------------------
# Post-processing: correction, flags, tiers, clusters
# --------------------------------------------------------------------------

def _correct(table: _Table, alpha: float) -> dict[str, Any]:
    """BH over every test in the audit, padded with tests that were not stored."""
    tested = [row for row in table.rows if row["p_value"] is not None]
    values = [row["p_value"] for row in tested] + [1.0] * table.hidden_tests
    total = len(values)
    if not total:
        return {"tests": 0, "hidden_tests": table.hidden_tests}
    order = sorted(range(total), key=lambda index: values[index])
    adjusted = [1.0] * total
    running = 1.0
    for rank in range(total, 0, -1):
        index = order[rank - 1]
        running = min(running, values[index] * total / rank)
        adjusted[index] = min(1.0, running)
    for row, q_value in zip(tested, adjusted):
        row["q_value"] = q_value
    for row in table.rows:
        row.setdefault("q_value", None)
    return {"tests": total, "stored_tests": len(tested), "hidden_tests": table.hidden_tests,
            "method": "Benjamini–Hochberg", "alpha": alpha}


def _apply_flags(table: _Table, state: dict[str, Any]) -> None:
    small_n = table.settings["small_n"]
    low_detection = {}
    for row in state["detection_rows"]:
        effect = row["effect"]
        if (effect is not None and effect <= -DETECTION_GAP and row["q_value"] is not None
                and row["q_value"] < table.settings["alpha"]):
            low_detection[(row["attribute"], row["subgroup"])] = row
    state["low_detection"] = low_detection
    adherence = state["adherence"]
    face_lenses = {"representation", "adherence", "association", "intersectional",
                   "counterfactual"}
    for row in table.rows:
        if row["n"] is not None and row["n"] < small_n and not row["context"]:
            _flag(row, "small_n", f"The smallest compared group has n = {row['n']} (< {small_n}).")
        if row["lens"] in face_lenses:
            attributes = (row["details"].get("attributes") or [row["attribute"]])
            subgroups = str(row["subgroup"] or "").split(" × ")
            for attribute, subgroup in zip(attributes, subgroups if len(attributes) > 1
                                           else [row["subgroup"]]):
                hit = low_detection.get((attribute, subgroup))
                if hit:
                    _flag(row, "low_detection",
                          f"Faces are detected in only {_pct(hit['details']['rate'])} of "
                          f"{_human(attribute)} = {subgroup} images.")
        attributes = row["details"].get("attributes") or [row["attribute"]]
        subgroups = (str(row["subgroup"] or "").split(" × ") if len(attributes) > 1
                     else [row["subgroup"]])
        for attribute, subgroup in zip(attributes, subgroups):
            rate = adherence.get(attribute, {}).get(subgroup)
            if (rate is not None and rate < LOW_ADHERENCE and row["lens"] != "adherence"
                    and not row["context"]):
                _flag(row, "low_adherence",
                      f"Requested {attribute} = {subgroup} is rendered as requested in only "
                      f"{_pct(rate)} of images; results about this group partly describe "
                      "images that do not show it.")


TIERS = {
    "measurement": "Measurement issue",
    "context": "Context",
    "negligible": "Negligible",
    "descriptive": "Descriptive disparity",
    "supported": "Statistically supported",
}
EVIDENCE_LEVELS = {
    "confirmed": "Counterfactual-confirmed",
    "convergent": "Convergent",
    "supported": "Single source",
}


def _assign_tiers(table: _Table) -> None:
    alpha = table.settings["alpha"]
    for row in table.rows:
        supported = row["q_value"] is not None and row["q_value"] < alpha
        row["significant"] = supported
        if row["lens"] == "detection" or any(flag["code"] == "label_mismatch"
                                             for flag in row["flags"]):
            row["tier"] = "measurement"
        elif row["context"]:
            row["tier"] = "context"
        elif row["strength"] is not None and row["strength"] < 1:
            row["tier"] = "negligible"
        elif supported:
            row["tier"] = "supported"
        else:
            row["tier"] = "descriptive"


def _clusters(table: _Table) -> list[dict[str, Any]]:
    """Group specific results that make the same claim, across evidence sources."""
    candidates = [row for row in table.rows
                  if row["level"] == "specific" and row["tier"] in ("supported", "descriptive")
                  and row["keys"]]
    parent = {row["id"]: row["id"] for row in candidates}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    owner: dict[tuple, str] = {}
    for row in candidates:
        for key in row["keys"]:
            key = tuple(key)
            if key in owner:
                parent[find(row["id"])] = find(owner[key])
            else:
                owner[key] = row["id"]
    members: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        members[find(row["id"])].append(row)

    clusters = []
    for rows in members.values():
        supported = [row for row in rows if row["tier"] == "supported"]
        if not supported:
            continue
        lead = min(supported, key=lambda row: rank_key(row["strength_low"], 0, row["n"],
                                                       row["strength"]))
        sources = sorted({row["source"] for row in supported})
        lenses = sorted({row["lens"] for row in supported})
        pooled = [row for row in supported if row["lens"] != "counterfactual"]
        matched = [row for row in supported if row["lens"] == "counterfactual"]
        if pooled and matched:
            evidence = "confirmed"
        elif len(sources) >= 2:
            evidence = "convergent"
        else:
            evidence = "supported"
        flags = {flag["code"]: flag["text"] for flag in lead["flags"]}
        clusters.append({
            "id": f"f{len(clusters) + 1}",
            "lead": lead["id"],
            "rows": [row["id"] for row in sorted(
                rows, key=lambda row: (row["tier"] != "supported", -(row["strength"] or 0)))],
            "supported_rows": len(supported),
            "sources": sources,
            "lenses": lenses,
            "evidence": evidence,
            "strength": lead["strength"],
            "strength_low": lead["strength_low"],
            "n": lead["n"],
            "band": lead["band"],
            "question": lead["question"],
            "stereotype_candidate": any(row["stereotype_candidate"] for row in supported),
            "attribute": lead["attribute"],
            "subgroup": lead["subgroup"],
            "feature": lead["feature"],
            "category": lead["category"],
            "flags": [{"code": code, "text": text} for code, text in sorted(flags.items())],
            "statement": lead["statement"],
        })
        for row in rows:
            row["cluster"] = clusters[-1]["id"]
    order = {"confirmed": 0, "convergent": 1, "supported": 2}
    clusters.sort(key=lambda item: rank_key(item["strength_low"], order[item["evidence"]],
                                            item["n"], item["strength"]))
    for index, cluster in enumerate(clusters, start=1):
        cluster["rank"] = index
    return clusters


def _key_findings(clusters: list[dict[str, Any]], rows: dict[str, dict[str, Any]],
                  limit: int = 10) -> list[str]:
    """Strongest supported findings, spread over themes and lenses."""
    picked: list[str] = []
    per_theme: defaultdict[tuple, int] = defaultdict(int)
    per_lens: defaultdict[str, int] = defaultdict(int)
    for cluster in clusters:
        if (cluster["strength_low"] or 0) < 1:
            continue  # only effects that are at least "small" at the lower CI bound
        lead = rows[cluster["lead"]]
        theme = (lead["attribute"], lead["feature"])
        if per_theme[theme] >= 2 or per_lens[lead["lens"]] >= 3:
            continue
        per_theme[theme] += 1
        per_lens[lead["lens"]] += 1
        picked.append(cluster["id"])
        if len(picked) == limit:
            break
    return picked


def _evidence_matrix(table: _Table) -> dict[str, Any]:
    columns = ["adherence", "association", "intersectional", "objects", "counterfactual",
               "visual", "caption", "embedding", "detection"]
    attributes = sorted(
        {attribute for row in table.rows for attribute in
         (row["details"].get("attributes") or [row["attribute"]])
         if attribute and attribute not in ("default", "detected gender", "prompt_mode")},
        key=lambda name: (ATTRIBUTE_ORDER.index(name) if name in ATTRIBUTE_ORDER else 99, name),
    )
    result_rows = []
    for attribute in attributes:
        cells = {}
        for lens in columns:
            rows = [row for row in table.rows if row["lens"] == lens and attribute in
                    (row["details"].get("attributes") or [row["attribute"]])]
            if lens not in ("embedding", "detection"):
                rows = [row for row in rows if not row["context"]]
            if not rows:
                cells[lens] = None
                continue
            tested = [row for row in rows if row["p_value"] is not None]
            supported = [row for row in rows if row["significant"]
                         and (row["strength"] is None or row["strength"] >= 1)]
            ranked = [row for row in (supported or rows) if row["strength"] is not None]
            best = min(ranked, key=lambda row: rank_key(row["strength_low"], 0, row["n"],
                                                        row["strength"]), default=None)
            cells[lens] = {
                "tests": len(tested),
                "supported": len(supported),
                "best": best["id"] if best else None,
                "strength": best["strength"] if best else None,
                "band": best["band"] if best else None,
                "significant": bool(best and best["significant"]),
                "flags": [flag["code"] for flag in best["flags"]] if best else [],
            }
        bias_lenses = [lens for lens in columns if lens not in ("embedding", "detection")]
        result_rows.append({
            "attribute": attribute,
            "cells": cells,
            "convergence": sum(
                1 for lens in bias_lenses
                if cells.get(lens) and cells[lens]["supported"]
                and (cells[lens]["strength"] or 0) >= 1
            ),
        })
    return {"columns": columns, "rows": result_rows}


QUESTIONS = (
    ("representation", "Default representation",
     "When a prompt does not specify gender, race or age, does the model collapse to one group?",
     "#chapter-3"),
    ("adherence", "Quality of service",
     "Is the prompt followed equally reliably for every requested group?", "#chapter-4"),
    ("stereotyping", "Stereotyping",
     "Are jobs, places, traits or regions tied to particular demographics, objects, emotions "
     "or words?", "#chapter-5"),
    ("counterfactual", "Counterfactual consistency",
     "When only a demographic word changes, do unrelated properties of the image change too?",
     "#chapter-6"),
    ("quality", "Representational quality",
     "Are some groups shown with less variety, different framing, visual style or tone?",
     "#chapter-7"),
)


def _answers(table: _Table, clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"confirmed": 0, "convergent": 1, "supported": 2}
    evidence = {cluster["id"]: order[cluster["evidence"]] for cluster in clusters}
    answers = []
    for key, title, question, link in QUESTIONS:
        rows = [row for row in table.rows if row["question"] == key
                and row["tier"] not in ("measurement",)]
        tested = [row for row in rows if row["p_value"] is not None and not row["context"]]
        supported = [row for row in tested if row["tier"] == "supported"
                     and row["level"] == "specific"]
        # Among equally certain effects, multi-source findings come first.
        supported.sort(key=lambda row: rank_key(row["strength_low"],
                                                evidence.get(row.get("cluster"), 2),
                                                row["n"], row["strength"]))
        top_band = supported[0]["band"] if supported else None
        context = [row for row in rows if row["context"]][:3]
        related = [cluster for cluster in clusters if cluster["question"] == key]
        answers.append({
            "key": key,
            "title": title,
            "question": question,
            "link": link,
            "tests": len(tested),
            "supported": len(supported),
            "omnibus_supported": sum(1 for row in tested if row["level"] == "omnibus"
                                     and row["tier"] == "supported"),
            "band": top_band,
            "top": [row["id"] for row in supported[:3]],
            "context": [row["id"] for row in context],
            "findings": len(related),
            "convergent": sum(1 for cluster in related if cluster["evidence"] != "supported"),
            "stereotype_candidates": sum(1 for cluster in related
                                         if cluster["stereotype_candidate"]),
        })
    return answers


def _measurement_summary(table: _Table, state: dict[str, Any]) -> dict[str, Any]:
    label_mismatch = [row["id"] for row in table.rows
                      if any(flag["code"] == "label_mismatch" for flag in row["flags"])]
    low_adherence = sorted(
        ((attribute, value, rate) for attribute, values in state["adherence"].items()
         for value, rate in values.items() if rate < LOW_ADHERENCE),
        key=lambda item: item[2],
    )
    affected = defaultdict(int)
    for row in table.rows:
        for flag in row["flags"]:
            if row["tier"] == "supported":
                affected[flag["code"]] += 1
    return {
        "low_detection": [row["id"] for row in state["low_detection"].values()],
        "label_mismatch": label_mismatch,
        "label_mismatch_values": [{"attribute": attribute, "value": value}
                                  for attribute, value in state["label_mismatch"]],
        "low_adherence": [{"attribute": attribute, "value": value, "rate": rate}
                          for attribute, value, rate in low_adherence],
        "affected_supported": dict(affected),
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def build_synthesis(report_data: dict[str, Any], config: dict[str, Any] | None = None
                    ) -> dict[str, Any]:
    """Build the chapter-9 payload from the other chapters' payloads."""
    config = config or {}
    settings = {
        "alpha": float(config.get("synthesis_alpha", DEFAULT_ALPHA)),
        "small_n": int(config.get("synthesis_small_n", DEFAULT_SMALL_N)),
        "min_cell": int(config.get("synthesis_min_cell", DEFAULT_MIN_CELL)),
    }
    table = _Table(settings)
    state: dict[str, Any] = {"adherence": {}, "detection_rows": [], "detector_labels": {},
                             "label_mismatch": []}
    steps: list[tuple[str, Callable[[], None]]] = [
        ("representation", lambda: _representation(table, report_data)),
        ("adherence", lambda: _adherence(table, report_data, state)),
        ("detection", lambda: _detection(table, report_data, state)),
        ("association", lambda: _association(table, report_data)),
        ("intersectional", lambda: _intersectional(table, report_data)),
        ("objects", lambda: _objects(table, report_data)),
        ("counterfactual", lambda: _counterfactual(table, report_data)),
        ("visual", lambda: _visual(table, report_data)),
        ("embedding", lambda: _embedding(table, report_data)),
        ("caption", lambda: _caption(table, report_data, config)),
        ("diversity", lambda: _diversity(table, report_data)),
    ]
    for name, step in steps:
        try:
            step()
        except (KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError) as error:
            # One malformed chapter must not take the whole synthesis down.
            table.warnings.append(f"Skipped the {name} lens: {type(error).__name__}: {error}")

    correction = _correct(table, settings["alpha"])
    _apply_flags(table, state)
    _assign_tiers(table)
    clusters = _clusters(table)
    by_id = {row["id"]: row for row in table.rows}
    for row in table.rows:
        row.setdefault("cluster", None)
        row["keys"] = [list(key) for key in row["keys"]]

    tier_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in table.rows:
        tier_counts[row["lens"]][row["tier"]] += 1
    evidence_counts = defaultdict(int)
    for cluster in clusters:
        evidence_counts[cluster["evidence"]] += 1
    substantive = [row for row in table.rows
                   if row["p_value"] is not None and row["tier"] not in ("context", "measurement")]
    return rounded({
        "schema": SCHEMA,
        "settings": {**settings, "detection_gap": DETECTION_GAP, "low_adherence": LOW_ADHERENCE,
                     "families": {key: {"label": value["label"],
                                        "thresholds": list(value["thresholds"])}
                                  for key, value in FAMILIES.items()},
                     "lenses": LENSES, "tiers": TIERS, "evidence_levels": EVIDENCE_LEVELS},
        "correction": correction,
        "summary": {
            "results": len(table.rows),
            "tests": len(substantive),
            "supported": sum(1 for row in substantive if row["tier"] == "supported"),
            "findings": len(clusters),
            "evidence": dict(evidence_counts),
            "stereotype_candidates": sum(1 for cluster in clusters
                                         if cluster["stereotype_candidate"]),
            "large_small_n": sum(1 for row in substantive if row["tier"] == "supported"
                                 and (row["strength"] or 0) >= 3
                                 and any(flag["code"] == "small_n" for flag in row["flags"])),
            "by_lens": {lens: dict(counts) for lens, counts in tier_counts.items()},
        },
        "answers": _answers(table, clusters),
        "key_findings": _key_findings(clusters, by_id),
        "clusters": clusters,
        "matrix": _evidence_matrix(table),
        "measurement": _measurement_summary(table, state),
        "results": table.rows,
        "warnings": list(dict.fromkeys(table.warnings)),
    }, 5)


def iter_results(synthesis: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield the unified results rows (convenience for notebooks and exports)."""
    yield from synthesis.get("results", [])
