"""Chapter 9 data: cross-analysis synthesis of chapters 2-8.

Unlike the other chapter builders this one does not read image records. It
reads the ``report_data`` payloads of the other chapters and turns every
effect they measured into one row of a common **results table**
(``table.ResultTable``):

    lens, attribute, subgroup, reference, feature, category, metric,
    effect (native units) + 95% CI, standardized magnitude + CI,
    strength score, n, p-value, BH q-value, flags, tier, statement

One extractor per lens fills the table (``face_lenses``,
``association_lenses``, ``image_lenses``, ``caption_lens``). On top of that
table it computes (``postprocess`` and ``summaries``)

* a Benjamini-Hochberg false-discovery-rate correction over *all* tests of
  the audit (one family, so a result is never "significant" only because it
  was looked at in isolation);
* **evidence clusters**: results from different evidence sources (face
  detector, object detector, captions, pixel statistics, matched
  counterfactual pairs) that make the same claim about the same subgroup;
* an **evidence matrix** (prompt attribute x lens);
* data-driven answers to the five fairness questions of the reading guide;
* measurement-reliability caveats (face-detection gaps, label mismatches,
  poorly followed requests) that are attached to the affected results.

Everything is derived from the artifact, so a different model or dataset
produces different findings. The function is cheap and deterministic, so the
analyzer rebuilds it on every run.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable

from ..stats_utils import rounded
from .association_lenses import association_results, intersectional_results, object_results
from .caption_lens import caption_results
from .constants import (
    DEFAULT_ALPHA,
    DEFAULT_MIN_CELL,
    DEFAULT_SMALL_N,
    DETECTION_GAP,
    EVIDENCE_LEVELS,
    FAMILIES,
    LENSES,
    LOW_ADHERENCE,
    SCHEMA,
    TIERS,
)
from .face_lenses import adherence_results, detection_results, representation_results
from .image_lenses import (
    counterfactual_results,
    diversity_results,
    embedding_results,
    visual_results,
)
from .postprocess import apply_flags, assign_tiers, build_clusters, correct_p_values
from .summaries import answer_questions, evidence_matrix, key_findings, measurement_summary
from .table import ResultTable


def build_synthesis(report_data: dict[str, Any], config: dict[str, Any] | None = None
                    ) -> dict[str, Any]:
    """Build the chapter-9 payload from the other chapters' payloads."""
    config = config or {}
    settings = {
        "alpha": float(config.get("synthesis_alpha", DEFAULT_ALPHA)),
        "small_n": int(config.get("synthesis_small_n", DEFAULT_SMALL_N)),
        "min_cell": int(config.get("synthesis_min_cell", DEFAULT_MIN_CELL)),
    }
    table = ResultTable(settings)
    state: dict[str, Any] = {"adherence": {}, "detection_rows": [], "detector_labels": {},
                             "label_mismatch": []}
    steps: list[tuple[str, Callable[[], None]]] = [
        ("representation", lambda: representation_results(table, report_data)),
        ("adherence", lambda: adherence_results(table, report_data, state)),
        ("detection", lambda: detection_results(table, report_data, state)),
        ("association", lambda: association_results(table, report_data)),
        ("intersectional", lambda: intersectional_results(table, report_data)),
        ("objects", lambda: object_results(table, report_data)),
        ("counterfactual", lambda: counterfactual_results(table, report_data)),
        ("visual", lambda: visual_results(table, report_data)),
        ("embedding", lambda: embedding_results(table, report_data)),
        ("caption", lambda: caption_results(table, report_data, config)),
        ("diversity", lambda: diversity_results(table, report_data)),
    ]
    for name, step in steps:
        try:
            step()
        except (KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError) as error:
            # One malformed chapter must not take the whole synthesis down.
            table.warnings.append(f"Skipped the {name} lens: {type(error).__name__}: {error}")

    correction = correct_p_values(table, settings["alpha"])
    apply_flags(table, state)
    assign_tiers(table)
    clusters = build_clusters(table)
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
        "answers": answer_questions(table, clusters),
        "key_findings": key_findings(clusters, by_id),
        "clusters": clusters,
        "matrix": evidence_matrix(table),
        "measurement": measurement_summary(table, state),
        "results": table.rows,
        "warnings": list(dict.fromkeys(table.warnings)),
    }, 5)


def iter_results(synthesis: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield the unified results rows (convenience for notebooks and exports)."""
    yield from synthesis.get("results", [])
