"""Chapter 9 post-processing: FDR correction, caveat flags, tiers and the
clustering of results into findings."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..stats_utils import benjamini_hochberg
from .constants import DETECTION_GAP, LOW_ADHERENCE
from .table import ResultTable, add_flag, human, pct, rank_key


def correct_p_values(table: ResultTable, alpha: float) -> dict[str, Any]:
    """BH over every test in the audit, padded with tests that were not stored."""
    tested = [row for row in table.rows if row["p_value"] is not None]
    values = [row["p_value"] for row in tested] + [1.0] * table.hidden_tests
    total = len(values)
    if not total:
        return {"tests": 0, "hidden_tests": table.hidden_tests}
    adjusted = benjamini_hochberg(values)
    for row, q_value in zip(tested, adjusted):
        row["q_value"] = q_value
    for row in table.rows:
        row.setdefault("q_value", None)
    return {"tests": total, "stored_tests": len(tested), "hidden_tests": table.hidden_tests,
            "method": "Benjamini–Hochberg", "alpha": alpha}


def apply_flags(table: ResultTable, state: dict[str, Any]) -> None:
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
            add_flag(row, "small_n",
                     f"The smallest compared group has n = {row['n']} (< {small_n}).")
        if row["lens"] in face_lenses:
            attributes = (row["details"].get("attributes") or [row["attribute"]])
            subgroups = str(row["subgroup"] or "").split(" × ")
            for attribute, subgroup in zip(attributes, subgroups if len(attributes) > 1
                                           else [row["subgroup"]]):
                hit = low_detection.get((attribute, subgroup))
                if hit:
                    add_flag(row, "low_detection",
                             f"Faces are detected in only {pct(hit['details']['rate'])} of "
                             f"{human(attribute)} = {subgroup} images.")
        attributes = row["details"].get("attributes") or [row["attribute"]]
        subgroups = (str(row["subgroup"] or "").split(" × ") if len(attributes) > 1
                     else [row["subgroup"]])
        for attribute, subgroup in zip(attributes, subgroups):
            rate = adherence.get(attribute, {}).get(subgroup)
            if (rate is not None and rate < LOW_ADHERENCE and row["lens"] != "adherence"
                    and not row["context"]):
                add_flag(row, "low_adherence",
                         f"Requested {attribute} = {subgroup} is rendered as requested in only "
                         f"{pct(rate)} of images; results about this group partly describe "
                         "images that do not show it.")


def assign_tiers(table: ResultTable) -> None:
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


def build_clusters(table: ResultTable) -> list[dict[str, Any]]:
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
