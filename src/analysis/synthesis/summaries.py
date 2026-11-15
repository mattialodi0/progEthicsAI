"""Chapter 9 summaries: key findings, evidence matrix, answers to the fairness
questions and measurement caveats."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .constants import ATTRIBUTE_ORDER, LOW_ADHERENCE, QUESTIONS
from .table import ResultTable, rank_key, reliable_band


def key_findings(clusters: list[dict[str, Any]], rows: dict[str, dict[str, Any]],
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


def evidence_matrix(table: ResultTable) -> dict[str, Any]:
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


def answer_questions(table: ResultTable, clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        # Verdict: the largest disparity the data reliably shows, i.e. the band
        # at the lower CI bound of the top-ranked result. With hundreds of
        # tests the largest point estimate is almost always "large"; the
        # lower bound and the counts below say how much of that is solid.
        top_band = reliable_band(supported[0]) if supported else None
        reliable = {name: sum(1 for row in supported if row["band_low"] == name)
                    for name in ("large", "medium")}
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
            "reliable": reliable,
            "top": [row["id"] for row in supported[:3]],
            "context": [row["id"] for row in context],
            "findings": len(related),
            "convergent": sum(1 for cluster in related if cluster["evidence"] != "supported"),
            "stereotype_candidates": sum(1 for cluster in related
                                         if cluster["stereotype_candidate"]),
        })
    return answers


def measurement_summary(table: ResultTable, state: dict[str, Any]) -> dict[str, Any]:
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
