"""Distinctive caption words per group: TF-IDF rankings and tested associations."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any

from ..records import detected_value, record_mode
from ..stats_utils import normal_p
from .parser import WORD_CATEGORIES


def _group_value(
    record: dict[str, Any], attribute: str, config: dict[str, Any]
) -> str | None:
    if attribute == "prompt_mode":
        return record_mode(record, config)
    features = record.get("prompt_feats")
    value = features.get(attribute) if isinstance(features, dict) else None
    if value is None or not str(value).strip():
        return None
    return str(value).strip().lower()


def records_by_group(
    records: list[dict[str, Any]], attribute: str, config: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = _group_value(record, attribute, config)
        if value is not None:
            groups[value].append(record)
    return dict(groups)


def records_by_detected(
    records: list[dict[str, Any]], attribute: str
) -> dict[str, list[dict[str, Any]]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = detected_value(record, attribute)
        if value is not None:
            groups[value].append(record)
    return dict(groups)


def tfidf(
    grouped_records: dict[str, list[dict[str, Any]]],
    parsed: dict[int, dict[str, Any]],
    top_terms: int,
    min_group_size: int,
) -> dict[str, dict[str, Any]]:
    """Rank terms per group, separately for each word category.

    Each group (all captions of one attribute value) is one document;
    tf = count / document length and idf = log(n_documents / document
    frequency), matching ``nltk.text.TextCollection``.
    """
    groups = {
        group: records for group, records in grouped_records.items()
        if len(records) >= min_group_size
    }
    if not groups:
        return {}
    result: dict[str, dict[str, Any]] = {
        group: {"observations": len(records), "top_terms": {}}
        for group, records in groups.items()
    }
    for category in WORD_CATEGORIES:
        counts = {
            group: Counter(
                word for record in records
                for word in parsed[id(record)]["words"][category]
            )
            for group, records in groups.items()
        }
        document_frequency = Counter(
            term for counter in counts.values() for term in counter
        )
        n_documents = len(counts)
        for group, counter in counts.items():
            length = sum(counter.values())
            ranked = sorted(
                (
                    (
                        term,
                        count / length * math.log(n_documents / document_frequency[term]),
                        count,
                    )
                    for term, count in counter.items()
                ),
                key=lambda item: (-item[1], -item[2], item[0]),
            )[:top_terms]
            result[group]["top_terms"][category] = [
                {"term": term, "score": round(score, 6), "frequency": count}
                for term, score, count in ranked
            ]
    return result


def term_association_tests(
    grouped_records: dict[str, list[dict[str, Any]]],
    parsed: dict[int, dict[str, Any]],
    *,
    min_documents: int,
    min_group_documents: int,
    keep: int,
) -> dict[str, dict[str, Any]]:
    """Terms over-represented in one group relative to the other groups.

    Unit of analysis is the caption: a term is present or absent in each
    caption (any word category). For every group, the share of its captions
    containing the term is compared with the share in the captions of the
    *other values of the same attribute* — the same prompt context — using
    the log odds ratio with a 0.5 continuity correction, its Wald 95%
    interval and a two-sided z-test. Unlike TF-IDF scores these are
    comparable across groups and carry their own uncertainty.
    """
    documents = {
        group: [
            {word for words in parsed[id(record)]["words"].values() for word in words}
            for record in records
        ]
        for group, records in grouped_records.items()
    }
    if len(documents) < 2:
        return {}
    totals = Counter(term for docs in documents.values() for doc in docs for term in doc)
    total_documents = sum(len(docs) for docs in documents.values())
    result: dict[str, dict[str, Any]] = {}
    for group, docs in documents.items():
        in_group = Counter(term for doc in docs for term in doc)
        n_group = len(docs)
        n_rest = total_documents - n_group
        if not n_group or not n_rest:
            continue
        rows = []
        tested = 0
        for term, count in in_group.items():
            if count < min_group_documents or totals[term] < min_documents:
                continue
            tested += 1
            rest = totals[term] - count
            a, b = count + 0.5, n_group - count + 0.5
            c, d = rest + 0.5, n_rest - rest + 0.5
            log_odds = math.log(a * d / (b * c))
            se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            z = log_odds / se
            if z <= 0:
                continue
            rows.append({
                "term": term,
                "group_documents": count,
                "group_n": n_group,
                "group_share": round(count / n_group, 6),
                "rest_documents": rest,
                "rest_n": n_rest,
                "rest_share": round(rest / n_rest, 6),
                "log_odds": round(log_odds, 6),
                "ci_low": round(log_odds - 1.96 * se, 6),
                "ci_high": round(log_odds + 1.96 * se, 6),
                "z": round(z, 4),
                "p_value": normal_p(z),
            })
        rows.sort(key=lambda item: (-item["z"], item["term"]))
        # ``tested`` counts every eligible term, including those not kept, so
        # a multiple-testing correction can account for the whole family.
        result[group] = {"tested": tested, "terms": rows[:keep]}
    return result
