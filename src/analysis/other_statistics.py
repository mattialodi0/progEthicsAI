"""Analysis statistics not tied to a single report renderer."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .general_statistics import _detected_value, _record_mode


PROMPT_ATTRIBUTES = (
    "prompt_mode", "gender", "race", "age", "emotion", "job", "workplace",
    "continent", "status", "toxic_trait", "positive_trait",
)
DETECTED_ATTRIBUTES = ("gender", "race", "age", "emotion")
DEFAULT_MIN_GROUP_SIZE = 2
DEFAULT_TOP_TERMS = 5


def _nltk_tokens(caption: str) -> list[str]:
    try:
        from nltk.tokenize import word_tokenize
    except ImportError as error:
        raise RuntimeError("Chapter 8 requires the nltk package") from error
    return [
        token.lower() for token in word_tokenize(caption, preserve_line=True)
        if token.isalpha()
    ]


def _nltk_documents(records: list[dict[str, Any]]) -> tuple[list[list[str]], list[str]]:
    try:
        from nltk import FreqDist
    except ImportError as error:
        raise RuntimeError("Chapter 8 requires the nltk package") from error
    documents: list[list[str]] = []
    for record in records:
        caption = record.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            continue
        tokens = _nltk_tokens(caption)
        if tokens:
            documents.append(tokens)
    vocabulary = sorted(FreqDist(token for document in documents for token in document))
    return documents, vocabulary


def _group_value(
    record: dict[str, Any], attribute: str, config: dict[str, Any]
) -> str | None:
    if attribute == "prompt_mode":
        return _record_mode(record, config)
    features = record.get("prompt_feats")
    value = features.get(attribute) if isinstance(features, dict) else None
    if value is None or not str(value).strip():
        return None
    return str(value).strip().lower()


def _tfidf(
    grouped_records: dict[str, list[dict[str, Any]]],
    vocabulary: list[str],
    top_terms: int,
    min_group_size: int,
) -> dict[str, dict[str, Any]]:
    try:
        from nltk import FreqDist
        from nltk.text import TextCollection
    except ImportError as error:
        raise RuntimeError("Chapter 8 requires the nltk package") from error

    groups = {
        group: [
            _nltk_tokens(record["caption"])
            for record in records
            if isinstance(record.get("caption"), str) and record["caption"].strip()
        ]
        for group, records in grouped_records.items()
        if len(records) >= min_group_size
    }
    # Treat each variable value as one document, while retaining image counts.
    documents = {
        group: [token for image in images for token in image]
        for group, images in groups.items()
    }
    collection = TextCollection(list(documents.values()))
    result = {}
    for group, document in documents.items():
        frequencies = FreqDist(document)
        ranked = sorted(
            (
                (term, collection.tf_idf(term, document), frequencies[term])
                for term in vocabulary if frequencies[term]
            ),
            key=lambda item: (-item[1], -item[2], item[0]),
        )[:top_terms]
        result[group] = {
            "observations": len(grouped_records[group]),
            "top_terms": [
                {
                    "term": term,
                    "score": round(float(score), 6),
                    "document_frequency": count,
                }
                for term, score, count in ranked
            ],
        }
    return result


def _records_by_group(
    records: list[dict[str, Any]], attribute: str, config: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = _group_value(record, attribute, config)
        if value is not None:
            groups[value].append(record)
    return dict(groups)


def build_caption_analysis(
    data: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Compute NLTK vocabulary and TF-IDF rankings for chapter 8."""
    min_group_size = max(
        1, int(config.get("caption_min_group_size", DEFAULT_MIN_GROUP_SIZE))
    )
    top_terms = max(1, int(config.get("caption_top_terms", DEFAULT_TOP_TERMS)))
    records = [
        record for record in data
        if isinstance(record.get("caption"), str) and record["caption"].strip()
    ]
    _, vocabulary = _nltk_documents(records)

    prompt_associations = {
        attribute: _tfidf(
            _records_by_group(records, attribute, config),
            vocabulary,
            top_terms,
            min_group_size,
        )
        for attribute in PROMPT_ATTRIBUTES
    }

    detected_associations: dict[str, Any] = {}
    for prompt_attribute in PROMPT_ATTRIBUTES:
        conditioned: dict[str, Any] = {}
        prompt_groups = _records_by_group(records, prompt_attribute, config)
        for detected_attribute in DETECTED_ATTRIBUTES:
            output_groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
            for prompt_value, prompt_records in prompt_groups.items():
                detected_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
                for record in prompt_records:
                    detected_value = _detected_value(record, detected_attribute)
                    if detected_value is not None:
                        detected_groups[detected_value].append(record)
                ranked = _tfidf(
                    dict(detected_groups), vocabulary, top_terms, min_group_size
                )
                if ranked:
                    output_groups[prompt_value] = ranked
            conditioned[detected_attribute] = output_groups
        detected_associations[prompt_attribute] = conditioned

    return {
        "n_captions": len(records),
        "vocabulary": vocabulary,
        "vocabulary_size": len(vocabulary),
        "min_group_size": min_group_size,
        "top_terms": top_terms,
        "prompt_associations": prompt_associations,
        "detected_associations": detected_associations,
    }
