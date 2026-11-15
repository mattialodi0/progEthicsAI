"""Chapter 8 data: caption analysis.

Every caption is tokenized and part-of-speech tagged with NLTK. Stopwords
(NLTK's English list plus a small list of captioning boilerplate) are removed,
and the remaining content words are lemmatized and split into three word
categories: adjectives, verbs/actions and names/objects (nouns). On top of
these descriptive words the chapter computes

* TF-IDF rankings per word category, for prompt values and for detected
  attributes conditioned on prompt values;
* tested term associations (log odds ratio of each word between a group and
  the other values of the same attribute);
* the sentiment of the descriptive words of every image (VADER), aggregated
  per attribute value;
* the vocabulary diversity per attribute value and word category (rarefied
  richness, Simpson diversity and type-token ratio).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .lexical import lexical_profile, mean_ci
from .parser import (
    DEFAULT_DOMAIN_STOPWORDS,
    DEFAULT_SENTIMENT_NEUTRAL_WORDS,
    WORD_CATEGORIES,
    WORD_CATEGORY_LABELS,
    CaptionParser,
)
from .terms import records_by_detected, records_by_group, term_association_tests, tfidf


PROMPT_ATTRIBUTES = (
    "prompt_mode", "gender", "race", "age", "emotion", "job", "workplace",
    "continent", "status", "toxic_trait", "positive_trait",
)
DETECTED_ATTRIBUTES = ("gender", "race", "age", "emotion")
DEFAULT_MIN_GROUP_SIZE = 2
DEFAULT_TOP_TERMS = 5
DEFAULT_SENTIMENT_WORDS = 5
# Term associations: minimum captions containing a term, overall and in the
# group, and how many over-represented terms are kept per group.
DEFAULT_TERM_MIN_DOCUMENTS = 5
DEFAULT_TERM_MIN_GROUP_DOCUMENTS = 3
DEFAULT_TERM_ASSOCIATIONS = 10


def _setting(config: dict[str, Any], key: str, default: int) -> int:
    return max(1, int(config.get(key, default)))


def _parse_captions(
    records: list[dict[str, Any]], parser: CaptionParser
) -> tuple[dict[int, dict[str, Any]], Counter, dict[str, Counter], int, int]:
    """Parse and score every caption.

    Returns the parsed captions keyed by ``id(record)``, the removed stopword
    counts, the vocabulary per word category and the total and other-POS
    token counts.
    """
    parsed: dict[int, dict[str, Any]] = {}
    removed = Counter()
    vocabulary = {category: Counter() for category in WORD_CATEGORIES}
    total_tokens = other_tokens = 0
    for record in records:
        item = parser.parse(record["caption"])
        descriptive = [word for words in item["words"].values() for word in words]
        item["sentiment"] = {"all": parser.compound(descriptive)}
        for category, words in item["words"].items():
            item["sentiment"][category] = parser.compound(words)
            vocabulary[category].update(words)
        parsed[id(record)] = item
        removed.update(item["removed"])
        total_tokens += item["total"]
        other_tokens += item["other"]
    return parsed, removed, vocabulary, total_tokens, other_tokens


def _detected_associations(
    prompt_groups: dict[str, dict[str, list[dict[str, Any]]]],
    parsed: dict[int, dict[str, Any]],
    top_terms: int,
    min_group_size: int,
) -> dict[str, Any]:
    """TF-IDF of detected groups within each prompt value (e.g. doctors
    detected as female vs doctors detected as male)."""
    result: dict[str, Any] = {}
    for prompt_attribute, groups in prompt_groups.items():
        conditioned: dict[str, Any] = {}
        for detected_attribute in DETECTED_ATTRIBUTES:
            output_groups: dict[str, Any] = {}
            for prompt_value, prompt_records in groups.items():
                ranked = tfidf(
                    records_by_detected(prompt_records, detected_attribute),
                    parsed, top_terms, min_group_size,
                )
                if ranked:
                    output_groups[prompt_value] = ranked
            conditioned[detected_attribute] = output_groups
        result[prompt_attribute] = conditioned
    return result


def build_caption_analysis(
    data: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Compute caption vocabularies, TF-IDF, sentiment and diversity for chapter 8."""
    min_group_size = _setting(config, "caption_min_group_size", DEFAULT_MIN_GROUP_SIZE)
    top_terms = _setting(config, "caption_top_terms", DEFAULT_TOP_TERMS)
    sentiment_words = _setting(config, "caption_sentiment_words", DEFAULT_SENTIMENT_WORDS)
    parser = CaptionParser(
        config.get("caption_extra_stopwords", DEFAULT_DOMAIN_STOPWORDS),
        config.get("caption_sentiment_neutral_words", DEFAULT_SENTIMENT_NEUTRAL_WORDS),
    )
    records = [
        record for record in data
        if isinstance(record.get("caption"), str) and record["caption"].strip()
    ]
    parsed, removed, vocabulary, total_tokens, other_tokens = _parse_captions(records, parser)

    prompt_groups = {
        attribute: records_by_group(records, attribute, config)
        for attribute in PROMPT_ATTRIBUTES
    }
    detected_groups = {
        attribute: records_by_detected(records, attribute)
        for attribute in DETECTED_ATTRIBUTES
    }
    term_settings = {
        "min_documents": _setting(config, "caption_term_min_documents",
                                  DEFAULT_TERM_MIN_DOCUMENTS),
        "min_group_documents": _setting(config, "caption_term_min_group_documents",
                                        DEFAULT_TERM_MIN_GROUP_DOCUMENTS),
        "keep": _setting(config, "caption_term_associations", DEFAULT_TERM_ASSOCIATIONS),
    }

    def profiles(groups_by_attribute: dict[str, Any]) -> dict[str, Any]:
        return {
            attribute: lexical_profile(groups, parsed, parser, min_group_size, sentiment_words)
            for attribute, groups in groups_by_attribute.items()
        }

    all_scores = [item["sentiment"]["all"] for item in parsed.values()
                  if item["sentiment"]["all"] is not None]
    return {
        "n_captions": len(records),
        "min_group_size": min_group_size,
        "top_terms": top_terms,
        "word_categories": WORD_CATEGORY_LABELS,
        "stopwords": {
            "nltk": parser.nltk_stopwords,
            "domain": parser.domain_stopwords,
            "total_tokens": total_tokens,
            "removed_tokens": sum(removed.values()),
            "other_pos_tokens": other_tokens,
            "most_removed": [
                {"term": term, "frequency": count}
                for term, count in removed.most_common(30)
            ],
        },
        "vocabulary": {
            category: sorted(counter) for category, counter in vocabulary.items()
        },
        "vocabulary_size": {
            category: len(counter) for category, counter in vocabulary.items()
        },
        "overall_sentiment": mean_ci(all_scores),
        "sentiment_neutral_words": parser.neutral_words,
        "prompt_associations": {
            attribute: tfidf(groups, parsed, top_terms, min_group_size)
            for attribute, groups in prompt_groups.items()
        },
        "detected_associations": _detected_associations(prompt_groups, parsed, top_terms,
                                                        min_group_size),
        "lexical_profiles": {"prompt": profiles(prompt_groups),
                             "detected": profiles(detected_groups)},
        "term_associations": {
            attribute: term_association_tests(groups, parsed, **term_settings)
            for attribute, groups in prompt_groups.items()
        },
        "term_association_settings": term_settings,
    }
