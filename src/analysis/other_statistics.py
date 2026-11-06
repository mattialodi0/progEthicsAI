"""Analysis statistics not tied to a single report renderer.

Chapter 8 caption analysis
--------------------------
Every caption is tokenized and part-of-speech tagged with NLTK. Stopwords
(NLTK's English list plus a small list of captioning boilerplate) are removed,
and the remaining content words are lemmatized and split into three word
categories: adjectives, verbs/actions and names/objects (nouns). On top of
these descriptive words the chapter computes

* TF-IDF rankings per word category, for prompt values and for detected
  attributes conditioned on prompt values;
* the sentiment of the descriptive words of every image (VADER), aggregated
  per attribute value;
* the vocabulary diversity per attribute value and word category (rarefied
  richness, Simpson diversity and type-token ratio).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable

from .general_statistics import _detected_value, _record_mode


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

# Word categories, keyed by the Penn Treebank tag prefix and the WordNet POS
# used for lemmatization.
WORD_CATEGORIES = {
    "adjectives": ("JJ", "a"),
    "verbs": ("VB", "v"),
    "nouns": ("NN", "n"),
}
WORD_CATEGORY_LABELS = {
    "adjectives": "Adjectives",
    "verbs": "Verbs / actions",
    "nouns": "Names / objects",
}

# Words produced by the captioning template ("The image shows ...", "The
# overall mood of the image is ...") that describe the caption rather than the
# image, plus spatial function words the tagger often mislabels as nouns or
# adjectives ("behind", "next"). They are removed in addition to NLTK's
# English stopwords and can be overridden with ``caption_extra_stopwords``.
DEFAULT_DOMAIN_STOPWORDS = (
    "image", "picture", "photo", "photograph", "portrait", "show", "depict",
    "appear", "seem", "also", "overall", "visible", "suggest", "behind", "next",
)

# Words VADER scores through a sense they do not have in image captions
# ("grey hair", "hand", "tank top", "loose waves", "hard hat", "lying down",
# "safety vest"), including the race-linked "dreadlock". They are treated as
# neutral so that hair, clothing or pose descriptions do not create spurious
# sentiment gaps between groups. Override with
# ``caption_sentiment_neutral_words``.
DEFAULT_SENTIMENT_NEUTRAL_WORDS = (
    "grey", "hand", "top", "hard", "highlight", "loose", "low", "number",
    "cut", "create", "extend", "easel", "lying", "dreadlock", "allow",
    "united", "play", "intricate", "natural", "safety", "blind", "impression",
)

# VADER thresholds commonly used to label a compound score.
SENTIMENT_THRESHOLD = 0.05


def _require_nltk() -> Any:
    try:
        import nltk
    except ImportError as error:
        raise RuntimeError("Chapter 8 requires the nltk package") from error
    return nltk


def _ensure_nltk_resource(path: str, package: str) -> None:
    nltk = _require_nltk()
    for candidate in (path, f"{path}.zip"):
        try:
            nltk.data.find(candidate)
            return
        except LookupError:
            continue
    if not nltk.download(package, quiet=True):
        raise RuntimeError(
            f"Chapter 8 requires the NLTK resource '{package}'. "
            f"Install it with: python -m nltk.downloader {package}"
        )


class _CaptionParser:
    """Tokenize, tag, filter and lemmatize captions."""

    def __init__(
        self,
        extra_stopwords: Iterable[str],
        neutral_words: Iterable[str] = DEFAULT_SENTIMENT_NEUTRAL_WORDS,
    ):
        nltk = _require_nltk()
        _ensure_nltk_resource("tokenizers/punkt_tab", "punkt_tab")
        _ensure_nltk_resource("taggers/averaged_perceptron_tagger_eng",
                              "averaged_perceptron_tagger_eng")
        _ensure_nltk_resource("corpora/stopwords", "stopwords")
        _ensure_nltk_resource("corpora/wordnet", "wordnet")
        _ensure_nltk_resource("sentiment/vader_lexicon", "vader_lexicon")
        from nltk.corpus import stopwords
        from nltk.sentiment import SentimentIntensityAnalyzer
        from nltk.stem import WordNetLemmatizer

        self._tokenize = nltk.word_tokenize
        self._tag = nltk.pos_tag
        self._lemmatize = WordNetLemmatizer().lemmatize
        self.nltk_stopwords = sorted(set(stopwords.words("english")))
        self.domain_stopwords = sorted(
            {str(word).strip().lower() for word in extra_stopwords if str(word).strip()}
            - set(self.nltk_stopwords)
        )
        self._stopwords = set(self.nltk_stopwords) | set(self.domain_stopwords)
        self.sentiment = SentimentIntensityAnalyzer()
        self.neutral_words = sorted(
            {str(word).strip().lower() for word in neutral_words if str(word).strip()}
        )
        self._neutral = set(self.neutral_words)

    def parse(self, caption: str) -> dict[str, Any]:
        """Return the descriptive words of a caption, split by word category."""
        words: dict[str, list[str]] = {category: [] for category in WORD_CATEGORIES}
        removed: list[str] = []
        other = 0
        total = 0
        tagged = self._tag(self._tokenize(caption, preserve_line=True))
        for token, tag in tagged:
            if not token.isalpha():
                continue
            total += 1
            token = token.lower()
            if token in self._stopwords:
                removed.append(token)
                continue
            for category, (prefix, wordnet_pos) in WORD_CATEGORIES.items():
                if tag.startswith(prefix):
                    lemma = self._lemmatize(token, wordnet_pos)
                    if lemma in self._stopwords:
                        removed.append(token)
                    else:
                        words[category].append(lemma)
                    break
            else:
                other += 1
        return {"words": words, "removed": removed, "other": other, "total": total}

    def compound(self, words: list[str]) -> float | None:
        """VADER compound score of a bag of descriptive words."""
        if not words:
            return None
        scored = [word for word in words if word not in self._neutral]
        if not scored:
            return 0.0
        return float(self.sentiment.polarity_scores(" ".join(scored))["compound"])

    def valence(self, word: str) -> float:
        if word in self._neutral:
            return 0.0
        return float(self.sentiment.lexicon.get(word, 0.0))


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------

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


def _records_by_group(
    records: list[dict[str, Any]], attribute: str, config: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = _group_value(record, attribute, config)
        if value is not None:
            groups[value].append(record)
    return dict(groups)


def _records_by_detected(
    records: list[dict[str, Any]], attribute: str
) -> dict[str, list[dict[str, Any]]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = _detected_value(record, attribute)
        if value is not None:
            groups[value].append(record)
    return dict(groups)


# --------------------------------------------------------------------------
# TF-IDF
# --------------------------------------------------------------------------

def _tfidf(
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


# --------------------------------------------------------------------------
# Tested term associations
# --------------------------------------------------------------------------

def _term_associations(
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
    from .stats_utils import normal_p

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


# --------------------------------------------------------------------------
# Sentiment and vocabulary diversity
# --------------------------------------------------------------------------

def _mean_ci(values: list[float]) -> dict[str, Any]:
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None,
                "positive_share": None, "negative_share": None}
    mean = sum(values) / n
    if n > 1:
        variance = sum((value - mean) ** 2 for value in values) / (n - 1)
        half_width = 1.96 * math.sqrt(variance / n)
    else:
        half_width = 0.0
    return {
        "n": n,
        "mean": round(mean, 4),
        "sd": round(math.sqrt(variance), 4) if n > 1 else 0.0,
        "ci_low": round(mean - half_width, 4),
        "ci_high": round(mean + half_width, 4),
        "positive_share": round(sum(v > SENTIMENT_THRESHOLD for v in values) / n, 4),
        "negative_share": round(sum(v < -SENTIMENT_THRESHOLD for v in values) / n, 4),
    }


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _rarefied_richness(counter: Counter, depth: int) -> float | None:
    """Expected number of distinct words in a random sample of ``depth`` tokens.

    Unlike the raw number of distinct words, this is comparable across groups
    of different sizes because every group is evaluated at the same depth.
    """
    total = sum(counter.values())
    if depth <= 0 or total < depth:
        return None
    log_all = _log_comb(total, depth)
    expected = 0.0
    for count in counter.values():
        rest = total - count
        absent = 0.0 if rest < depth else math.exp(_log_comb(rest, depth) - log_all)
        expected += 1.0 - absent
    return expected


def _diversity(counter: Counter, depth: int | None) -> dict[str, Any]:
    tokens = sum(counter.values())
    types = len(counter)
    simpson = (
        1.0 - sum(count * (count - 1) for count in counter.values()) / (tokens * (tokens - 1))
        if tokens > 1 else None
    )
    rarefied = _rarefied_richness(counter, depth) if depth else None
    return {
        "tokens": tokens,
        "types": types,
        "type_token_ratio": round(types / tokens, 4) if tokens else None,
        "rarefied_types": round(rarefied, 2) if rarefied is not None else None,
        "simpson": round(simpson, 4) if simpson is not None else None,
    }


def _lexical_profiles(
    grouped_records: dict[str, list[dict[str, Any]]],
    parsed: dict[int, dict[str, Any]],
    parser: _CaptionParser,
    min_group_size: int,
    sentiment_words: int,
) -> dict[str, Any]:
    """Sentiment and vocabulary diversity for every value of one attribute."""
    groups = {
        group: records for group, records in grouped_records.items()
        if len(records) >= min_group_size
    }
    counters = {
        group: {
            category: Counter(
                word for record in records
                for word in parsed[id(record)]["words"][category]
            )
            for category in WORD_CATEGORIES
        }
        for group, records in groups.items()
    }
    # Common rarefaction depth: the smallest group's token count, per category.
    depth = {
        category: min(
            (sum(counter[category].values()) for counter in counters.values()),
            default=0,
        )
        for category in WORD_CATEGORIES
    }
    output: dict[str, Any] = {}
    for group, records in groups.items():
        sentiment = {}
        for category in ("all", *WORD_CATEGORIES):
            scores = [
                parsed[id(record)]["sentiment"][category] for record in records
                if parsed[id(record)]["sentiment"][category] is not None
            ]
            sentiment[category] = _mean_ci(scores)
        all_words = Counter()
        for counter in counters[group].values():
            all_words.update(counter)
        polar = [(word, count, parser.valence(word)) for word, count in all_words.items()
                 if parser.valence(word)]
        output[group] = {
            "observations": len(records),
            "sentiment": sentiment,
            "sentiment_words": {
                "positive": [
                    {"term": word, "frequency": count, "valence": valence}
                    for word, count, valence in sorted(
                        (item for item in polar if item[2] > 0),
                        key=lambda item: (-item[1], item[0]))[:sentiment_words]
                ],
                "negative": [
                    {"term": word, "frequency": count, "valence": valence}
                    for word, count, valence in sorted(
                        (item for item in polar if item[2] < 0),
                        key=lambda item: (-item[1], item[0]))[:sentiment_words]
                ],
            },
            "diversity": {
                category: _diversity(counters[group][category], depth[category])
                for category in WORD_CATEGORIES
            },
        }
    return {"rarefaction_depth": depth, "groups": output}


# --------------------------------------------------------------------------
# Chapter builder
# --------------------------------------------------------------------------

def build_caption_analysis(
    data: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Compute caption vocabularies, TF-IDF, sentiment and diversity for chapter 8."""
    min_group_size = max(
        1, int(config.get("caption_min_group_size", DEFAULT_MIN_GROUP_SIZE))
    )
    top_terms = max(1, int(config.get("caption_top_terms", DEFAULT_TOP_TERMS)))
    sentiment_words = max(
        1, int(config.get("caption_sentiment_words", DEFAULT_SENTIMENT_WORDS))
    )
    parser = _CaptionParser(
        config.get("caption_extra_stopwords", DEFAULT_DOMAIN_STOPWORDS),
        config.get("caption_sentiment_neutral_words", DEFAULT_SENTIMENT_NEUTRAL_WORDS),
    )
    records = [
        record for record in data
        if isinstance(record.get("caption"), str) and record["caption"].strip()
    ]

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

    prompt_groups = {
        attribute: _records_by_group(records, attribute, config)
        for attribute in PROMPT_ATTRIBUTES
    }
    detected_groups = {
        attribute: _records_by_detected(records, attribute)
        for attribute in DETECTED_ATTRIBUTES
    }

    prompt_associations = {
        attribute: _tfidf(groups, parsed, top_terms, min_group_size)
        for attribute, groups in prompt_groups.items()
    }

    detected_associations: dict[str, Any] = {}
    for prompt_attribute, groups in prompt_groups.items():
        conditioned: dict[str, Any] = {}
        for detected_attribute in DETECTED_ATTRIBUTES:
            output_groups: dict[str, Any] = {}
            for prompt_value, prompt_records in groups.items():
                ranked = _tfidf(
                    _records_by_detected(prompt_records, detected_attribute),
                    parsed, top_terms, min_group_size,
                )
                if ranked:
                    output_groups[prompt_value] = ranked
            conditioned[detected_attribute] = output_groups
        detected_associations[prompt_attribute] = conditioned

    term_settings = {
        "min_documents": max(1, int(config.get(
            "caption_term_min_documents", DEFAULT_TERM_MIN_DOCUMENTS))),
        "min_group_documents": max(1, int(config.get(
            "caption_term_min_group_documents", DEFAULT_TERM_MIN_GROUP_DOCUMENTS))),
        "keep": max(1, int(config.get("caption_term_associations", DEFAULT_TERM_ASSOCIATIONS))),
    }
    term_associations = {
        attribute: _term_associations(groups, parsed, **term_settings)
        for attribute, groups in prompt_groups.items()
    }

    lexical_profiles = {
        "prompt": {
            attribute: _lexical_profiles(groups, parsed, parser, min_group_size, sentiment_words)
            for attribute, groups in prompt_groups.items()
        },
        "detected": {
            attribute: _lexical_profiles(groups, parsed, parser, min_group_size, sentiment_words)
            for attribute, groups in detected_groups.items()
        },
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
        "overall_sentiment": _mean_ci(all_scores),
        "sentiment_neutral_words": parser.neutral_words,
        "prompt_associations": prompt_associations,
        "detected_associations": detected_associations,
        "lexical_profiles": lexical_profiles,
        "term_associations": term_associations,
        "term_association_settings": term_settings,
    }
