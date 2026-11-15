"""Caption sentiment and vocabulary diversity per group."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

from .parser import WORD_CATEGORIES, CaptionParser


# VADER thresholds commonly used to label a compound score.
SENTIMENT_THRESHOLD = 0.05


def mean_ci(values: list[float]) -> dict[str, Any]:
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


def lexical_profile(
    grouped_records: dict[str, list[dict[str, Any]]],
    parsed: dict[int, dict[str, Any]],
    parser: CaptionParser,
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
            sentiment[category] = mean_ci(scores)
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
