"""NLTK caption parsing: tokenize, tag, remove stopwords, lemmatize, score."""

from __future__ import annotations

from typing import Any, Iterable


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


class CaptionParser:
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
