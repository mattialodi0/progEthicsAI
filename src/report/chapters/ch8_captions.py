"""Renderer for chapter 8: caption and semantic associations."""

from __future__ import annotations

from typing import Any

from ..explain import chapter_header, findings, guide
from ..insights import chapter_eight
from ..static_files import script_json, static_text

_SCRIPT = static_text("chapter_8.js")
_STYLE = static_text("chapter_8.css")


def render_chapter_eight(chapter_data: dict[str, Any]) -> str:
    """Render precomputed caption word associations, sentiment and diversity."""
    serialized = script_json(chapter_data)
    return f"""
  <style>
{_STYLE}  </style>
  {chapter_header(8, "Caption and semantic associations",
      "How are different groups <em>described</em>: which words, which tone, how much "
      "variety?",
      "<p>Each image was described by an automatic captioning model. Captions capture "
      "content that no structured detector measures — clothing, activities, settings, "
      "expressions, mood — so they can reveal stereotyped depictions that chapters 3–7 "
      "miss. The approach is the same as before: compare the descriptions of different "
      "prompt groups and look for systematic differences.</p>")}
  {findings(chapter_eight(chapter_data))}
  <p>Captions are tokenized and part-of-speech tagged with NLTK.
  Stopwords are removed and the remaining descriptive words are lemmatized and
  split into three categories: <strong>adjectives</strong>,
  <strong>verbs / actions</strong> and <strong>names / objects</strong> (nouns).
  These words are used to rank distinctive terms, to measure the sentiment of
  each image description and to compare vocabulary diversity across attribute
  values. Groups smaller than <span id="chapter-8-min-group"></span> images are
  excluded.</p>
  <div class="metrics" id="chapter-8-metrics"></div>

  <section id="chapter-8-prompt" aria-labelledby="chapter-8-prompt-heading">
    <h3 id="chapter-8-prompt-heading">8.1 Top TF-IDF words by prompt value</h3>
    <p>For each prompt variable, the highest-scoring words of each category are
    shown for every variable value. TF-IDF is calculated across the values of
    that variable (each value is one document), so a term scores high when it
    is frequent for one value and uncommon for the others. Words used for every
    value score zero and are never shown.</p>
    {guide(
        read="One card per value of the selected prompt variable, with its most "
             "distinctive adjectives, verbs and nouns, ranked by TF-IDF; the bar and number "
             "give how often the word occurs. Words seen fewer than 3 times are hidden. "
             "Hover a word for its TF-IDF score.",
        look=["Words that describe the person rather than what the prompt asked for: "
              "appearance (“beautiful”, “muscular”), clothing (“hijab”, “suit”), "
              "condition (“dirty”, “poor”), or role (“assistant”) that appear for some "
              "groups only.",
              "For non-demographic variables (jobs, traits, continents): words that encode "
              "a demographic (“woman”, “bearded”, “turban”) or a stereotyped setting.",
              "Negative or loaded vocabulary concentrated in some groups."],
        caution="TF-IDF favours words that are unique to one value, including harmless "
                "ones (a stethoscope for a doctor). A word must be unexpected given the "
                "prompt to count as a stereotype; check its frequency, not only its score.",
    )}
    <div class="ch8-controls">
      <label>Prompt variable <select id="chapter-8-prompt-select"></select></label>
    </div>
    <div id="chapter-8-prompt-cards" class="plot-grid"></div>
  </section>

  <section id="chapter-8-attributes" aria-labelledby="chapter-8-attributes-heading">
    <h3 id="chapter-8-attributes-heading">8.2 Top TF-IDF words by detected attribute</h3>
    <p>Detected groups are compared within each prompt-variable value. Select a
    prompt variable to account for its composition before comparing detected
    gender, race, age, or emotion groups.</p>
    {guide(
        read="Cards compare images with the same prompt value but a different detected "
             "attribute, e.g. doctors detected as female vs doctors detected as male.",
        look="Different vocabulary for detected groups under the same prompt: the model "
             "(or the captioner) describes, e.g., female doctors with appearance words and "
             "male doctors with activity words.",
        caution="Both the detected attribute and the caption are model outputs, and the "
                "captioner can introduce its own bias. Groups with few images produce "
                "unstable word lists.",
    )}
    <div class="ch8-controls">
      <label>Prompt variable <select id="chapter-8-conditioning-select"></select></label>
      <label>Detected attribute <select id="chapter-8-detected-select"></select></label>
    </div>
    <div id="chapter-8-attribute-cards" class="plot-grid"></div>
  </section>

  <section id="chapter-8-sentiment" aria-labelledby="chapter-8-sentiment-heading">
    <h3 id="chapter-8-sentiment-heading">8.3 Sentiment of image descriptions</h3>
    <p>The descriptive words of each caption (stopwords removed) are scored with
    the VADER lexicon, giving a compound score from &minus;1 (negative) to +1
    (positive) per image. Scores of at least &plusmn;0.05 count as positive or
    negative. The chart shows the mean per attribute value with a 95% confidence
    interval; choose a word category to score only adjectives, verbs or nouns.
    VADER was built for general English, so neutral descriptive words such as
    most clothing and objects carry no sentiment, and words such as
    <em>smiling</em> or <em>serious</em> dominate the score.</p>
    {guide(
        read="Dots are mean sentiment per group with a 95% confidence interval; the "
             "horizontal line is 0 (neutral). The table lists the words driving the score.",
        look=["Groups whose intervals do not overlap: their images are described in a "
              "systematically more positive or negative tone.",
              "Tone differences for demographic attributes (gender, race, age, continent) "
              "matter more than for emotion or status prompts, where a difference is "
              "requested (a “sad person” should read as sad)."],
        caution="Sentiment comes from a general-purpose lexicon applied to short captions; "
                "it reflects word choice, not a judgement of the image.",
    )}
    <details><summary>Words treated as neutral (<span id="chapter-8-neutral-count"></span>)</summary>
      <p class="note">VADER scores these words through a sense they do not have in
      image captions (<em>grey</em> hair, tank <em>top</em>, <em>loose</em> waves,
      <em>hard</em> hat, <em>lying</em> down). Several describe hair or clothing that
      differ by gender, age or race (for example <em>dreadlock</em>), so scoring them
      would create sentiment gaps between groups that do not come from the image's
      mood. Configure with <code>caption_sentiment_neutral_words</code>.</p>
      <p class="ch8-wordlist" id="chapter-8-neutral-list"></p>
    </details>
    <div class="ch8-controls">
      <label>Attribute <select id="chapter-8-sentiment-attribute"></select></label>
      <label>Word category <select id="chapter-8-sentiment-category"></select></label>
    </div>
    <div class="plot-card"><div id="chapter-8-sentiment-plot"></div></div>
    <div class="ch8-table-wrap" id="chapter-8-sentiment-table"></div>
  </section>

  <section id="chapter-8-diversity" aria-labelledby="chapter-8-diversity-heading">
    <h3 id="chapter-8-diversity-heading">8.4 Vocabulary diversity</h3>
    <p>Diversity of the descriptive vocabulary for each attribute value and word
    category. <strong>Rarefied unique words</strong> is the expected number of
    distinct words in a random sample of the same number of words from every
    group (the size of the smallest group), so it can be compared across groups of
    different sizes. <strong>Simpson diversity</strong> is the probability that two
    randomly drawn words differ. The raw type-token ratio is listed for reference
    but falls as groups grow, so compare it only between groups of similar size.</p>
    {guide(
        read="Grouped bars: one colour per word category, one group per attribute value.",
        look="Groups with clearly lower vocabulary diversity are described in a more "
             "repetitive, formulaic way — the textual counterpart of the low visual "
             "diversity in chapter 3.2.",
    )}
    <div class="ch8-controls">
      <label>Attribute <select id="chapter-8-diversity-attribute"></select></label>
      <label>Measure <select id="chapter-8-diversity-metric">
        <option value="rarefied_types">Rarefied unique words</option>
        <option value="simpson">Simpson diversity</option>
        <option value="type_token_ratio">Type-token ratio</option>
      </select></label>
    </div>
    <div class="plot-card"><div id="chapter-8-diversity-plot"></div></div>
    <div class="ch8-table-wrap" id="chapter-8-diversity-table"></div>
  </section>

  <section id="chapter-8-vocabulary" aria-labelledby="chapter-8-vocabulary-heading">
    <h3 id="chapter-8-vocabulary-heading">8.5 Stopwords and caption vocabulary</h3>
    <p class="note">Documentation of the text preprocessing, for reproducibility: which
    words were removed before the analyses above and the full vocabulary that remained.</p>
    <p id="chapter-8-stopword-summary"></p>
    <div class="plot-card"><div id="chapter-8-stopword-plot"></div></div>
    <details><summary>Captioning stopwords (<span id="chapter-8-domain-count"></span>)</summary>
      <p class="note">Words removed in addition to the NLTK list because they come
      from the captioning template (&ldquo;The image shows&hellip;&rdquo;) rather than
      from the image, or because the tagger mislabels them. Configure with
      <code>caption_extra_stopwords</code>.</p>
      <p class="ch8-wordlist" id="chapter-8-domain-list"></p>
    </details>
    <details><summary>NLTK English stopwords (<span id="chapter-8-nltk-count"></span>)</summary>
      <p class="ch8-wordlist" id="chapter-8-nltk-list"></p>
    </details>
    <div id="chapter-8-vocabulary-lists"></div>
  </section>

  <script>
    const chapterEightData = {serialized};
{_SCRIPT}  </script>
"""
