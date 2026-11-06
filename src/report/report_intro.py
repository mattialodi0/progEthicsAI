"""Renderer for the "How to read this report" guide shown before chapter 1."""

from __future__ import annotations

from .explain import callout


_LENSES = (
    ("Representation", "Who does the model draw when you do not say who?",
     "Chapter 3", "#chapter-3"),
    ("Adherence", "Does the model draw what you asked for, equally well for every group?",
     "Chapter 4", "#chapter-4"),
    ("Association", "Which demographics, objects, emotions and words travel together?",
     "Chapters 5, 7, 8", "#chapter-5"),
    ("Counterfactual", "What else changes when only one word of the prompt changes?",
     "Chapter 6", "#chapter-6"),
    ("Synthesis", "Which disparities are large, reliable and confirmed by several kinds of "
     "evidence?", "Chapter 9", "#chapter-9"),
    ("Method", "How were the numbers produced and how far can they be trusted?",
     "Chapter 10", "#chapter-10"),
)

_GLOSSARY = (
    ("n", "Number of observations (images or matched pairs) behind a value. "
     "With n = 10 a single image moves a percentage by 10 points: treat small-n "
     "results as hints, not findings."),
    ("Percentage points (pp)", "The arithmetic difference between two percentages. "
     "60% vs 40% is a 20 pp gap (a 50% relative increase)."),
    ("95% confidence interval (CI)", "Range of values compatible with the data. If the "
     "intervals of two groups do not overlap, the difference is unlikely to be noise. "
     "An interval for a difference that excludes 0 indicates a reliable direction."),
    ("p-value", "Probability of a pattern at least this strong if there were no real "
     "association. Small p (&lt; 0.05) says an effect is probably not noise; it does "
     "<em>not</em> say the effect is large. With many tests, some p &lt; 0.05 occur by chance."),
    ("Cramér's V", "Strength of association between two categorical variables, from 0 "
     "(independent) to 1 (one determines the other). Conventional reading: &lt; 0.1 "
     "negligible, 0.1–0.3 weak, 0.3–0.5 moderate, &gt; 0.5 strong."),
    ("Lift", "P(item | group) / P(item). 1 = the item is as common in the group as overall; "
     "3 = three times more common. Lifts from items seen in 1–2 images are unstable."),
    ("Normalized entropy", "How evenly a categorical feature is spread over its categories: "
     "1 = perfectly even, 0 = only one category."),
    ("Cosine distance", "1 − cosine similarity of two CLIP embeddings: 0 = semantically "
     "identical, larger = more different. Typical image pairs fall between 0.1 and 0.5."),
    ("Silhouette score", "How well images cluster by a grouping in embedding space, from −1 "
     "to 1. Near 0 = groups overlap; &gt; 0.25 = noticeable structure; &gt; 0.5 = clearly separate."),
    ("Adherence / accuracy", "Share of images whose detected attribute equals the requested one."),
)


def render_reading_guide() -> str:
    """Return the guide that explains the report structure and the statistics."""
    lenses = "".join(
        f'<a class="lens" href="{href}"><strong>{name}</strong>'
        f"<span>{question}</span><em>{chapters}</em></a>"
        for name, question, chapters, href in _LENSES
    )
    glossary = "".join(f"<dt>{term}</dt><dd>{text}</dd>" for term, text in _GLOSSARY)
    return f"""
  <section id="reading-guide" class="reading-guide" aria-labelledby="reading-guide-heading">
    <h2 id="reading-guide-heading">How to read this report</h2>
    <p>This is an <strong>output audit</strong> of a text-to-image (T2I) model. Images are
    generated from controlled prompt templates (for example <em>“A picture of a
    [race] [job]”</em>), and every image is measured automatically: a face-analysis model
    estimates gender, race, age and emotion, an object detector lists visible items, a
    captioning model describes the scene, and CLIP turns each image into an embedding
    vector. Fairness questions are then answered by comparing those measurements across
    prompt groups. The audit looks at the model through four lenses, and chapter 9 combines
    them:</p>
    <div class="lens-grid">{lenses}</div>
    <h3>Fairness questions this report helps answer</h3>
    <ol class="questions">
      <li><strong>Default representation</strong> — when a prompt does not specify
        gender, race or age, does the model collapse to one group? (ch. 3)</li>
      <li><strong>Quality of service</strong> — is the prompt followed equally reliably for
        every requested group, or are some groups “overwritten” by the model? (ch. 4)</li>
      <li><strong>Stereotyping</strong> — are jobs, places, traits or regions systematically
        tied to particular demographics, objects, emotions or words? (ch. 5, 8)</li>
      <li><strong>Counterfactual consistency</strong> — when only a demographic word
        changes, do unrelated properties of the image change too? (ch. 6)</li>
      <li><strong>Representational quality</strong> — are some groups shown with less
        variety, different framing or a different visual style? (ch. 3.2, 7)</li>
    </ol>
    <div class="guide">
      {callout("look", [
        "<strong>Size before significance.</strong> Look first at the gap (percentage "
        "points, Cramér's V, lift); a p-value only says it is unlikely to be noise.",
        "<strong>Sample size.</strong> Every chart shows n on hover. Gaps in groups of "
        "10 images can be one or two images.",
        "<strong>Consistency.</strong> A pattern that appears in several chapters (e.g. in "
        "detected demographics, captions and counterfactual pairs) is much stronger evidence "
        "than a single chart.",
        "<strong>The right baseline.</strong> Compare a group with the other groups under "
        "the same prompt template and prompt mode, not with an ideal 50/50 world.",
      ], title="Judging the strength of evidence")}
      {callout("caution", [
        "All demographic labels are <strong>predicted by detectors</strong>, not ground "
        "truth. Detectors have their own biases and may err more for some groups, which can "
        "create or hide disparities. Race and gender categories are coarse and do not "
        "reflect how people self-identify.",
        "Images whose face could not be detected are excluded from face-based results "
        "(see chapter 2).",
        "The report describes this model, these prompts and these settings. It does not "
        "prove why the model behaves this way, nor how it behaves on other prompts.",
        "Whether a measured difference is <em>harmful</em> is a normative judgement that "
        "requires context; the report supplies the evidence for that judgement.",
      ], title="Limits of every chart in this report")}
    </div>
    <details class="defs"><summary>Glossary of statistics used throughout the report</summary>
      <dl class="glossary">{glossary}</dl>
    </details>
    <p class="note">Charts are interactive: hover for exact values and sample sizes,
    click a legend entry to hide it, double-click to isolate it, drag to zoom and
    double-click the plot to reset. Selectors above a chart change what it shows.</p>
  </section>
"""
