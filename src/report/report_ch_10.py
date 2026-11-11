"""Renderer for chapter 10: methodology, limitations and reproducibility.

The chapter explains *how* every number in the report was produced: the
audit pipeline, the models that turn images into measurements, the statistics
used per chapter, the multiple-testing correction and evidence classification
of chapter 9, and what the audit cannot establish. Facts that depend on the
audited run (image counts, detection rates, thresholds, recorded settings,
measurement caveats) are read from the analysis artifact; the description of
the method itself is static because it is fixed by the code.
"""

from __future__ import annotations

from typing import Any

from .explain import badge, callout, chapter_header, esc, guide, metric_defs, pct

# Model name, role, what it outputs, where it is used, main limitation.
_MODELS = (
    ("DeepFace", "Face analysis",
     "Face detection (MTCNN backend, aligned, a face is required), then per-face "
     "<strong>age</strong> (a number), <strong>gender</strong>, <strong>race</strong> "
     "(six classes) and <strong>emotion</strong> (seven classes) with a probability per class. "
     "The dominant class is used, and age is cut into young (&lt; 30), middle-aged (30–49) and "
     "old (50+). Age in years (4.1, 6.1, 7.1) and the class probabilities (2.4, 4.1) are used "
     "only as checks on those labels.",
     "Chapters 2–7, 9",
     "Trained on photographs of real people, so it can be less accurate on synthetic, stylised "
     "or non-frontal faces and for some groups. Gender is binary, race has six coarse classes "
     "and emotion is read from facial expression only. Every demographic value is a "
     "<em>perceived</em> label, not an identity."),
    ("YOLO (YOLO11-nano, Ultralytics)", "Object detector",
     "Bounding boxes, confidence and COCO class for every detected object. Only the "
     "<code>person</code> boxes are used: they give the <strong>number of people</strong> and "
     "the <strong>share of the image covered by people</strong>.",
     "Chapters 6, 7",
     "The smallest and fastest YOLO variant: it can miss small or occluded people. Its "
     "80-class vocabulary has no concept of many job-related objects."),
    ("Florence-2 (base, Microsoft)", "Captioning and open-vocabulary detection",
     "A free-text <strong>caption</strong> (most detailed caption task, beam search of 3) "
     "and a list of <strong>object labels</strong> from its object-detection task. Labels "
     "and caption are also searched for prompt words that the face model cannot see "
     "(job, workplace, status, traits).",
     "Chapters 5.3, 8, 9",
     "A generative model: captions can omit details, hallucinate objects and reuse "
     "template phrases (“The image shows…”). Its wording about people carries the biases of "
     "its own training data, so caption differences mix the generator's behaviour with the "
     "captioner's."),
    ("CLIP (ViT-B/32, OpenAI weights, via open_clip)", "Joint image–text embedding",
     "A 512-dimensional unit vector per image (and per prompt). Distances between vectors "
     "measure semantic dissimilarity.",
     "Chapters 3.2, 6, 7",
     "CLIP encodes both content and style, and has documented social biases of its own. "
     "Distances are relative: they compare groups with each other, not with an absolute "
     "standard of diversity."),
    ("VADER and NLTK", "Caption text processing",
     "Tokenisation, part-of-speech tagging, lemmatisation, stopword removal and a lexicon "
     "<strong>sentiment</strong> score of the descriptive words of each caption.",
     "Chapter 8",
     "VADER was built for social-media text; some words carry a sentiment they do not have "
     "in an image description (a short list of neutral words is overridden)."),
    ("Pixel statistics", "Low-level image properties",
     "Brightness, saturation and contrast of each image.",
     "Chapter 7",
     "Describes style and lighting, not content; differences can reflect the setting "
     "(an office vs a construction site) rather than the person depicted."),
)

_PIPELINE = (
    ("1", "Prompt generation",
     "Prompts are built from templates such as <em>“A picture of a [race] [job]”</em> in "
     "three <strong>modes</strong>: <em>control</em> (four attributes fixed: gender, race, "
     "age, job), <em>single variable</em> (one attribute changes) and <em>double variable</em> "
     "(two attributes change). Every prompt is rendered several times (the number per prompt is "
     "in chapter 2)."),
    ("2", "Image generation",
     "The audited text-to-image model renders every prompt. Generation settings (seed, "
     "steps, guidance scale, resolution) should be recorded in the metadata so the run "
     "can be repeated."),
    ("3", "Feature extraction",
     "Each image is passed through the models of the table above. Results are stored next "
     "to the image in one metadata file. A face that is not found simply leaves the "
     "face-based fields empty."),
    ("4", "Preprocessing",
     "Whole-word patterns find which attribute values each prompt <em>requests</em> "
     "(gender, race, age, emotion, job, workplace, continent, status, traits). "
     "“man/woman” and “male/female” are merged. Person counts and face/person area ratios are "
     "derived from the detector boxes; an image is <em>valid</em> when DeepFace found a face. "
     "Detected age in years and the class probabilities are averaged over the faces of an "
     "image."),
    ("5", "Analysis",
     "One builder per chapter turns the records into statistics (next section) and writes "
     "them to a versioned JSON artifact. Chapter 9 reads the other chapters' results and "
     "never touches the images."),
    ("6", "Report",
     "This page is rendered from the artifact only: no model is run and no image is read, "
     "so rendering is deterministic and cheap."),
)

_CHAPTER_METHODS = (
    ("2 · Coverage", "Counts per prompt mode, attribute value and prompt; detection rate per "
     "feature and per group (Wilson 95% intervals). Share of faces whose most likely gender, "
     "race or emotion class has a probability of at least 60%.",
     "Shows where the sample is thin, and where the detector fails or is unsure."),
    ("3 · Representation", "Detected gender, race, age and emotion distributions for prompts "
     "that do not request that feature; normalised Shannon entropy; mean pairwise CLIP "
     "distance within a prompt group.",
     "Compare with the other groups, not with an ideal population."),
    ("4 · Adherence", "Share of images whose detected value equals the requested one (the "
     "detector label “latino hispanic” is read as “hispanic”). Confusion matrices per value, "
     "combinations of attributes. Two checks: detected age in years per requested age term "
     "(Hedges' g between neighbouring terms), and adherence recomputed with the class "
     "probabilities (mean probability of the requested class; confident labels only).",
     "A low value can be a model failure or a detector failure; the checks help tell which."),
    ("5 · Associations", "Prompt attribute × detected feature contingency tables: Pearson "
     "χ², Cramér's V (raw and bias-corrected, bootstrap interval) and adjusted standardised "
     "residuals per cell (|r| &gt; 1.96 marks a deviating cell). Object prevalence per context "
     "with Fisher's exact test and lift. Intersections are compared with the expectation "
     "from their constituent groups.",
     "V measures strength, not direction: read the residuals for direction."),
    ("6 · Counterfactuals", "Matched pairs: prompts identical except one attribute value. "
     "Differences of unrelated properties (detected demographics, detected age in years, "
     "number of people, face size, CLIP distance) are tested with paired t-tests, "
     "two-proportion tests and "
     "Newcombe intervals.",
     "Because the rest of the prompt is fixed, template effects are ruled out."),
    ("7 · Visual features", "Kruskal–Wallis tests (ε²) and Hedges' g for pixel/layout "
     "statistics and for detected age in years (prompts without an age only); silhouette score of CLIP embeddings against a label-permutation null; "
     "cross-validated linear probe that tries to read the group from the embedding.",
     "A group that is linearly separable in CLIP space is shown differently."),
    ("8 · Captions", "Lemmatised vocabulary per group, TF-IDF distinctive words, presence "
     "tests of words per group against all others, VADER sentiment with a t-interval, "
     "vocabulary entropy.",
     "Words rest on few captions per group: check n before quoting a word."),
    ("9 · Synthesis", "Every effect in a common table, one Benjamini–Hochberg correction "
     "for the whole audit, a standardised strength, evidence clusters and tiers "
     "(described below).",
     "The only chapter that ranks findings against each other."),
)

_STYLE = """
  <style>
    .method-steps { counter-reset: none; display: grid; gap: .8rem;
      grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr)); margin: 1rem 0; }
    .method-step { background: var(--accent-soft); border-radius: 8px; padding: .8rem 1rem; }
    .method-step h4 { margin: 0 0 .3rem; }
    .method-step .step-no { background: var(--accent); border-radius: 50%; color: #fff;
      display: inline-block; font-size: .8rem; height: 1.4rem; line-height: 1.4rem;
      margin-right: .4rem; text-align: center; width: 1.4rem; }
    .method-step p { font-size: .93rem; margin: 0; }
    .model-table td, .model-table th { font-size: .9rem; }
    .model-table td:first-child { white-space: nowrap; }
    .method-limit { border-left: 4px solid var(--caution); margin: .8rem 0;
      padding: .1rem 0 .1rem 1rem; }
    .method-limit h4 { margin: 0 0 .25rem; }
    .method-limit p { margin: .2rem 0; }
  </style>
"""


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _pipeline() -> str:
    steps = "".join(
        f'<div class="method-step"><h4><span class="step-no">{number}</span>{esc(title)}</h4>'
        f"<p>{text}</p></div>"
        for number, title, text in _PIPELINE
    )
    return f"""
  <section id="chapter-10-pipeline">
    <h3>10.1 The audit pipeline</h3>
    <p>The audit is an <strong>output audit</strong>: it treats the text-to-image model as a
    black box, controls what it is asked for and measures what it returns. It needs no access
    to the model's weights or training data, and so cannot explain <em>why</em> a pattern
    exists, only <em>that</em> it exists.</p>
    <div class="method-steps">{steps}</div>
  </section>"""


def _models(config: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td><strong>{name}</strong><br><span class='note'>{esc(role)}</span></td>"
        f"<td>{output}</td><td>{esc(used)}</td><td>{limit}</td></tr>"
        for name, role, output, used, limit in _MODELS
    )
    declared = config.get("extraction_models")
    declared_html = ""
    if isinstance(declared, list) and declared:
        declared_html = (
            f'<p class="note">Extraction models declared in the audit configuration: '
            f"{esc(', '.join(str(item) for item in declared))}. The table above describes "
            "the extraction code that produced the stored metadata; if the two differ, "
            "trust the code and update the configuration.</p>"
        )
    return f"""
  <section id="chapter-10-models">
    <h3>10.2 Measurement models</h3>
    <p>The audited model is judged through <strong>other models</strong>. Everything the
    report says about “who” or “what” is in an image is the opinion of one of the models
    below, so each is a source of measurement error as well as of information.</p>
    <div class="table-wrap"><table class="model-table">
      <thead><tr><th>Model</th><th>What it produces</th><th>Used in</th>
      <th>Main limitation</th></tr></thead><tbody>{rows}</tbody></table></div>
    {declared_html}
    {callout("caution", [
        "<strong>Detector error is not independent of the audited groups.</strong> A face "
        "model that is less accurate for some groups can create a disparity that is not in the "
        "generator, or hide one that is. Chapter 9 therefore attaches a caveat to every "
        "result about a subgroup with a lower face-detection rate, and cross-checks "
        "face-based findings with a second instrument (objects, captions, counterfactual "
        "pairs) whenever possible.",
        "<strong>Race and gender are coarse perceived categories</strong> chosen because "
        "the detector outputs them. They do not reflect how people identify, and they omit "
        "non-binary people and many ethnic groups.",
    ], title="Why the measurements are uncertain")}
  </section>"""


def _statistics(settings: dict[str, Any], correction: dict[str, Any]) -> str:
    alpha = settings.get("alpha", 0.05)
    small_n = settings.get("small_n", 30)
    families = settings.get("families") or {}
    chapter_rows = "".join(
        f"<tr><th scope='row'>{esc(name)}</th><td>{method}</td><td>{note}</td></tr>"
        for name, method, note in _CHAPTER_METHODS
    )
    family_text = ", ".join(
        f"{esc(value['label'])} ({' / '.join(f'{t:g}' for t in value['thresholds'])})"
        for value in families.values()
    )
    tests = correction.get("tests")
    tests_text = f"{tests:,} tests" if isinstance(tests, int) else "all tests"
    hidden = correction.get("hidden_tests")
    hidden_text = (f", {hidden:,} of them screened candidates (objects and caption words) "
                   "that were tested but not stored" if isinstance(hidden, int) and hidden else "")
    return f"""
  <section id="chapter-10-statistics">
    <h3>10.3 How the analysis works</h3>
    <h4>One comparison standard</h4>
    <p>Almost every result compares a <strong>subgroup with the other values of the same
    prompt attribute</strong> (for example <em>nurse</em> against all other jobs), under the
    same prompt mode, using only images whose prompt does <em>not</em> request the measured
    feature. Without that rule, “female doctors are detected as female” would count as
    bias. Prompts that mention a gender, race or age are therefore used for adherence and
    counterfactuals, not for default-representation questions.</p>
    <h4>Per chapter</h4>
    <div class="table-wrap"><table><thead><tr><th>Chapter</th><th>Method</th>
    <th>How to use it</th></tr></thead><tbody>{chapter_rows}</tbody></table></div>
    <h4>From many tests to a few findings (chapter 9)</h4>
    <ol>
      <li><strong>Common effect size.</strong> Each effect keeps its natural unit (percentage
      points, mean difference, V) with a 95% interval, and is also converted to a
      standardised magnitude: {family_text}. The magnitude is mapped to a
      <em>strength</em> where 1 = small, 2 = medium, 3 = large.</li>
      <li><strong>Multiple-testing correction.</strong> The p-values of {tests_text}{hidden_text}
      are corrected <em>together</em> with the Benjamini–Hochberg false-discovery-rate
      procedure. A result is <em>statistically supported</em> when its q-value is below
      {alpha} <em>and</em> its strength is at least small; this keeps the expected share of
      false discoveries among supported results at or below {pct(alpha, 0)}, and stops
      tiny effects in large samples from counting.</li>
      <li><strong>Conservative ranking.</strong> Results are ranked by the strength at the
      <em>lower</em> end of their interval, so a huge gap seen in 8 images does not outrank a
      solid one seen in 200. The verdict of each fairness question is the band of its
      top-ranked result at that lower end (the largest <em>reliable</em> disparity found),
      reported with the number of results that are reliably large or moderate, because a
      maximum over many tests says little on its own.</li>
      <li><strong>Evidence clusters.</strong> Results that make the same claim about the same
      subgroup are merged. A cluster is <em>counterfactual-confirmed</em> when a matched-pair
      result agrees with a pooled one, <em>convergent</em> when two or more different
      instruments agree, and <em>single source</em> otherwise.</li>
      <li><strong>Tiers.</strong> Every result is <em>statistically supported</em>,
      a <em>descriptive disparity</em> (visible but not corroborated), <em>negligible</em>,
      <em>context</em> (explains other results, not a bias on its own) or a
      <em>measurement issue</em>. Results with fewer than {small_n} images in their smallest
      group carry a “small n” flag.</li>
    </ol>
    {callout("read", [
        "<strong>Size before significance.</strong> With enough images every difference is "
        "“significant”; the strength and its interval say whether it matters.",
        "<strong>Absence of a supported result is not evidence of fairness.</strong> It "
        "means the audit, at this sample size, could not distinguish the group from noise.",
        "<strong>Stereotype candidates</strong> are supported links between a social-group "
        "attribute and non-requested content. They are patterns for a human reviewer, not "
        "verdicts.",
    ], title="Reading the classification")}
    {metric_defs([
        ("Benjamini–Hochberg q-value", "p-value adjusted so that the expected proportion of "
         "false discoveries among results with q ≤ α is at most α.",
         "Use q, not p, to decide whether a result survives the audit."),
        ("Wilson / Newcombe interval", "Score-based 95% intervals for a proportion and for "
         "the difference of two proportions.", "Well behaved with small n and near 0% / 100%."),
        ("Cohen's h", "Effect size for two proportions, 2·(asin√p₁ − asin√p₂).",
         "0.2 small · 0.5 medium · 0.8 large."),
        ("Hedges' g / d<sub>z</sub>", "Standardised mean difference (independent / paired "
         "samples), with small-sample correction.", "0.2 small · 0.5 medium · 0.8 large."),
        ("Cramér's V (bias-corrected)", "Strength of association of a contingency table; "
         "Bergsma's correction removes the upward bias of sparse tables.",
         "Cohen's w = V·√(min(r, c) − 1) is used on the common scale."),
        ("Silhouette vs null", "How well labelled groups cluster in CLIP space compared with "
         "randomly permuted labels.", "Only structure above the null is reported."),
    ], title="Statistics glossary (methods)")}
  </section>"""


def _limitations(report_data: dict[str, Any]) -> str:
    summary = report_data.get("chapter_1") or {}
    synthesis = report_data.get("chapter_9") or {}
    settings = synthesis.get("settings") or {}
    measurement = synthesis.get("measurement") or {}
    affected = measurement.get("affected_supported") or {}
    generated = _number(summary.get("generated_images")) or 0
    valid = _number(summary.get("valid_images")) or 0
    prompts = summary.get("num_prompts")
    per_prompt = (f"about {generated / prompts:.0f} images per prompt"
                  if generated and isinstance(prompts, (int, float)) and prompts else
                  "few images per prompt")
    small_n = settings.get("small_n", 30)

    data_notes = [
        f"<strong>Sample size.</strong> {int(generated):,} images from {esc(prompts)} prompts "
        f"({per_prompt}). Uncertainty is quantified with intervals, but a group of 10 images "
        "moves a percentage by 10 points per image."
    ]
    if affected.get("small_n"):
        data_notes.append(
            f"{affected['small_n']} supported result(s) rest on a smallest group of fewer than "
            f"{small_n} images and are flagged “small n”; treat them as leads to confirm "
            "with more images."
        )
    if generated:
        data_notes.append(
            f"<strong>Face-based results exclude {int(generated - valid):,} images "
            f"({pct((generated - valid) / generated)})</strong> in which no face was found. "
            "These images are not random: see the detection-gap caveats in chapter 9.6."
        )

    measurement_notes = []
    mismatch = measurement.get("label_mismatch_values") or []
    if mismatch:
        measurement_notes.append(
            "<strong>Vocabulary mismatch.</strong> Requested values the detector cannot "
            "output (" + ", ".join(f"{esc(item['attribute'])} = {esc(item['value'])}"
                                   for item in mismatch)
            + ") have an adherence of 0% by construction and are excluded from "
            "quality-of-service rankings."
        )
    low = measurement.get("low_adherence") or []
    if low:
        measurement_notes.append(
            "<strong>Poorly followed requests.</strong> "
            + ", ".join(f"{esc(item['attribute'])} = {esc(item['value'])} ({pct(item['rate'], 0)})"
                        for item in low)
            + ". Results conditioned on these prompts partly describe images that do not "
            "show the requested group."
        )

    blocks = [
        ("Scope of the prompts", [
            "Prompts are short, English and template-based. Real users write longer and more "
            "varied prompts, and the model may behave differently on them.",
            "Only the attributes and values listed in the configuration were tested; "
            "other groups, languages, cultures and intersections are not covered.",
            "The audit measures one model version with one set of generation settings. It does "
            "not describe other versions, settings, safety filters or prompt rewriters.",
        ]),
        ("Data and sample", data_notes),
        ("Measurement", measurement_notes + [
            "All demographic and emotion values are <strong>predictions</strong>; error "
            "rates were not measured on this data and may differ across groups.",
            "Perceived age comes from a regression whose error is several years and whose "
            "values are compressed towards 30–40; age groups (young / middle-aged / old) are "
            "derived from it and are coarse. Many faces lie within a few years of a cut-off, "
            "so results in years (4.1, 6.1, 7.1) are more stable than the groups, but the "
            "years are a relative scale, not literal ages.",
            "Race and emotion labels are often narrow wins between two classes (confidence "
            "table in 2.4). The class probabilities are not calibrated, so "
            "probability-weighted adherence (4.1) is a sensitivity check and not a corrected "
            "estimate; association tests keep using the most likely class.",
            "Emotion and sentiment words describe appearance, not inner states.",
            "Objects and captions come from models with their own vocabularies and biases; "
            "a word absent from a caption is not proof that the object is absent.",
            "Image-level features (CLIP distance, pixel statistics) mix content, style and "
            "composition: a difference may not be about the person.",
        ]),
        ("Statistics", [
            "Images generated from the same prompt are not independent observations; "
            "intervals that treat them as independent can be somewhat optimistic.",
            "The false-discovery rate is controlled over the tests the audit ran, not over "
            "every question one could ask of the data; the correction counts screened "
            "candidates as well as the stored results.",
            "Thresholds for small / medium / large effects are conventions, not measures "
            "of harm.",
            "The silhouette and distance statistics of chapter 7 use a deterministic subsample "
            "of the embeddings to bound computation.",
        ]),
        ("Interpretation", [
            "A disparity is not necessarily a harm: whether it is acceptable depends on the "
            "use, the audience and the baseline one considers fair. The audit supplies "
            "evidence for that judgement; it does not make it.",
            "A match with a real-world statistic (for example the share of women among "
            "nurses) is not an ethical justification either; models can reproduce and "
            "amplify the world they were trained on.",
            "Absence of evidence is not evidence of absence: groups with a null result may "
            "still be treated unfairly in ways this audit could not see.",
        ]),
    ]
    html_blocks = "".join(
        f'<div class="method-limit"><h4>{esc(title)}</h4>'
        + "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul></div>"
        for title, items in blocks
    )
    return f"""
  <section id="chapter-10-limitations">
    <h3>10.4 Limitations</h3>
    <p>Items that quote numbers are computed from this audit; the others hold for any run of
    this pipeline.</p>
    {html_blocks}
  </section>"""


def _reproducibility(report_data: dict[str, Any]) -> str:
    summary = report_data.get("chapter_1") or {}
    coverage = report_data.get("chapter_2") or {}
    modes = coverage.get("mode_counts") or {}
    settings = summary.get("generation_settings") or {}
    recorded = (
        "Recorded: " + ", ".join(f"{esc(key)} = {esc(value)}" for key, value in settings.items())
        if settings else
        badge("not recorded", "mid") + " Seed, steps, guidance scale and resolution are "
        "missing from the metadata, so these images cannot be regenerated exactly."
    )
    mode_text = ", ".join(f"{esc(mode)}: {count}" for mode, count in modes.items()) or "n/a"
    return f"""
  <section id="chapter-10-reproducibility">
    <h3>10.5 Reproducibility</h3>
    <table><tbody>
      <tr><th scope="row">Audited model</th><td>{esc(summary.get("model_name", "?"))}
        (version: {esc(summary.get("model_version", "?"))})</td></tr>
      <tr><th scope="row">Audit date</th><td>{esc(summary.get("audit_date", "?"))}</td></tr>
      <tr><th scope="row">Images by prompt mode</th><td>{mode_text}</td></tr>
      <tr><th scope="row">Generation settings</th><td>{recorded}</td></tr>
      <tr><th scope="row">Analysis</th><td>Deterministic given the metadata and the
        configuration: bootstrap, permutation and subsampling steps use fixed seeds. The
        artifact stores a fingerprint of both, and the analyzer warns when a kept chapter was
        built from a different dataset or configuration.</td></tr>
      <tr><th scope="row">Commands</th><td><code>python -m src.analysis.analyzer</code> builds
        the data, <code>python -m src.report.report_builder assets/analysis.json</code>
        renders this report from it.</td></tr>
    </tbody></table>
    <p class="note">To audit another model, generate images for the same prompts, run the
    extraction notebook on them and rerun the two commands: every statement in chapters 1
    and 9 is computed from the artifact and updates automatically.</p>
  </section>"""


def render_chapter_ten(report_data: dict[str, Any], config: dict[str, Any] | None = None) -> str:
    """Render the methodology chapter from the analysis artifact payload."""
    synthesis = report_data.get("chapter_9") or {}
    config = config or {}
    return f"""
  {_STYLE}
  {chapter_header(10, "Methodology, limitations and explanations",
      "How were these numbers produced, and how far can they be trusted?",
      "<p>This chapter documents the audit so that a reader can judge it: the pipeline, the "
      "models that turn images into measurements, the statistics behind every chapter, how "
      "findings are classified, and what the evidence cannot show. Read it before quoting a "
      "number from the report.</p>")}
  {_pipeline()}
  {_models(config)}
  {_statistics(synthesis.get("settings") or {}, synthesis.get("correction") or {})}
  {_limitations(report_data)}
  {_reproducibility(report_data)}
"""
