"""Chapter 10: static description of the audit method (fixed by the code)."""


# Model name, role, what it outputs, where it is used, main limitation.
MODELS = (
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
PIPELINE = (
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
CHAPTER_METHODS = (
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
