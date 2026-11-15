"""Chapter 9 settings: thresholds, lenses, effect-size families, vocabularies."""

from __future__ import annotations

from typing import Any


SCHEMA = 2
DEFAULT_ALPHA = 0.05
# Results whose smallest compared group is below this are flagged "small n".
DEFAULT_SMALL_N = 30
# Cells/groups smaller than this are not turned into specific results.
DEFAULT_MIN_CELL = 5
# A subgroup whose face-detection rate is this much below the rest (and the
# gap is supported) gets a measurement caveat on every face-based result.
DETECTION_GAP = 0.10
# Requested values followed less often than this weaken results about them.
LOW_ADHERENCE = 0.5
DEMOGRAPHIC = ("gender", "race", "age")
FACE_FEATURES = ("gender", "race", "age", "emotion")
# Numerical features (chapters 6 and 7) that estimate a demographic.
DEMOGRAPHIC_FEATURES = ("detected_age",)
# Prompt attributes that name a social group or a social judgement. An
# association that links one of them to non-requested content is a
# *candidate* stereotype (a pattern for a human to review, not a verdict).
SOCIAL_ATTRIBUTES = ("gender", "race", "age", "continent", "status",
                     "toxic_trait", "positive_trait")
ATTRIBUTE_ORDER = ("gender", "race", "age", "emotion", "job", "workplace", "continent",
                   "status", "toxic_trait", "positive_trait", "prompt_mode")
LENSES: dict[str, dict[str, str]] = {
    "representation": {"label": "Default representation", "source": "Face detector",
                       "link": "#chapter-3-distributions", "chapter": "3.1"},
    "adherence": {"label": "Prompt adherence", "source": "Face detector",
                  "link": "#chapter-4", "chapter": "4"},
    "association": {"label": "Demographic association", "source": "Face detector",
                    "link": "#chapter-5-single-variable", "chapter": "5.1"},
    "intersectional": {"label": "Intersectional deviation", "source": "Face detector",
                       "link": "#chapter-5-intersectional", "chapter": "5.4"},
    "objects": {"label": "Object association", "source": "Object detector",
                "link": "#chapter-5-stereotypical", "chapter": "5.3"},
    "counterfactual": {"label": "Matched counterfactual", "source": "Matched pairs",
                       "link": "#chapter-6", "chapter": "6"},
    "visual": {"label": "Image features", "source": "Pixel statistics",
               "link": "#chapter-7-features", "chapter": "7.1"},
    "caption": {"label": "Caption language", "source": "Caption model",
                "link": "#chapter-8", "chapter": "8"},
    "embedding": {"label": "Embedding separability", "source": "CLIP embedding",
                  "link": "#chapter-7-embeddings", "chapter": "7.2"},
    "diversity": {"label": "Output diversity", "source": "CLIP embedding",
                  "link": "#chapter-3-diversity", "chapter": "3.2"},
    "detection": {"label": "Face detection", "source": "Face detector",
                  "link": "#chapter-2-detections", "chapter": "2.4"},
}
# Conventional small / medium / large thresholds per effect-size family.
FAMILIES: dict[str, dict[str, Any]] = {
    "h": {"label": "Cohen's h", "thresholds": (0.2, 0.5, 0.8)},
    "g": {"label": "Hedges' g", "thresholds": (0.2, 0.5, 0.8)},
    "dz": {"label": "Cohen's d_z", "thresholds": (0.2, 0.5, 0.8)},
    "w": {"label": "Cohen's w", "thresholds": (0.1, 0.3, 0.5)},
    "epsilon2": {"label": "ε² (Kruskal–Wallis)", "thresholds": (0.01, 0.06, 0.14)},
    "silhouette": {"label": "Silhouette", "thresholds": (0.1, 0.25, 0.5)},
}
BANDS = ("negligible", "small", "medium", "large")
# Above the "large" threshold the score keeps growing linearly (so very large
# effects can still be ranked against each other) up to this cap.
MAX_STRENGTH = 6.0
# Caption words that express a value of a detected dimension. Used to link
# caption evidence to face-detector evidence ("nurse" captions saying
# "woman" corroborate "nurse" faces detected as female). Override with
# ``synthesis_concept_lexicon`` in the config.
DEFAULT_CONCEPTS: dict[str, dict[str, tuple[str, ...]]] = {
    "gender": {
        "female": ("woman", "women", "female", "girl", "lady", "dress", "skirt", "lipstick"),
        "male": ("man", "men", "male", "boy", "guy", "gentleman", "beard", "mustache",
                 "moustache"),
    },
    "age": {
        "old": ("elderly", "old", "older", "aged", "senior", "wrinkle", "wrinkled"),
        "young": ("young", "youthful", "teenage", "teenager", "child", "kid"),
    },
    "emotion": {
        "happy": ("smile", "smiling", "happy", "cheerful", "joyful", "grin", "laugh",
                  "laughing", "joy"),
        "sad": ("sad", "somber", "sombre", "sorrow", "melancholy", "tear", "cry", "crying",
                "gloomy"),
        "angry": ("angry", "anger", "furious", "frown", "scowl", "rage"),
        "neutral": ("serious", "neutral", "stern", "expressionless"),
        "fear": ("afraid", "fear", "scared", "anxious", "worried", "nervous"),
    },
}
TIERS = {
    "measurement": "Measurement issue",
    "context": "Context",
    "negligible": "Negligible",
    "descriptive": "Descriptive disparity",
    "supported": "Statistically supported",
}
EVIDENCE_LEVELS = {
    "confirmed": "Counterfactual-confirmed",
    "convergent": "Convergent",
    "supported": "Single source",
}
QUESTIONS = (
    ("representation", "Default representation",
     "When a prompt does not specify gender, race or age, does the model collapse to one group?",
     "#chapter-3"),
    ("adherence", "Quality of service",
     "Is the prompt followed equally reliably for every requested group?", "#chapter-4"),
    ("stereotyping", "Stereotyping",
     "Are jobs, places, traits or regions tied to particular demographics, objects, emotions "
     "or words?", "#chapter-5"),
    ("counterfactual", "Counterfactual consistency",
     "When only a demographic word changes, do unrelated properties of the image change too?",
     "#chapter-6"),
    ("quality", "Representational quality",
     "Are some groups shown with less variety, different framing, visual style or tone?",
     "#chapter-7"),
)
