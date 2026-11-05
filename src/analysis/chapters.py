"""One data builder per report chapter.

Every builder takes the *preprocessed* records and the config and returns the
JSON-serializable ``report_data["chapter_N"]`` payload. Builders never read
another chapter's output, so any subset of chapters can be regenerated on its
own (see ``analyzer.save_analysis``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from .bias_statistics import collect_bias_stats
from .counterfactual_statistics import build_counterfactual_analysis
from .coverage import build_coverage
from .general_statistics import demographic_composition, diversity_metrics, invalid_images
from .other_statistics import build_caption_analysis
from .prompt_adherence import prompt_adherence
from .prompt_statistics import collect_prompt_stats
from .visual_features import build_visual_feature_analysis

ChapterBuilder = Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, Any]]


def build_summary(data: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Chapter 1: executive-summary metadata and headline counts."""
    prompt_stats = collect_prompt_stats(data, config)
    generated_images = prompt_stats["num_images"]
    failed_features = invalid_images(data)["count"]
    settings = config.get("generation_settings", config.get("generation", {}))
    return {
        "model_name": next(
            (config[key] for key in ("model_name", "model", "T2I_model")
             if config.get(key) not in (None, "")),
            "Not specified",
        ),
        "model_version": next(
            (config[key] for key in ("model_version", "version")
             if config.get(key) not in (None, "")),
            "Not specified",
        ),
        "generation_settings": settings if isinstance(settings, dict) else {},
        "audit_date": config.get(
            "audit_date", datetime.now().astimezone().date().isoformat()
        ),
        "num_prompts": prompt_stats["num_unique_prompts"],
        "generated_images": generated_images,
        # "valid" means at least one face was detected (see preprocessing).
        "valid_images": max(0, generated_images - failed_features),
        "failed_feature_detections": failed_features,
    }


def build_demographic_diversity(
    data: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Chapter 3: demographic composition and diversity metrics."""
    diversity = diversity_metrics(data, config)
    return {
        "dimensions": ["gender", "race", "age", "emotion"],
        "composition": demographic_composition(data, config),
        "diversity": diversity,
        "prompt_modes": sorted(diversity["available_prompt_modes"]),
    }


def build_prompt_adherence(
    data: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Chapter 4: requested-versus-detected attribute adherence."""
    return prompt_adherence(data, [], config)


CHAPTER_BUILDERS: dict[str, ChapterBuilder] = {
    "chapter_1": build_summary,
    "chapter_2": build_coverage,
    "chapter_3": build_demographic_diversity,
    "chapter_4": build_prompt_adherence,
    "chapter_5": collect_bias_stats,
    "chapter_6": build_counterfactual_analysis,
    "chapter_7": build_visual_feature_analysis,
    "chapter_8": build_caption_analysis,
}

CHAPTER_TITLES = {
    "chapter_1": "Executive summary",
    "chapter_2": "Dataset and generation coverage",
    "chapter_3": "Representation and diversity",
    "chapter_4": "Prompt adherence",
    "chapter_5": "Conditional and intersectional bias",
    "chapter_6": "Counterfactual fairness",
    "chapter_7": "Visual feature and embedding analysis",
    "chapter_8": "Caption and semantic association analysis",
}


def normalize_chapter(name: str | int) -> str:
    """Accept ``3``, ``"3"``, ``"ch3"`` or ``"chapter_3"``; return ``"chapter_3"``."""
    text = str(name).strip().lower()
    for prefix in ("chapter_", "chapter", "ch_", "ch"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    key = f"chapter_{text}"
    if key not in CHAPTER_BUILDERS:
        raise ValueError(
            f"Unknown chapter {name!r}; expected one of "
            + ", ".join(key.split("_")[1] for key in CHAPTER_BUILDERS)
        )
    return key
