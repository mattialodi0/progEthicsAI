from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .preprocessing import preprocess
from .prompt_statistics import collect_prompt_stats
from .general_statistics import collect_general_stats
from .bias_statistics import collect_bias_stats
from ..report.report_ch_2 import build_coverage
from ..report.report_ch_3 import build_demographic_diversity
from ..report.report_ch_7 import build_visual_feature_analysis
from .other_statistics import build_caption_analysis
from .counterfactual_statistics import build_counterfactual_analysis


def analyze(data: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """
    Compute all data required by the report and return a JSON-serializable artifact.

    Args:
        data: The data to analyze.
        config: The configuration for the analysis.
    """
    verbose = config.get('verbose', 0)
    if verbose > 0:
        print("Running fairness analysis...")
    
    # Preprocess the data
    preprocessed_data = preprocess(data)

    # Collect prompt statistics
    prompt_stats = collect_prompt_stats(preprocessed_data, config=config)

    # Collect general statistics
    general_stats = collect_general_stats(preprocessed_data, config=config)
    bias_stats = collect_bias_stats(preprocessed_data, config=config)


    analysis_results = {
        'prompt_stats': prompt_stats,
        'general_stats': general_stats,
        'bias_stats': bias_stats
    }

    # These are report-specific data transformations, but are computed here so
    # rendering never needs the original metadata.
    report_data = {
        "chapter_1": _build_summary(preprocessed_data, analysis_results, config),
        "chapter_2": build_coverage(preprocessed_data, config),
        "chapter_3": build_demographic_diversity(analysis_results, config),
        "chapter_4": general_stats.get("prompt_adherence", {}),
        "chapter_5": bias_stats,
        "chapter_6": build_counterfactual_analysis(preprocessed_data),
        "chapter_7": build_visual_feature_analysis(preprocessed_data, config),
        "chapter_8": build_caption_analysis(preprocessed_data, config),
    }
    artifact = {
        "schema_version": 1,
        "config": config,
        "analysis_results": analysis_results,
        "report_data": report_data,
    }
    # Fail at analysis time rather than producing an artifact that the report
    # builder cannot load later.
    json.dumps(artifact, ensure_ascii=True)
    return artifact


def _build_summary(
    data: list[dict[str, Any]],
    analysis_results: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    prompt_stats = analysis_results.get("prompt_stats", {})
    general_stats = analysis_results.get("general_stats", {})
    invalid_images = general_stats.get("invalid_images", {})
    generated_images = prompt_stats.get("num_images", len(data))
    failed_features = invalid_images.get("count", 0)
    if not isinstance(generated_images, int) or generated_images < 0:
        raise ValueError("prompt_stats.num_images must be a non-negative integer")
    if not isinstance(failed_features, int) or failed_features < 0:
        raise ValueError("general_stats.invalid_images.count must be a non-negative integer")
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
        "num_prompts": prompt_stats.get("num_unique_prompts", 0),
        "generated_images": generated_images,
        "valid_images": max(0, generated_images - failed_features),
        "failed_feature_detections": failed_features,
    }


def save_analysis(
    data: list[dict[str, Any]],
    config: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Compute the analysis once and persist it for a later report build."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(analyze(data, config), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":

    config_path = './assets/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    metadata_json_path = './assets/all_metadata.json.gz'
    with gzip.open(metadata_json_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    
    config["verbose"] = 2
    save_analysis(data, config, "./assets/analysis.json")