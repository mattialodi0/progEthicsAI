"""Build the HTML fairness-audit report."""

from __future__ import annotations

from datetime import datetime
import html
import json
from pathlib import Path
from typing import Any


_CONFIGURATION_KEYS = {
    "T2I_model",
    "model",
    "model_name",
    "model_version",
    "version",
    "generation_settings",
    "generation",
}


def _first_config_value(config: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None and value != "":
            return value
    return "Not specified"


def _generation_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("generation_settings", config.get("generation"))
    if isinstance(settings, dict):
        return settings

    return {
        key: config[key]
        for key in sorted(config)
        if key in _CONFIGURATION_KEYS
        and key not in {"T2I_model", "model", "model_name", "model_version", "version"}
    }


def _summary(
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

    return {
        "model_name": _first_config_value(config, "model_name", "model", "T2I_model"),
        "model_version": _first_config_value(config, "model_version", "version"),
        "generation_settings": _generation_settings(config),
        "audit_date": config.get(
            "audit_date", datetime.now().astimezone().date().isoformat()
        ),
        "num_prompts": prompt_stats.get("num_unique_prompts", 0),
        "generated_images": generated_images,
        "valid_images": max(0, generated_images - failed_features),
        "failed_feature_detections": failed_features,
    }


def _metric_card(label: str, value: Any) -> str:
    return (
        '<div class="metric">'
        f'<span class="metric-label">{html.escape(label)}</span>'
        f'<strong class="metric-value">{html.escape(str(value))}</strong>'
        "</div>"
    )


def build_report(
    data: list[dict[str, Any]],
    analysis_results: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """Write the first chapter of the interactive audit report.

    The report is a standalone HTML file and does not require Plotly or a
    running notebook.  Later chapters can reuse the embedded ``reportData``
    object to add charts and analyses without changing the executive summary.

    Args:
        data: Image metadata records used by the audit.
        analysis_results: Results returned by the analysis collectors.
        config: Audit and report configuration.

    Returns:
        The path of the generated HTML report.
    """
    if not isinstance(data, list):
        raise TypeError("data must be a list")
    if not isinstance(analysis_results, dict):
        raise TypeError("analysis_results must be a dictionary")
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")

    summary = _summary(data, analysis_results, config)
    report_data = {"chapter": "Executive summary", **summary}
    serialized_data = json.dumps(report_data, ensure_ascii=True).replace("</", "<\\/")
    settings_json = json.dumps(summary["generation_settings"], indent=2, ensure_ascii=True)

    output_dir = Path(config.get("report_dir", "report"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / config.get("report_filename", "fairness_report.html")
    if report_path.suffix.lower() != ".html":
        raise ValueError("report_filename must use the .html extension")

    report_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fairness audit - Executive summary</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
    body {{ background: #f4f6f8; color: #17202a; margin: 0; }}
    main {{ margin: 0 auto; max-width: 960px; padding: 2rem 1rem 4rem; }}
    .eyebrow {{ color: #52606d; font-size: .8rem; font-weight: 700;
      letter-spacing: .08em; text-transform: uppercase; }}
    h1 {{ margin-bottom: .35rem; }} h2 {{ margin-top: 2rem; }}
    .subtitle {{ color: #52606d; margin-top: 0; }}
    .metrics {{ display: grid; gap: 1rem; grid-template-columns:
      repeat(auto-fit, minmax(170px, 1fr)); }}
    .metric, section, details {{ background: white; border: 1px solid #d9e2ec;
      border-radius: 8px; box-shadow: 0 2px 5px #102a430d; }}
    .metric {{ padding: 1rem; }} .metric-label {{ color: #52606d; display: block;
      font-size: .9rem; }} .metric-value {{ display: block; font-size: 1.6rem;
      margin-top: .35rem; }} section, details {{ padding: 1rem 1.25rem; }}
    details {{ margin-top: 1rem; }} summary {{ cursor: pointer; font-weight: 700; }}
    pre {{ background: #f4f6f8; border-radius: 6px; overflow: auto; padding: 1rem; }}
    .placeholder {{ border-left: 4px solid #829ab1; color: #52606d; padding-left: 1rem; }}
    .note {{ color: #52606d; font-size: .95rem; }}
  </style>
</head>
<body>
<main>
  <div class="eyebrow">Chapter 1</div>
  <h1>Executive summary</h1>
  <p class="subtitle">Image-generation fairness audit - audit date: {html.escape(str(summary["audit_date"]))}</p>

  <section aria-labelledby="model-heading">
    <h2 id="model-heading">Model and audit configuration</h2>
    <p><strong>Model:</strong> {html.escape(str(summary["model_name"]))}</p>
    <p><strong>Version:</strong> {html.escape(str(summary["model_version"]))}</p>
    <details>
      <summary>Show generation settings</summary>
      <pre>{html.escape(settings_json)}</pre>
    </details>
  </section>

  <h2>Audit coverage</h2>
  <div class="metrics">
    {_metric_card("Prompts", summary["num_prompts"])}
    {_metric_card("Generated images", summary["generated_images"])}
    {_metric_card("Valid images", summary["valid_images"])}
    {_metric_card("Failed feature detections", summary["failed_feature_detections"])}
  </div>

  <section aria-labelledby="fairness-heading">
    <h2 id="fairness-heading">Fairness overview</h2>
    <p class="placeholder">Fairness findings will be added in a later chapter.</p>
  </section>

  <section aria-labelledby="scope-heading">
    <h2 id="scope-heading">What this audit establishes</h2>
    <p>This audit summarizes the generated sample and the feature detections
    available from the configured analysis pipeline. It can identify patterns
    and disparities in this dataset under these prompts and generation settings.</p>
    <p class="note">It cannot establish that the model is fair or unfair in
    general, prove causal bias, or predict results for prompts, populations,
    models, or settings not represented in this audit.</p>
  </section>
</main>
<script>
  // Expose structured results for the interactive chapters added later.
  const reportData = {serialized_data};
</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return report_path