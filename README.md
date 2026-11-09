# A framework to measure fairness for text-to-image generative models

## Overview
The code in this repository helps to analyze the fairness of any Text-to-Image model.
The process is composed of the following phases:
1) prompts genereation
2) images generation
3) features extraction
4) feature analysis
5) report generation
Once they are over, the final result will be a report containing informations, evaluations, conclusions about the model fairness.

## Prompts generation
## Images generation
## Features extraction

Prompt generation is cheap (CPU); generation and extraction are the compute-heavy steps
(GPU). Clone the repo in a Colab runtime and run them there; only the resulting folder is
needed locally.

```bash
!git clone <repo-url> && cd progEthics && pip install -r requirements-colab.txt
# 0) prompts (skip to reuse the files already in assets/)
!python -m src.generation.prompt_generator --mode 0 --dest assets   # control_prompts.txt
!python -m src.generation.prompt_generator --mode 1 --dest assets   # single_var_prompts.txt
!python -m src.generation.prompt_generator --mode 2 --dest assets   # double_var_prompts.txt
RUN=/content/drive/MyDrive/ethics/PixArtSigma
!python -m src.generation.generate_images --model pixart-sigma --out $RUN   # images + metadata
!python -m src.extraction.extract_features --run $RUN                       # -> all_metadata.json.gz
```

- `src/generation/prompt_generator.py` builds the prompts of one mode (0 = control, 1 = single
  variable, 2 = double variable) from the variable lists at the top of the file, one prompt per
  line, repeated for the number of images wanted. Edit those lists to change the audited
  jobs, races, etc. before generating.
- `src/generation/generate_images.py` reads the prompt files (`assets/*_prompts.txt`, one
  per prompt mode) and writes `images/` and `generation_metadata.jsonl`. It is an *example*
  adapter: to audit another model add an entry to `MODELS` or pass `--model module:Class`
  (subclass of `ImageGenerator`); nothing else in the pipeline changes.
- `src/extraction/extract_features.py` runs independent stages (`image_metrics`, `faces`
  [DeepFace], `florence` [caption + objects], `clip` [embeddings], `yolo` [objects]) and merges
  them into a single `all_metadata.json.gz`. Choose stages with `--stages`.
- Both steps are resumable (re-run the same command after a disconnect) and keep
  `<run>/config.json` up to date (model, generation settings, stage status and counters,
  extraction models, environment). The analysis and report read that same file.

Locally, once `<run>/all_metadata.json.gz` exists:

Copy the run folder locally (keep its `images/` folder next to `all_metadata.json.gz` to get
image thumbnails in the qualitative examples of chapter 8).

```powershell
python -m src.analysis.analyzer --config <run>/config.json --data <run>/all_metadata.json.gz --output <run>/analysis.json
```
## Feature Analysis and reporting

Analysis and HTML rendering are separate steps. The analysis step reads the
metadata, performs preprocessing and feature statistics, and writes a
versioned JSON artifact. The report step reads only that artifact, so it does
not reload the dataset or repeat expensive analysis.

From the repository root:

```powershell
python -m src.analysis.analyzer                  # build the data for every chapter
python -m src.analysis.analyzer --chapters 3 7   # rebuild only chapters 3 and 7
python -m src.analysis.analyzer --list           # when each chapter was last built
python -m src.report.report_builder assets/analysis.json --no-wait
```

Each report chapter has its own data builder (`src/analysis/chapters.py`), so
after the first full run you can rebuild just the chapters you are working on.
The others are kept from the existing artifact. A chapter missing from the
artifact is always built. If you change the config or the dataset, the
analyzer warns about kept chapters that were built from a different one.
`--config`, `--data` and `--output` override the default `assets/` paths.

For library or notebook use:

```python
from src.analysis.analyzer import save_analysis
from src.report.report_builder import build_report

artifact_path = save_analysis(data, config, "assets/analysis.json")                   # all chapters
artifact_path = save_analysis(data, config, "assets/analysis.json", chapters=[5, 6])  # partial update
report_path = build_report(artifact_path, open_report=True, wait_for_close=True)
```

Features we can measure automatically:
- race: DeepFace
- gender: DeepFace
- demographic: DeepFace 
- emotion: DeepFace
- caption: Florence-2
- items: Florence-2
- feature vector: CLIP

Biases we are interested in reporting:
- race
- gender
- demographic
- culture
- socioeconomic
- actractiveness
- emotion


### Report

1. Executive summary - The most important findings, disparities, and caveats.
2. Dataset and generation coverage - Prompt design, sample sizes, demographic coverage, and data quality.
3. Representation and diversity - Who appears in the generated images and how varied the outputs are.
4. Prompt adherence - Whether generated images match the requested attributes.
5. Conditional and intersectional bias - How attributes and demographic combinations affect outcomes.
6. Counterfactual fairness - What changes when one prompt attribute is changed while others remain fixed.
7. Visual feature and embedding analysis - Differences in measurable image properties and semantic representation.
8. Caption and semantic association analysis - What image descriptions reveal about recurring attributes, objects, and contexts.
9. Cross-analysis and key findings - Combine evidence from multiple measurements to identify robust patterns.
0. Methodology, limitations, and appendix - Definitions, uncertainty, robustness checks, and reproducibility.



