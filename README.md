# Ethics in AI project: measuring fairness in text-to-image models

A framework to audit the fairness of any text-to-image (T2I) generative model. It generates
a controlled set of prompts, renders them with the model under test, extracts demographic and
semantic features from the images, analyzes them statistically and produces an interactive
HTML report on representation, bias and stereotypes.

The report keeps representation, prompt adherence, associations and counterfactual comparisons
separate on purpose: they measure different aspects of fairness and are not collapsed into a
single score.

## Pipeline

```
prompts  ->  images  ->  features  ->  analysis  ->  report
 (CPU)       (GPU)       (GPU)         (CPU)         (CPU)
```

| # | Phase | Module | Input | Output |
|---|-------|--------|-------|--------|
| 1 | Prompt generation | `src/generation/prompt_generator.py` | variable lists in the file | `assets/*_prompts.txt` |
| 2 | Image generation | `src/generation/generate_images.py` | prompt files | `<run>/images/`, `generation_metadata.jsonl`, `config.json` |
| 3 | Feature extraction | `src/extraction/extract_features.py` | a run folder | `<run>/all_metadata.json.gz` |
| 4 | Analysis | `src/analysis/analyzer.py` | metadata + config | `<run>/analysis.json` |
| 5 | Report | `src/report/report_builder.py` | `analysis.json` | `<run>/report/` (HTML) |

Image generation and feature extraction are compute-heavy, so run steps 1-3 in a Colab GPU
runtime (ready-made notebooks are in `notebooks/`) and copy the run folder locally for steps 4-5.

## Repository layout

```
assets/               generated prompt files (git-ignored, regenerate with step 1)
notebooks/            Colab notebooks: imge_generation, feature_extraction, feature_analysis
src/
  run_config.py       shared config.json handling (atomic, resumable, preserves manual keys)
  generation/         prompt_generator.py, generate_images.py, models/ (one adapter per model)
  extraction/         extract_features.py
  analysis/           analyzer.py + chapters.py and the statistics modules
                      (coverage, bias, counterfactual, prompt adherence, embeddings,
                       visual features, synthesis, preprocessing, ...)
  report/             report_builder.py, one report_ch_*.py per chapter, insights/explain texts
                      readme.md: design notes for the report structure
requirements.txt          local analysis / report dependencies
requirements-colab-generation.txt  GPU dependencies for step 2 (generation)
requirements-colab.txt    GPU dependencies for step 3 (extraction)
```

## Installation

Local (analysis and report):

```bash
pip install -r requirements.txt
```

Colab / GPU: generation and extraction need different `transformers` versions, so install
the matching file before each step (each step runs in its own process, no runtime restart needed):

```bash
!git clone <repo-url> && cd progEthics
!pip install -U -r requirements-colab-generation.txt   # before step 2
!pip install -r requirements-colab.txt                 # before step 3
```

Extraction pins `transformers` to 4.49.0 because the Florence-2 remote code breaks on newer
versions; generation needs >= 4.51 for the FLUX.2-klein text encoder (Qwen3).

## Usage

All commands are run from the repository root. `<run>` is the folder holding one audit run
(one model, one set of prompts).

### 1. Prompt generation

Writes one prompt file per mode, one prompt per line, each repeated for the number of images wanted.

```bash
python -m src.generation.prompt_generator --mode 0 --dest assets   # control_prompts.txt
python -m src.generation.prompt_generator --mode 1 --dest assets   # single_var_prompts.txt
python -m src.generation.prompt_generator --mode 2 --dest assets   # double_var_prompts.txt
```

| Mode | Prompts | Purpose |
|------|---------|---------|
| 0 | control: all variables specified | check the accuracy of the face-attribute extractor |
| 1 | single variable (job, workplace, age, gender, race, emotion, continent, status, toxic / positive trait) | base distributions and primary biases |
| 2 | double variable (e.g. gender x job, race x job, age x gender) | intersectional bias and counterfactual comparisons |

The audited vocabularies (jobs, races, ages, emotions, ...) are the lists at the top of
`prompt_generator.py`. Edit them before generating to change what is audited. Options:
`--mode/-m`, `--dest/-d`, `--filename/-f`.

### 2. Image generation

```bash
python -m src.generation.generate_images --model pixart-sigma --out <run>
python -m src.generation.generate_images --list-models
```

Built-in models: `sd15`, `pixart-sigma`, `flux-schnell`, `flux2-klein-4b`, one file each in
`src/generation/models/`. To audit another model add a file there and register it in
`MODELS` (`models/__init__.py`), or pass `--model package.module:Class` (a subclass of `ImageGenerator`); nothing
else in the pipeline changes.

Other options: `--prompts MODE=PATH` (repeatable, defaults to `assets/*_prompts.txt`),
`--seed` (image *i* uses `seed + i`), `--batch-size`, `--limit N` (smoke test).

### 3. Feature extraction

```bash
python -m src.extraction.extract_features --run <run>
python -m src.extraction.extract_features --run <run> --stages faces clip
python -m src.extraction.extract_features --run <run> --merge-only
```

Independent stages, merged into `all_metadata.json.gz`:

| Stage | Model | Features |
|-------|-------|----------|
| `image_metrics` | none | brightness, saturation, contrast |
| `faces` | DeepFace | perceived age, gender, race, emotion per face |
| `florence` | Florence-2 | detailed caption, open-vocabulary objects |
| `clip` | CLIP | normalized image and prompt embeddings |
| `yolo` | YOLO11-nano | COCO objects, number of people |

Steps 2 and 3 are resumable: re-run the same command after a disconnect. Both keep
`<run>/config.json` up to date (model, generation settings, stage status and counters,
extraction models, environment), and the analysis and report read that same file.

### 4. Analysis

Copy the run folder locally. Keep its `images/` folder next to `all_metadata.json.gz` to get
image thumbnails in the qualitative examples of chapter 8.

```bash
python -m src.analysis.analyzer --config <run>/config.json --data <run>/all_metadata.json.gz --output <run>/analysis.json
python -m src.analysis.analyzer --chapters 3 7    # rebuild only chapters 3 and 7
python -m src.analysis.analyzer --list            # when each chapter was last built
```

The analyzer preprocesses the metadata, computes the statistics and writes a versioned JSON
artifact. Each chapter has its own builder (`src/analysis/chapters.py`), so after the first full
run you can rebuild only the chapters you are working on; the others are kept from the existing
artifact, and a missing chapter is always built. If the config or dataset changed, the analyzer
warns about kept chapters built from a different one. Chapter 9 is derived from the other
chapters and is rebuilt on every run. Without arguments the paths default to
`assets/config.json`, `assets/all_metadata.json.gz` and `assets/analysis.json`.

### 5. Report

```bash
python -m src.report.report_builder -i <run>/analysis.json --output-dir <run>/report --no-wait
```

The report step reads only the analysis artifact, so it never reloads the dataset or repeats
expensive analysis. Options: `--input/-i`, `--output-dir/-o`, `--no-open`, `--no-wait`.

### From Python / notebooks

```python
from src.analysis.analyzer import save_analysis
from src.report.report_builder import build_report

artifact_path = save_analysis(data, config, "assets/analysis.json")                   # all chapters
artifact_path = save_analysis(data, config, "assets/analysis.json", chapters=[5, 6])  # partial update
report_path = build_report(artifact_path, open_report=True, wait_for_close=True)
```

## What is measured

Features extracted automatically:

- **Race, gender, age, emotion** of the depicted people: DeepFace
- **Caption and objects**: Florence-2
- **Image / prompt embeddings**: CLIP
- **People count and COCO objects**: YOLO
- **Low-level image properties**: brightness, saturation, contrast

Bias dimensions analyzed (conditional distributions such as P(gender | job)):

- gender, racial, age and emotional bias, conditioned on prompt attributes
- prompt adherence: whether the requested gender, age and race are the ones detected
- toxic, positive and attractiveness stereotypes from captions
- object and item associations
- counterfactual fairness: what changes when a single prompt attribute is swapped

## The report

1. **Executive summary**: the most important findings, disparities and caveats.
2. **Dataset and generation coverage**: prompt design, sample sizes, demographic coverage, data quality.
3. **Representation and diversity**: who appears in the images and how varied the outputs are.
4. **Prompt adherence**: whether images match the requested attributes.
5. **Conditional and intersectional bias**: how attributes and demographic combinations affect outcomes.
6. **Counterfactual fairness**: what changes when one prompt attribute changes and the others stay fixed.
7. **Visual feature and embedding analysis**: differences in image properties and semantic representation.
8. **Caption and semantic association analysis**: recurring attributes, objects and contexts in descriptions.
9. **Cross-analysis and key findings**: evidence combined across measurements to identify robust patterns.
10. **Methodology, limitations and appendix**: definitions, uncertainty, robustness, reproducibility.

The design rationale for each chapter is in [src/report/readme.md](src/report/readme.md).

## Limitations

Demographic labels are *perceived* attributes predicted by automatic models (DeepFace, Florence-2,
CLIP) that have their own errors and biases. The control prompts (mode 0) exist to estimate this
extractor error; interpret group gaps with it in mind. Results describe the audited model under
the chosen prompts, vocabularies and seeds, and do not generalize beyond them.
