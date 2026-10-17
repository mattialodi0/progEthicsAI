# A framework to measure fairness for text-to-image generative models

## Two-step analysis and reporting

Analysis and HTML rendering are separate steps. The analysis step reads the
metadata, performs preprocessing and feature statistics, and writes a
versioned JSON artifact. The report step reads only that artifact, so it does
not reload the dataset or repeat expensive analysis.

From the repository root:

```powershell
python -m src.analysis.analyzer
python -m src.report.report_builder .\assets\analysis.json --no-wait
```

For library or notebook use:

```python
from src.analysis.analyzer import save_analysis
from src.report.report_builder import build_report

artifact_path = save_analysis(data, config, "assets/analysis.json")
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

Other statistics:
- diversity (both intra-group and coverage)
- consistency
- wording influence


## Report

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


demographics distributions
diversity
prompt adherence
conditional distributions
    single variable-attribute association matrices
    double variable-attribute association matrices
    items stereotypical association
    intersectional analysis
    couterfactual analysis
        from feats
        from embeddings
image feats ditributions
embeddings
    cluster separability
captions
    prompt-word associations
    word-attributes associations
