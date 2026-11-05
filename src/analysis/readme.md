# Structure for a comprehensive T2I fairness report

I would build the report as a model fairness audit, rather than simply a collection of plots. The goal is to move from describing what the model generates, to identifying systematic disparities, to testing whether those disparities depend on prompt wording, demographic combinations, or the model's internal semantic representations.

Since you already have prompt metadata, extracted image features, captions, and embeddings, you can build a fairly rich report without generating additional images. I would organize it into 10 main sections, with an interactive dashboard where readers can filter by job, gender, race, age, emotion, and other prompt attributes.

## Report overview

01\. Executive summary
The most important findings, disparities, and caveats.

02\. Dataset and generation coverage
Prompt design, sample sizes, demographic coverage, and data quality.

03\. Representation and diversity
Who appears in the generated images and how varied the outputs are.

04\. Prompt adherence
Whether generated images match the requested attributes.

05\. Conditional and intersectional bias
How attributes and demographic combinations affect outcomes.

06\. Counterfactual fairness
What changes when one prompt attribute is changed while others remain fixed.

07\. Visual feature and embedding analysis
Differences in measurable image properties and semantic representation.

08\. Caption and semantic association analysis
What image descriptions reveal about recurring attributes, objects, and contexts.

09\. Cross-analysis and key findings
Combine evidence from multiple measurements to identify robust patterns.

10\. Methodology, limitations, and appendix
Definitions, uncertainty, robustness checks, and reproducibility.

The key design decision is to keep representation, prompt adherence, associations, and counterfactual comparisons separate. They measure different aspects of fairness and should not be collapsed into one score prematurely.

## 1. Executive summary

This is the first page and should be understandable without reading the rest of the report.

### What to include

- Model name, version, generation settings, and audit date.
- Number of prompts, generated images, valid images, and failed feature detections.
- Most important disparities found.
- Attributes and subgroups most affected.
- Strongest and weakest prompt adherence.
- A concise explanation of what the audit can and cannot establish.

### Recommended visualizations

Fairness overview

A compact set of headline metrics: demographic representation, adherence, diversity, and largest observed subgroup gaps.

Largest observed disparities

Ranked horizontal bar chart of the largest measured differences, with uncertainty intervals where available.

Avoid inventing an overall fairness score unless you can justify its normalization, weighting, and interpretation. A summary of several distinct indicators is more informative.

## 2. Dataset and generation coverage

This section explains what was actually tested. It is essential because apparent fairness can be an artifact of an unbalanced prompt set, missing detections, or unequal sample sizes.

### Include

- Number of prompts by mode: single-variable, double-variable, and control.
- Counts by job, workplace, gender, race, age, status, emotion, continent, and other prompt attributes.
- Images generated per prompt and any repeated generations.
- Valid detection rate for each extracted feature.
- Missing captions, embeddings, demographic detections, and failed generations.
- Distribution of generation settings, if they vary.

### Plots

Prompt coverage bar charts

Show the number of prompts and images in each category. Separate prompt counts from image counts.

Feature availability heatmap

Rows are extracted features; columns are prompt modes or demographic groups. Color represents the proportion of valid observations.

Detection success by subgroup

Compare successful and failed detections. This can reveal measurement bias in your analysis pipeline.

Important: distinguish a missing detection from a genuine absence of an attribute. For example, a face detector failing to identify a face does not mean the image contains no person.

## 3. Demographic representation and diversity

This section answers two different questions:

1. Who does the model generate?
2. How varied are the generated outputs?

Keep these questions distinct: a model can generate a balanced number of demographic categories while producing nearly identical images within each category.

### 3.1 Demographic distributions

Analyze detected gender, race, age group, and other available attributes.

Plots to include:

- Overall demographic proportions.
- Stacked bars of demographic composition by prompt type.
- Requested versus detected demographic distributions.
- Demographic proportions by job or workplace.

The most useful visualization is often a requested-versus-generated composition plot, since it immediately shows where the generated distribution diverges from the prompt specification.

Interpretation should account for the prompts you supplied. If 70% of your prompts request one category, a 70% output share is not automatically evidence of bias.

### 3.2 Diversity

Use several complementary measures.

| Measure                            | What it tells you                                                      |
| ---------------------------------- | ---------------------------------------------------------------------- |
| Category entropy                   | How evenly a categorical feature is distributed                        |
| Normalized entropy                 | Allows comparison across features with different numbers of categories |
| Simpson diversity                  | How likely two randomly sampled outputs are to differ in category      |
| Unique caption or object frequency | Variety in described content                                           |
| Embedding dispersion               | Semantic or visual variation between generated outputs                 |
| Pairwise embedding similarity      | How repetitive outputs are within a group                              |

Plots:

- Diversity scores by job, demographic group, and prompt mode.
- Pairwise similarity distributions for each subgroup.
- Rare-category frequency charts.
- Diversity versus sample count.

Be careful with interpretation: higher diversity is not universally better. A prompt that explicitly specifies a narrow appearance should naturally produce less variation than an open-ended prompt.

## 4. Prompt adherence

This section measures whether the image-generation model follows the attributes explicitly requested in the prompt.

For your framework, this should be one of the core sections because it connects the input prompt to the measured image features.

### Include

- Overall adherence by attribute.
- Adherence by prompt mode.
- Adherence by demographic category.
- Adherence to combinations of two attributes.
- Conflicts between specified attributes and detected features.
- Uncertainty or invalid detections.

### Recommended plots

Requested-versus-detected heatmaps

For example, requested gender versus detected gender. Include a separate row or annotation for unknown detections.

Adherence by subgroup

Compare the probability of correctly representing an attribute across categories.

Single versus double-variable prompts

Test whether specifying a second attribute reduces adherence to the first.

For categorical features, define adherence consistently. For a requested category \\(a\\), a simple estimate is:

\\[ A(a)=\frac{\text{valid outputs detected as }a} {\text{all valid outputs requesting }a} \\]

Also report the invalid-detection rate separately, so excluding failed detections does not conceal a subgroup-specific failure.

For semantic adherence, use your existing text-image embeddings or similarity measures as a complementary metric. Embedding similarity is not proof that a particular attribute was represented correctly.

## 5. Conditional distributions and association analysis

This should probably be the largest section of the report. It is where you investigate whether attributes such as gender, race, or age systematically co-occur with other properties of the generated images.

### 5.1 Single-variable–attribute association matrices

Measure the association between one specified prompt attribute and each detected image feature.

Examples:

- Prompt gender versus detected emotion.
- Prompt race versus detected age.
- Prompt gender versus objects mentioned in captions.
- Prompt job versus detected gender, age, or scene attributes.

Plots: heatmaps with one row per prompt attribute category and one column per output feature or category.

Use an appropriate statistic for each data type:

- Categorical–categorical: conditional proportions, Cramér's V, or an appropriate contingency-table effect size.
- Categorical–continuous: group distributions, standardized mean differences, or suitable rank-based effect sizes.
- Embedding vectors: a separately defined distance or distributional comparison.

For binary or categorical outcomes, conditional proportions are often easier to interpret than a matrix of p-values.

### 5.2 Double-variable–attribute association matrices

Investigate what happens when two prompt attributes are specified together, such as white woman, older man, or female doctor.

Examples:

- Gender × race versus detected emotion.
- Gender × job versus caption objects.
- Race × age versus embedding distribution.
- Job × workplace versus generated scene.

Plots:

- Faceted association heatmaps.
- Grouped or stacked conditional-probability bars.
- Difference-in-differences plots comparing joint groups.
- Heatmaps with demographic combinations on the rows and output features on the columns.

Do not simply reuse the single-variable associations and label them intersectional. The joint analysis should investigate whether the combination behaves differently from what would be expected from its constituent attributes.

### 5.3 Stereotypical item associations

Focus on specific associations between prompts and generated content.

Examples include whether certain occupations disproportionately co-occur with particular clothing, objects, environments, emotions, or other caption terms.

Recommended plots:

- Top associated objects or caption terms for each job.
- A job × object association heatmap.
- Lift plots showing which associations are overrepresented relative to a defined baseline.
- A ranked chart of the strongest positive and negative associations.

A useful measure is lift:

\\[ \operatorname{Lift}(A,B)= \frac{P(B\mid A)}{P(B)} \\]

Here, \\(A\\) might be a job category and \\(B\\) a caption term or detected object.

A lift above 1 means the item is more prevalent in the selected group than in the reference population. It does not by itself establish that an association is harmful or stereotypical. That judgment requires contextual interpretation.

### 5.4 Intersectional analysis

This section should go beyond a list of individual associations.

Ask whether the output for a joint group differs from the outputs for its constituent groups. For example, does a particular occupation generate different visual or semantic characteristics for different gender–race combinations?

Plots:

- Intersectional heatmaps, with rows representing group combinations.
- Small-multiple charts comparing the same feature across joint groups.
- Forest plots of group differences with confidence intervals.
- Sample-size annotations for every intersection.

Where sample sizes allow, fit a model with interaction terms, such as:

\\[ Y=\beta_0+\beta_1G+\beta_2R+\beta_3(G\times R) \\]

Here, \\(G\\) and \\(R\\) represent two prompt attributes and \\(Y\\) is a measured output feature. The interaction term tests whether the association involving one attribute changes with the other.

This is particularly useful for distinguishing an intersectional pattern from an ordinary demographic difference.

### 5.5 A common comparison standard

Across these subsections, establish a clear reference distribution. For example, compare groups within the same occupation and prompt template when investigating demographic differences.

Otherwise, a difference attributed to gender might actually reflect differences in the jobs, environments, or prompt wording used for each gender.

## 6. Counterfactual analysis

Counterfactual analysis should have its own section because it asks a more controlled question than ordinary conditional analysis:

What changes when one prompt attribute changes while the rest of the prompt remains fixed?

For example, compare prompts with the same job, workplace, and wording that differ only in the specified gender.

### 6.1 Counterfactual analysis from extracted features

Compare paired prompts or outputs using your categorical and numerical feature columns.

Possible metrics:

- Change in detected attribute category.
- Change in emotion or age estimates.
- Change in object or scene frequency.
- Change in a numerical image feature.
- Difference in adherence rates between counterfactual groups.

Plots:

- Paired difference plots.
- Before-versus-after scatterplots.
- Heatmaps of attribute changes.
- Distribution plots of paired numerical differences.

Keep the prompts matched on everything except the attribute under investigation. Where possible, generate repeated outputs for each condition, because individual image generations can differ substantially.

### 6.2 Counterfactual analysis from embeddings

Use the embeddings to measure whether changing a prompt attribute changes the semantic representation of the output beyond the intended attribute.

Potential measurements include:

- Cosine distance between paired image embeddings.
- Difference in similarity to a shared reference prompt.
- Distances between group embedding distributions.
- Changes in embedding projections or cluster membership.

Plots:

- Distributions of paired embedding distances.
- PCA or UMAP projections colored by counterfactual condition.
- Paired embedding comparisons grouped by occupation.
- Difference between within-condition and between-condition distances.

A crucial caveat: a larger embedding distance does not automatically mean more bias. Changing a requested attribute should change at least some aspects of the image. The interesting question is whether unrelated properties also change systematically, and whether the embedding metric can distinguish those changes from the intended one.

Also, if your model is stochastic, embeddings of separately generated images reflect both prompt effects and random generation variation. Treating paired outputs as true image-level counterfactuals requires care.

## 7. Image features and embedding-space analysis

This section examines output characteristics independently of whether they match a particular prompt.

### 7.1 Image-feature distributions

Include every feature family you extracted that has a meaningful fairness interpretation, but organize them by type.

- Demographic estimates: age, gender, race.
- Expression and emotion.
- Visual properties: brightness, saturation, color, composition, if available.
- Detected objects, people, or environmental features.
- Quality and detection confidence.

Plots:

- Histograms or density plots for numerical features.
- Stacked bars for categorical features.
- Violin or box plots for subgroup comparisons.
- Correlation matrices for numerical features.
- Feature distributions split by prompt mode and demographic group.

Do not treat automated demographic predictions as unquestionable ground truth. Errors in face analysis, especially unequal error rates across groups, can create or amplify apparent model disparities.

### 7.2 Embedding distributions and cluster separability

This subsection explores whether groups occupy different regions of the embedding space.

Plots:

- PCA for a relatively interpretable linear projection.
- UMAP for exploratory nonlinear structure.
- Group-colored scatterplots.
- Cluster composition by demographic or prompt category.
- Within-group versus between-group distance distributions.

You can also measure cluster separability with metrics such as silhouette score, provided the clustering and distance metric are appropriate.

A classifier trained to predict prompt attributes from embeddings can be another diagnostic, but its accuracy should be interpreted carefully: high separability means the information is encoded in the embeddings, not that the encoding is necessarily unfair.

For embedding visualizations, make the same projection available across comparable groups. Fitting a different projection to each group would make visual comparisons unreliable.

## 8. Caption and semantic association analysis

Captions provide a bridge between low-level image features and richer semantic content. They can reveal attributes that are not covered by your structured feature extractors.

### 8.1 Prompt-word associations

Examine which words appear in captions given a prompt attribute or combination.

Examples:

- Occupation versus caption terms.
- Gender versus clothing or activity terms.
- Race versus environment descriptions.
- Prompt mode versus the frequency of contextual details.

Plots:

- Top terms per prompt category.
- TF-IDF or normalized term-frequency heatmaps.
- Term association scores and lift charts.
- Word co-occurrence networks, if the graph remains readable.

### 8.2 Word-attribute associations

Instead of looking only at word frequencies, measure associations between caption terms and detected image attributes.

For example, does a particular word co-occur more often with one detected gender or age category than another, after accounting for the prompt?

Plots:

- Caption term × detected attribute heatmaps.
- Log-odds or lift plots.
- Selected word co-occurrence graphs.
- Caption examples illustrating strong quantitative associations.

Use normalization and minimum-frequency thresholds. Very rare words can produce extreme-looking association scores simply because they appear once or twice.

### 8.3 Qualitative caption examples

Include a small, curated set of examples to help readers understand the quantitative findings.

For each example, display:

- The original prompt.
- The generated image.
- The automatic caption.
- Relevant extracted features.
- The reason the example was selected.

These examples should illustrate an already measured pattern, not substitute for statistical evidence. Since captions are model-generated or automatically inferred, they can themselves contain systematic errors.

## 9. Cross-analysis and key findings

This is where the report becomes more than ten separate collections of charts.

Create a findings section that connects evidence across the previous sections.

For example:

- A demographic representation difference is also associated with different caption terms.
- A job-related association appears in both structured object features and caption analysis.
- A joint demographic disparity is visible in conditional distributions and persists in matched counterfactual comparisons.
- A subgroup has lower adherence and a higher feature-detection failure rate, which weakens confidence in the apparent distribution.

### Recommended visualizations

Evidence matrix

Rows are candidate findings; columns are feature, caption, embedding, adherence, and counterfactual evidence. Show effect direction and strength where comparable.

Ranked effect sizes

Show the largest well-supported differences, with uncertainty intervals and sample sizes.

Effect versus reliability

Compare effect magnitude with the amount of available evidence, highlighting large effects based on small samples.

I would avoid automatically assigning labels such as “fair” and “unfair” to every measured difference. Instead, classify findings as descriptive disparities, statistically supported associations, potential stereotypes, or robustness-tested counterfactual differences. Then explain the evidence behind each conclusion.

## 10. Methodology, limitations, and appendix

This is necessary if you want the report to be reproducible or suitable for an academic project.

### Methodology

Document:

- Prompt templates, variable definitions, and prompt-generation procedure.
- Model version, inference settings, and random seeds.
- Feature extractors and captioning models, including versions.
- Embedding model and similarity metric.
- Statistical tests, effect-size definitions, and confidence-interval methods.
- Treatment of missing detections, repeated generations, and duplicate prompts.
- Multiple-comparison correction, where applicable.

### Limitations

Explicitly discuss:

- Limited prompt coverage and possible sampling imbalance.
- Errors in automated demographic and emotion detection.
- Captioning-model and embedding-model biases.
- Sensitivity to prompt wording and generation randomness.
- Small samples in intersectional groups.
- The distinction between correlation, representation, and normative judgments about fairness.
- Why the audit cannot establish the model's training-data causes or internal causal mechanisms from outputs alone.

### Appendix

Include detailed tables, complete association matrices, feature definitions, prompt lists, extended plots, statistical results, and reproducibility information. Keep the main report focused on interpretable results.

## Making the report interactive

I would build it in Python using Plotly and Dash for a browser-based dashboard, or Plotly with Quarto for a report that combines narrative, code, and interactive figures. Streamlit is another good option if you prefer to build the interface quickly.

A useful structure is a report with a narrative overview plus an interactive analysis explorer.

## Proposed interactive explorer

Example interface specification — the controls below illustrate how the report could be organized, rather than querying your dataset.

Prompt mode

Single-variable prompts

Double-variable prompts

Control prompts

Primary attribute

Gender

Race

Age

Job

Analysis type

Representation and diversity

Prompt adherence

Conditional associations

Counterfactual analysis

Embedding analysis

Caption analysis

Comparison dimension

Job

Workplace

Emotion

Race × gender

CURRENT VIEW CONFIGURATION

### Associations

Prompt mode: Double-variable

Primary attribute: Gender

Comparison: Job

Design this analysis view

### Interactive features worth implementing

| Feature                   | Why it matters                                                      |
| ------------------------- | ------------------------------------------------------------------- |
| Global filters            | Reuse the same filters across all charts                            |
| Click-to-filter           | Select a group in one plot and update related plots                 |
| Hover tooltips            | Show counts, percentages, effect sizes, and uncertainty             |
| Sample-size display       | Make unreliable small groups immediately visible                    |
| Image and caption gallery | Inspect examples behind a statistical result                        |
| Download filtered data    | Allow reproducibility and further analysis                          |
| Export to HTML/PDF        | Produce a shareable report for readers who do not use the dashboard |

For the static deliverable, export the narrative and selected plots to HTML or PDF. Keep interactive plots in HTML because PDF cannot preserve most dashboard interactions.

## Suggested implementation priority

You do not need to build every chart at once. I would implement them in this order:

1. Core: dataset coverage, demographic distributions, diversity, and prompt adherence.
2. Main fairness analysis: single- and double-variable associations, stereotypical associations, and intersectional comparisons.
3. Advanced analysis: matched counterfactuals from extracted features and embeddings.
4. Semantic analysis: caption-word associations, embedding-space separability, and qualitative examples.
5. Synthesis: evidence matrix, robustness checks, uncertainty estimates, and final findings.

One final recommendation: structure the underlying analysis results as a reusable table with columns such as `analysis_type`, `prompt_mode`, `attribute`, `subgroup`, `feature`, `effect_size`, `confidence_interval`, `sample_size`, and `p_value` where applicable. Keep the metric definition and comparison baseline alongside each result. This will make it much easier to build a consistent report, rank findings, add interactive filters, and regenerate the report when you evaluate another model.