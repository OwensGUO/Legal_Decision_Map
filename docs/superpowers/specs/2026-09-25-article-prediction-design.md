# Article Prediction and Unified Metrics Design

## Scope

Add defendant-level conviction-article prediction to the existing charge and sentence model.
CAIL uses `meta.relevant_articles`; CMDL uses each target defendant's
`outcomes[].judgment[].article`. CMDL's case-level `relevant_articles` are not copied to every
defendant because they cannot be attributed reliably. A reserved `sentencing_articles` field
remains empty until defendant-level supervision is available.

## Labels and leakage control

Article labels are normalized as `criminal_law:<article>[:<paragraph>...]`; for example, CAIL
article `234` becomes `criminal_law:234`, and CMDL `303-2` becomes
`criminal_law:303:2`. The processed metadata contains an `article_vocabulary`.

Training uses `fact_strict` when present so cited charges and articles cannot leak into any task.
Older processed fixtures without strict text continue to fall back to `fact_conservative`.

## Model and training

The predictor adds a multi-label article head. It concatenates the pooled fact representation
with sigmoid charge probabilities, giving the article task a differentiable, soft dependency on
the charge task. Sentence-by-charge predictions additionally consume sigmoid article
probabilities, so sentence supervision can use the predicted legal basis without hard error
propagation. Article supervision uses binary cross entropy and has a configurable loss weight.

Static prediction JSONL adds true/predicted article vectors, probabilities, and labels. The
majority baseline predicts the most frequent training article so all experiment variants retain
the same output schema.

## Required evaluation

Charge and article prediction are multi-label tasks. For each task, report:

- accuracy as exact-set match;
- macro precision;
- macro recall;
- macro F1.

Also report micro precision, recall, and F1 plus Hamming accuracy.

Sentence prediction remains a mixed classification/regression task. The required four
classification metrics operate on a deterministic final-sentence class:

- `death`, `life`, `detention`, `control`, `exempt`, or `unknown`;
- fixed-term bins `0-6`, `7-12`, `13-24`, `25-36`, `37-60`, `61-120`, and `121+` months.

Sentence accuracy, macro precision, macro recall, and macro F1 are reported over those classes.
Existing eligible-case MAE, log-MAE, and three-month tolerance accuracy remain supplementary
regression metrics.

## Compatibility and verification

Evaluation retains legacy charge metric aliases (`macro_f1`, `micro_f1`, `exact_match`) and can
read older prediction rows without article fields. New model-generated rows always contain all
three task outputs. Unit tests cover label extraction, normalization, soft hierarchical heads,
article loss routing, prediction export, and hand-checked classification metrics.
