# Article Prediction and Unified Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add defendant-level conviction-article prediction and require accuracy, macro precision, macro recall, and macro F1 for charge, article, and sentence tasks.

**Architecture:** Normalize source article labels into the data schema, train a soft charge-conditioned multi-label article head, and feed its probabilities into the sentence head. Export all predictions in one static JSONL and evaluate multi-label charge/article outputs plus deterministic categorical sentence bins, while retaining regression metrics.

**Tech Stack:** Python 3.12, PyTorch 2.10, NumPy 2.2, pytest, Ruff.

## Global Constraints

- CAIL article supervision comes from `meta.relevant_articles`.
- CMDL defendant-level supervision comes only from `outcomes[].judgment[].article`.
- CMDL case-level `relevant_articles` must not be copied to individual defendants.
- Model inputs prefer `fact_strict` to prevent charge/article leakage.
- All three prediction tasks must report accuracy, macro precision, macro recall, and macro F1.
- Sentence regression metrics remain available in addition to categorical metrics.

---

### Task 1: Article labels in the processed data

**Files:**
- Create: `src/legal_landscape/data/articles.py`
- Modify: `src/legal_landscape/data/schema.py`
- Modify: `src/legal_landscape/data/cail.py`
- Modify: `src/legal_landscape/data/cmdl.py`
- Modify: `src/legal_landscape/data/build.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `normalize_criminal_article(value: object) -> str`,
  `CaseUnit.conviction_articles`, `CaseUnit.sentencing_articles`, and metadata
  `article_vocabulary`.

- [x] Write tests proving CAIL `234` becomes `criminal_law:234`, CMDL `303-2` becomes
  `criminal_law:303:2`, and case-level CMDL articles do not become defendant labels.
- [x] Run `pytest tests/test_data.py -q` and confirm the new assertions fail because article fields
  are absent.
- [x] Implement normalization, adapter extraction, schema serialization, and vocabulary output.
- [x] Re-run `pytest tests/test_data.py -q` and confirm it passes.

### Task 2: Hierarchical article head and article loss

**Files:**
- Modify: `src/legal_landscape/models/heads.py`
- Modify: `src/legal_landscape/models/predictor.py`
- Modify: `src/legal_landscape/models/losses.py`
- Test: `tests/test_models_losses.py`

**Interfaces:**
- Consumes: integer `num_articles` and multi-hot `article_targets`.
- Produces: `article_logits`, `article_probabilities`, and `LossBreakdown.article`.

- [x] Write tests for article output shape, charge-conditioned article logits, article-conditioned
  sentence predictions, and article BCE activation/empty-mask behavior.
- [x] Run `pytest tests/test_models_losses.py -q` and confirm failures identify the missing head and
  loss.
- [x] Add the soft hierarchical heads and article loss routing with configurable `article` weight.
- [x] Re-run `pytest tests/test_models_losses.py -q` and confirm it passes.

### Task 3: Training targets and static prediction export

**Files:**
- Modify: `src/legal_landscape/training/train.py`
- Modify: `configs/model/qwen35_9b_qlora.yaml`
- Test: `tests/test_training.py`

**Interfaces:**
- Consumes: processed `conviction_articles` and metadata `article_vocabulary`.
- Produces: static rows containing `true_articles`, `predicted_articles`,
  `article_probabilities`, and human-readable article labels.

- [x] Write tests showing model text prefers `fact_strict` and static rows export multi-label
  article predictions.
- [x] Run `pytest tests/test_training.py -q` and confirm failures are caused by missing article
  arguments/fields.
- [x] Add article targets, loss wiring, distributed prediction gathering, run metadata, and majority
  article baseline output.
- [x] Re-run `pytest tests/test_training.py -q` and confirm it passes.

### Task 4: Required metrics for all three tasks

**Files:**
- Modify: `src/legal_landscape/evaluation/static_metrics.py`
- Modify: `scripts/evaluate_model.py`
- Test: `tests/test_evaluation.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `multilabel_metrics`, `multiclass_metrics`, `sentence_class`, and prefixed static metric
  keys for charge, article, and sentence.

- [x] Write hand-checked tests for accuracy, macro precision, macro recall, and macro F1 on both
  multi-label and multi-class inputs; add an end-to-end CLI assertion for all twelve required keys.
- [x] Run the focused tests and confirm they fail because the metrics are absent.
- [x] Implement reusable metrics, sentence bins, static aggregation, bootstrap endpoints, and
  backward-compatible charge aliases.
- [x] Re-run the focused tests and confirm they pass.

### Task 5: Documentation and repository verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents: label sources, strict input policy, hierarchical prediction, sentence bins, and metric
  names.

- [x] Update the README to describe article prediction and the required evaluation outputs.
- [x] Run `python -m compileall -q src scripts tests`.
- [x] Run `pytest -q`.
- [x] Run `ruff check .`.
- [x] Review `git diff --check` and `git status --short` for unintended changes.
