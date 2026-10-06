# Step 9 — Use of Generative AI

## Summary

Generative AI was used in this capstone as a **development assistant**, not as
part of the delivered solution. The fraud-detection system contains no large
language model (LLM) component: no chatbot, LLM-backed recommender, or
generated-text feature is used for training, scoring, explanation, or fairness
auditing.

## How Generative AI was used

AI coding assistants integrated in the development environment (Cursor and
OpenAI Codex) supported the following tasks:

| Area | Use | Author responsibility |
|---|---|---|
| Code completion | Inline suggestions while writing data-profiling, feature-engineering, modelling, and audit code in `src/` | Accepted only after reading the code and checking it against the leakage and reproducibility rules in `README.md` |
| Unit tests | Drafting test cases in `tests/` for temporal-split integrity, strict-past features, capacity selection, metric consistency, fairness metrics, and artifact presence | Expected values come from the generated artifacts and analytical rules, not from the assistant |
| Documentation and formatting | Editing report prose, summarising generated tables, and scripting the layout of the final report and slide decks | Every reported number is read from files produced by the project code; no metric was written by hand or invented |
| Review | Checking deliverables against the rubric and flagging inconsistencies between reports, slides, and generated results | Findings were verified against the code and artifacts before any change was made |

## What Generative AI was not used for

- Generating, labelling, or augmenting data. All analysis uses the unmodified
  public Sparkov dataset.
- Producing model predictions, metrics, or fairness results. These come only
  from the reproducible scripts listed in `README.md`.
- Making analytical decisions without review. Problem framing, metric choice,
  the temporal evaluation design, model selection, and the deployment
  recommendation were decided by the author and are recorded in each step's
  `decisions.md`.

## Verification safeguards

- All headline results regenerate from versioned code and configuration in
  `src/` and `configs/`.
- The automated test suite (`make test`) checks the critical temporal, leakage,
  metric, and artifact behaviour.
- Reported figures in the report and presentations were cross-checked against
  the machine-readable outputs in `deliverables/`.

## Limitations of AI assistance

AI suggestions can be plausible but wrong, for example by introducing subtle
data leakage or misreporting a metric. This risk was managed by treating every
suggestion as unverified until it was tested or traced to a generated artifact.
