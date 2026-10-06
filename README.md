# Explainable and Fair Credit Card Fraud Detection

An end-to-end machine-learning project that ranks credit-card transactions by
fraud risk so that a fraud team with limited review capacity can focus on the
transactions most likely to be fraudulent. It covers problem framing, data
auditing, leakage-safe feature engineering, temporal model evaluation,
explainability, fairness auditing, and a business case.

Capstone project for the Postgraduate Diploma in Artificial Intelligence and
Machine Learning, by Ryan Rey Corpin.

**Repository:** [github.com/rrcorpin/credit-card-fraud-detection](https://github.com/rrcorpin/credit-card-fraud-detection)

> **Academic benchmark only.** The model is trained on synthetic data and is
> not intended for production use without validation on real data,
> governance, security, privacy, and regulatory review.

## Key results

Results are from a single evaluation on the supplied test period
(June–December 2020), which was not used for any modelling decision.

| Measure | Result |
|---|---:|
| PR-AUC (primary metric) | 0.8584 |
| ROC-AUC | 0.9960 |
| Fraud cases detected when reviewing the top 1% of transactions | 92.54% (1,985 of 2,145) |
| Fraudulent value detected at 1% review | 98.61% ($1.12M of $1.13M) |
| Precision at 1% review | 35.71% (the maximum possible is 38.6% at this fraud rate) |
| Illustrative net benefit at 1% review, base cost assumptions | $718,803 (benefit-cost ratio 12.3×) |
| Gender gap in fraud recall (female versus male cardholders) | 7.17 percentage points |

The model is a strong ranked review aid, but the confirmed subgroup gap,
uncalibrated scores, and synthetic data mean the recommendation is a governed
shadow pilot, not automated customer declines.

## Problem

Fraud makes up only about 0.5% of transactions, so a model that never flags
fraud would still be 99.5% accurate. The project is therefore framed as
**supervised binary classification used for risk ranking**: each transaction
receives a score, and the highest-scoring share is sent for review. Models are
selected by **PR-AUC** and judged operationally by recall, precision, and
fraud-value capture at a fixed review capacity, alongside false-positive burden
and fairness across demographic groups.

## Dataset

The public [Sparkov Credit Card Transaction Fraud Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
contains synthetic cardholder, merchant, and transaction records.

| Split | Rows | Fraud cases | Fraud rate | Period |
|---|---:|---:|---:|---|
| `fraudTrain.csv` | 1,296,675 | 7,506 | 0.5789% | Jan 2019 – Jun 2020 |
| `fraudTest.csv` | 555,719 | 2,145 | 0.3860% | Jun 2020 – Dec 2020 |

Full-row profiling found no missing values, duplicate transaction IDs, invalid
targets, or invalid coordinates. The full data dictionary is in
[`deliverables/02_data_understanding/data_dictionary.csv`](deliverables/02_data_understanding/data_dictionary.csv).

## Approach

1. **Temporal evaluation.** The training file is split chronologically into
   development (Jan 2019 – Mar 2020) and validation (Mar – Jun 2020) periods.
   The supplied test file was used once, after every choice was frozen.
2. **Leakage-safe features.** Cardholder, merchant, and category history uses
   only strictly earlier transactions, and transactions sharing a timestamp
   never enter one another's history. Encoders and scalers are fitted on
   development data only. Names, addresses, and card numbers are excluded.
3. **Feature engineering.** Log amounts, cyclical time features,
   customer-to-merchant distance, amount relative to the cardholder's history,
   time since the card's previous transaction, and frequency encodings. Mutual information is used for
   feature selection; PCA was evaluated and rejected.
4. **Models.** A prevalence baseline, logistic regression, decision trees,
   random forest, and histogram gradient boosting, with balanced class weights
   and chronological hyperparameter tuning. Gradient boosting performed best.
5. **Explainability.** TreeSHAP (global and local), partial dependence, and ICE
   plots.
6. **Fairness audit.** Selection rate, true- and false-positive rates, and
   precision by gender, age band, and gender-by-age intersection, with 95%
   confidence intervals. Two post-processing mitigations were tested and
   rejected because each improved one fairness measure while worsening others.
7. **Business case.** Cost sensitivity across 27 combinations of recovery
   rate, review cost, and customer-friction cost.

## Repository structure

```text
├── configs/          Versioned settings for every pipeline stage
├── data/
│   ├── raw/          Sparkov CSVs (downloaded separately; see data/raw/README.md)
│   └── processed/    Reserved for derived data
├── deliverables/     Reports, figures, and tables for each capstone step
│   ├── 01_problem_framing/
│   ├── 02_data_understanding/
│   ├── 03_preprocessing_eda_features/
│   ├── 04_model_implementation/
│   ├── 05_explainability_ethics_fairness/
│   ├── 06_presentations/
│   ├── 07_github_repository/
│   ├── 09_generative_ai_use/
│   └── final_submission/   Final report (PDF, DOCX), both presentations, and repository link
├── models/           Fitted preprocessors and classifiers (see models/README.md)
├── notebooks/        EDA.ipynb: executed data-profiling and EDA notebook
├── src/              Profiling, feature, modelling, fairness, and evaluation code
├── tests/            Automated checks for temporal, leakage, metric, and artifact logic
├── Makefile          One command per pipeline stage
└── requirements.txt  Pinned dependency versions
```

## Getting started

Developed and tested on Python 3.14. The raw data needs about 700 MB of disk
space.

```bash
python -m venv .venv
source .venv/bin/activate
make setup
```

Then download the dataset from
[Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) and place
`fraudTrain.csv` and `fraudTest.csv` in `data/raw/sparkov/`.
[`data/raw/README.md`](data/raw/README.md) gives step-by-step instructions for
downloading in a web browser or with the Kaggle CLI, and the checksums for
confirming the files match the ones used here.

## Reproducing the results

Each stage writes its figures, tables, and metrics to `deliverables/` and its
fitted artifacts to `models/`. Every random process uses seed 42.

| Command | Stage |
|---|---|
| `make profile` | Full-row data profile and data dictionary |
| `make step3` | EDA, feature engineering, feature selection, and PCA |
| `make step4` | Train and compare the eight candidate models |
| `make step4-enhancements` | Chronological tuning, feature-set comparison, and calibration |
| `make step5` | SHAP, PDP/ICE, robustness, and fairness audit |
| `make final-eval` | Refit the frozen winner and evaluate the test period |
| `make test` | Run the automated test suite |

`make final-eval` reproduces the one-time test evaluation. Its results should
not be used to revisit model, feature, or threshold choices.

## Reports and presentations

- [Final report (PDF)](deliverables/final_submission/Ryan_Rey_Corpin_Capstone_Final_Report.pdf)
- [Technical presentation](deliverables/final_submission/Ryan_Rey_Corpin_Capstone_Technical_Presentation.pptx)
  and [business presentation](deliverables/final_submission/Ryan_Rey_Corpin_Capstone_Business_Presentation.pptx)
- Detailed step reports, mapped to the capstone rubric, in
  [`deliverables/`](deliverables/README.md)

## Limitations

- The data is synthetic; performance on real transactions is unproven.
- The 1% review capacity is an assumption, not a measured staffing limit. The
  cost model favours a smaller 0.5% queue in 24 of 27 scenarios.
- Scores are rankings, not calibrated probabilities. Isotonic calibration was
  tested but is not part of the frozen model.
- A 98-feature variant outperformed the frozen 25-feature model on validation
  (PR-AUC 0.9166 versus 0.8906) but has no independent test result.
- The dataset has only binary gender and no race or socioeconomic attributes,
  so those fairness audits were not possible.

## Use of Generative AI

AI coding assistants supported development; the solution itself contains no
LLM component. See
[`deliverables/09_generative_ai_use/generative_ai_usage.md`](deliverables/09_generative_ai_use/generative_ai_usage.md).

## License

This repository does not yet include a software license. The dataset remains
subject to its publisher's terms and is not redistributed here.
