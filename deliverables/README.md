# Capstone Deliverables

Each numbered folder maps to one capstone rubric step and contains the written
report for that step, its decision record, and the figures and tables generated
by the code in `../src/`. Submission-ready files are in
[`final_submission/`](final_submission/).

## Rubric map

| Step | Rubric area | Points | Primary evidence |
|---:|---|---:|---|
| 1 | Problem understanding and framing | 10 | [`01_problem_framing/problem_framing.md`](01_problem_framing/problem_framing.md) |
| 2 | Data collection and understanding | 10 | [`02_data_understanding/data_understanding.md`](02_data_understanding/data_understanding.md) |
| 3 | Preprocessing, EDA, and feature engineering | 10 | [`03_preprocessing_eda_features/eda_feature_engineering_report.md`](03_preprocessing_eda_features/eda_feature_engineering_report.md) |
| 4 | Model implementation and comparison | 20 | [`04_model_implementation/model_implementation_report.md`](04_model_implementation/model_implementation_report.md) |
| 5 | Critical thinking, ethical AI, and bias auditing | 20 | [`05_explainability_ethics_fairness/bias_fairness_analysis.md`](05_explainability_ethics_fairness/bias_fairness_analysis.md) |
| 6 | Final presentation and communication | 10 | Two reviewed 8–12-slide decks in [`06_presentations/`](06_presentations/) |
| 7 | GitHub profile and upload | 15 | Public repository at [github.com/rrcorpin/credit-card-fraud-detection](https://github.com/rrcorpin/credit-card-fraud-detection), structured with `src/`, `notebooks/`, `data/`, and `models/`; see [`07_github_repository/github_repository.md`](07_github_repository/github_repository.md) |
| 9 | Use of Generative AI (optional) | — | [`09_generative_ai_use/generative_ai_usage.md`](09_generative_ai_use/generative_ai_usage.md) |

## Step 1 evidence

- [`01_problem_framing/decisions.md`](01_problem_framing/decisions.md): explicit framing and metric decisions
- Clear financial-services business context
- Supervised binary-classification task definition
- Unit of analysis, target, prediction boundary, and leakage rule
- Primary metric: PR-AUC
- Operational technical metrics and business KPIs
- Stakeholder impacts, acceptance criteria, fairness criteria, scope, and
  limitations

## Step 2 evidence

- [`02_data_understanding/data_understanding.md`](02_data_understanding/data_understanding.md): reviewed written deliverable
- [`02_data_understanding/decisions.md`](02_data_understanding/decisions.md): dataset, integrity, outlier, and evaluation decisions
- [`02_data_understanding/data_dictionary.csv`](02_data_understanding/data_dictionary.csv): complete 23-variable dictionary
- [`02_data_understanding/data_profile.json`](02_data_understanding/data_profile.json): machine-readable full-row profile
- [`../notebooks/EDA.ipynb`](../notebooks/EDA.ipynb): executed analysis notebook
- [`../src/data_profile.py`](../src/data_profile.py): reusable profiling implementation
- [`../configs/data_profile.json`](../configs/data_profile.json): versioned configuration
- [`../tests/test_data_profile_artifacts.py`](../tests/test_data_profile_artifacts.py): integrity tests

## Step 3 evidence

- [`03_preprocessing_eda_features/eda_feature_engineering_report.md`](03_preprocessing_eda_features/eda_feature_engineering_report.md): reviewed EDA and feature-engineering report
- [`03_preprocessing_eda_features/decisions.md`](03_preprocessing_eda_features/decisions.md): explicit analytical decisions and rationale
- `03_preprocessing_eda_features/figures/`: eleven reproducible publication-ready figures
- `03_preprocessing_eda_features/tables/`: EDA, selection, importance, and PCA tables
- [`03_preprocessing_eda_features/analysis_summary.json`](03_preprocessing_eda_features/analysis_summary.json): machine-readable results
- [`03_preprocessing_eda_features/feature_manifest.json`](03_preprocessing_eda_features/feature_manifest.json): candidate and selected feature specification
- [`../src/features.py`](../src/features.py): reusable strict-past feature and preprocessing implementation
- [`../src/step3_analysis.py`](../src/step3_analysis.py): reproducible analysis and figure generation

## Step 4 evidence

- [`04_model_implementation/model_implementation_report.md`](04_model_implementation/model_implementation_report.md): validation protocol, model comparison, operating analysis, selection, and limitations
- [`04_model_implementation/decisions.md`](04_model_implementation/decisions.md): explicit modelling decisions and consequences
- `04_model_implementation/metrics.json`: machine-readable complete metrics and environment versions
- `04_model_implementation/tables/`: model comparison, capacity metrics, and selected-model validation scores
- `04_model_implementation/figures/`: six reproducible validation figures
- [`../configs/step4_models.json`](../configs/step4_models.json): all candidate hyperparameters and evaluation settings
- [`../src/modeling.py`](../src/modeling.py) and [`../src/step4_modeling.py`](../src/step4_modeling.py): reusable modelling and experiment implementation
- `../models/step4/`: fitted preprocessor, feature manifest, and eight serialized models (committed; see [`../models/README.md`](../models/README.md))
- [`../tests/test_modeling.py`](../tests/test_modeling.py) and [`../tests/test_step4_artifacts.py`](../tests/test_step4_artifacts.py): metric, threshold, isolation, and artifact checks
- [`04_model_implementation/enhancements/reviewer_response.md`](04_model_implementation/enhancements/reviewer_response.md): chronological tuning, 98- versus 25-feature comparison, calibration, and fairness-mitigation transfer
- [`04_model_implementation/final_test/final_test_report.md`](04_model_implementation/final_test/final_test_report.md): one-time final test evaluation and illustrative cost sensitivity

## Step 5 evidence

- [`05_explainability_ethics_fairness/bias_fairness_analysis.md`](05_explainability_ethics_fairness/bias_fairness_analysis.md): final-report-ready explainability, bias, fairness, mitigation, and governance section
- [`05_explainability_ethics_fairness/decisions.md`](05_explainability_ethics_fairness/decisions.md): explicit audit and mitigation decisions
- `05_explainability_ethics_fairness/audit_summary.json`: machine-readable protocol, robustness, fairness, mitigation, and explainability results
- `05_explainability_ethics_fairness/tables/`: group metrics with Wilson intervals, intersections, SHAP results, local cases, mitigation comparisons, and temporal checks
- `05_explainability_ethics_fairness/figures/`: eleven reproducible SHAP, PDP/ICE, fairness, mitigation, and robustness figures
- [`../configs/step5_fairness.json`](../configs/step5_fairness.json): audit configuration and eligibility thresholds
- [`../src/step5_fairness.py`](../src/step5_fairness.py): reproducible audit implementation using SHAP and Fairlearn
- [`../tests/test_step5_fairness.py`](../tests/test_step5_fairness.py): uncertainty, capacity, missing-attribute, mitigation, and artifact checks

## Step 6 evidence

- [`06_presentations/Ryan_Rey_Corpin_Capstone_Technical_Presentation.pptx`](06_presentations/Ryan_Rey_Corpin_Capstone_Technical_Presentation.pptx): 12-slide technical deck
- [`06_presentations/Ryan_Rey_Corpin_Capstone_Business_Presentation.pptx`](06_presentations/Ryan_Rey_Corpin_Capstone_Business_Presentation.pptx): 10-slide executive deck
- [`final_submission/Ryan_Rey_Corpin_Capstone_Final_Report.docx`](final_submission/Ryan_Rey_Corpin_Capstone_Final_Report.docx) and [`final_submission/Ryan_Rey_Corpin_Capstone_Final_Report.pdf`](final_submission/Ryan_Rey_Corpin_Capstone_Final_Report.pdf): submission-ready report

## Step 7 evidence

- [`07_github_repository/github_repository.md`](07_github_repository/github_repository.md): repository link, required-structure map, deliverables, and reproduction commands
- [`final_submission/Ryan_Rey_Corpin_Capstone_GitHub_Repository.txt`](final_submission/Ryan_Rey_Corpin_Capstone_GitHub_Repository.txt): repository link for the submission portal
- [`../README.md`](../README.md): open-source project README with setup and reproduction instructions

## Step 9 evidence

- [`09_generative_ai_use/generative_ai_usage.md`](09_generative_ai_use/generative_ai_usage.md): how AI assistants were used during development

## Final submission

[`final_submission/`](final_submission/) contains only the files submitted for
grading: the final report (DOCX and PDF), both presentations, and the GitHub
repository link. They follow the
`Your_Name_Assignment_Name` naming convention and are listed with their
verification status in
[`final_submission/submission_manifest.md`](final_submission/submission_manifest.md).
