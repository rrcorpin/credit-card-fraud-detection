# Step 4 Decision Log

| Decision | Choice | Rationale and consequence |
|---|---|---|
| Task | Supervised binary classification | `is_fraud` is a labelled binary outcome at transaction level. |
| Holdout policy | Do not access the supplied test file in Step 4 | Prevents model, feature, and threshold choices from adapting to final test labels. |
| Validation | Later 20% of the original training period | Tests temporal generalization and matches the prediction setting better than a random split. |
| Primary metric | Average precision (PR-AUC) | Appropriate for a 0.5931% validation prevalence and rewards useful minority-class ranking. |
| Supporting metrics | ROC-AUC, Brier score, log loss, F1, precision, recall, FPR, and fraud-value recall | Captures ranking, score quality, classification, workload, and financial exposure. |
| Initial feature branch | Step 3 top-25 mutual-information features | Provides a documented, development-only selection branch and reduces training complexity. A later controlled ablation found the full 98-feature branch stronger on outer validation (0.9166 versus 0.8906 PR-AUC), so the compact branch is not claimed to be predictively optimal. |
| Protected attributes | Exclude gender and age from primary models | Reserve them for Step 5 auditing; removal alone is not assumed to guarantee fairness. |
| Imbalance handling | Balanced class weights; no validation resampling | Makes rare fraud consequential during fitting without distorting evaluation prevalence. |
| Baseline | Dummy prior plus logistic regression | Establishes no-skill and interpretable linear reference points. |
| Nonlinear candidates | Pruned decision tree, random forest, histogram gradient boosting | Tests progressively richer nonlinearities and interactions on tabular data. |
| Hyperparameter exploration | Initial controlled comparison plus development-only randomized search | Four randomized draws per top family plus the prespecified booster were tuned on a chronological inner split; outer validation stayed clean and early stopping was disabled. |
| Provisional winner | Flexible histogram gradient boosting | Highest validation PR-AUC: 0.8914. |
| Operational scenario | Rank and review exactly the top 1% | Provides a transparent capacity-constrained comparison; it is an assumption, not a final business recommendation. |
| Tie handling | Stable score ordering and an exact review count | Makes fixed-capacity metrics deterministic even when scores tie. |
| Probability use | Raw scores for ranking; isotonic probabilities only with monitoring | Later-slice isotonic recalibration reduced outer-validation Brier loss by 26.3% but slightly reduced PR-AUC. |
| Final model status | Provisional | Step 5 must explain and audit it; final test performance remains unseen. |
