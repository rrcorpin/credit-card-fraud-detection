# Step 4 Enhancement — Tuning, Representation, Calibration, and Fairness Transfer

## Purpose

This reproducible enhancement addresses reviewer comments without using the
outer chronological validation block during hyperparameter search. It also
preserves the integrity of the already completed one-time test evaluation: no
new model is promoted using knowledge of the test labels.

## Chronological tuning design

The original development period was divided chronologically into 829,872
inner-training and 207,468 inner-tuning transactions. Four randomized
configurations were sampled for random forest and histogram gradient boosting;
the original flexible boosting configuration was included as a prespecified
reference. The later 259,335-row outer validation block was not used during
search. Early stopping was disabled, removing the original random 10% internal
holdout concern.

| Family | Best inner-tuning PR-AUC | Best configuration |
|---|---:|---|
| Histogram gradient boosting | 0.8737 | learning rate 0.08; 200 iterations; 63 leaves; minimum leaf 25; L2 0.5 |
| Random forest | 0.8283 | 180 trees; depth 20; minimum leaf 20; 50% features; 80% sample |

The search independently selected the same flexible boosting capacity used by
the locked model. When refitted on the full development period, tuned boosting
also outperformed tuned random forest on outer validation (PR-AUC 0.8906 versus
0.8630). Complete candidates are in
`tables/chronological_randomized_search.csv`.

## Representation ablation

Using the tuned booster and identical chronological partitions:

| Representation | Features | PR-AUC | Precision at 1% | Recall at 1% | Fraud-value recall at 1% |
|---|---:|---:|---:|---:|---:|
| Full engineered representation | 98 | **0.9166** | **55.28%** | **93.24%** | **97.66%** |
| Mutual-information selection | 25 | 0.8906 | 54.01% | 91.09% | 96.82% |

The full representation is the predictive winner on outer validation. The
original top-25 model remains the only model evaluated in the locked one-time
test, so these post-review findings are not used to rewrite that test result.
The full representation is recommended for a future shadow candidate and must
be confirmed on genuinely new out-of-time data before promotion.

PCA is explicitly rejected as a predictive branch. Step 3 showed that 13 of 20
numeric components are required for 95% variance, the first two components have
substantial class overlap, categorical signals would be discarded, and the
components are less actionable for fraud investigators.

## Calibration experiment

The booster was trained on inner-training data, an isotonic mapping was fitted
only on the later inner-tuning slice, and both score versions were assessed on
outer validation.

| Score | PR-AUC | Brier loss | Log loss |
|---|---:|---:|---:|
| Raw class-weighted score | **0.8775** | 0.002244 | 0.009177 |
| Isotonic probability | 0.8669 | **0.001654** | **0.007019** |

Recalibration lowers Brier loss by 26.3% and log loss by 23.5%, while isotonic
ties slightly reduce ranking metrics. Therefore the operational policy should
continue ranking raw scores; calibrated probabilities may support probability-
based decisions only after current-period calibration monitoring.

## Fairness mitigation transfer to test

The validation-derived gender cutoffs (F 0.4230; M 0.9105) were applied
unchanged to the later test period.

| Policy | Review rate | Precision | Recall | Value recall | Gender TPR gap | DP ratio |
|---|---:|---:|---:|---:|---:|---:|
| Global test-period top 1% | 1.00% | 35.71% | **92.54%** | **98.61%** | 7.17 pp | 0.898 |
| Validation-derived group cutoffs | 0.83% | **42.01%** | 90.44% | 97.14% | **1.08 pp** | 0.405 |

The TPR gap reduction transfers, but demographic parity, false-positive, and
precision disparities worsen, capacity changes, and group-specific treatment
raises governance concerns. The mitigation remains rejected for deployment.

## Reproduction

```bash
.venv/bin/python -m src.step4_enhancements --project-root .
```

Configuration is in `configs/step4_enhancements.json`; machine-readable tables,
summary JSON, and fitted enhancement artifacts are saved with this report and
under `models/step4_enhancements/`.
