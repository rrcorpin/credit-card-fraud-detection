# Final Test Evaluation and Business Scenario

## Frozen evaluation protocol

The Step 4 winner, feature list, and hyperparameters were frozen before the
supplied test labels were used. The preprocessor and selected model were refit
on all 1,296,675 `fraudTrain.csv` transactions. Strict-past behavioral features
for test transactions used the combined chronological history, so training
transactions could contribute prior history while no future transaction or
label contributed to an earlier feature.

The supplied `fraudTest.csv` period was evaluated once. It contains 555,719
transactions and 2,145 fraud cases from 21 June through 31 December 2020.

## Final ranking performance

| Metric | Validation | Final test |
|---|---:|---:|
| PR-AUC | 0.8914 | **0.8584** |
| ROC-AUC | 0.9968 | **0.9960** |
| Brier score | 0.00475 | **0.00481** |

PR-AUC falls by 0.0330 on the later test period, while ROC-AUC and Brier score
remain similar. The decline supports the earlier warning about temporal shift
and/or overfitting, but the model preserves strong rare-event ranking.

## Capacity trade-off on final test

| Review capacity | Reviews | Precision | Fraud recall | False-positive rate | Fraud-value recall |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 2,779 | 65.67% | 85.08% | 0.172% | 95.78% |
| **1.0%** | **5,558** | **35.71%** | **92.54%** | **0.645%** | **98.61%** |
| 2.0% | 11,115 | 18.48% | 95.76% | 1.637% | 99.27% |

At the declared 1% scenario, the system detects 1,985 of 2,145 fraudulent
transactions and captures $1,117,603.70 of $1,133,324.68 labelled fraudulent
value. It flags 3,573 legitimate transactions and misses 160 fraud cases.

This is a review-ranking result. It does not imply that every flagged fraud
would be blocked or recovered, that every false positive would be declined, or
that the raw score is a calibrated probability.

The top-1% policy is period-relative: transactions are ranked using only the
test period's scores, without labels, and exactly the highest 1% are selected.
The resulting cutoff is 0.4692 versus 0.5782 on validation. The different
numeric cutoff reflects score-distribution drift and is not label leakage.

## Fairness confirmation

The test period confirms the direction of the validation gender finding:

| Group | TPR | FPR | Precision | Value recall |
|---|---:|---:|---:|---:|
| Female (`F`) | 89.26% | 0.711% | 32.50% | 97.81% |
| Male (`M`) | 96.43% | 0.566% | 40.07% | 99.39% |

The gender TPR gap narrows from 8.65 points in validation to 7.17 points in
test but remains material. The test disparate-impact ratio is 0.898 and the
equalized-odds difference is 0.0717. The result strengthens the conclusion that
fairness requires further investigation rather than a one-off validation fix.

The under-18 test group contains no labelled fraud, so equalized-odds metrics
are undefined for that group. Eligible age bands show a 6.75-point TPR gap and
a disparate-impact ratio of 0.699.

## Illustrative cost and ROI sensitivity

The business analysis compares the model-assisted review scenario with no
intervention. The base assumptions are:

- 70% of the labelled fraud value in reviewed fraud transactions is avoided or
  recovered.
- Each reviewed transaction costs $5.
- Each false positive creates an additional $10 of customer-friction cost.

Under those assumptions during the synthetic test period:

| Item | Result |
|---|---:|
| Avoided fraud loss | $782,322.59 |
| Review and friction cost | $63,520.00 |
| Illustrative net benefit | **$718,802.59** |
| Benefit-cost ratio | **12.32×** |
| Scenario ROI | **1,131.6%** |

Across 27 combinations of 50%–90% recovery, $2–$10 review cost, and $5–$25
false-positive friction cost, the 1% policy has an illustrative net benefit of
$413,896.85–$976,862.33 and a benefit-cost ratio of 3.86×–34.71×.

These are scenario results, not a production ROI forecast. They exclude model
development, integration, infrastructure, monitoring, governance, and broader
customer or reputational effects. The synthetic dataset also cannot establish
real fraud-loss recovery. A real institution must replace every assumption
with internal operational evidence before funding or deployment decisions.

## Recommendation

The evidence supports a controlled shadow-mode pilot, not automated customer
declines. A pilot should validate the ranking on real transaction streams,
measure analyst capacity and recovery, recalibrate scores, investigate subgroup
errors, and obtain model-risk, compliance, privacy, and fraud-operations
approval. Promotion beyond shadow mode should require predefined performance,
fairness, cost, drift, and incident thresholds.
