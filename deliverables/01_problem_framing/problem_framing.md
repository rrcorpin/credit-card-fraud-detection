# Step 1 — Problem Understanding and Framing

## 1. Business context

Credit-card issuers and payment providers must evaluate a continuous stream of
transactions while fraud represents only a very small share of total activity.
Missing fraud creates direct or reimbursable financial loss, investigation and
chargeback expense, and loss of customer trust. Conversely, incorrectly
flagging legitimate activity can decline or delay genuine purchases, create
customer-service workload, and disproportionately burden particular customer
groups.

The operational problem is therefore not simply to maximize predictive
accuracy. It is to rank transactions by risk so that a limited intervention or
manual-review capacity captures substantial fraudulent activity and value while
controlling false positives, customer friction, cost, and unfair subgroup
outcomes.

## 2. Problem statement

Develop an explainable machine-learning system that estimates the probability
that an individual credit-card transaction is fraudulent using information
available at or before the transaction time. The resulting risk score will
support a policy that prioritizes transactions for review, challenge, or other
proportionate intervention.

The system will be evaluated chronologically to approximate future operation.
It must demonstrate performance on rare fraud, quantify the operational and
financial consequences of its decision threshold, explain important drivers and
errors, and audit whether fraud protection or false-positive burden differs
materially across demographic groups.

## 3. Data-science task

This is a **supervised binary classification** task:

- Unit of analysis: one credit-card transaction.
- Target: `is_fraud`.
- Positive class (`1`): fraudulent transaction.
- Negative class (`0`): legitimate transaction.
- Model output: continuous fraud-risk score or calibrated probability.
- Decision output: an operational action determined by a threshold or ranked
  review capacity.

Classification is appropriate because historical labelled examples are
available and the required output distinguishes two known outcomes. Anomaly
detection may be considered as an auxiliary experiment, but it is not the
primary framing because labels permit direct supervised learning and
cost-sensitive evaluation.

## 4. Stakeholders and decisions

| Stakeholder | Decision or impact |
|---|---|
| Fraud operations | Which transactions should be prioritized within available review capacity? |
| Risk and finance leaders | Does the system reduce expected loss after investigation and customer-friction costs? |
| Customers | Are legitimate payments unnecessarily challenged or declined, and is fraud protection equitable? |
| Model risk, compliance, and audit | Is the model reproducible, explainable, monitored, and supported by appropriate controls? |
| Data science and engineering | Is the pipeline leakage-safe, temporally valid, calibrated, and maintainable? |

## 5. Prediction boundary and leakage rule

Only information that would be available at or before the transaction timestamp
may be used. Cardholder, merchant, or category history must be calculated from
strictly earlier transactions. Future fraud labels, future aggregates, and
preprocessing fitted on validation or test data are prohibited.

The supplied final test period will remain excluded from feature, model,
hyperparameter, and threshold selection. Its labels may be used only for the
final evaluation. Dataset-level descriptive profiling is allowed but must not
inform model selection.

## 6. Technical success metrics

### Primary model-selection metric

**Area under the precision-recall curve (PR-AUC)** will be the primary model
selection metric. Fraud prevalence is approximately 0.5%, so ordinary accuracy
can be very high for a model that detects no fraud. PR-AUC directly evaluates
the trade-off between capturing fraudulent transactions and maintaining useful
precision under severe class imbalance.

### Operational metrics

- **Recall at a fixed false-positive rate or review rate:** proportion of fraud
  captured under an explicit intervention constraint.
- **Precision at review capacity:** proportion of reviewed transactions that are
  fraudulent when only the top-ranked transactions can be investigated.
- **Fraud-value detection rate:** share of total fraudulent monetary value
  captured.
- **Expected cost or net benefit:** value-weighted false-negative loss plus
  review and false-positive costs under stated assumptions.

### Supporting metrics

- ROC-AUC for ranking discrimination, reported alongside rather than instead of
  PR-AUC.
- Precision, recall, F1 score, and confusion matrix at the selected threshold.
- Brier score and calibration plots to test probability reliability.
- Metric stability across time periods.
- Inference time and model size where operationally relevant.

Accuracy will be reported only for completeness and will not be used to select
the model.

## 7. Business KPIs

The business case will be expressed using measurable, scenario-based KPIs:

| KPI | Definition |
|---|---|
| Fraud capture rate | Detected fraudulent transactions / all fraudulent transactions |
| Fraud-value detection rate | Value of detected fraud / total fraudulent value |
| False-positive rate | Legitimate transactions flagged / all legitimate transactions |
| Customer-friction rate | Legitimate transactions challenged or reviewed / all legitimate transactions |
| Review yield | Detected fraud cases / reviewed transactions |
| Review workload | Transactions requiring intervention per period |
| Net benefit | Avoided fraud loss minus review, intervention, and false-positive costs |
| Cost per detected fraud | Total review/intervention cost / detected fraud cases |

Financial KPIs will not be claimed until cost and recovery assumptions are
explicit. Results will include sensitivity analysis rather than one unsupported
ROI estimate.

## 8. Fairness and responsible-AI success criteria

The model should not provide meaningfully different fraud protection or service
quality without a defensible reason. Audits will compare gender and age groups,
and sufficiently sized intersections, using:

- True-positive-rate/recall parity and fraud-value-detection parity for fraud
  protection.
- False-positive-rate and precision parity for service quality and burden.
- Equalized-odds differences or ratios where appropriate.
- Group sample sizes and uncertainty intervals to avoid overinterpreting noise.

Protected attributes will be retained for auditing even if excluded from the
production feature set. Removing a protected attribute will not be treated as
proof of fairness because other variables may act as proxies.

## 9. Acceptance criteria

The project will be considered technically successful if it:

1. Beats a prevalence-aware dummy baseline on validation and the untouched
   temporal test period using PR-AUC.
2. Produces a useful fraud-capture and review-yield trade-off at a clearly stated
   operating constraint.
3. Demonstrates that the selected threshold has positive expected value across
   a reasonable range of cost assumptions, or transparently concludes that it
   does not.
4. Uses raw scores only for ranking unless a separate, later-slice calibration
   stage demonstrates probability reliability; any probability interpretation
   must report Brier/log-loss evidence and ongoing recalibration requirements.
5. Documents subgroup performance, material disparities, limitations, and at
   least one feasible mitigation experiment.
6. Is reproducible from configuration, source code, and documented data-acquisition
   steps.

No arbitrary numerical target is declared before baseline modelling and cost
assumptions are established. This avoids defining success using unsupported
values or test-set knowledge.

## 10. Scope and limitations

### In scope

- Transaction-level supervised fraud scoring
- Temporal validation and rare-event evaluation
- Leakage-safe behavioral feature engineering
- Multiple-model comparison and tuning
- Threshold and cost analysis
- Explainability, error analysis, and group fairness auditing

### Out of scope

- Direct production deployment or real-time payment authorization
- Claims of effectiveness on a specific bank's customers
- Investigation of fraud rings through graph methods
- Identity verification, anti-money-laundering case management, or legal guilt
  determination

The Sparkov data is synthetic. It supports reproducible experimentation but
cannot by itself establish production performance, legal compliance, or
generalization to a real institution.
