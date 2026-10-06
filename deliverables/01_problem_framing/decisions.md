# Step 1 Decision Record

| ID | Decision | Rationale | Consequence |
|---|---|---|---|
| S1-D01 | Detect fraud at the individual credit-card transaction level. | This directly matches the assignment's finance example and makes the unit of analysis unambiguous. | Account-opening fraud and credit-default prediction are out of scope. |
| S1-D02 | Frame the task as supervised binary classification. | Historical labels identify legitimate (`0`) and fraudulent (`1`) transactions. | Models must output a fraud-risk score before an operational threshold is applied. |
| S1-D03 | Use PR-AUC as the primary model-selection metric. | Fraud prevalence is approximately 0.5%; accuracy can be misleading under this imbalance. | ROC-AUC and accuracy are supporting measures, not selection criteria. |
| S1-D04 | Evaluate ranked review capacity, fraud/value recall, false-positive burden, and expected net benefit. | A statistically strong model may still be operationally or financially unsuitable. | Threshold selection requires explicit review-capacity and cost assumptions. |
| S1-D05 | Treat explainability and group fairness as acceptance criteria. | Fraud controls can deny or delay legitimate transactions and create unequal burdens. | Gender and derived age must remain available for audit even if excluded from the primary model. |

