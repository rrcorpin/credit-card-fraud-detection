# Step 5 Decision Log

| Decision | Choice | Rationale and consequence |
|---|---|---|
| Audit split | Chronological validation only | Preserves the supplied test for final evaluation and avoids adapting governance claims to test labels. |
| Operating policy | Global top-1% review queue | Matches Step 4's declared scenario and permits equal-workload comparisons. |
| Explainability | TreeSHAP, PDP, and ICE | Combines global, local, average, and heterogeneous model-behavior views. |
| SHAP sample | 1,500 fraud plus 1,500 legitimate validation rows | Makes rare fraud patterns visible; importance is conditional on this enriched sample. |
| SHAP interpretation | Log-odds model contributions, not causal effects | Prevents explanations from being presented as proof or intervention effects. |
| Primary audit attributes | Reported binary gender and derived age bands | These are the demographic attributes available in Sparkov and were excluded from model inputs. |
| Race audit | Unavailable | The dataset contains no race field; no proxy is invented. |
| Socioeconomic audit | Unavailable | City population is not a valid measure of socioeconomic status. |
| Contextual proxy audit | Development-derived city-population tertiles | Screens geography-related variation without relabelling it as socioeconomic status. |
| Uncertainty | Wilson 95% intervals for rate metrics | Prevents point estimates—especially small age/intersection groups—from being overinterpreted. |
| Intersection eligibility | At least 5,000 rows and 20 fraud cases | Reduces unstable comparisons while retaining material gender-by-age groups. |
| Fairness definitions | DP difference/ratio, EO difference/ratio, TPR, FPR, precision, value recall | Separates review burden, protection, and review yield; no single metric is treated as universal. |
| Mitigation 1 | Equal review rate within gender at fixed total workload | Tests demographic-parity post-processing; rejected because the TPR gap remains. |
| Mitigation 2 | Validation-tuned equal-opportunity thresholds at fixed workload | Quantifies an upper-bound trade-off; rejected because other disparities worsen and evaluation is optimistic. |
| Protected-attribute thresholds | Diagnostic only | Would require separate validation plus legal, compliance, and governance approval. |
| Current disposition | Do not claim fairness or production readiness | Material gender, age, and intersectional gaps require investigation and additional mitigation experiments. |

