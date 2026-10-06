# Step 2 Decision Record

| ID | Decision | Rationale | Consequence |
|---|---|---|---|
| S2-D01 | Use the public Sparkov dataset and disclose that it is synthetic. | It provides transaction-level labels, interpretable features, temporal coverage, and demographic audit attributes. | Results support academic benchmarking but cannot establish production performance. |
| S2-D02 | Preserve the downloaded ZIP and raw CSVs without modification. | Immutable source data and cryptographic hashes improve provenance and reproducibility. | Cleaning and transformations occur in code; raw data is excluded from Git because of size. |
| S2-D03 | Use chronological evaluation rather than random splitting. | Fraud prevalence changes over time and future information must not influence earlier predictions. | The supplied test file remains untouched until final evaluation. |
| S2-D04 | Do not delete IQR-flagged amounts or city populations automatically. | Extreme transaction values and large-city populations are plausible and potentially informative. | Step 3 uses transformations and robust methods rather than blanket row deletion. |
| S2-D05 | Investigate, rather than silently delete, transactions with derived age below 18. | These may be synthetic-generation artifacts or authorized-user scenarios. | Age plausibility is documented and age is reserved primarily for fairness analysis. |
| S2-D06 | Treat `zip` and `cc_num` as identifiers/categories, not continuous quantities. | Their numeric storage does not imply meaningful arithmetic distance. | Raw card number is excluded; ZIP requires categorical treatment. |

