# Step 2 — Data Collection and Understanding

## 1. Dataset selection and provenance

The project uses the public **Sparkov Credit Card Transaction Fraud Dataset**,
downloaded from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection).
It contains synthetic cardholder, merchant, and transaction records created for
fraud-detection experimentation. It was selected because each row is an
individual financial transaction, the target is available, the two-year time
span permits chronological evaluation, fraud is realistically rare, and the
data contains interpretable temporal, monetary, merchant, geographic, and
demographic variables.

The synthetic origin protects real cardholders and makes the analysis
reproducible, but it also limits external validity. Patterns or performance in
this dataset cannot be assumed to represent a particular bank, country, payment
network, or current fraud environment.

### Local acquisition and integrity

The unmodified archive and extracted CSVs are stored under `data/raw/` and
excluded from version control because of their size.

| File | Size | SHA-256 |
|---|---:|---|
| `sparkov-fraud-detection.zip` | approximately 202 MB | `4e32829b9ba5a6b17af707c513c15204011044df0e8971e431cedca3d8c0a8a1` |
| `fraudTrain.csv` | approximately 335 MB | `fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d` |
| `fraudTest.csv` | approximately 143 MB | `12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0` |

The ZIP integrity test passed before extraction. The reproducible profiler
recomputes the CSV fingerprints whenever it runs.

## 2. Reproducible profiling method

The profile is generated from every row—not a notebook sample—using:

```bash
.venv/bin/python -m src.data_profile --project-root .
```

The implementation reads each file in configured chunks of 200,000 rows and
writes:

- `deliverables/02_data_understanding/data_profile.json`: machine-readable counts, schema,
  hashes, distributions, integrity checks, and profiling methodology.
- `deliverables/02_data_understanding/data_dictionary.csv`: complete variable dictionary.

The configuration is stored in `configs/data_profile.json`. The same process is
called from `notebooks/EDA.ipynb`, so interactive and command-line results use
one implementation.

## 3. Dataset overview

| Split | Transactions | Fraud | Legitimate | Fraud rate | Imbalance | Period |
|---|---:|---:|---:|---:|---:|---|
| Train | 1,296,675 | 7,506 | 1,289,169 | 0.5789% | 1 fraud : 171.8 legitimate | 2019-01-01 00:00:18 to 2020-06-21 12:13:37 |
| Test | 555,719 | 2,145 | 553,574 | 0.3860% | 1 fraud : 258.1 legitimate | 2020-06-21 12:14:25 to 2020-12-31 23:59:34 |
| **Total** | **1,852,394** | **9,651** | **1,842,743** | **0.5210%** | **1 fraud : 190.9 legitimate** | **2019-01-01 to 2020-12-31** |

There are 23 raw columns, one of which (`Unnamed: 0`) is a sequential CSV export
index. After removing that index, the data contains 21 candidate/source
variables and one target.

The test fraud prevalence is about 33.3% lower than the training prevalence.
This is early evidence of temporal distribution shift and supports
chronological, rather than random, evaluation. The test labels are described
here for dataset documentation only; they remain prohibited from feature,
model, hyperparameter, and threshold selection.

## 4. Feature families and types

| Family | Variables | Analytical treatment |
|---|---|---|
| Target | `is_fraud` | Binary integer: 0 legitimate, 1 fraudulent |
| Time | `trans_date_trans_time`, `unix_time` | Chronological ordering; derive hour, weekday, recency, and past-only history |
| Transaction value | `amt` | Continuous US-dollar amount; examine log/robust transformations without deleting legitimate high values |
| Merchant | `merchant`, `category`, `merch_lat`, `merch_long` | Categorical merchant context and numeric location |
| Cardholder geography | `city`, `state`, `zip`, `lat`, `long`, `city_pop` | Mixed categorical/numeric; geography treated as potential demographic proxy |
| Cardholder attributes | `gender`, `dob`, `job` | Gender and derived age support fairness auditing; occupation is a possible proxy feature |
| Direct/entity identifiers | `cc_num`, `first`, `last`, `street`, `trans_num` | Direct identifiers excluded as raw predictors; `cc_num` may key strictly past-only behavior; `trans_num` supports integrity checks |
| Export artifact | `Unnamed: 0` | Drop before modelling |

Raw integer storage does not always imply a continuous variable. In particular,
`cc_num` is an entity key and `zip` is categorical.

## 5. Cardinality and allowed values

The training file contains:

- 983 synthetic card identifiers
- 693 merchants
- 14 transaction categories
- 894 cities and 51 state/DC abbreviations
- 970 ZIP codes
- 494 occupation labels
- 968 dates of birth
- Two gender labels: `F` and `M`

The 14 transaction categories are:

`entertainment`, `food_dining`, `gas_transport`, `grocery_net`, `grocery_pos`,
`health_fitness`, `home`, `kids_pets`, `misc_net`, `misc_pos`, `personal_care`,
`shopping_net`, `shopping_pos`, and `travel`.

The test file contains the same merchants, gender labels, and transaction
categories. It contains 50 state/DC values rather than 51 because Delaware does
not appear in the test period. This is a minor categorical-distribution change
that the future encoder must handle safely.

## 6. Missing values, duplicates, and validity checks

Full-row profiling found:

- Zero missing values in either file.
- Zero invalid target values; all labels are 0 or 1.
- Zero duplicate transaction identifiers within either file.
- Zero transaction-identifier overlap between the train and test files.
- A sequential export index in both files; it is redundant and will be removed.
- Zero invalid timestamps or dates of birth.
- Zero non-positive transaction amounts or city populations.
- Zero cardholder or merchant coordinates outside valid latitude/longitude
  bounds.
- Zero dates of birth after their corresponding transaction timestamp.

Age derived from date of birth and transaction time ranges from approximately
13.9 to 95.6 years in training and 15.4 to 96.2 years in testing. There are
15,555 training transactions and 6,292 test transactions associated with a
derived age below 18. These are transaction counts, not unique people. They may
represent synthetic-generation artifacts or authorized-user scenarios and will
be investigated rather than silently deleted.

Duplicate checks use exact sets of deterministic 64-bit pandas hashes of
transaction IDs. Hash collision risk is negligible but not mathematically zero;
this implementation detail is recorded in the JSON profile.

## 7. Numeric distributions and outliers

### Transaction amount

| Split | Minimum | Median | 99th percentile | Maximum | Tukey-IQR flags |
|---|---:|---:|---:|---:|---:|
| Train | $1.00 | $47.52 | $545.99 | $28,948.90 | 67,290 (5.19%) |
| Test | $1.00 | $47.29 | $519.85 | $22,768.11 | 27,778 (5.00%) |

The amount distribution is strongly right-skewed. The training upper Tukey fence
is $193.38 and the test upper fence is $193.08; because the lower fences are
negative and the observed minimum is $1.00, the flags are upper-tail values.
High-value payments may be genuine or highly informative for fraud and will not
be removed solely because they are statistical outliers. Step 3 will compare a
`log1p` or robust transformation and inspect fraud prevalence by amount range.

### City population

| Split | Minimum | Median | 99th percentile | Maximum | Tukey-IQR flags |
|---|---:|---:|---:|---:|---:|
| Train | 23 | 2,456 | 1,577,385 | 2,906,700 | 242,674 (18.72%) |
| Test | 23 | 2,408 | 1,577,385 | 2,906,700 | 104,151 (18.74%) |

City population is also heavily right-skewed. Large-city values are meaningful,
not data errors, so they will be transformed, binned, or robustly modelled
rather than deleted.

The observed coordinate ranges fall inside valid geographic bounds. Coordinates
will later support a customer-to-merchant distance feature; univariate IQR rules
are not appropriate for geographic longitude or latitude.

## 8. Data dictionary

The complete 23-variable dictionary—including role, analytical type, units or
format, allowed values/ranges, and description—is stored at:

`deliverables/02_data_understanding/data_dictionary.csv`

The dictionary explicitly distinguishes direct identifiers, entity keys,
quasi-identifiers, predictive variables, protected audit attributes, ordering
variables, and the target.

## 9. Initial implications for modelling

1. **Severe imbalance:** accuracy is inappropriate as the primary metric;
   PR-AUC, recall at capacity, precision at capacity, and value-based measures
   are required.
2. **Temporal shift:** the lower future-period fraud rate and categorical changes
   require chronological validation and drift analysis.
3. **No missingness does not eliminate pipeline safeguards:** encoders and
   production-oriented preprocessing should still handle unseen categories and
   future missing values safely.
4. **Identifiers require care:** names, street address, raw card number,
   transaction ID, and export index must not be direct predictive features.
5. **Outliers are potentially informative:** amount and population tails should
   not be automatically removed.
6. **Protected/proxy attributes:** gender and age enable fairness auditing;
   geography and occupation may act as proxies and require sensitivity checks.
7. **Past-only behavior:** any cardholder or merchant aggregate must use only
   transactions strictly preceding the scored transaction.

## 10. Limitations

- The dataset is synthetic, and its generative rules may introduce patterns not
  present in live payment networks.
- Only binary gender labels are available, which does not represent the full
  range of gender identities.
- The data does not document all operational processes that produced the fraud
  labels, investigation delays, losses, recoveries, or customer interventions.
- Cost and ROI analysis will therefore require explicit scenarios and
  sensitivity analysis rather than claims about actual bank savings.
- Full profiling establishes structure and integrity but is not a substitute for
  the visual and relational EDA planned for Step 3.
