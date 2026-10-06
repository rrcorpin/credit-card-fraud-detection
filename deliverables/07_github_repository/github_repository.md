# Step 7 — GitHub Profile and Upload

## Repository link

**<https://github.com/rrcorpin/credit-card-fraud-detection>**

The repository is public and is organised like an open-source project. The root
[`README.md`](../../README.md) explains the problem, dataset, approach, results,
setup, and how to reproduce every output.

## Required structure

| Required | Location in repository | Contents |
|---|---|---|
| `src/` for scripts | [`src/`](../../src/) | Data profiling, strict-past feature engineering, model training and comparison, Step 4 enhancements, explainability and fairness audit, and final test evaluation |
| `notebooks/` | [`notebooks/EDA.ipynb`](../../notebooks/EDA.ipynb) | Executed data-profiling and exploratory data analysis notebook |
| `data/` | [`data/raw/README.md`](../../data/raw/README.md), `data/processed/` | Download instructions (browser or Kaggle CLI) and SHA-256 checksums for the Sparkov dataset; the raw CSVs are not redistributed |
| `models/` | [`models/`](../../models/) | Fitted preprocessors and classifiers for Step 4, the Step 4 enhancements, and the final model (see [`models/README.md`](../../models/README.md)) |

Supporting folders: `configs/` (versioned settings for each pipeline stage),
`tests/` (automated checks), and `deliverables/` (reports, figures, tables, and
presentations for each rubric step).

## Deliverables

| Deliverable | Location |
|---|---|
| GitHub repository link | <https://github.com/rrcorpin/credit-card-fraud-detection> |
| Final report | [`final_submission/Ryan_Rey_Corpin_Capstone_Final_Report.pdf`](../final_submission/Ryan_Rey_Corpin_Capstone_Final_Report.pdf) and [`.docx`](../final_submission/Ryan_Rey_Corpin_Capstone_Final_Report.docx) |
| Reproducible code | `src/`, `configs/`, `Makefile`, pinned `requirements.txt`, and `tests/` |

## Reproducibility

After downloading the dataset as described in
[`data/raw/README.md`](../../data/raw/README.md):

```bash
python -m venv .venv
source .venv/bin/activate
make setup
make profile step3 step4 step4-enhancements step5 final-eval eda
make test
```

A clean-copy reproduction reproduced every result table and metric file
exactly and regenerated all figures; the 26 automated tests pass.
