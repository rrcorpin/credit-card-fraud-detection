# Raw data

The Sparkov Credit Card Transaction Fraud Dataset is not committed because of
its size (about 500 MB extracted). Source:
[Kaggle — Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection).

Both download options below need a free Kaggle account. The code expects this
layout, relative to the project root:

```text
data/raw/sparkov/fraudTrain.csv
data/raw/sparkov/fraudTest.csv
```

## Option 1: Download in a web browser (no Kaggle CLI needed)

1. Sign in to Kaggle and open the
   [dataset page](https://www.kaggle.com/datasets/kartik2112/fraud-detection).
2. Click **Download** to save the dataset as a ZIP file (about 200 MB).
3. Create the folder `data/raw/sparkov/` inside the project.
4. Extract the ZIP and move `fraudTrain.csv` and `fraudTest.csv` into
   `data/raw/sparkov/`. The ZIP contains only these two files.
5. Verify the checksums as described below.

## Option 2: Download with the Kaggle CLI

Install the CLI with `pip install kaggle`, then create an API token from
**Settings → API** on your Kaggle account and save it as described in the
[Kaggle API documentation](https://www.kaggle.com/docs/api). From the project
root:

```bash
mkdir -p data/raw/sparkov
kaggle datasets download -d kartik2112/fraud-detection \
  -p data/raw/sparkov --unzip
```

## Verify the files

Confirm the downloaded files are identical to the ones used in this project by
comparing their SHA-256 checksums with the table below.

macOS or Linux:

```bash
shasum -a 256 data/raw/sparkov/fraudTrain.csv data/raw/sparkov/fraudTest.csv
```

Windows (PowerShell):

```powershell
Get-FileHash data\raw\sparkov\fraudTrain.csv, data\raw\sparkov\fraudTest.csv -Algorithm SHA256
```

PowerShell prints the checksum in uppercase; the comparison is not case
sensitive.

| File | Rows | SHA-256 |
|---|---:|---|
| `sparkov/fraudTrain.csv` | 1,296,675 | `fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d` |
| `sparkov/fraudTest.csv` | 555,719 | `12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0` |

If a checksum differs, the dataset on Kaggle has changed and the reported
results may not reproduce exactly.

Files in this folder are treated as immutable. All derived data is produced by
the code in `src/`.
