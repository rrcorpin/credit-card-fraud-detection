# Model artifacts

All fitted artifacts are committed (about 10 MB in total) and can be
regenerated from the versioned configurations:

| Folder | Contents | Regenerate with |
|---|---|---|
| `step4/` | Development-period preprocessor, 25-column feature selection, and all eight Step 4 candidate classifiers | `make step4` |
| `step4_enhancements/` | Chronologically tuned boosters (25 and 98 features), inner preprocessor, and isotonic calibrator | `python -m src.step4_enhancements --project-root .` |
| `final/` | Frozen winner refit on all of `fraudTrain.csv`, with its preprocessor and feature order, used once for the final test | `make final-eval` |

`deliverables/04_model_implementation/tables/model_artifact_manifest.csv`
records each Step 4 artifact's size and SHA-256 checksum. A complete scoring
workflow must apply `src.features.engineer_features`, the fitted preprocessor,
the recorded feature selection, and then the classifier, in that order.

The Step 4 and enhancement models are development artifacts. Only `final/`
corresponds to the reported one-time test result, and no artifact is
production-ready.
