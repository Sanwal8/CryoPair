# Input formats

## Patient features for fitting

Save NumPy archives with named arrays using `numpy.savez`. Store identifiers as Unicode strings so the archives open with `allow_pickle=False`.

| File | Arrays |
| --- | --- |
| FFPE or frozen features | `patient` (unique IDs), `features` (finite floating-point matrix, patients × features) |
| Expression | `patient` (unique IDs), `gene` (unique column names), `values` (finite floating-point matrix, patients × genes) |

The two files for each fit must contain exactly the same patients. Row order can differ. Each inner validation fold needs at least 10 patients and the other folds together need at least 30. Use the full set of original FFPE development features to fit a study-style reader. The CRISP slide predictor expects a reader with 2,560 feature coefficients per gene. See the [synthetic example](../tests/test_fitting.py) for how to write the archives and apply the fitted coefficients.

## Tile and feature arrays for training

Prepare the arrays before running the trainer. The study used 224 × 224 RGB tiles at 0.5 µm per pixel. All teacher features come from the original CRISP encoder. The trainer reads arrays directly from disk.

| File | Shape and meaning |
| --- | --- |
| `patients.csv` | One row per patient with unique `patient` and `role`. Use `train` for training rows. Supply `cancer` for shuffled pairing. |
| `ffpe_mean.npy` | All patients × 2,560. Mean original FFPE feature over the retained tiles for each patient, float32. |
| `sample_features.npy` | All patients × 16 × 2,560 in the study. Eight features per preparation in adequately tiled patients, FFPE followed by frozen, stored as float16. Used for per-feature standardization. |
| `done.npy`, `valid.npy` | One boolean per patient per file. A training row must satisfy both flags. |
| `frozen_tiles.npy` | Training rows × 256 × 224 × 224 × 3, RGB uint8. |
| `ffpe_tiles.npy` | Training rows × 128 × 224 × 224 × 3, RGB uint8. |
| `ffpe_tile_features.npy` | Training rows × 128 × 2,560, float16. Each vector corresponds to the FFPE tile at the same indices. |

Full-length arrays follow the CSV order. Training-only arrays follow that order after selecting all rows with role `train`, before filtering on the two flags. If the pixel arrays are in a separate folder, pass it with `--tiles` and include a matching `patients.csv` there. The source package starts from these prepared arrays. It does not include the cohort download or pack-building pipeline.
