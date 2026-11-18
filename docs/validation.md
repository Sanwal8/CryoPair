# Validation

These tests compare the software with the study pipeline, using the study inputs on one Windows workstation with an NVIDIA RTX 4090. They establish agreement between implementations. Prediction accuracy against measured expression is evaluated in the manuscript, not here.

| Test | Outcome | Scope |
| --- | --- | --- |
| A | Verified, 102 comparisons | From saved patient features, all 17 model configurations reproduced the saved study predictions for six cohorts (1,949 patients) with a largest difference of 0.0. |
| B | Verified, 10 checks | One EAGLE frozen slide read again at its original tile coordinates through CRISP, CryoPair-CRISP and CRISP with PAIRMAP. |
| C | Verified, 21 checks | The standalone prediction command on the same slide. The largest difference from the saved predictions was 0.0000019 across 14,926 genes. |
| D | Verified, 85 checks | Five EAGLE and five BLGSP patients, 15 slides, 52 prediction runs. The largest difference from the saved predictions was 0.0000067. All slides had a measured pixel size; the five BLGSP patients had two slides each. |
| C2 | Verified, 21 checks | Test C repeated with the released prediction command (version 0.2.1). The largest difference from the saved predictions was 0.0000019. |
| E | Verified, 12 checks | The five readers and the CRISP map refitted by the standalone command had coefficients identical to the study's. Predictions were identical for 1,947 patients with complete features. |
| F | Failed, 8 of 9 checks passed | A 20-step training comparison. The failed check applied a relative tolerance to the retention term, which is close to zero by construction. Starting adapters, the first batch and the export of adapters into the prediction command passed. |
| F2 | Measured | The same 20 training steps run twice with the study's training code and twice with the standalone command. |
| G | Measured | One forward pass of the first training batch with the study's encoder loader and the standalone loader, each in the notebook process and in a fresh Python process. |

Differences are in log2(TPM + 1) unless stated otherwise.

## Training comparison

The largest relative loss difference was 0.0129% between the two runs of the study's training code and 0.0132% between the two standalone runs. Between the two implementations it was at most 0.0198%. Adapter weights after 20 steps differed by 0.088% and 0.090% within an implementation and by 0.089% to 0.093% between implementations. The between-implementation differences are similar in size to the run-to-run differences, although the largest of them is above the largest run-to-run difference. These are descriptive comparisons of four runs.

Before the first optimizer update, the retention term was 6.07e-8 in both runs of the study's code and 3.78e-7 in both standalone runs. This difference is repeatable and its cause has not been identified; all recorded numerical settings were the same. It is about 0.00005% of the loss. Test F2 does not change the outcome of test F.

Test G looked for the source of this difference. Both encoder loaders had identical weights and produced identical inference features on the first batch in the notebook and in a fresh Python process. Small differences occurred in gradient-enabled forward passes in the fresh process (retention term 6.1e-8 to 7.7e-8, on a total loss of 0.58). The larger first-step retention difference observed in F2 (3.8e-7) did not recur, and its cause remains unresolved.

A complete run has 3,672 steps for 7,340 paired patients. At the measured 5.6 seconds per step this extrapolates to 5.7 hours before overhead. Only the first 20 steps were compared. Agreement of the final adapters after four epochs has not been measured.

## Prediction comparison

Reading the saved tile coordinates again tests the encoder, the pooling and the reader on a fixed input. Drawing new tiles also changes which tissue is represented. Across 30 comparisons of an extra seed with the default seed (ten patients, three extra seeds each), the median of the mean absolute change per gene was 0.0215 for CRISP, 0.0144 for CryoPair-CRISP and 0.0131 for CRISP with PAIRMAP. The largest change of a single gene was 0.70. Ten patients give a descriptive check, not a general estimate of sampling stability.

## Released files and tested files

`cryopair_fit.py` and `cryopair_train.py` differ from the files used in tests E, F and F2 only in the reported version string. `cryopair_predict.py` is byte for byte the file used in test C2. It differs from the file used in test D by one added check: it refuses adapters marked as the shuffled control or as an unfinished run. Files without these marks, including the study adapters, are read as before. The ten-patient test D was not repeated with this file.

## Not covered

- Slide prediction with encoders other than CRISP. The other encoder families were tested from saved features only.
- Slides whose pixel size is inferred from the objective magnification.
- Building the training arrays from slides.
- A complete training run and the shuffled-pair control on the study data.
- Any computer other than the test workstation.
- Calibration of absolute expression for an individual patient.

## Standalone feature extraction

On one EAGLE frozen slide, `cryopair-encode` v0.2.1 reproduced all 2,560 pooled CRISP features of `cryopair-predict` exactly under default tile sampling. The post-release check on 18 November 2026 used the 256 tiles selected by default sampling for the original Test C slide on the study RTX 4090. The maximum and mean absolute feature differences were both zero. FFPE slides, multi-patient runs of `cryopair-encode`, and other hardware remain untested.
