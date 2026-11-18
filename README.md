# CryoPair

CryoPair transfers predictors trained on formalin-fixed, paraffin-embedded (FFPE) histology to frozen sections. It uses frozen and FFPE slides of the same patients to adapt an image encoder (CryoPair) or to fit a linear map between the two kinds of features (PAIRMAP), so that an RNA predictor fitted on FFPE slides can be applied to frozen sections without refitting.

This repository holds the software described in the manuscript "Patient-paired transfer of predictors based on pathology foundation models to frozen sections". It is research software and is not intended for diagnostic or clinical use.

## What this release contains

| Command | What it does |
| --- | --- |
| `cryopair-predict` | Predicts RNA expression for one patient from frozen H&E slides with CRISP, CryoPair-CRISP or CRISP with PAIRMAP |
| `cryopair-fit` | Fits an RNA reader or a PAIRMAP from patient-level features |
| `cryopair-train` | Trains CryoPair adapters for CRISP from paired frozen and FFPE tiles |

The release is source code only. It does not include encoder weights, trained adapters, fitted readers, PAIRMAP coefficients, slides or patient data. The underlying encoders are distributed under terms that do not clearly permit redistribution of files derived from them, so these files have to be produced by each user: obtain the base weights from their providers, then fit a reader (and, if wanted, a map and adapters) with the commands above.

## Requirements

- Python 3.11 or later.
- Fitting needs only NumPy and runs on a CPU.
- Slide prediction and training need a CUDA GPU, PyTorch 2.5 or later and timm 1.0.19. Slide prediction also needs the OpenSlide library to read slides; training reads prepared arrays and does not use it.

The software was tested on one workstation (Windows 10, NVIDIA RTX 4090, Python 3.11.1, PyTorch 2.5.1 with CUDA 12.1, timm 1.0.19). On that machine a prediction for one slide took about 40 seconds including model loading, fitting a reader for 14,926 genes on 1,015 patients took under 30 seconds, and a training step took 5.6 seconds, which extrapolates to about 6 hours for a complete run. Other systems have not been tested.

## Install

```sh
git clone https://github.com/Sanwal8/CryoPair.git
cd CryoPair
python -m pip install .
python -m unittest discover -s tests -v
```

The tests use synthetic data. They check that the fitting command is installed and works and, when PyTorch is installed, which adapter files the prediction command accepts. They do not need a GPU or any weights, and they say nothing about prediction accuracy.

For slide prediction and training, install PyTorch for your CUDA version first, and for slide prediction also the native OpenSlide library for your platform, then:

```sh
python -m pip install ".[slides,train]"
```

## Get the encoders

Both checkpoints are needed for slide prediction and for training. Each has to be requested from its provider under its own terms.

- [Virchow2](https://huggingface.co/paige-ai/Virchow2): non-commercial research use only, no redistribution, no diagnostic or clinical use.
- [CRISP](https://huggingface.co/FTZhou/CRISP): MIT license, gated download. CRISP is loaded on top of the Virchow2 architecture, so both files are required.

## Fit a reader and a PAIRMAP

The usual starting point is a cohort of your own with FFPE slides and measured expression. Encode the FFPE slides with the original encoder, average the tile features per patient, and fit a reader:

```sh
cryopair-fit reader --features ffpe.npz --targets rna.npz --space CRISP --units "log2(TPM + 1)" --out reader.npz
```

If you also have frozen and FFPE features of the same patients, fit a PAIRMAP:

```sh
cryopair-fit pairmap --frozen frozen.npz --ffpe ffpe_paired.npz --backbone CRISP --out map.npz
```

Rows are matched by patient ID, so the two files of a fit may list patients in different orders. The reader and the map may come from different cohorts; in the study the reader was fitted on 1,015 development patients and the map on 7,340 paired training patients. File formats are described in [docs/inputs.md](docs/inputs.md).

## Train adapters

```sh
cryopair-train --pack training_pack --virchow2 virchow2/model.safetensors --crisp CRISP.pth --out paired_run
```

The command reads prepared tile and feature arrays (see [docs/inputs.md](docs/inputs.md)); it does not build them from slides. Add `--steps 20` for a short trial. Rerunning the same command in the same output folder continues from the last checkpoint. `--pairing shuffled` trains the control in which each patient is paired with another patient of the same cancer type; use a separate output folder for it. Adapters from a short trial or from the shuffled control are marked as such in the adapter file, and `cryopair-predict` refuses them, so that only a finished paired run is reported as CryoPair-CRISP.

GPU training is not bit-reproducible. After 20 steps, repeated runs on the test machine differed by about 0.09% in adapter weights. Agreement after the full 3,672 steps has not been measured.

## Predict RNA for one patient

```sh
cryopair-predict --slides frozen.svs --models crisp cryopair-crisp crisp-pairmap \
    --virchow2 virchow2/model.safetensors --crisp CRISP.pth --reader reader.npz \
    --adapters paired_run/adapters.npz --pairmap map.npz --out predictions
```

For CRISP alone, give `--models crisp` and leave out `--adapters` and `--pairmap`. For a patient with several slides, list all of them after `--slides`; their tiles are pooled into one patient feature. The output folder must be new or empty.

| Output file | Content |
| --- | --- |
| `predicted_rna.csv` | One row per gene, one column per model |
| `patient_features.npz` | The pooled patient feature of each encoder state |
| `tiles_slide<n>.npz` | The tile coordinates used for each slide; they can be passed back with `--tiles` to repeat a run on the same tiles |
| `run.json` | Slides used, tile counts, pixel size and its source, file hashes of the weights and fitted files, software versions |

Up to 256 tissue tiles are drawn per slide. The draw depends only on the slide file, so repeating a run gives the same result; `--seed` draws another sample.

## Reading the predictions

- Values are on the scale of the reader's targets, log2(TPM + 1) for the study readers.
- The study evaluated how well predictions rank patients within a cancer type. Absolute expression for an individual patient has not been calibrated.
- Predictions depend on which tiles are sampled. In ten patients, redrawing the tiles changed the average gene by about 0.01 to 0.05 and single genes by up to 0.7 on the log2(TPM + 1) scale.
- If a slide records no pixel size, it is inferred from the objective magnification and this is noted in `run.json`. That path has not been tested.

## Intended use

CryoPair is meant for research on frozen-section histology: studying how predictors trained on FFPE slides behave on frozen sections, and testing patient-paired adaptation and mapping on new cohorts.

It is not a medical device. It must not be used to diagnose, treat or manage patients, and its outputs have not been validated for any clinical decision. The Virchow2 terms exclude diagnostic and clinical use in any case.

## Validation

The three commands were compared with the study pipeline on the study data; the details and limits are in [docs/validation.md](docs/validation.md). In short: the prediction command reproduced the saved study predictions for ten patients from two cohorts, the fitting command reproduced the five study readers and the CRISP map exactly, and the training command agreed with the study's training code within 0.020% of the loss over the first 20 steps. A complete training run, other encoders from slides and a second computer have not been tested.

## License and terms

The CryoPair source code is released under the [MIT License](LICENSE). The license covers this code only. Encoder weights, and any adapters, readers or maps derived with them, remain subject to the terms of the respective encoder providers.

## Citation

If you use this software, please cite the manuscript (in preparation) and this repository; see [CITATION.cff](CITATION.cff).

## Contact

Sanwal Ahmad Zafar, Cedars-Sinai Medical Center (sanwal.ahmadzafar@csmc.edu). Questions and problems can also be raised as GitHub issues.
