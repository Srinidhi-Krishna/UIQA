# Underwater Image Quality Assessment with a Degradation-Aware Multi-Scale Transformer

No-reference underwater image quality assessment (UIQA) that predicts the human mean opinion score (MOS) **and** shows which regions drive the score. Evaluated on **UWIQA** and **SAUD**.

Authors: Cheemakurthi Swetha, KR Srinidhi Rajamane (Shiv Nadar University Chennai)

## Method in brief

- Each image is resized to 256x256 and split into non-overlapping patches at three scales (32, 64, 128 px; 64 + 16 + 4 = 84 tokens).
- A shared ImageNet-pretrained ResNet-50 with squeeze-and-excitation encodes every patch.
- A small CNN degradation estimator (pretrained on synthetic haze, colour cast and blur with known levels, no MOS) conditions the tokens through FiLM.
- A 4-layer Transformer models cross-patch, cross-scale context.
- Three heads predict patch quality, saliency and degradation; fused importance weights drive scale-adaptive pooling to give the image score.
- Only image-level MOS is used as human supervision. The degradation head uses handcrafted pseudo-labels.
- Training loss: Huber + ranking + PLCC + degradation pseudo-label loss.
- Inference: stratified K-fold ensemble (K = 10), 4-view flip averaging, five-parameter logistic calibration.
- The same pipeline and hyperparameters are used for both datasets.

## Results

Held-out 20% test split of each dataset, single trained ensemble (seed 42), no confidence interval. EDANet values are as published and were obtained under a possibly different protocol, so comparisons are indicative only.

### UWIQA (179 test images)

| Method | PLCC | SROCC | KRCC | RMSE |
|---|---|---|---|---|
| EDANet (as published) | 0.8156 | 0.8104 | 0.6661 | 0.0860 |
| Ours (raw) | 0.8248 | 0.8072 | 0.6666 | 0.0867 |
| Ours (calibrated) | 0.8265 | 0.8072 | 0.6668 | 0.0842 |

Competitive with EDANet: higher PLCC and KRCC and lower RMSE, slightly lower SROCC. The differences are small.

### SAUD (210 test images)

| Method | PLCC | SROCC | KRCC | RMSE (normalised 0-1 scale) |
|---|---|---|---|---|
| EDANet (as published) | 0.8792 | 0.8517 | 0.7459 | 9.3911  |
| Ours (raw) | 0.7638 | 0.7498 | 0.5594 | 11.62 |
| Ours (calibrated) | 0.7733 | 0.7499 | 0.5595 | 11.44 |

On SAUD the model is clearly below EDANet's published numbers (about 0.10 lower in PLCC and SROCC, 0.19 lower in KRCC).

> TODO: SAUD's MOS scale differs from UWIQA's. Our RMSE above is on the normalised 0-1 scale. To compare with EDANet's 9.3911, map predictions and labels back to the original MOS scale and report that value here (as in the paper's results table).

Error pattern on SAUD (calibrated predictions): predictions are compressed toward the middle of the range (slope of prediction vs. MOS about 0.56). Mean absolute error is about 0.055 for MOS 0.5-0.7 (99 test images) but about 0.28 for MOS above 0.85 (9 images), where the model underpredicts. See `results/saud/test_audit.csv` for the per-image errors.

Raw outputs for both datasets are in [`results/`](results/).

## Repository layout

```
.
├── UWIQA_HybridModel.ipynb      # full pipeline on UWIQA: data, model, training, K-fold ensemble, calibration, evaluation
├── SAUD_HybridModel.ipynb       # same pipeline on SAUD (rename to match your notebook file)
├── requirements.txt
├── results/
│   ├── uwiqa/                   # test_summary_raw.csv, test_summary_calibrated.csv, test_predictions_all_seeds.csv
│   └── saud/                    # same three files, plus test_audit.csv (per-image errors, sorted worst first)
├── paper/
│   └── paper.tex                # IEEE-format report (figures not included)
├── LICENSE
└── .gitignore
```

## Setup

```bash
git clone https://github.com/Srinidhi-Krishna/UIQA.git
cd UIQA
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from <https://pytorch.org/get-started/locally/> if the default `pip install torch` does not give you GPU support.

## Data

Neither dataset is included in this repository. Obtain them from their original authors and follow their terms.

**UWIQA** (890 images, MOS in [0, 1]): Yang et al., 2021, *Signal Processing: Image Communication*, vol. 94, 116218. Place a zip next to the notebook:

```
UWIQA Dataset.zip
└── UWIQA/              # images 0001.png, 0002.png, ...
└── IQA Value.xlsx      # MOS labels
```

**SAUD** (1,000 enhanced underwater images: 10 enhancement methods applied to 100 scenes; filenames follow `g<group>_s<scene>_<method>.png`): Jiang et al., 2022, *IEEE Trans. Circuits Syst. Video Technol.*, vol. 32, no. 9, pp. 5959-5974. Put the images and MOS file where the SAUD notebook's `CFG` paths point. MOS labels are normalised to [0, 1] for training.

The first cells of each notebook unzip the data and set `CFG["DATA_DIR"]` and `CFG["ANNOT_CSV"]`.

## Running

1. Open the notebook for the dataset you want in Jupyter, VS Code or Google Colab.
2. Run the cells in order. Key settings are in the `CFG` dictionary: `KFOLD = 10`, `SEEDS = [42]`, `CNN_BACKBONE = "resnet50"`.
3. The main experiment trains one model per fold until early stopping, then writes the summary CSVs to `./outputs/` and checkpoints to `./checkpoints/` (both git-ignored).

A GPU is strongly recommended. The reported runs used a single NVIDIA T4 on Google Colab with mixed precision.

## Reproducibility notes

- Split: stratified by MOS class (width 0.1 on the normalised scale), seed 42, about 20% fixed test set; the remaining pool is re-partitioned into 10 stratified folds. The test set has 179 images for UWIQA and 210 for SAUD.
- SAUD: the 210 test images are 21 complete scenes (all ten enhanced versions of each), so no scene is shared between the test set and the training pool.
- Class balancing in the training sampler uses 9 MOS classes for UWIQA and 11 for SAUD.
- Calibration is fitted on out-of-fold validation predictions only; the test split is used only for the final scores.
- The calibration mapping is not constrained to be monotonic.

## Limitations

- Patch quality, importance and degradation maps are model attributions, not physical measurements; there are no patch-level human labels.
- Degradation supervision is synthetic or handcrafted and has not been validated against human judgements.
- The synthetic degradations (haze, colour cast, blur) may not capture the artefacts typical of SAUD's enhanced images; this is a possible reason for the gap to EDANet there and has not been tested.
- Predictions on SAUD are compressed toward mid-range scores, so very high and very low quality images are the least reliable.
- The backbone is pretrained on terrestrial ImageNet images.
- Results come from one training seed with no confidence interval.

## Citation

TODO: add the paper reference (and link) once available.

## License

Code is released under the MIT License (see `LICENSE`). The datasets have their own terms; please follow them.
