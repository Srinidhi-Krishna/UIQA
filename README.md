# Patch-Wise Underwater Image Quality Assessment with a Degradation-Aware Multi-Scale Transformer

No-reference underwater image quality assessment (UIQA) that predicts the human mean opinion score (MOS) **and** shows which regions drive the score.

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

## Results (UWIQA, held-out 20% test split, 179 images)

| Method | PLCC | SROCC | KRCC | RMSE |
|---|---|---|---|---|
| EDANet (as published, protocol may differ) | 0.8156 | 0.8104 | 0.6661 | 0.0860 |
| Ours (raw) | 0.8248 | 0.8072 | 0.6666 | 0.0867 |
| Ours (calibrated) | 0.8265 | 0.8072 | 0.6668 | 0.0842 |

Single trained ensemble (seed 42), no confidence interval. Differences from EDANet are small and the protocols may differ, so treat the comparison as indicative. Raw outputs are in [`results/uwiqa/`](results/uwiqa/).

Second dataset (SAUD / UIED): *coming soon.*

## Repository layout

```
.
├── UWIQA_HybridModel.ipynb      # full pipeline: data, model, training, K-fold ensemble, calibration, evaluation
├── requirements.txt
├── results/
│   └── uwiqa/                   # test_summary_raw.csv, test_summary_calibrated.csv, test_predictions_all_seeds.csv
├── paper/
│   └── paper.tex                # IEEE-format report (figures not included)
├── LICENSE
└── .gitignore
```

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from <https://pytorch.org/get-started/locally/> if the default `pip install torch` does not give you GPU support.

## Data

The dataset is **not** included in this repository. Obtain UWIQA from its original authors (Yang et al., 2021, *Signal Processing: Image Communication*, vol. 94, 116218) and place a zip next to the notebook:

```
UWIQA Dataset.zip
└── UWIQA/              # images 0001.png, 0002.png, ...
└── IQA Value.xlsx      # MOS labels
```

The first cells of the notebook unzip it and set `CFG["DATA_DIR"]` and `CFG["ANNOT_CSV"]`.

## Running

1. Open `UWIQA_HybridModel.ipynb` in Jupyter, VS Code or Google Colab.
2. Run the cells in order. Key settings are in the `CFG` dictionary: `KFOLD = 10`, `SEEDS = [42]`, `CNN_BACKBONE = "resnet50"`.
3. The main experiment trains one model per fold until early stopping, then writes the summary CSVs to `./outputs/` and checkpoints to `./checkpoints/` (both git-ignored).

A GPU is strongly recommended. The reported runs used a single NVIDIA T4 on Google Colab with mixed precision.

## Reproducibility notes

- Split: stratified by MOS class (width 0.1), seed 42, 20% fixed test set; the remaining 80% is re-partitioned into 10 stratified folds.
- Calibration is fitted on out-of-fold validation predictions only; the test split is used only for the final scores.
- The calibration mapping is not constrained to be monotonic.

## Limitations

- Patch quality, importance and degradation maps are model attributions, not physical measurements; there are no patch-level human labels.
- Degradation supervision is synthetic or handcrafted and has not been validated against human judgements.
- The backbone is pretrained on terrestrial ImageNet images.
- Results come from one training seed.

## Citation

TODO: add the paper reference (and link) once available.

## License

Code is released under the MIT License (see `LICENSE`). The UWIQA dataset has its own terms; please follow them.
