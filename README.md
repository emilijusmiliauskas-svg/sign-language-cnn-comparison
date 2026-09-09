# Sign Language A/B/C: What 981 Images Can and Can't Teach a CNN

A comparison of **five configurations** — three CNNs trained from scratch, two transfer-learning backbones — on a small custom dataset of hand-sign photographs for the letters **A**, **B**, and **C**.

Every number below comes from [`experiments/run_experiments.py`](experiments/run_experiments.py): one data pipeline, one evaluation, one global seed, and **each configuration differing from the baseline in exactly one respect**. The raw results are committed as JSON in [`experiments/results/`](experiments/results/).

---

## The Problem

981 training images. 199 validation, 199 test. Three classes.

That is a *tiny* dataset by computer vision standards, and it sets up the central question: when you have barely a thousand images, is it worth building a CNN from scratch at all, or should you always reach for a pretrained backbone?

## Results

| Configuration | Params | Test acc | Macro F1 | Train |
|---|---:|:---:|:---:|---:|
| Scratch CNN, light augmentation *(baseline)* | 250,691 | 78.4% | 0.772 | 15 min |
| Scratch CNN, heavy augmentation | 250,691 | 77.4% | 0.762 | 18 min |
| Scratch CNN, no class weighting | 250,691 | 83.9% | 0.833 | 18 min |
| **Transfer — MobileNetV2** | 2,261,827 | **97.0%** | 0.967 | 34 min |
| **Transfer — InceptionResNetV2** | 54,341,347 | **99.5%** | 0.994 | 37 min |

Trained on CPU (Apple M4), seed 42 throughout.

**Transfer learning wins decisively, and it isn't close.** Roughly 9× the parameters bought 19 points. A further 24× bought 2.5 more — real, but sharply diminishing.

InceptionResNetV2 misclassified exactly **one image out of 199**.

## Why Accuracy Alone Overstates Everything Here

The test split is **imbalanced — 43 / 64 / 92** — while training is near-balanced (333 / 340 / 308). Guessing the majority class alone scores **46.2%**.

That is why macro F1 is reported beside every accuracy: it weights all three classes equally and refuses to be flattered by the majority. For the scratch models the two diverge meaningfully; for the transfer models they converge, which is itself evidence the transfer models learned all three classes rather than leaning on the common one.

## The Finding That Mattered

The original from-scratch CNN scored **81.4% on the held-out test set** but only **63% when submitted to the class competition**. Same model, same weights, an 18-point collapse.

The cause was not overfitting. It was **train/serve skew**:

- Resizing, rescaling, and augmentation all lived in the **`tf.data` pipeline**, not in the model.
- The competition harness fed raw 256×256 images straight to `model.predict()`.
- None of the preprocessing ran. The model received inputs it had never seen in that form.

The fix is to move every preprocessing step **inside the model** as Keras layers — `Rescaling`, `Resizing`, and the augmentation layers — so training and inference traverse an identical graph and the model is self-contained at the serialization boundary. Every configuration in this repository now does that.

This is the kind of bug a test-set score is structurally incapable of catching, because the test set flows through the same pipeline the training data does. Only deployment surfaces it.

## The Claim That Did Not Survive Re-Testing

The original write-up concluded that **aggressive augmentation cost 16 points** — that test accuracy fell from 81.4% to 65.3% when the reference notebook's horizontal flip, ±51° rotation and heavy contrast jitter were adopted.

Under controlled conditions that effect **almost entirely disappears**:

| | Test acc | Macro F1 |
|---|:---:|:---:|
| Light augmentation | 78.4% | 0.772 |
| Heavy augmentation | 77.4% | 0.762 |

**One point.** On a 199-image test set, one point is two images.

The original comparison changed three things at once — it fixed the skew bug, adopted heavy augmentation, *and* dropped `class_weight` — so the 16-point drop could never have been attributed to augmentation in the first place. Re-running with a single variable changed shows that augmentation was not the culprit. What actually caused the original collapse remains unidentified; the most likely candidates are the interaction with the skew fix and ordinary run-to-run variance in an unseeded setup.

Removing class weighting looks like it *helps* — 78.4% to 83.9%. I would not claim that as a real effect. The training set is near-balanced, so the computed weights range only 0.96–1.06, and a 5.5-point swing from weights that close to 1.0 is far more plausibly variance than causation. Establishing it would take repeated runs across several seeds, which is the obvious next experiment and is not done here.

## What Each Model Gets Wrong

Confusion matrices, rows = true class **A / B / C**:

```
Scratch, no class weighting        MobileNetV2                InceptionResNetV2
   [[37,  4,  2],                  [[42,  1,  0],             [[43,  0,  0],
    [ 5, 57,  2],                   [ 1, 63,  0],              [ 0, 64,  0],
    [ 8, 11, 73]]                   [ 2,  2, 88]]              [ 1,  0, 91]]
```

The scratch model's errors concentrate in the bottom row: **C is mistaken for A eight times and for B eleven times**, a recall of 0.79 against 0.86 and 0.89 for the other two. C is the open curved hand — the shape closest to a partially-formed A or B, and the one that suffers most under rotation.

## Reproducing

```bash
export SIGN_DATA_DIR=/path/to/processed
python experiments/run_experiments.py                      # all five
python experiments/run_experiments.py --only mobilenetv2   # one
```

Each run writes `experiments/results/<name>.json` with accuracy, macro and weighted F1, the confusion matrix, per-class precision/recall, full training history, and wall-clock time.

## Notebooks

[`notebooks/`](notebooks/) holds the original exploration — four of sixteen variants, committed with outputs intact so the training curves and sample grids render on GitHub:

```
01_cnn_from_scratch.ipynb              4-block CNN, 96×96 input, Adam + L2
02_cnn_heavier_augmentation.ipynb      the skew fix bundled with heavy augmentation
03_transfer_mobilenetv2.ipynb          frozen backbone then fine-tuning
04_transfer_inceptionresnetv2.ipynb    largest backbone
```

They are kept as the record of how the work actually proceeded, confounds and all. **They are not the source of the numbers above** — the harness is. Their dataset path now reads `SIGN_DATA_DIR`, and notebook 04 fetches ImageNet weights rather than a local `.h5`.

**Architecture of the scratch CNN:** four Conv→BatchNorm→ReLU→MaxPool→Dropout blocks (32→64→128→128 filters) into global average pooling and a 64-unit dense head, L2 weight decay 1e-4, Adam. Reaching 78–84% from 250k parameters and under a thousand images is a reasonable showing — it just cannot compete with features learned from ImageNet.

## Data

`data/samples/` contains **24 example images** — eight per class, downscaled — so the repository shows what the model sees.

The full dataset is not included: ~1 GB of hand photographs stored as a `tf.data` snapshot. See [`data/README.md`](data/README.md) for the expected layout and pixel conventions.

## Models

```
models/
├── scratch_light_aug.keras          3.0 MB
├── scratch_heavy_aug.keras          3.0 MB
└── scratch_no_class_weight.keras    3.0 MB
```

The transfer checkpoints are excluded: MobileNetV2 is 28 MB and InceptionResNetV2 is **655 MB**, far past GitHub's 100 MB file limit. The harness refetches ImageNet weights automatically, so both rebuild from scratch.

One serialization note: the backbone preprocessing is a `Rescaling` layer rather than a `Lambda` wrapping `preprocess_input`. Both compute the same [-1, 1] transform, but a `Lambda` holding a function reference **cannot be deserialized** by Keras 3 — which is the failure the original notebook 04 worked around by saving to `.h5`.

## Requirements

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

TensorFlow/Keras 3, scikit-learn, matplotlib, seaborn, pillow. The full suite takes about 2 hours on CPU; a GPU shortens the transfer runs considerably.

## What I'd Do Next

- **Repeat every configuration across 5 seeds** and report mean ± std. Single runs cannot separate a 1-point difference from noise, and the class-weighting result above needs exactly this before it means anything.
- **Sweep augmentation strength as a continuous parameter** rather than testing two presets, now that a controlled harness makes it a one-line change.
- **Report confidence intervals.** At n=199 the 95% interval on 97.0% is roughly ±2.4 points, so MobileNetV2 and InceptionResNetV2 are not cleanly separable despite a 2.5-point gap.
- **Add a deployment smoke test** — feed a raw 256×256 image to the saved model and assert a sane prediction. That single check would have caught the skew bug before submission.

## License

MIT
