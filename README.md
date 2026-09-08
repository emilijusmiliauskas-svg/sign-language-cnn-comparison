# Sign Language A/B/C: What 981 Images Can and Can't Teach a CNN

A comparison of **four convolutional architectures** — two built from scratch, two using transfer learning — on a small custom dataset of hand-sign photographs for the letters **A**, **B**, and **C**.

The interesting result isn't the winning accuracy. It's the gap between a model's test score and how it behaved when it was actually deployed.

---

## The Problem

981 training images. 199 validation, 199 test. Three classes.

That is a *tiny* dataset by computer vision standards, and it sets up the central question: when you have barely a thousand images, is it worth building a CNN from scratch at all, or should you always reach for a pretrained backbone?

## Results

| # | Model | Params | Test Accuracy |
|---|-------|-------:|:-------------:|
| 01 | CNN from scratch (v4) | 250,947 | 81.4% |
| 02 | CNN from scratch, heavier augmentation (v7) | 250,947 | 65.3% |
| 03 | Transfer — MobileNetV2 | 2,422,339 | **97.5%** |
| 04 | Transfer — InceptionResNetV2 | 54,533,859 | **99.0%** |

**Transfer learning wins decisively, and it isn't close.** A frozen ImageNet backbone with a small trained head reached 97.5% with roughly 10× the parameters of the scratch model — and 216× the parameters bought only a further 1.5 points.

## The Finding That Mattered

The from-scratch CNN scored **81.4% on the held-out test set** but only **63% when submitted to the class competition**. Same model, same weights, an 18-point collapse.

The cause was not overfitting. It was **train/serve skew**:

- Resizing, rescaling, and augmentation all lived in the **`tf.data` pipeline**, not in the model.
- The competition harness fed raw 256×256 images straight to `model.predict()`.
- None of the preprocessing ran. The model received inputs it had never seen in that form.

The fix was to move every preprocessing step **inside the model** as Keras layers — `Rescaling`, `Resizing`, and the augmentation layers — so that training and inference traverse an identical graph and the model is self-contained at the serialization boundary.

This is the kind of bug a test-set score is structurally incapable of catching, because the test set flows through the same pipeline the training data does. Only deployment surfaces it.

## The Second Finding: Augmentation Made It Worse

Notebook 02 fixed the skew bug *and* adopted aggressive augmentation — horizontal flip, ±51° rotation, heavy contrast jitter — copied from the course's reference notebook.

Test accuracy fell from **81.4% to 65.3%**.

With only 981 images the augmentation was too strong for the signal available: at ±51° rotation, and especially under horizontal flip, the distinction between hand signs starts to break down. Macro-average F1 dropped to 0.51, well below the weighted 0.60, meaning at least one class had largely collapsed. Augmentation is regularization, and regularization on a tiny dataset can cost more than the overfitting it prevents.

Three things changed at once between 01 and 02, which is itself the methodological lesson. Beyond the skew fix and the augmentation, notebook 01 passes `class_weight` to `fit()` and notebook 02 does not. With three simultaneous changes, the 16-point drop cannot be attributed to any one of them — the comparison shows *that* v7 was worse, not *why*.

## Notebooks

```
notebooks/
├── 01_cnn_from_scratch.ipynb              81.4% — 4-block CNN, 96×96 input, Adam + L2
├── 02_cnn_heavier_augmentation.ipynb      65.3% — skew fixed, augmentation overdone
├── 03_transfer_mobilenetv2.ipynb          97.5% — frozen backbone + fine-tuning, 11.3 min
└── 04_transfer_inceptionresnetv2.ipynb    99.0% — largest backbone, preprocessing in-model
```

All notebooks are committed with outputs intact, so the training curves, sample grids, and confusion matrices render directly on GitHub.

**Architecture of the scratch CNN (01):** four Conv→BatchNorm→ReLU→MaxPool→Dropout blocks (32→64→128→128 filters) into global average pooling and a 64-unit dense head, with L2 weight decay at 1e-4 and Adam. Reaching 81.4% from 250k parameters and under a thousand images is a reasonable showing — it just cannot compete with features learned from ImageNet.

## Trained Models

```
models/
├── cnn_from_scratch.keras            3.1 MB
└── cnn_heavier_augmentation.keras    3.1 MB
```

The two transfer-learning checkpoints (11 MB and 30 MB) are excluded from the repository.

Notebook 03 (MobileNetV2) fetches its ImageNet weights through Keras automatically. **Notebook 04 does not** — it loads the InceptionResNetV2 backbone from a local file at `~/Downloads/InceptionResNetV2_notop.h5`, which is not in this repository. To run it, either download the `notop` weights to that path or change the `weights=` argument to `"imagenet"` so Keras fetches them.

## Data

The dataset is **not included** — roughly 1 GB of photographs of human hands, which is both too large for version control and not mine to publish.

The notebooks expect the following layout, with one subdirectory per class:

```
processed/
├── train/   A/  B/  C/     981 images
├── val/     A/  B/  C/     199 images
└── test/    A/  B/  C/     199 images
```

Point `PROCESSED_DIR` at that directory and the notebooks run end to end. Note that `PROCESSED_DIR` is currently **hardcoded to `~/Downloads/processed`** in notebooks 01 and 03 — change it before running.

## Requirements

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

TensorFlow/Keras 3, scikit-learn, matplotlib, seaborn. Notebook 04 fine-tunes a 54M-parameter backbone and wants a GPU; notebooks 01–03 are comfortable on CPU (MobileNetV2 took 11.3 minutes).

## What I'd Do Next

- **Separate the two changes in v7.** Fix the skew and hold augmentation constant, then sweep augmentation strength independently. As committed, the experiment confounds them.
- **Sweep augmentation as a parameter** rather than adopting a reference configuration wholesale — rotation range in particular.
- **Set a global random seed.** Notebooks 02 and 04 seed their augmentation layers (`RANDOM_SEED = 42`), but no notebook calls `keras.utils.set_random_seed()` or `tf.random.set_seed()`, so weight initialisation and shuffling are unseeded. None of these numbers reproduce exactly on a re-run.
- **Report confidence intervals.** On a 199-image test set, one misclassification moves accuracy by half a point; 97.5% and 99.0% are not meaningfully distinguishable at this sample size.
- **Test the deployment path explicitly** — a check that feeds a raw image to the saved model and asserts a sane prediction would have caught the skew bug before submission.

## License

MIT
