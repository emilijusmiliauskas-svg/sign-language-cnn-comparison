"""
Controlled re-run of the sign-language A/B/C comparison.

The original notebooks changed several things at once between runs, so the
measured differences could not be attributed to any single cause. This harness
fixes that: every run shares one data pipeline, one evaluation, and one seed,
and each configuration changes exactly one variable against the baseline.

All preprocessing (rescale, resize, augment) lives INSIDE the model, so the
saved model is self-contained and training and inference traverse the same
graph. That is the train/serve skew fix from the original v7 work, applied
uniformly here rather than bundled with other changes.

Usage:
    python experiments/run_experiments.py --only scratch_light_aug
    python experiments/run_experiments.py            # all five
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import layers  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    f1_score,
)

# --- Reproducibility ---------------------------------------------------------
# The original notebooks seeded some augmentation layers but never set a global
# seed, so weight initialisation and shuffling were unseeded and no reported
# number reproduced. One call covers python, numpy and tensorflow.
RANDOM_SEED = 42
keras.utils.set_random_seed(RANDOM_SEED)

# --- Paths -------------------------------------------------------------------
# Override with SIGN_DATA_DIR rather than editing the file.
DATA_DIR = Path(
    os.environ.get("SIGN_DATA_DIR", os.path.expanduser("~/Downloads/processed"))
)
RESULTS_DIR = Path(__file__).parent / "results"
MODELS_DIR = Path(__file__).parent.parent / "models"

CLASS_NAMES = ["A", "B", "C"]
NUM_CLASSES = 3
SOURCE_SIZE = 256
BATCH_SIZE = 32


# --- Data --------------------------------------------------------------------
def load_split(split: str) -> tf.data.Dataset:
    """Load one tf.data snapshot. Pixels arrive as float32 in [0, 255]."""
    path = DATA_DIR / split
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset at {path}. Set SIGN_DATA_DIR to the directory holding "
            "train/, val/ and test/."
        )
    return tf.data.Dataset.load(str(path))


def batched(ds: tf.data.Dataset, shuffle: bool = False) -> tf.data.Dataset:
    if shuffle:
        ds = ds.shuffle(1024, seed=RANDOM_SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def labels_of(ds: tf.data.Dataset) -> np.ndarray:
    return np.array([int(label) for _, label in ds.as_numpy_iterator()])


# --- Preprocessing head (inside the model) -----------------------------------
def preprocessing_layers(img_size: int, augment: str) -> list[layers.Layer]:
    """
    Rescale, resize, then augment — all as model layers.

    'light' is the configuration the from-scratch v4 run used.
    'heavy'  matches the course reference notebook that v7 adopted:
             horizontal flip, +/-51 degrees, strong contrast jitter.
    """
    head: list[layers.Layer] = [
        layers.Rescaling(1.0 / 255),
        layers.Resizing(img_size, img_size),
    ]

    if augment == "light":
        head += [
            layers.RandomRotation(0.08, seed=RANDOM_SEED),
            layers.RandomZoom(0.1, seed=RANDOM_SEED),
            layers.RandomTranslation(0.1, 0.1, seed=RANDOM_SEED),
            layers.RandomContrast(0.2, seed=RANDOM_SEED),
        ]
    elif augment == "heavy":
        head += [
            layers.RandomFlip("horizontal", seed=RANDOM_SEED),
            layers.RandomRotation(0.142, seed=RANDOM_SEED),  # ~51 degrees
            layers.RandomZoom(0.2, seed=RANDOM_SEED),
            layers.RandomContrast(1.0, seed=RANDOM_SEED),
        ]
    elif augment != "none":
        raise ValueError(f"unknown augmentation: {augment}")

    return head


# --- Architectures -----------------------------------------------------------
def build_scratch(img_size: int, augment: str) -> keras.Model:
    """The 4-block CNN from notebook 01: 32/64/128/128 -> GAP -> Dense(64)."""
    weight_decay = 1e-4
    model = keras.Sequential(
        [keras.Input(shape=(SOURCE_SIZE, SOURCE_SIZE, 3))]
        + preprocessing_layers(img_size, augment),
        name="scratch_cnn",
    )
    for filters, dropout in [(32, 0.25), (64, 0.25), (128, 0.30), (128, 0.30)]:
        model.add(
            layers.Conv2D(
                filters,
                3,
                padding="same",
                kernel_regularizer=keras.regularizers.l2(weight_decay),
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(dropout))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    return model


def build_transfer(backbone_name: str, img_size: int, augment: str) -> keras.Model:
    """Frozen ImageNet backbone plus a small head. Weights fetched by Keras."""
    if backbone_name == "mobilenetv2":
        backbone = keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
        )
    elif backbone_name == "inceptionresnetv2":
        backbone = keras.applications.InceptionResNetV2(
            include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
        )
    else:
        raise ValueError(backbone_name)

    backbone.trainable = False

    inputs = keras.Input(shape=(SOURCE_SIZE, SOURCE_SIZE, 3))
    x = inputs
    # Resize and augment in [0, 255] space, then apply the backbone's own
    # preprocessing, which expects that range.
    x = layers.Resizing(img_size, img_size)(x)
    if augment == "light":
        x = layers.RandomRotation(0.08, seed=RANDOM_SEED)(x)
        x = layers.RandomZoom(0.1, seed=RANDOM_SEED)(x)
        x = layers.RandomContrast(0.2, seed=RANDOM_SEED)(x)
    # Both backbones expect [-1, 1]. A Rescaling layer does exactly what their
    # preprocess_input does, and unlike a Lambda it survives save/load — the
    # serialization problem the original notebook 04 worked around.
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="backbone_preprocess")(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name=backbone_name)
    model.backbone = backbone  # kept for the fine-tuning phase
    return model


# --- Experiment definitions --------------------------------------------------
# Each entry differs from 'scratch_light_aug' in exactly one respect.
CONFIGS: dict[str, dict] = {
    "scratch_light_aug": dict(
        arch="scratch", img_size=96, augment="light",
        class_weight=True, epochs=120,
        note="Baseline: 4-block CNN, light augmentation, class weights on.",
    ),
    "scratch_heavy_aug": dict(
        arch="scratch", img_size=96, augment="heavy",
        class_weight=True, epochs=120,
        note="Isolates augmentation strength. Only 'augment' differs from baseline.",
    ),
    "scratch_no_class_weight": dict(
        arch="scratch", img_size=96, augment="light",
        class_weight=False, epochs=120,
        note="Isolates class weighting. Only 'class_weight' differs from baseline.",
    ),
    "mobilenetv2": dict(
        arch="mobilenetv2", img_size=224, augment="light",
        class_weight=True, epochs=15, finetune_epochs=30,
        note="Transfer learning, 2.2M-parameter backbone.",
    ),
    "inceptionresnetv2": dict(
        arch="inceptionresnetv2", img_size=224, augment="light",
        class_weight=True, epochs=15, finetune_epochs=5,
        note="Transfer learning, 54M-parameter backbone.",
    ),
}


def class_weights_from(y: np.ndarray) -> dict[int, float]:
    counts = np.bincount(y, minlength=NUM_CLASSES)
    total = counts.sum()
    return {i: float(total / (NUM_CLASSES * c)) for i, c in enumerate(counts)}


def run(name: str, cfg: dict, train_ds, val_ds, test_ds, y_train, y_test) -> dict:
    print(f"\n{'=' * 70}\n{name}\n{cfg['note']}\n{'=' * 70}", flush=True)
    keras.utils.set_random_seed(RANDOM_SEED)

    if cfg["arch"] == "scratch":
        model = build_scratch(cfg["img_size"], cfg["augment"])
    else:
        model = build_transfer(cfg["arch"], cfg["img_size"], cfg["augment"])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    weights = class_weights_from(y_train) if cfg["class_weight"] else None
    ckpt = MODELS_DIR / f"{name}.keras"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_accuracy", save_best_only=True, verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=12, min_lr=1e-5, verbose=0
        ),
    ]

    started = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        class_weight=weights,
        callbacks=callbacks,
        verbose=2,
    )
    histories = {k: [float(x) for x in v] for k, v in history.history.items()}

    # Phase 2 for transfer models: unfreeze and fine-tune at a lower rate.
    if cfg.get("finetune_epochs"):
        model.backbone.trainable = True
        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg["finetune_epochs"],
            class_weight=weights,
            callbacks=callbacks,
            verbose=2,
        )
        for k, v in ft.history.items():
            histories.setdefault(k, []).extend(float(x) for x in v)

    elapsed = time.time() - started

    # Evaluate the checkpointed best, not the final epoch.
    best = keras.models.load_model(str(ckpt))
    probs = best.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)

    accuracy = float((y_pred == y_test).mean())
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    print(
        f"\n{name}: test accuracy {accuracy:.4f} | macro F1 {macro_f1:.4f} "
        f"| {elapsed / 60:.1f} min",
        flush=True,
    )
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3, zero_division=0))

    return {
        "name": name,
        "note": cfg["note"],
        "config": {k: v for k, v in cfg.items() if k != "note"},
        "params": int(best.count_params()),
        "test_accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "per_class": classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
        "train_minutes": round(elapsed / 60, 2),
        "epochs_ran": len(histories.get("loss", [])),
        "history": histories,
        "seed": RANDOM_SEED,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="Run only these configurations")
    args = parser.parse_args()

    names = args.only or list(CONFIGS)
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown configuration(s): {unknown}")

    raw_train, raw_val, raw_test = (load_split(s) for s in ("train", "val", "test"))
    y_train, y_test = labels_of(raw_train), labels_of(raw_test)
    train_ds = batched(raw_train, shuffle=True)
    val_ds, test_ds = batched(raw_val), batched(raw_test)

    print(
        f"train={len(y_train)}  val={raw_val.cardinality().numpy()}  "
        f"test={len(y_test)}  seed={RANDOM_SEED}"
    )
    print(f"test class counts: {np.bincount(y_test, minlength=NUM_CLASSES).tolist()}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        result = run(name, CONFIGS[name], train_ds, val_ds, test_ds, y_train, y_test)
        out = RESULTS_DIR / f"{name}.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
