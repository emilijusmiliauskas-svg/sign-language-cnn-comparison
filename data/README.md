# Data

The full dataset is **not** in this repository: ~1 GB of hand photographs stored
as a `tf.data` snapshot (serialised tensors, not image files).

## Sample

`samples/` holds 24 images — eight each of A, B and C, drawn from the test split
and downscaled to 128×128 — so you can see what the model is looking at without
the full download.

## Full dataset layout

The notebooks and the experiment harness both load `tf.data` snapshots:

```
processed/
├── train/   981 images
├── val/     199 images
└── test/    199 images
```

Each element is a `(256, 256, 3) float32` image with pixels in **[0, 255]** and an
`int32` label, where `0 = A`, `1 = B`, `2 = C`.

Point the code at it with:

```bash
export SIGN_DATA_DIR=/path/to/processed
```

Both splits used for training are close to balanced (train: 333 / 340 / 308).
The **test split is not** — 43 / 64 / 92 — so a majority-class guess scores 46.2%,
and accuracy alone overstates performance. Macro F1 is reported alongside it
throughout for that reason.
