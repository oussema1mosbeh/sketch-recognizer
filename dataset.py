"""Shared data loading, so every model is trained and tested on exactly the same split."""
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
CATEGORIES = sorted(p.stem for p in DATA.glob("*.npy"))


def load_split(seed=42):
    """Return (x_train, y_train), (x_val, y_val), (x_test, y_test).

    x: uint8 images of shape (n, 28, 28); y: category index into CATEGORIES.
    80% train (to learn), 10% validation (to make choices), 10% test (touched once, at the end).
    """
    xs, ys = [], []
    for label, name in enumerate(CATEGORIES):
        images = np.load(DATA / f"{name}.npy")
        xs.append(images)
        ys.append(np.full(len(images), label))
    x, y = np.concatenate(xs), np.concatenate(ys)

    order = np.random.default_rng(seed).permutation(len(x))  # shuffle so classes are mixed
    x, y = x[order], y[order]
    a, b = int(len(x) * .8), int(len(x) * .9)
    return (x[:a], y[:a]), (x[a:b], y[a:b]), (x[b:], y[b:])
