"""Step 1: look at the doodles before training anything.

Run with:  .venv/Scripts/python.exe 01_explore.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = Path("data")
CHARTS = Path("charts")
CHARTS.mkdir(exist_ok=True)
CATEGORIES = sorted(p.stem for p in DATA.glob("*.npy"))

# Each file is an array of shape (15000, 28, 28): 15,000 tiny grayscale images.
# 0 = empty (black), 255 = full stroke (white).
data = {name: np.load(DATA / f"{name}.npy") for name in CATEGORIES}
print(f"{len(CATEGORIES)} categories, {sum(len(v) for v in data.values()):,} drawings")
print(f"one drawing: shape {data['cat'][0].shape}, values {data['cat'].min()}..{data['cat'].max()}\n")

# How much "ink" does each category use? Busy drawings (bicycle) vs simple ones (sun).
print("Average ink per drawing (% of pixels used):")
for name in CATEGORIES:
    ink = (data[name] > 0).mean() * 100
    print(f"  {name:>9}: {ink:5.1f}%  {'#' * int(ink)}")

# --- 8 random drawings of every category ---------------------------------------
rng = np.random.default_rng(0)
fig, axes = plt.subplots(len(CATEGORIES), 8, figsize=(8, len(CATEGORIES) * 1.05))
fig.patch.set_facecolor("#140d18")
for r, name in enumerate(CATEGORIES):
    for c, i in enumerate(rng.choice(len(data[name]), 8, replace=False)):
        ax = axes[r, c]
        ax.imshow(data[name][i], cmap="magma")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if c == 0:
            ax.set_ylabel(name, rotation=0, ha="right", va="center", color="#f6dbc0", fontsize=9)
fig.suptitle("What people draw (8 random doodles per category)", color="#f8f4e9")
fig.tight_layout()
fig.savefig(CHARTS / "examples.png", dpi=110, facecolor=fig.get_facecolor())

# --- The "average drawing" of each category ------------------------------------
# Averaging 15,000 drawings shows the shape most people draw. If two averages look
# alike (cat and dog?), the model will probably confuse them.
fig, axes = plt.subplots(3, 5, figsize=(9, 5.6))
fig.patch.set_facecolor("#140d18")
for ax, name in zip(axes.ravel(), CATEGORIES):
    ax.imshow(data[name].mean(axis=0), cmap="magma")
    ax.set_title(name, color="#f6dbc0", fontsize=10)
    ax.axis("off")
fig.suptitle("The average of 15,000 drawings per category", color="#f8f4e9")
fig.tight_layout()
fig.savefig(CHARTS / "averages.png", dpi=110, facecolor=fig.get_facecolor())
print("\nCharts saved to charts/")
