"""Step 6: download drawings of things the model does NOT know, to teach it "none of these".

Two separate groups from Quick, Draw!:
- OUTLIER_TRAIN: shown during training as "not one of the 15" (outlier exposure)
- OUTLIER_TEST:  never used in training, only to measure how well "I don't know" works
  on shapes the model has truly never seen.

Data: Quick, Draw! by Google, CC BY 4.0 - https://github.com/googlecreativelab/quickdraw-dataset
Run with:  .venv/Scripts/python.exe 06_download_unknown.py
"""
import ast
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

OUTLIER_TRAIN = ["star", "smiley face", "moon", "cloud", "rainbow", "umbrella", "cup", "lightning", "butterfly",
                 "face", "circle", "square", "triangle", "hexagon", "line", "zigzag", "squiggle", "snowflake",
                 "ice cream", "cake", "crown", "diamond", "mushroom", "spider", "octopus"]
OUTLIER_TEST = ["key", "pizza", "light bulb", "hat", "shoe", "scissors", "snail", "cactus", "book", "candle",
                "envelope", "mountain", "lollipop", "anchor", "door"]
PER_TRAIN, PER_TEST = 3_000, 1_000
PIXELS = 28 * 28
URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"
OUT = Path("data/unknown")
OUT.mkdir(parents=True, exist_ok=True)


def fetch(url, first_byte, last_byte):
    req = urllib.request.Request(url, headers={"Range": f"bytes={first_byte}-{last_byte}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


for group, names, per in (("train", OUTLIER_TRAIN, PER_TRAIN), ("test", OUTLIER_TEST, PER_TEST)):
    for name in names:
        target = OUT / f"{group}_{name.replace(' ', '_')}.npy"
        if target.exists():
            continue
        url = URL.format(urllib.parse.quote(name))
        try:
            head = fetch(url, 0, 255)
        except urllib.error.HTTPError as e:
            print(f"  {name}: not in Quick, Draw! ({e.code}), skipped")
            continue
        header_len = int.from_bytes(head[8:10], "little")
        info = ast.literal_eval(head[10:10 + header_len].decode("latin1"))
        start, n = 10 + header_len, min(per, info["shape"][0])
        images = np.frombuffer(fetch(url, start, start + n * PIXELS - 1), dtype=np.uint8).reshape(n, 28, 28)
        np.save(target, images)
        print(f"{group:>5} {name:>12}: {n:,} drawings")

print("\nDone. Files are in data/unknown/")
