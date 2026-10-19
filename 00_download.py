"""Step 0: download the doodles.

Google's Quick, Draw! files hold 100,000+ drawings per category. We only need
the first 15,000 of each, so we ask the server for just the first part of each
file (an HTTP "Range" request) instead of downloading everything.

Data: Quick, Draw! by Google, CC BY 4.0 - https://github.com/googlecreativelab/quickdraw-dataset
Run with:  .venv/Scripts/python.exe 00_download.py
"""
import ast
import urllib.request
from pathlib import Path

import numpy as np

CATEGORIES = ["cat", "dog", "fish", "bicycle", "car", "airplane", "house", "tree",
              "flower", "sun", "apple", "banana", "guitar", "clock", "eye"]
PER_CATEGORY = 15_000
PIXELS = 28 * 28  # every drawing is a 28 x 28 grayscale image = 784 bytes
URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"
OUT = Path("data")
OUT.mkdir(exist_ok=True)


def fetch(url, first_byte, last_byte):
    req = urllib.request.Request(url, headers={"Range": f"bytes={first_byte}-{last_byte}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


for name in CATEGORIES:
    target = OUT / f"{name}.npy"
    if target.exists():
        print(f"{name:>9}: already downloaded")
        continue
    url = URL.format(name)
    # A .npy file starts with a small text header (shape, type), then the raw pixels.
    head = fetch(url, 0, 255)
    header_len = int.from_bytes(head[8:10], "little")
    info = ast.literal_eval(head[10:10 + header_len].decode("latin1"))
    start = 10 + header_len
    available = info["shape"][0]
    n = min(PER_CATEGORY, available)
    raw = fetch(url, start, start + n * PIXELS - 1)
    images = np.frombuffer(raw, dtype=np.uint8).reshape(n, 28, 28)
    np.save(target, images)
    print(f"{name:>9}: {n:,} drawings (of {available:,} available)")

print("\nDone. Files are in data/")
