"""Step 2: baselines - simple models that set the score a neural network must beat.

Run with:  .venv/Scripts/python.exe 02_baseline.py
"""
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from dataset import CATEGORIES, load_split

(x_train, y_train), _, (x_test, y_test) = load_split()
print(f"train {len(x_train):,} | test {len(x_test):,} drawings, {len(CATEGORIES)} categories\n")

# 1. Random guessing: with 15 categories, you are right 1 time in 15.
print(f"Random guess accuracy:      {1 / len(CATEGORIES):.1%}")

# 2. Logistic regression: one weight per pixel per category.
#    It sees each drawing as a flat list of 784 numbers - it has no idea which
#    pixels are neighbours, so it can't recognise shapes, only "ink in this spot".
#    (Trained on 45,000 drawings to keep it quick; it barely improves with more.)
X = lambda x: x.reshape(len(x), -1) / 255.0   # 28x28 image -> 784 numbers between 0 and 1
start = time.time()
model = LogisticRegression(max_iter=300)
model.fit(X(x_train[:45_000]), y_train[:45_000])
accuracy = (model.predict(X(x_test)) == y_test).mean()
print(f"Logistic regression:        {accuracy:.1%}   (trained in {time.time() - start:.0f}s)")
print("\nThis is the score the neural network has to beat.")
