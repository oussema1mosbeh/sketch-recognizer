"""Step 4: the final, honest test on drawings nothing has seen yet.

Run with:  .venv/Scripts/python.exe 04_evaluate.py
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from dataset import CATEGORIES, load_split
from model import SketchCNN

device = "cuda" if torch.cuda.is_available() else "cpu"
_, _, (x_test, y_test) = load_split()

model = SketchCNN(len(CATEGORIES)).to(device)
model.load_state_dict(torch.load("models/cnn.pt", map_location=device))
model.eval()

with torch.no_grad():
    x = torch.tensor(x_test, device=device).unsqueeze(1).float() / 255
    probs = torch.cat([model(x[i:i + 2048]).softmax(1) for i in range(0, len(x), 2048)]).cpu().numpy()
pred = probs.argmax(1)
conf = probs.max(1)

# --- Overall and per-category accuracy --------------------------------------------
print(f"TEST accuracy: {(pred == y_test).mean():.2%} on {len(y_test):,} unseen drawings")
top3 = (np.argsort(-probs, 1)[:, :3] == y_test[:, None]).any(1).mean()
print(f"Top-3 accuracy: {top3:.2%}  (right answer among the model's 3 best guesses)\n")

print("Per category (hardest first):")
per = {name: (pred[y_test == i] == i).mean() for i, name in enumerate(CATEGORIES)}
for name, acc in sorted(per.items(), key=lambda kv: kv[1]):
    print(f"  {name:>9}: {acc:6.2%}  {'#' * int((acc - .85) * 200)}")

# --- Confusion matrix: what gets mistaken for what? ---------------------------------
cm = confusion_matrix(y_test, pred, normalize="true")
pairs = [(cm[i, j], CATEGORIES[i], CATEGORIES[j]) for i in range(len(CATEGORIES))
         for j in range(len(CATEGORIES)) if i != j]
print("\nMost common mix-ups:")
for share, true, guess in sorted(pairs, reverse=True)[:6]:
    print(f"  {true:>9} drawn, model said {guess:<9} {share:.1%} of the time")

fig, ax = plt.subplots(figsize=(8.5, 7.5))
fig.patch.set_facecolor("#140d18")
ax.imshow(np.sqrt(cm), cmap="magma")  # square root makes small mix-ups visible next to the 95% diagonal
ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=60, ha="right", color="#f6dbc0")
ax.set_yticks(range(len(CATEGORIES)), CATEGORIES, color="#f6dbc0")
ax.set_xlabel("model guessed", color="#a796ad")
ax.set_ylabel("actually drawn", color="#a796ad")
for i in range(len(CATEGORIES)):
    for j in range(len(CATEGORIES)):
        if cm[i, j] >= .01:
            ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=7,
                    color="#140d18" if cm[i, j] > .5 else "#f8f4e9")
ax.set_title("Confusion matrix (% of each category)", color="#f8f4e9")
fig.tight_layout()
fig.savefig("charts/confusion.png", dpi=110, facecolor=fig.get_facecolor())

# --- Gallery of mistakes: the model's most confident wrong answers ------------------
wrong = np.where(pred != y_test)[0]
worst = wrong[np.argsort(-conf[wrong])][:24]
fig, axes = plt.subplots(3, 8, figsize=(11, 4.8))
fig.patch.set_facecolor("#140d18")
for ax, i in zip(axes.ravel(), worst):
    ax.imshow(x_test[i], cmap="magma")
    ax.set_title(f"drawn: {CATEGORIES[y_test[i]]}\nguess: {CATEGORIES[pred[i]]} {conf[i]:.0%}",
                 color="#f6dbc0", fontsize=7.5)
    ax.axis("off")
fig.suptitle("The model's most confident mistakes", color="#f8f4e9")
fig.tight_layout()
fig.savefig("charts/mistakes.png", dpi=110, facecolor=fig.get_facecolor())
print("\nCharts saved: charts/confusion.png, charts/mistakes.png")
