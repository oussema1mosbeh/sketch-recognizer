"""Step 3: train the CNN on the GPU.

Run with:  .venv/Scripts/python.exe 03_train.py
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import CATEGORIES, load_split
from model import SketchCNN

EPOCHS = 15          # one epoch = the model sees every training drawing once
BATCH = 512          # drawings per learning step
LR = 3e-3            # learning rate: how big each correction is
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

(x_tr, y_tr), (x_va, y_va), _ = load_split()   # the test set stays hidden until step 4
to_t = lambda x: torch.tensor(x, dtype=torch.uint8, device=device).unsqueeze(1)  # all data fits on the GPU
x_tr, x_va = to_t(x_tr), to_t(x_va)
y_tr, y_va = torch.tensor(y_tr, device=device), torch.tensor(y_va, device=device)


def augment(x):
    """Data augmentation: randomly tilt, shift and resize each drawing a little.

    People draw things slightly rotated, off-centre or bigger/smaller. Showing the
    model these variations teaches it the SHAPE matters, not the exact position.
    """
    n = len(x)
    angle = (torch.rand(n, device=device) - .5) * 0.35          # about +-10 degrees
    scale = 1 + (torch.rand(n, device=device) - .5) * 0.2       # 90%..110%
    shift = (torch.rand(n, 2, device=device) - .5) * 0.16       # about +-2 pixels
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1),
                         torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    correct = sum((model(x[i:i + 2048].float() / 255).argmax(1) == y[i:i + 2048]).sum().item()
                  for i in range(0, len(x), 2048))
    return correct / len(x)


model = SketchCNN(len(CATEGORIES)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
steps = EPOCHS * (len(x_tr) // BATCH)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)  # warm up, then slow down

history, best, start = [], 0, time.time()
Path("models").mkdir(exist_ok=True)
for epoch in range(1, EPOCHS + 1):
    model.train()
    order = torch.randperm(len(x_tr), device=device)
    total_loss = 0
    for i in range(len(x_tr) // BATCH):
        idx = order[i * BATCH:(i + 1) * BATCH]
        x = augment(x_tr[idx].float() / 255)
        loss = F.cross_entropy(model(x), y_tr[idx], label_smoothing=0.05)  # how wrong were we?
        opt.zero_grad()
        loss.backward()       # work out how each weight should change
        opt.step()            # change them a little
        sched.step()
        total_loss += loss.item()

    train_loss = total_loss / (len(x_tr) // BATCH)
    val_acc = accuracy(model, x_va, y_va)
    history.append({"epoch": epoch, "loss": train_loss, "val_acc": val_acc})
    saved = ""
    if val_acc > best:  # keep the version that does best on drawings it never trained on
        best = val_acc
        torch.save(model.state_dict(), "models/cnn.pt")
        saved = "  <- best so far, saved"
    print(f"epoch {epoch:2d} | loss {train_loss:.3f} | validation accuracy {val_acc:.2%}{saved}")

print(f"\nBest validation accuracy: {best:.2%}  (trained in {time.time() - start:.0f}s)")
Path("models/history.json").write_text(json.dumps(history, indent=1))

# --- Learning curves --------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
fig.patch.set_facecolor("#140d18")
ep = [h["epoch"] for h in history]
for ax, key, title, color in [(a1, "loss", "Training loss (lower = better)", "#f6dbc0"),
                              (a2, "val_acc", "Validation accuracy", "#b4688f")]:
    ax.set_facecolor("#140d18")
    ax.plot(ep, [h[key] for h in history], color=color, lw=2, marker="o", ms=4)
    ax.set_title(title, color="#f8f4e9")
    ax.tick_params(colors="#a796ad")
    for s in ax.spines.values():
        s.set_color("#2b1f31")
    ax.set_xlabel("epoch", color="#a796ad")
a2.axhline(0.729, color="#a796ad", ls="--", lw=1)
a2.text(ep[-1], 0.729, " baseline 72.9%", color="#a796ad", va="bottom", ha="right", fontsize=9)
a2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
fig.tight_layout()
fig.savefig("charts/training.png", dpi=110, facecolor=fig.get_facecolor())
print("Learning curves saved to charts/training.png")
