"""Step 7: teach the model to say "I don't know this shape".

The problem: the last layer shares 100% of its confidence between the 15 categories,
so a heart becomes "apple, 89%". Nobody ever showed the model "this is NONE of them".

The fix, outlier exposure: train on the 15 categories as before, and ALSO show it
drawings of other things (stars, clouds, smiley faces...) with the target "no idea":
its confidence should be spread evenly over the 15 categories for those.
The app then says "I don't know" when the top confidence is below a threshold.

Honest measurement:
- the threshold is chosen on validation data (known drawings + held-out outlier drawings)
- the result is measured on the test set AND on 14 shapes never used anywhere in training
  (key, pizza, light bulb, hat...)

Run with:  .venv/Scripts/python.exe 07_unknown.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import CATEGORIES, load_split
from model import SketchCNN

EPOCHS, BATCH, LR = 15, 512, 3e-3
OE_WEIGHT = 0.5          # how much "say no idea for these" counts, next to "get the 15 right"
KEEP_KNOWN = 0.95        # the threshold keeps 95% of real drawings of the 15 categories
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
UNK = Path("data/unknown")

(x_tr, y_tr), (x_va, y_va), (x_te, y_te) = load_split()


def load_group(group, split_seed=7):
    xs, names = [], []
    for f in sorted(UNK.glob(f"{group}_*.npy")):
        xs.append(np.load(f)); names.append(f.stem[len(group) + 1:].replace("_", " "))
    return xs, names


oe_parts, oe_names = load_group("train")
rng = np.random.default_rng(7)
oe_tr, oe_va = [], []
for part in oe_parts:                        # 90% of each outlier category to train, 10% to tune the threshold
    part = part[rng.permutation(len(part))]
    cut = int(len(part) * .9)
    oe_tr.append(part[:cut]); oe_va.append(part[cut:])
oe_tr, oe_va = np.concatenate(oe_tr), np.concatenate(oe_va)
unk_parts, unk_names = load_group("test")    # never used in training or tuning
unk_te = np.concatenate(unk_parts)
print(f"outlier exposure: {len(oe_tr):,} drawings from {len(oe_names)} categories; "
      f"never-seen test shapes: {len(unk_te):,} from {len(unk_names)} categories ({', '.join(unk_names)})")

to_t = lambda x: torch.tensor(x, dtype=torch.uint8, device=device).unsqueeze(1)
X_tr, X_oe = to_t(x_tr), to_t(oe_tr)
Y_tr = torch.tensor(y_tr, device=device)


def augment(x):
    """Same random tilts, shifts and zooms as step 3."""
    n = len(x)
    angle = (torch.rand(n, device=device) - .5) * 0.35
    scale = 1 + (torch.rand(n, device=device) - .5) * 0.2
    shift = (torch.rand(n, 2, device=device) - .5) * 0.16
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    return F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)


@torch.no_grad()
def confidence(model, x):
    """Top probability for each drawing, and the predicted category."""
    model.eval()
    probs = torch.cat([model(to_t(x[i:i + 4096]).float() / 255).softmax(1) for i in range(0, len(x), 4096)])
    top, pred = probs.max(1)
    return top.cpu().numpy(), pred.cpu().numpy()


def auroc(known, unknown):
    """Chance that a random known drawing gets a higher confidence than a random unknown one (1.0 = perfect)."""
    s = np.concatenate([known, unknown])
    ranks = s.argsort().argsort() + 1
    return (ranks[:len(known)].sum() - len(known) * (len(known) + 1) / 2) / (len(known) * len(unknown))


def report(name, model):
    va_k, _ = confidence(model, x_va)
    va_u, _ = confidence(model, oe_va)
    thr = float(np.quantile(va_k, 1 - KEEP_KNOWN))          # chosen on validation only
    te_k, te_pred = confidence(model, x_te)
    te_u, unk_pred = confidence(model, unk_te)
    kept = te_k >= thr
    res = {
        "model": name, "threshold": thr,
        "accuracy (all test drawings)": float((te_pred == y_te).mean()),
        "known drawings kept": float(kept.mean()),
        "accuracy of kept answers": float((te_pred[kept] == y_te[kept]).mean()),
        "never-seen shapes answered 'I don't know'": float((te_u < thr).mean()),
        "separation AUROC (never-seen vs known)": float(auroc(te_k, te_u)),
        "per never-seen shape": {n: float((confidence(model, p)[0] < thr).mean()) for n, p in zip(unk_names, unk_parts)},
    }
    print(f"\n{name}")
    for k, v in res.items():
        if k == "per never-seen shape":
            print("  'I don't know' rate per never-seen shape: " + ", ".join(f"{n} {r:.0%}" for n, r in v.items()))
        elif k != "model":
            print(f"  {k:<44} {v:.2%}" if k != "threshold" else f"  {k:<44} {v:.3f}")
    return res


# --- Before: the current model, with a confidence threshold only ------------------------
base = SketchCNN(len(CATEGORIES)).to(device)
base.load_state_dict(torch.load("models/cnn.pt", map_location=device))
results = [report("BEFORE: current model + confidence threshold", base)]

# --- After: retrain with outlier exposure ----------------------------------------------
model = SketchCNN(len(CATEGORIES)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
steps = EPOCHS * (len(X_tr) // BATCH)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
best = -1
print("\ntraining with outlier exposure...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    order = torch.randperm(len(X_tr), device=device)
    for i in range(len(X_tr) // BATCH):
        idx = order[i * BATCH:(i + 1) * BATCH]
        oe_idx = torch.randint(0, len(X_oe), (BATCH // 2,), device=device)
        x = augment(torch.cat([X_tr[idx], X_oe[oe_idx]]).float() / 255)
        logits = model(x)
        known_loss = F.cross_entropy(logits[:BATCH], Y_tr[idx], label_smoothing=0.05)
        # for the outliers, the target is "equal confidence in all 15": cross-entropy to the uniform distribution
        oe_loss = -logits[BATCH:].log_softmax(1).mean(1).mean()
        loss = known_loss + OE_WEIGHT * oe_loss
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    # model choice on validation: accuracy on known drawings + how many held-out outliers it rejects
    va_k, va_pred = confidence(model, x_va)
    va_u, _ = confidence(model, oe_va)
    thr = np.quantile(va_k, 1 - KEEP_KNOWN)
    score = (va_pred == y_va).mean() + (va_u < thr).mean()
    tag = ""
    if score > best:
        best = score
        torch.save(model.state_dict(), "models/cnn_oe.pt")
        tag = "  <- best, saved"
    print(f"epoch {epoch:2d} | val accuracy {(va_pred == y_va).mean():.2%} | held-out outliers rejected {(va_u < thr).mean():.2%}{tag}")

model.load_state_dict(torch.load("models/cnn_oe.pt", map_location=device))
results.append(report("AFTER: retrained with outlier exposure", model))
Path("models/unknown_results.json").write_text(json.dumps(results, indent=1))
