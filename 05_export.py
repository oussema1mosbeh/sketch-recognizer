"""Step 5: export the trained CNN to ONNX so it can run inside a web browser.

ONNX is a standard file format for neural networks. PyTorch writes it, and
ONNX Runtime Web runs it in the browser - no Python, no server.

Run with:  .venv/Scripts/python.exe 05_export.py
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from dataset import CATEGORIES, load_split
from model import SketchCNN

OUT = Path("demo")
OUT.mkdir(exist_ok=True)


class WithProbabilities(torch.nn.Module):
    """Wrap the CNN so the exported file outputs probabilities (0..1, summing to 1)
    instead of raw scores. The web page can then show '87% cat' directly."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).softmax(dim=1)


net = SketchCNN(len(CATEGORIES))
net.load_state_dict(torch.load("models/cnn.pt", map_location="cpu"))
model = WithProbabilities(net).eval()   # eval(): turn off dropout, use learned BatchNorm stats

example = torch.zeros(1, 1, 28, 28)     # one 28x28 grayscale drawing, values 0..1
torch.onnx.export(model, (example,), OUT / "model.onnx", input_names=["drawing"],
                  output_names=["probabilities"], opset_version=17, dynamo=False)
(OUT / "labels.json").write_text(json.dumps(CATEGORIES))
print(f"Exported demo/model.onnx ({(OUT / 'model.onnx').stat().st_size / 1e6:.1f} MB) and demo/labels.json")

# --- Check: does the ONNX file give the same answers as PyTorch? ----------------------
_, _, (x_test, y_test) = load_split()
x = x_test[:2000, None].astype(np.float32) / 255
session = ort.InferenceSession(str(OUT / "model.onnx"))
onnx_probs = np.concatenate([session.run(None, {"drawing": x[i:i + 1]})[0] for i in range(len(x))])
with torch.no_grad():
    torch_probs = model(torch.from_numpy(x)).numpy()

same = (onnx_probs.argmax(1) == torch_probs.argmax(1)).mean()
print(f"\nChecked on {len(x):,} test drawings:")
print(f"  same top answer as PyTorch: {same:.2%}")
print(f"  largest probability difference: {np.abs(onnx_probs - torch_probs).max():.1e}")
print(f"  accuracy of the ONNX model: {(onnx_probs.argmax(1) == y_test[:len(x)]).mean():.2%}")
