"""The neural network: a small CNN (convolutional neural network)."""
import torch
from torch import nn


def block(c_in, c_out):
    """Two 3x3 convolutions, then halve the image size.

    A convolution slides a small 3x3 filter over the image. Early filters learn
    simple things (edges, stroke directions); later ones combine them into parts
    (ears, wheels, petals). BatchNorm keeps training stable, ReLU adds non-linearity,
    MaxPool shrinks the image so the next block sees a bigger area at once.
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SketchCNN(nn.Module):
    def __init__(self, n_classes=15):
        super().__init__()
        self.features = nn.Sequential(
            block(1, 32),     # 28x28 -> 14x14
            block(32, 64),    # 14x14 -> 7x7
            block(64, 128),   # 7x7   -> 3x3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 128 x 3 x 3 = 1,152 numbers
            nn.Dropout(0.3),                    # randomly hide 30% during training -> less overfitting
            nn.Linear(128 * 3 * 3, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),          # one score per category
        )

    def forward(self, x):
        """x: (batch, 1, 28, 28) floats in 0..1 -> (batch, n_classes) scores."""
        return self.classifier(self.features(x))


if __name__ == "__main__":
    n = sum(p.numel() for p in SketchCNN().parameters())
    print(f"SketchCNN has {n:,} learnable numbers ({n * 4 / 1e6:.1f} MB)")
