import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ClfImg(nn.Module):
    """
    MNIST image-to-digit classifier. Roughly based on the encoder from:
    https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(  # input shape (3, 28, 28)
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 14, 14)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> (128, 7, 7)
            nn.Dropout2d(0.5),
            nn.ReLU(),
            Flatten(),  # -> (12544)
            nn.Linear(12544, 128),  # -> (128)
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 10)  # -> (10)
        )

    def forward(self, x):
        h = self.encoder(x)
        return F.log_softmax(h, dim=-1)