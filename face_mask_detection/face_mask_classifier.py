import pytorch_lightning as pl
from torch import Tensor
from torch.nn import (Conv2d, Linear, MaxPool2d, ReLU,
                      Sequential)


class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.convLayer1 = Sequential(
            Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )
        self.convLayer2 = Sequential(
            Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )
        self.convLayer3 = Sequential(
            Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(3, 3)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )
        self.linearLayers = Sequential(
            Linear(in_features=2048, out_features=1024),
            ReLU(),
            Linear(in_features=1024, out_features=2),
        )

    def forward(self, x: Tensor):
        out = self.convLayer1(x)
        out = self.convLayer2(out)
        out = self.convLayer3(out)
        out = out.view(-1, 2048)
        out = self.linearLayers(out)
        return out
