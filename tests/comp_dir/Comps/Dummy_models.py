
from PTLF.utils import Model


import torch.nn as nn

class DummyModel(Model):
    def __init__(self):
        super().__init__()
    
    def _setup(self, args):
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)