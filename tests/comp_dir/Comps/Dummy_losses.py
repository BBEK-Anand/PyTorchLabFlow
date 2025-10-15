from PTLF.utils import Loss


from torch import nn

class DummyLoss(Loss):
    def __init__(self):
        super().__init__()
    
    def _setup(self, args):
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.criterion(x)