
from PTLF.utils import Optimizer

import torch.optim as optim

class DummyOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self.args = {'model_parameters'}
    def _setup(self,args):
       
        self.optimizer = optim.Adam(args['model_parameters'])
        
    def step(self, **kwargs):
        self.optimizer.step(**kwargs)

    def zero_grad(self):
        self.optimizer.zero_grad()
