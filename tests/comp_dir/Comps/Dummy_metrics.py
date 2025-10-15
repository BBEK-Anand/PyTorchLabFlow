import os
from PTLF.utils import Metric
import torch



class DummyMetric(Metric):
    def __init__(self):
        super().__init__()
    
    def _setup(self,args):
        pass

    def forward(self, x):
        return 1

