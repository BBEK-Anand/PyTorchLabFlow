import os
from PTLF.utils import DataSet
import torch



class DummyDataSet(DataSet):
    def __init__(self):
        super().__init__()
    
    def _setup(self,args):
        pass
    def collate_fn(batch):
        pass
    def __len__(self):
        return 5

    def __getitem__(self, idx):
       
        x = [3,4,5,1]
        label = [3]
        return x, label

