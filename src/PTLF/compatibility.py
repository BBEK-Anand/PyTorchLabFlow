"""
    This module provides a framework for validating, training, and testing PyTorch models
    within an experiment-driven pipeline. It defines three main classes:

    - `Checker`: A base class for running model forward passes and optionally tracking outputs.
    - `ExpChecker`: Extends `Checker` to support training and validation using specified metrics and optimizers.
    - `TestChecker`: Extends `Checker` for testing saved models on a specified dataset using an experiment pipeline.

    The module relies on utility components such as:
    - `load_component` for dynamic component loading
    - `PipeLine` for experiment management
    - `Model`, a typing alias or interface for PyTorch models

    Classes internally use `Tracker` subclasses to collect metrics and predictions for inspection or logging.

    Typical usage involves instantiating one of the checker classes with a config dictionary (`args`)
    and calling its `check()` method to execute the evaluation logic.
"""

from typing import cast
from copy import deepcopy
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .utils import Model, load_component
from .experiment import PipeLine


class Checker:
    """
        Base class for model evaluation and debugging.
        Provides a `_forward` method for running data through the model
        and tracking inputs, labels, logits, and other diagnostic data.
    """
    def __init__(self):
        """Initialize the Checker with appropriate device (CUDA or CPU)."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _forward(self, dataloader, tracker=None, desc="running"):
        """
            Perform a forward pass on the given dataloader.

            Args:
                dataloader (DataLoader): The data loader to iterate over.
                tracker (object, optional): Tracker object to monitor metrics and collect outputs.
                desc (str): Description string for tqdm progress bar.

            Returns:
                bool: True if at least a few batches were processed, otherwise False.
        """
        loader_tqdm = tqdm(dataloader, desc=desc, leave=True)
        model = cast(Model, self.model)
        try:
            i = 0
            for batch_idx, datas in enumerate(loader_tqdm):
                inpts = [i.to(self.device) for i in datas[0]]
                lbls = [i.to(self.device) for i in datas[1]]
                logits = model(*inpts)
                idxes = [batch_idx for i in range(len(lbls))]
                lbls = torch.cat([
                        t.unsqueeze(1) if t.ndim == 1 else t
                        for t in lbls], dim=1)
                if len(datas) > 2:
                    idxes = datas[2]
                if tracker:
                    tracker.step(
                        self, inputs=inpts, labels=lbls, logits=logits, idxes=idxes
                    )
                if i >= 3:
                    return True
                i += 1
        except:
            raise

    def check(self):
        """
        Placeholder method to be implemented by subclasses.
        """


class ExpChecker(Checker):
    """
    Experimental checker class for running training and validation checks
    using specified model, datasets, optimizer, and metrics.
    """
    def __init__(self, args):
        """
        Initialize ExpChecker with components from configuration arguments.

        Args:
            args (dict): Dictionary containing model, dataset, loss, metrics, and optimizer specifications.
        """
        super().__init__()
        try:
            self.model = load_component(**args["model"]).to(self.device)
            dargs = deepcopy(args["dataset"])
            dargs["args"]["data_src"] = args["train_data_src"]
            ds = load_component(**dargs)
            self.trainloader = DataLoader(
                ds, batch_size=args["train_batch_size"], shuffle=False
            )
            dargs["args"]["data_src"] = args["val_data_src"]
            ds = load_component(**dargs)
            self.valoader = DataLoader(
                ds, batch_size=args["val_batch_size"], shuffle=False
            )
            self.metrics = {"loss": args["loss"], **args["metrics"]}
            optimizer = deepcopy(args["optimizer"])
            optimizer["args"]["model_parameters"] = self.model.parameters()
            self.optimizer = load_component(**optimizer)
            self.metrics = {
                i: load_component(**j).to(self.device) for i, j in self.metrics.items()
            }
        except:
            raise

    def check(self):
        """
        Run both training and validation forward passes while collecting metrics.
        Uses internal Tracker class to monitor performance and outputs.

        Returns:
            bool: True if both training and validation ran successfully.
        """
        class Tracker:
            """
                Initialize Tracker with metrics and optimizer.

                Args:
                    metrics (dict): Dictionary of metric components.
                    optimizer (Optimizer, optional): Optimizer to step with loss.
                    device (torch.device): Device to use.
            """
            def __init__(
                self, metrics, optimizer=None,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu") ):

                self.metrics = metrics
                self.device = device
                self.optimizer = optimizer
                self.keeps = {
                    "metrics": {i: 0 for i in metrics},
                    "records": {
                        "inputs": [],
                        "labels": None,
                        "logits": None,
                        "idxes": [],
                    },
                }

            def step(self, P, **kwargs):
                """
                Perform a single update step: accumulate outputs, compute metrics, and optimize if applicable.

                Args:
                    P (Checker): Parent checker object.
                    kwargs: Contains 'inputs', 'labels', 'logits', and optionally 'idxes'.
                """
                logits = kwargs["logits"]
                labels = kwargs["labels"]
                for key, val in kwargs.items():
                    if isinstance(val, list):
                        self.keeps["records"][key].extend(val)
                    else:
                        if self.keeps["records"][key] is None:
                            self.keeps["records"][key] = val.detach().clone()
                        else:
                            self.keeps["records"][key] = torch.cat(
                                [self.keeps["records"][key], val.detach()], dim=0
                            )

                metrics = {i: j(logits, labels) for i, j in self.metrics.items()}
                if self.optimizer:
                    metrics["loss"].backward()
                    self.optimizer.step(loss=metrics["loss"].item(), epoch=1)
                metrics["loss"] = metrics["loss"].item()
                for i in metrics:
                    self.keeps["metrics"][i] += metrics[i]

        self.trntracker = Tracker(
            metrics=self.metrics, optimizer=self.optimizer, device=self.device
        )
        self.vldtracker = Tracker(
            metrics=self.metrics, optimizer=None, device=self.device
        )
        self.model.train()
        trn = self._forward(self.trainloader, tracker=self.trntracker, desc="training")

        self.model.eval()
        with torch.no_grad():
            vld = self._forward(
                self.valoader, tracker=self.vldtracker, desc="validation"
            )
        return trn and vld
