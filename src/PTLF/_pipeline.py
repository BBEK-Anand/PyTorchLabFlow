"""
dummy
"""

from typing import TypedDict, Optional, Dict, List, Union, Any
import json
import os
from pathlib import Path
import time
from datetime import datetime
from copy import deepcopy
import shutil
import psutil
import traceback

import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .utils import (
    load_component,
    hash_args,
    get_invalid_loc_queries,
    Db,
    Component,
)
from .context import get_caller, get_shared_data


class CompsDict(TypedDict):
    """
    fgfvv
    """

    model: Component
    loss: Component
    optimizer: Component
    dataset: Component
    metrics: Dict[str, Component]


# _core.py
class PipeLine:
    """
    khgkjv
    """

    def __init__(self, pplid=None):
        """
        Initialize the pipeline with default settings and empty components.
        """
        self.settings = get_shared_data()

        self.pplid = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.comps: CompsDict = {}
        self.args = None

        self.trainDataLoader = None
        self.validDataLoader = None
        self.settings = get_shared_data()
        self.cnfg = None
        self._prepared = False
        self.__db = Db(db_path=f"{self.settings['data_path']}/ppls.db")
        self.__best = None
        if pplid:
            self.load(pplid=pplid)

    def _save_config(self) -> None:
        """
        Save the current experiment configuration to a JSON file.

        This method writes the configuration stored in `self.cnfg` to a config file,
        but only if the hash of the current arguments matches the stored experiment ID.
        This ensures consistency and prevents accidental overwrites due to argument changes.

        Raises
        ------
        ValueError
            If the current arguments do not match the stored experiment's arguments,
            indicating that the configuration has been modified since it was created.
        """
        if self.verify(args=self.cnfg["args"]) == self.cnfg["pplid"]:
            with open(self.get_path(of="config"), "w", encoding="utf-8") as out_file:
                json.dump(self.cnfg, out_file, indent=4)
        else:
            raise ValueError(
                f"can not save config for Experiment: {self.cnfg['pplid']}."
                "\n it's args has been changed"
            )

    def sync(self) -> None:
        """
        Synchronize and update the experiment configuration with the latest quick settings.

        Loads the quick configuration file associated with the current experiment ID,
        updates the main configuration (`self.cnfg`) with its contents, and then saves
        the updated configuration to disk.

        Side Effects
        ------------
        - Modifies the `self.cnfg` attribute by merging it with the quick configuration.
        - Writes the updated configuration to the config file.
        - Prints a success message indicating the experiment has been synced.

        Raises
        ------
        ValueError
            If the current arguments do not match the original configuration, preventing saving.
        """
        with open(self.get_path(of="quick"), "r", encoding="utf-8") as quick:
            quick = json.load(quick)
        self.cnfg.update({**quick})
        self._save_config()
        print(f"{self.pplid} synced successfully")

    def get_path(
        self,
        of: str,
        pplid: Optional[str] = None,
        epoch: Optional[int] = None
    ) -> str:
        """
            Generate a standardized file path for various experiment artifacts.

            Constructs and returns a file path based on the type of file (`of`), experiment ID,
            epoch number, and batch index, where applicable. Automatically creates necessary
            directories if they do not exist.

            Parameters
            ----------
            of : str
                The type of file to retrieve the path for. Supported values:
                - "config": Configuration file path.
                - "weight": Model weights file path.
                - "gradient": Saved gradients file path.
                - "history": Training history file path.
                - "quick": Quick config file path.
            pplid : str, optional
                Experiment ID. If not provided, uses the currently set `self.pplid`.
            epoch : int, optional
                Epoch number. Required for weight and gradient file paths.
                For weights, if not specified, the best epoch from config is used.

            Returns
            -------
            str
                Full path to the specified artifact as a string with forward slashes.

            Raises
            ------
            ValueError
                If `pplid` is not set or invalid.
                If required parameters (`epoch`, `batch`) are missing for gradient paths.
                If the `of` argument is not one of the supported values.
        """
        pplid = pplid or self.pplid
        if not pplid:
            raise ValueError("Experiment ID (pplid) must be provided.")

        base_path = Path(self.settings["data_path"])

        if of == "config":
            path = base_path / "Configs" / f"{pplid}.json"

        elif of == "weight":
            if epoch is None:
                raise ValueError(
                    "Epoch must be specified or defined in config under 'best.epoch'."
                )
            path = base_path / "Weights" / pplid / f"{pplid}_e{epoch}.pth"

        elif of == "history":
            path = base_path / "Histories" / f"{pplid}.csv"

        elif of == "quick":
            path = base_path / "Quicks" / f"{pplid}.json"

        else:
            raise ValueError(
                f"Invalid value for 'of': {of}. Supported values: "
                "'config', 'weight', 'gradient', 'history', 'quick'."
            )

        path = path.as_posix()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def load_model(self, epoch: Optional[Union[int, str]] = None) -> torch.nn.Module:
        """
        Load model weights from disk into the model component for a specified epoch.

        If `epoch` is set to 'last' or 'best', the corresponding epoch value from the
        experiment configuration is used. If the epoch is 0 and no weights exist yet,
        the model's current state is saved before loading. Weights are loaded with
        `strict=False` to allow partial loading of the model.

        Parameters
        ----------
        epoch : int or str, optional
            The epoch number or keyword ('last' or 'best') indicating which weights to load.
            - 'last': Loads the most recent training checkpoint.
            - 'best': Loads the checkpoint with the best validation performance.
            - int: Loads the checkpoint from the specified epoch.
            - None: Defaults to using the current epoch from config if available.

        Returns
        -------
        torch.nn.Module
            The model component with the loaded weights.

        Raises
        ------
        ValueError
            If the experiment configuration or weight file path is invalid or missing.

        Notes
        -----
        - Uses `torch.load(..., weights_only=True)` for loading weights.
        - Uses `strict=False` in `load_state_dict` to allow for minor mismatches.
        - Automatically saves the model if the requested epoch is 0 and no checkpoint exists.
        """
        if epoch == "last":
            epoch = self.cnfg["last"]["epoch"]
        if epoch == "best":
            epoch = self.cnfg["best"]["epoch"]

        t = self.get_path(of="weight", pplid=self.pplid, epoch=epoch)
        if epoch == 0 and not os.path.exists(t):
            torch.save(self.comps["model"].state_dict(), t)
        else:
            self.comps["model"].load_state_dict(
                torch.load(t, weights_only=True), strict=False
            )
        return self.comps["model"]

    def load(self, pplid: str, prepare: bool = False) -> None:
        """
        Load the experiment configuration and optionally prepare the pipeline.

        Retrieves the configuration file associated with the given experiment ID and sets
        it as the active configuration. Optionally, prepares the pipeline using the loaded
        settings (e.g., model, data loaders, etc.).

        Parameters
        ----------
        pplid : str
            The experiment ID whose configuration is to be loaded.
        prepare : bool, optional
            Whether to immediately prepare the pipeline using the loaded configuration.
            Defaults to False.

        Raises
        ------
        ValueError
            If the provided experiment ID does not exist in the experiment database.

        Side Effects
        ------------
        - Sets `self.cnfg` with the loaded configuration dictionary.
        - Updates `self.pplid` to the provided experiment ID.
        - Calls `self.prepare()` if `prepare` is True.
        """
        if not self.verify(pplid=pplid):
            raise ValueError(f"The pplid: {pplid} is not exists")
        with open(self.get_path(of="config", pplid=pplid), encoding="utf-8") as cnfg:
            cnfg = json.load(cnfg)
        self.cnfg = cnfg
        self.pplid = pplid
        if prepare:
            self.prepare()

    def _setup_dataloaders(self, args):
        """
        initiating dataloaders
        """
        self.comps["loss"] = load_component(**args["loss"]).to(self.device)

        self.trainDataLoader = DataLoader(
            **self._adjust_loader_params(mode="train", args=args)
        )
        self.validDataLoader = DataLoader(
            **self._adjust_loader_params(mode="val", args=args)
        )

    def reset(self):
        """
        reset
        """
        self.settings = None
        self.pplid = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.comps = {
            "model": None,
            "loss": None,
            "optimizer": None,
            "dataset": None,
            "metrics": {},
        }
        self.args = None
        self.trainDataLoader = None
        self.validDataLoader = None
        self.settings = get_shared_data()
        self.cnfg = None
        self._prepared = False
        self.__db = Db(db_path=f"{self.settings['data_path']}/exps.db")


    def verify(self, *, pplid: str = None, args: Dict = None) -> Union[str, bool]:
        """
        Check whether a given experiment ID exists in the experiment database.

        Queries the experiments table to verify whether the specified experiment ID is recorded.

        Parameters
        ----------
        pplid : str
            The experiment ID to check.

        Returns
        -------
        Union[str, bool]
            Returns the `pplid` if it exists in the database, otherwise returns `False`.

        Examples
        --------
        >>> pipeline.verify("exp_001")
        'exp_001'

        >>> pipeline.verify("nonexistent_exp")
        False
        """

        if pplid:
            result = self.__db.query(
                "SELECT 1 FROM ppls WHERE pplid = ? LIMIT 1", (pplid,)
            )
            if len(result) > 0:
                return pplid
        elif args:
            args_hash = hash_args(args)
            rows = self.__db.query(
                "SELECT pplid FROM ppls WHERE args_hash =? LIMIT 1", (args_hash,)
            )
            if rows:
                pplid = rows[0][0]
                return pplid
        return False

    def log(self):
        """
        in future versions
        """

    def _check_args(self, args):
        t = get_invalid_loc_queries(args)
        if t:
            raise ValueError(
                "Make sure all components are saved.\nReff: " + ", ".join(t)
            )
        t = self.verify(args=args)
        if t:
            raise ValueError(f"same configuration is already exists in: {t}")

    def new(
        self,
        pplid: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        prepare: bool = False,
    ) -> None:
        """
        Create a new experiment configuration and initialize its tracking files.

        Parameters
        ----------
        pplid : str, optional
            Unique experiment identifier. Raises ValueError if it already exists.
        args : dict, optional
            Configuration arguments for the experiment.
        prepare : bool, optional
            If True, calls `self.prepare()` after creation. Defaults to False.

        Raises
        ------
        ValueError
            If the experiment ID already exists or if monitor mode is invalid.
        KeyError
            If 'metrics' key is missing from settings.

        Behavior
        --------
        - Checks if the experiment ID already exists; raises an error if so.
        - Checks if the same configuration already exists using `verify`.
        - Initializes configuration dictionary with metadata.
        - Saves the configuration.
        - Creates an empty history CSV with columns for training and validation metrics and loss.
        - Initializes quick checkpoint file with default best and last epoch metrics.
        - Appends experiment metadata to the main experiments CSV.
        - Optionally calls `self.prepare()` if `prepare=True`.
        """
        if self.verify(pplid=pplid):
            raise ValueError(f"{pplid} is already exists  try  different id")
        self._check_args(args)
        t = {
            "pplid": pplid,
            "args": args,
            "used": "",
            "best": {"epoch": 0},
            "last": {"epoch": 0},
        }

        self.pplid = pplid
        self.cnfg = t

        metrics = self.settings.get("metrics")
        if not metrics:
            raise KeyError("'metrics' is missing from settings.")

        keys = [
            *[f"train_{m}" for m in metrics],
            "train_loss",
            "train_duration",
            *[f"val_{m}" for m in metrics],
            "val_loss",
            "val_duration",
        ]
        record = pd.DataFrame([], columns=["epoch", *keys])
        record.to_csv(self.get_path(of="history", pplid=self.pplid), index=False)

        strategy = self.settings["strategy"]
        l = {"epoch": 0, **{i: 0 for i in keys}}
        if strategy["mode"] == "min":
            l[strategy["monitor"]] = float(10000)
        elif strategy["mode"] == "max":
            l[strategy["monitor"]] = -float(10000)
        else:
            raise ValueError("monitor should be min or max")

        quick = {"last": l, "best": l}
        with open(
            self.get_path(of="quick", pplid=pplid), "w", encoding="utf-8"
        ) as out_file:
            json.dump(quick, out_file, indent=4)

        
        self.__db.execute(
            "INSERT INTO ppls (pplid, args_hash) VALUES (?, ?)",
            (pplid, hash_args(args)),
        )

        self._save_config()
        # Initialize logs.csv and exps.csv
        
        if prepare:
            self.prepare()

    def _adjust_loader_params(self, mode: str, args: Optional[dict] = None) -> dict:
        """
        Adjusts DataLoader parameters based on system resources and dataset size.

        Parameters
        ----------
        mode : str
            Either 'train' or 'val' to specify the data loader type.
        args : dict, optional
            Configuration dictionary containing dataset and batch size parameters.
            If None, uses `self.args`.

        Returns
        -------
        dict
            Parameters for DataLoader including dataset, batch_size,
            shuffle, num_workers, collate_fn, and pin_memory.

        Raises
        ------
        ValueError
            If mode is neither 'train' nor 'val'.
        """
        args = self.args if args is None else args
        loc = args["dataset"]["loc"]
        dsargs = args["dataset"]["args"]

        if mode in {"val", "train"}:
            dsargs["data_src"] = args[f"{mode}_data_src"]
            ds = load_component(loc=loc, args=dsargs)
            collate_fn = getattr(ds, "collate_fn", None) or None
            batch_size = args[mode + "_batch_size"]
            shuffle = not mode == "val  "
        else:
            raise ValueError(mode + "_data_src is not found")

        num_cpu_cores = os.cpu_count()

        if len(ds) < batch_size:
            batch_size = len(ds)
            print(
                "Warning: Dataset size is smaller than the batch size."
                f"Adjusting batch size to {batch_size}."
            )
            args.update({mode + "_batch_size": batch_size})
            if self.args is not None:
                self._save_config()

        pin_memory = batch_size >= 32  # Larger batches benefit more from pin_memory

        if batch_size < 16:
            num_workers = max(1, num_cpu_cores // 2)  # Fewer workers for small batches
        elif batch_size < 64:
            num_workers = num_cpu_cores
        else:
            num_workers = min(num_cpu_cores * 2, 16)

        system_memory_available = psutil.virtual_memory().available > 5 * 1024**3
        if not system_memory_available:
            num_workers = min(num_workers, 4)
            pin_memory = False  # Disable pin_memory to save memory
            print(
                f"memory available={psutil.virtual_memory().available}<={5 * 1024**3}"
                " --> pin_memory={pin_memory}"
            )

        num_workers = 0 if self.args is None else num_workers
        # Return the optimal settings for DataLoader

        return {
            "dataset": ds,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": pin_memory,
        }

    def prepare(self) -> None:
        """
        Prepare the experiment by loading model, optimizer, metrics, loss, and data loaders.

        Loads components according to current configuration, initializes data loaders,
        and sets the best metric value based on the stored history and strategy.

        Raises
        ------
        ValueError
            If strategy monitor mode is not 'min' or 'max'.

        Behavior
        --------
        - Loads model and moves it to device.
        - Loads optimizer with model parameters.
        - Loads metrics and loss functions to device.
        - Creates training and validation data loaders.
        - Loads last saved model weights.
        - Initializes the best metric value from saved checkpoints or sets default.
        - Sets internal flag `_prepared` to True on success.
        """
        if not self.cnfg:
            print("not initiated")
            return
        args = deepcopy(self.cnfg["args"])
        self.comps["model"] = load_component(**args["model"]).to(self.device)
        args["optimizer"]["args"]["model_parameters"] = self.comps["model"].parameters()

        self.comps["optimizer"] = load_component(**args["optimizer"])
        self.comps["metrics"] = {
            name: load_component(**comp).to(self.device)
            for name, comp in args["metrics"].items()
        }
        self.comps["loss"] = load_component(**args["loss"]).to(self.device)
        self._setup_dataloaders(args=args)
        self.comps["model"] = self.load_model(epoch="last")
        with open(self.get_path(of="quick"), encoding="utf-8") as fl:
            q = json.load(fl)
            t = self.settings["strategy"]["monitor"]
            self.__best = q["best"][t]
        self._prepared = True
        print("Data loaders are successfully created")

    def update(self, data: dict) -> bool:
        """
        Update the pipeline configuration and save state after an epoch.

        Parameters
        ----------
        data : dict
            Dictionary containing keys such as 'epoch', 'train_accuracy', 'train_loss',
            'val_accuracy', 'val_loss', and potentially other metrics and durations.

        Returns
        -------
        bool
            Returns True if the current epoch's validation metric improves over the best recorded,
            triggering a best model save; otherwise, False.

        Notes
        -----
        - Saves model weights after every epoch.
        - Appends training and validation metrics to the history CSV.
        - Updates the quick checkpoint file with last and best metrics.
        """

        torch.save(
            self.comps["model"].state_dict(),
            self.get_path(of="weight", epoch=data["epoch"]),
        )
        # print(f"Current Model Weights saved temporarily")
        best = False
        strategy = self.settings["strategy"]
        if strategy["mode"] == "min" and self.__best >= data[strategy["monitor"]]:
            best = True
        elif strategy["mode"] == "max" and self.__best <= data[strategy["monitor"]]:
            best = True

        metrics = list(self.settings["metrics"])
        metrics = (
            ["epoch"]
            + [f"train_{i}" for i in metrics]
            + ["train_loss", "train_duration"]
            + [f"val_{i}" for i in metrics]
            + ["val_loss", "val_duration"]
        )
        # print(metrics, {i:data[i] for i in metrics})
        record = pd.DataFrame([[data[i] for i in metrics]], columns=metrics)
        # print(record)
        record.to_csv(
            self.get_path(of="history", pplid=self.pplid),
            mode="a",
            header=False,
            index=False,
        )
        with open(self.get_path(of="quick"), encoding="utf-8") as q:
            quick = json.load(q)
        quick["last"] = data
        if best:
            print(
                f"Best Model Weights Updated: Epoch {data['epoch']} - Val Loss: {data['val_loss']}"
            )
            quick["best"] = data

        with open(self.get_path(of="quick"), "w", encoding="utf-8") as out_file:
            json.dump(quick, out_file, indent=4)
        return best

    def train(
        self,
        num_epochs: int = 5,
        self_patience: Optional[int] = None,
        verbose: Union[List[str], str] = None
    ) -> None:
        """
        Train the model for a specified number of epochs with optional early stopping.

        Parameters
        ----------
        num_epochs : int, optional
            Number of epochs to train. Default is 5.
        self_patience : int, optional
            Number of epochs to wait for improvement before early stopping. If None, equals num_epochs.
        verbose : list of str or str, optional
            Metrics to display live during training. Must be from the set of defined metrics.

        Notes
        -----
        - Uses early stopping based on the configured strategy and patience.
        - Automatically resumes from last epoch.
        - Saves best model weights and updates training history.
        - Avoids re-entrance if training is already running.
        """
        if not self._prepared:
            print(
                "Preparation Error. Execute prepare() or set prepare=True before training."
            )
            return

        with open(self.get_path(of="quick"), encoding="utf-8") as q:
            quick = json.load(q)

        start_epoch = quick["last"]["epoch"]
        end_epoch = start_epoch + num_epochs
        patience = self_patience or num_epochs
        epochs_without_improvement = (
            self.cnfg["last"]["epoch"] - self.cnfg["best"]["epoch"]
        )

        verbose = list(self.settings['metrics']) if verbose ==None else verbose
        if isinstance(verbose, str):
            verbose = [verbose]

        if not isinstance(verbose, list) or not all(
            item in self.settings["metrics"] for item in verbose
        ):
            print(f"Verbose should be in metrics: {list(self.settings['metrics'])}")
            return

        rows = self.__db.query(
            "SELECT logid FROM runnings WHERE pplid = ?", (self.pplid,)
        )
        if rows:
            print(f"pplid: {self.pplid} is running in logid: {rows[0][0]}")
            return

        try:
            self.__db.execute(
                "INSERT INTO runnings (pplid, logid) VALUES (?, ?)",
                (self.pplid, self.settings["logid"]),
            )

            for epoch in range(start_epoch, end_epoch):
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch} due to no improvement.")
                    return
                if not self.should_running:
                    return 
                train_metrics = self._forward(
                    mode="train", epoch=epoch, verbose=verbose
                )
                val_metrics = self._forward(mode="valid", epoch=epoch)

                metrics_record = {
                    "epoch": epoch + 1,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }

                if self.update(metrics_record):
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            print("Finished Training")
        except (RuntimeError, ValueError, KeyError) as e:
            print("Error in training loop:", e)
            traceback.print_exc()
        except BaseException as e:
            print("Unexpected error in training loop:", type(e).__name__, e)
            traceback.print_exc()
        finally:
            self.__db.execute("DELETE FROM runnings WHERE pplid = ?", (self.pplid,))

    def _forward(
        self,
        mode: str,
        epoch: int,
        verbose: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
            Perform a single epoch pass through the dataset.

            Parameters
            ----------
            mode : str
                Either "train" or "valid" to determine which loader and mode to use.
            epoch : int
                Current epoch number.
            verbose : list of str, optional
                Metrics to display live in the progress bar.

            Returns
            -------
            Dict[str, float]
                Dictionary of averaged metrics and loss for the epoch, including duration.

            Raises
            ------
            ValueError
                If mode is not 'train' or 'valid'.
        """
        if mode == "train":
            self.comps["model"].train()
            loader = self.trainDataLoader
        elif mode == "valid":
            self.comps["model"].eval()
            loader = self.validDataLoader
        else:
            raise ValueError("mode only can be 'train' or 'valid'")

        loader_tqdm = tqdm(loader, desc=f"Epoch {epoch+1}", leave=True)
        running_metrics = {m: 0.0 for m in self.settings["metrics"]}
        running_metrics["loss"] = 0.0
        start_time = time.perf_counter()

        for batch_idx, datas in enumerate(loader_tqdm):
            inpts = datas[0].to(self.device) 
            lbls = datas[1].to(self.device) 
            # lbls = torch.cat([
            #             t.unsqueeze(1) if t.ndim == 1 else t
            #             for t in lbls], dim=1)
            if mode == "train":
                self.comps["optimizer"].zero_grad()
                logits = self.comps["model"](inpts)
                loss = self.comps["loss"](logits, lbls)
                loss.backward()
                self.comps["optimizer"].step(loss=loss.item(), epoch=epoch)
            else:
                with torch.no_grad():
                    logits = self.comps["model"](inpts)
                    loss = self.comps["loss"](logits, lbls)

            for m in self.settings["metrics"]:
                running_metrics[m] += self.comps["metrics"][m](logits, lbls)
            running_metrics["loss"] += loss.item()

            if verbose:
                metrics_display = {
                    m: running_metrics[m] / (batch_idx + 1)
                    for m in running_metrics
                    if m in verbose
                }
                loader_tqdm.set_postfix(**metrics_display)

        averaged = {k: v / len(loader) for k, v in running_metrics.items()}
        averaged["duration"] = time.perf_counter() - start_time
        return averaged

    def is_running(self):
        """
        Check if the current process (identified by `pplid`) is currently running.

        Queries the `runnings` table for an entry with the matching `pplid`.

        Returns:
            int or bool: The `logid` of the running process if found, otherwise `False`.
        """
        rows = self.__db.query(
            "SELECT logid FROM runnings WHERE pplid = ?", (self.pplid,)
        )
        if rows:
            return rows[0][0]
        return False
    
    @property
    def should_running(self):
        """
        Determine whether the process should continue running.

        This checks the `parity` value for the current `pplid` in the `runnings` table.
        If the value is `'stop'`, the process should no longer continue.

        Returns:
            bool: `True` if the process should keep running, `False` if it should stop.
        """

        rows = self.__db.query(
            "SELECT parity FROM runnings WHERE pplid = ?", (self.pplid,)
        )
        if rows and rows[0][0]=='stop':
            return False
        return True

    def stop_running(self):
        """
        Mark the current running process to be stopped.

        If the process is currently running (i.e., has an associated `logid` in the `runnings` table),
        this updates the `parity` field to `'stop'`, signaling it to stop after the current iteration.
        Otherwise, it prints a message indicating that the process is not running.

        Returns:
            None
        """
        logid = self.is_running()
        if logid:
            self.__db.execute(
                "UPDATE runnings SET parity = ? WHERE logid = ?", ('stop', logid)
            )
            print(f"ppid:{self.pplid} will be stopped at logid:{logid} after current iteration")
        else:
            print("it is not running anywhere")
