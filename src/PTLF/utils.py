"""
This module provides
"""

import sys
import os
from typing import overload, List, Callable, Literal, Dict, Any, Union, Optional
import json
import inspect
import hashlib
from abc import ABC, abstractmethod
import warnings
import importlib
import sqlite3
from copy import deepcopy

import pandas as pd

from torch import nn
from torch.utils.data import Dataset

__all__ = [
    "load_component",
    "hash_args",
    "Component",
    "Model",
    "DataSet",
    "Loss",
    "Optimizer",
    "Metric",
    "Db",
    "extract_all_locs",
    "get_invalid_loc_queries",
]


class ComponentLoadError(Exception):
    """
    ComponentLoadError
    """


def load_component(
    loc: str, args: Optional[Dict[str, Any]] = None, setup: bool = True
) -> Callable:
    """
        Dynamically load and optionally initialize a component class.

        This utility imports a class from a given module path and instantiates it.
        If the class defines a `setup` method and `setup=True`, it calls `setup(args)`
        and returns the initialized component. Otherwise, it returns the raw instance.

        Parameters
        ----------
        loc : str
            Fully qualified class location in dot notation (e.g., 'CompBase.models.MyModel').
            If no dot is present, it is assumed the class is defined in `__main__`.

        args : dict, optional
            Dictionary of arguments to pass to the `setup()` method, if applicable.
            Defaults to an empty dict.

        setup : bool, optional
            Whether to invoke the component’s `setup` method after instantiation. Defaults to True.

        Returns
        -------
        Any
            An instance of the loaded class, either raw or configured via `setup()`.

        Raises
        ------
        ComponentLoadError
            If the specified class is not found in the target module.
        ImportError
            If the module cannot be imported.
    """
    args = args or {}

    # Parse module and class name
    if "." in loc:
        module_path, class_name = loc.rsplit(".", 1)
        if module_path in sys.modules:
            module = sys.modules[module_path]
            if getattr(module, "__spec__", None):
                module = importlib.reload(module)
        else:
            module = importlib.import_module(module_path)
    else:
        # No dot means class is in __main__
        module = sys.modules["__main__"]
        class_name = loc
        warnings.warn(
            f"{loc} component is not saved. "
            "Make sure to save it in an appropriate location before"
            "initiating an experiment, test, or report."
        )
    if not hasattr(module, class_name):
        raise ComponentLoadError(
            f"Class '{class_name}' not found in module '{module.__name__}'"
        )
    component_cls = getattr(module, class_name)
    component = component_cls()

    # If setup method exists and setup flag is True, call it with args
    if setup and hasattr(component, "_setup"):
        return component.setup(args)
    return component


class Component:
    """
        Base class for all components with dynamic loading capability.

        Attributes:
            loc (str): Location identifier for the component.
            args (dict): Expected keys for arguments.
    """

    def __init__(self, loc: str = None):
        self.loc = self.__class__.__name__ if loc is None else loc
        self.args = {}
        

    def check_args(self, args: dict) -> bool:
        """Check whether provided args contain all required keys."""
        return all(arg in args for arg in self.args)

    def setup(self, args: Dict[str, Any]) -> Optional[Any]:
        """
            Set up the component with provided arguments.

            Args:
                args: Dictionary of arguments to initialize the component.

            Returns:
                Optional[Any]: Initialized component or setup result.
        """
        if self.check_args(args):
            # print(154,args)
            try:
                self._setup(args)
                return self
            except AttributeError as exc:
                raise AttributeError(
                    f"Component '{self.loc}' does not implement '_setup'"
                ) from exc
        raise ValueError(f"Arguments {args} are incompatible with '{self.loc}'")

    def _setup(self, args: Dict[str, Any],P=None) -> Optional[Any]:
        """
            Private setup to be overridden in subclasses.

            Args:
                args: Dictionary of arguments.

            Returns:
                Optional[Any]
        """
        raise NotImplementedError(f"Component '{self.loc}' must implement '_setup'")


class Model(Component, nn.Module, ABC):
    """
        Abstract base class for PyTorch models with dynamic loading support.
    """

    def __init__(self):
        Component.__init__(self)
        nn.Module.__init__(self)


class DataSet(Component, Dataset, ABC):
    """
        Abstract base class for PyTorch datasets with dynamic loading support.
    """

    def __init__(self):
        Component.__init__(self)
        Dataset.__init__(self)


class Loss(Component, nn.Module, ABC):
    """
    Abstract base class for loss functions.
    """

    def __init__(self):
        Component.__init__(self)
        nn.Module.__init__(self)


class Optimizer(Component, nn.Module, ABC):
    """
    Abstract base class for optimizers.
    """

    def __init__(self):
        Component.__init__(self)
        nn.Module.__init__(self)
        self.model_parameters = None


class Metric(Component, nn.Module, ABC):
    """
    Abstract base class for metrics.
    """

    def __init__(self):
        Component.__init__(self)
        nn.Module.__init__(self)


class Db:
    """
        Lightweight SQLite wrapper with foreign key enforcement.

        Args:
            db_path (str): Path to the SQLite database file.

        Raises:
            FileNotFoundError: If the directory for the DB path doesn't exist.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

        # Check if directory exists
        dir_name = os.path.dirname(db_path)
        if dir_name and not os.path.isdir(dir_name):
            raise FileNotFoundError(f"Directory does not exist: {dir_name}")

        self._connect()

    def _connect(self) -> None:
        """Establishes a database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")

    def execute(self, query: str, params: tuple = ()) -> Optional[sqlite3.Cursor]:
        """
        Execute a SQL query (INSERT, UPDATE, DELETE).

        Args:
            query (str): SQL query string.
            params (tuple): Query parameters.

        Returns:
            sqlite3.Cursor or None
        """
        if not self.conn:
            raise ConnectionError("No database connection.")
        try:
            cur = self.conn.cursor()
            cur.execute(query, params)
            self.conn.commit()
            return cur
        except sqlite3.Error as e:
            print(f"[SQLITE ERROR] {e}")
            return None

    def query(self, query: str, params: tuple = ()) -> list:
        """
        Execute a SELECT query and fetch all results.

        Args:
            query (str): SQL query string.
            params (tuple): Query parameters.

        Returns:
            list: Query results.
        """
        cursor = self.execute(query, params)
        return cursor.fetchall() if cursor else []

    def close(self) -> None:
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



def hash_args(args: Dict[str, Any]) -> str:
    """
        Generate a SHA-256 hash from a dictionary of arguments.

        This is commonly used to uniquely identify a configuration or set of parameters.

        Parameters
        ----------
        args : dict
            The dictionary of arguments to be hashed. Must be JSON-serializable.

        Returns
        -------
        str
            A SHA-256 hash string representing the input dictionary.

        Raises
        ------
        TypeError
        If the dictionary contains non-serializable values.
    """
    dict_str = json.dumps(args, sort_keys=True, separators=(",", ":"))
    # print(dict_str)
    hhas  = hashlib.sha256(dict_str.encode()).hexdigest()
    # print(hhas)
    return hhas

def extract_all_locs(d: Union[Dict, List]) -> List[str]:
    """
    Recursively extract all 'loc' values from nested dictionaries or lists.
    """
    locs = []

    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, dict):
                if "loc" in v:
                    locs.append(v["loc"])
                    if 'args' in v:
                        
                        locs.extend(extract_all_locs(v['args']))
            elif isinstance(v, list):
                locs.extend(extract_all_locs(v))  # ✅ fix: process lists inside dicts
    elif isinstance(d, list):
        for item in d:
            locs.extend(extract_all_locs(item))  # ✅ recurse list items

    return locs

def get_invalid_loc_queries(d: Union[Dict, List], parent_key: str = "") -> List[str]:
    """
    Recursively identify keys with invalid 'loc' values (i.e., missing module path).
    Returns the full path to each invalid 'loc'.
    """
    queries = []

    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}>{k}" if parent_key else k

            if isinstance(v, dict):
                if "loc" in v:
                    loc_val = v["loc"]
                    if "." not in str(loc_val):
                        queries.append(new_key)
                # 👇 ensure traversal through *all* subkeys, not just "args"
                queries.extend(get_invalid_loc_queries(v, new_key))

            elif isinstance(v, list):
                for idx, item in enumerate(v):
                    item_key = f"{new_key}[{idx}]"
                    queries.extend(get_invalid_loc_queries(item, item_key))

    elif isinstance(d, list):
        for idx, item in enumerate(d):
            item_key = f"{parent_key}[{idx}]"
            queries.extend(get_invalid_loc_queries(item, item_key))

    return queries
