import pytest
from typing import Dict, Any, Optional
from torch.utils.data import Dataset
from torch import nn
from abc import ABC

from PTLF.utils import Component, Model, DataSet, Loss, Optimizer, Metric


class ComponentTestSubclass(Component):
    """Concrete subclass for testing Component base class."""

    def __init__(self):
        super().__init__()
        self.args = ["foo", "bar"]
        self.setup_called = False
        self.received_args = None

    def _setup(self, args: Dict[str, Any], P=None) -> Optional[Any]:
        self.setup_called = True
        self.received_args = args
        return "setup done"


def test_component_init_and_check_args():
    comp = ComponentTestSubclass()
    assert comp.loc == "ComponentTestSubclass"
    assert comp.args == ["foo", "bar"]

    # Missing args keys
    assert comp.check_args({"foo": 1}) is False
    assert comp.check_args({"foo": 1, "bar": 2}) is True


def test_component_setup_success():
    comp = ComponentTestSubclass()
    result = comp.setup({"foo": 1, "bar": 2})

    assert result == comp
    assert comp.setup_called is True
    assert comp.received_args == {"foo": 1, "bar": 2}


def test_component_setup_fails_bad_args():
    comp = ComponentTestSubclass()
    with pytest.raises(ValueError):
        comp.setup({"foo": 1})  # Missing 'bar'


def test_component_setup_missing_setup_method():
    class NoSetupComponent(Component):
        pass

    comp = NoSetupComponent()
    comp.args = []  # no required args
    with pytest.raises(NotImplementedError):
        comp.setup({})



def test_component_private_setup_not_implemented():
    class DummyComponent(Component):
        def __init__(self):
            super().__init__()
            self.args = []

    comp = DummyComponent()
    with pytest.raises(NotImplementedError):
        comp._setup({})


def test_model_initialization():
    model = Model()
    assert isinstance(model, Component)
    assert isinstance(model, nn.Module)
    assert hasattr(model, "loc")


def test_dataset_initialization():
    dataset = DataSet()
    assert isinstance(dataset, Component)
    assert isinstance(dataset, Dataset)
    assert hasattr(dataset, "loc")


def test_loss_initialization():
    loss = Loss()
    assert isinstance(loss, Component)
    assert isinstance(loss, nn.Module)
    assert hasattr(loss, "loc")


def test_optimizer_initialization():
    optimizer = Optimizer()
    assert isinstance(optimizer, Component)
    assert isinstance(optimizer, nn.Module)
    assert hasattr(optimizer, "loc")
    assert hasattr(optimizer, "model_parameters")
    assert optimizer.model_parameters is None


def test_metric_initialization():
    metric = Metric()
    assert isinstance(metric, Component)
    assert isinstance(metric, nn.Module)
    assert hasattr(metric, "loc")
