Concepts
========

In **PyTorchLabFlow**, **everything is a Component** — from models and datasets to losses, optimizers, and custom blocks. This modular design enables users to configure, nest, and reuse components dynamically, simplifying the construction and maintenance of complex deep learning workflows.

Component
---------

A **Component** is a reusable, self-contained code block that can represent any part of your ML workflow — a model, dataset, optimizer, loss function, metric, or even a custom processing step.

Each Component is defined by:

- ``loc``: A string specifying the Python import path of the component class.
- ``args``: A dictionary of parameters to configure or initialize the component.

The key power of Components is **infinite nesting**: any component can accept other components inside its ``args``. This enables deeply composable, declarative, and dynamically configurable systems.

Key Features
~~~~~~~~~~~~

- **Dynamic Loading**: Components are loaded at runtime using the ``loc`` path.
- **Flexible Composition**: Nested components allow reusable and interchangeable blocks.
- **Argument Validation**: Each component validates its required arguments before setup.
- **Single Interface**: Everything behaves consistently through the same `Component` interface.

Core Behavior
~~~~~~~~~~~~~

All components inherit from the abstract base class ``Component``, which provides core logic for:

- ``check_args(args: dict)``: Validates required arguments.
- ``setup(args: dict)``: Triggers component initialization via the subclass's ``_setup()`` method.

Specialized Component Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Built-in Component Types
   :widths: 20 20 60
   :header-rows: 1

   * - Type
     - Class
     - Inherits From
   * - Model
     - ``Model``
     - ``Component``, ``torch.nn.Module``
   * - Dataset
     - ``DataSet``
     - ``Component``, ``torch.utils.data.Dataset``
   * - Loss
     - ``Loss``
     - ``Component``, ``torch.nn.Module``
   * - Optimizer
     - ``Optimizer``
     - ``Component``, ``torch.nn.Module``
   * - Metric
     - ``Metric``
     - ``Component``, ``torch.nn.Module``

.. note::

   - ``DataSet``: Loads and optionally pre-processes data.
   - ``Model``: Wraps your PyTorch models with config-driven construction.
   - ``Loss``: Defines the loss function used for training.
   - ``Optimizer``: Standard PyTorch optimizers, wrapped for config-driven setup.
   - ``Metric``: Custom metrics for evaluation.

Example: Custom Model Component
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class MyModel(Model):
        def __init__(self):
            super().__init__()
            self.args = {"input_dim", "output_dim"}

        def _setup(self, args):
            self.linear = nn.Linear(args["input_dim"], args["output_dim"])
            return self

Example: Nesting Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    {
        "loc": "my_project.models.TransformerClassifier",
        "args": {
            "backbone": {
                "loc": "my_project.models.TransformerEncoder",
                "args": {
                    "dim": 256,
                    "depth": 6
                }
            },
            "num_classes": 10
        }
    }

A deeper discussion on the flexibility, nesting power, and real-world use of Components is available in this Medium article:
👉 `Introducing Component in PyTorchLabFlow <https://medium.com/@bbek-anand/introducing-component-in-pytorchlabflow-5dfcfe41498d>`_

To explore all available methods and internals of the base class, visit the `Component API <../api/utils.html#PTLF.utils.Component>`_


Pipeline
--------

A **Pipeline** organizes an entire machine learning experiment as a composition of Components. It tracks and manages configuration, reproducibility, and structure.

Each pipeline is uniquely identified by an experiment ID (`expid`). It keeps all hyperparameters, model setup, data loading, and training behavior in a single, reusable definition.

Creating a New Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create a new pipeline from scratch
    P = PipeLine()
    P.new(pplid='my_first_experiment', args=experiment_args)

You define your experiment structure by passing nested Components as the ``args``.

To explore the full functionality of the :class:`PTLF.experiment.PipeLine`, including experiment loading, checkpointing, and tracking, see the `PipeLine API <../api/experiment.html#PTLF.experiment.PipeLine>`_


.. ---

.. Next Steps
.. ==========

.. - 📖 **Explore the API Documentation**:
..   - :ref:`PTLF.component <api/component>`
..   - :ref:`PTLF.model <api/model>`
..   - :ref:`PTLF.data <api/data>`
..   - :ref:`PTLF.loss <api/loss>`
..   - :ref:`PTLF.optim <api/optim>`
..   - :ref:`PTLF.metrics <api/metrics>`
..   - :ref:`PTLF.experiment <api/experiment>`

.. - 🚀 **Jump to**: :doc:`../api/index` for full API index.
