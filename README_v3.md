# PipeLine: A Flexible and Configurable Training Pipeline for PyTorch

The `PipeLine` class provides a flexible and configurable framework for training PyTorch models. It allows you to dynamically load models, datasets, loss functions, optimizers, and metrics from specified module locations. Additionally, it supports configuration through JSON files, enabling easy reuse and sharing of training setups.

## Table of Contents

- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Setting Up the Pipeline](#setting-up-the-pipeline)
  - [Preparing Data](#preparing-data)
  - [Training the Model](#training-the-model)
  - [Saving and Loading Configurations](#saving-and-loading-configurations)
- [Methods](#methods)
  - [load_component](#load_component)
  - [load_optimizer](#load_optimizer)
  - [save_config](#save_config)
  - [setup](#setup)
  - [prepare_data](#prepare_data)
  - [update](#update)
  - [train](#train)
  - [validate](#validate)
- [Contributing](#contributing)
- [License](#license)

## Installation

To use the `PipeLine` class, you need to have Python and PyTorch installed. You can install PyTorch from the official [PyTorch website](https://pytorch.org/get-started/locally/).

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/pipeline.git
```

## Directory Structure

Your project directory should be structured as follows:

```
Your_directory
  ├── Libs
  │   ├── __init__.py
  │   ├── accuracies.py
  │   ├── datasets.py
  │   ├── losses.py
  │   ├── models.py
  │   └── optimizers.py
  ├── History
  ├── Configs
  ├── Weights
  ├── config.json
  └── README.md
```

- `Libs`: Contains the Python scripts for models, datasets, losses, optimizers, and accuracy metrics.
- `History`: Contains CSV files with training history for different pipelines.
- `Configs`: Contains JSON configuration files for different pipelines.
- `Weights`: Contains model weights for different pipelines.
- `config.json`: A sample configuration file.
- `README.md`: This README file.

## Usage

### Setting Up the Pipeline

You can configure the pipeline either programmatically or by using a configuration file. The setup process involves specifying the locations of the model, loss function, optimizer, and dataset.

#### Example

```python
from pipeline import PipeLine

pipeline = PipeLine(name="MyModel", config_path="Configs/mymodel_config.json")

pipeline.setup(
    name="MyModel",
    model_loc="Libs.models.MyModel",
    accuracy_loc="Libs.accuracies.accuracy",
    loss_loc="Libs.losses.CrossEntropyLoss",
    optimizer_loc="Libs.optimizers.Adam",
    dataset_loc="Libs.datasets.MyDataset",
    train_folder="data/train",
    train_batch_size=32,
    valid_folder="data/valid",
    valid_batch_size=32,
    history_path="History/mymodel_history.csv",
    weights_path="Weights/mymodel_weights.pth",
    config_path="Configs/mymodel_config.json",
    use_config=False,
    make_config=True,
    prepare=True
)
```

### Preparing Data

The `prepare_data` method sets up data loaders for training and validation datasets. You can customize the dataset by specifying its location and parameters.

#### Example

```python
pipeline.prepare_data(
    dataset_loc="Libs.datasets.MyDataset",
    train_folder="data/train",
    train_batch_size=32,
    valid_folder="data/valid",
    valid_batch_size=32
)
```

### Training the Model

To start training the model, use the `train` method and specify the number of epochs.

#### Example

```python
pipeline.train(num_epochs=10)
```

### Saving and Loading Configurations

The `PipeLine` class supports saving the configuration to a JSON file and loading it for future use.

#### Example

```python
# Save the current configuration
pipeline.save_config()

# Load the configuration from a file
pipeline.setup(config_path="Configs/mymodel_config.json", use_config=True)
```

## Methods

### load_component

Loads a component (model, loss function, metric, etc.) from a specified module location.

#### Parameters
- `module_loc` (str): The module location of the component.
- `**kwargs`: Additional arguments for component initialization.

#### Returns
- The instantiated component.

### load_optimizer

Loads an optimizer from a specified module location.

#### Parameters
- `module_loc` (str): The module location of the optimizer.
- `**kwargs`: Additional arguments for optimizer initialization.

#### Returns
- The instantiated optimizer.

### save_config

Saves the current configuration to a JSON file.

### setup

Configures the pipeline with the specified parameters or loads the configuration from a file.

#### Parameters
- `name` (str): The name of the pipeline.
- `model_loc` (str): The module location of the model.
- `accuracy_loc` (str): The module location of the accuracy metric.
- `loss_loc` (str): The module location of the loss function.
- `optimizer_loc` (str): The module location of the optimizer.
- `dataset_loc` (str): The module location of the dataset.
- `train_folder` (str): The path to the training data folder.
- `train_batch_size` (int): The batch size for training.
- `valid_folder` (str): The path to the validation data folder.
- `valid_batch_size` (int): The batch size for validation.
- `history_path` (str): The path to save the training history.
- `weights_path` (str): The path to save the model weights.
- `config_path` (str): The path to save the configuration file.
- `use_config` (bool): Whether to load the configuration from a file.
- `make_config` (bool): Whether to create a new configuration file.
- `prepare` (bool): Whether to prepare the data loaders.

---

##### Example Uses
----
###### case1
```python
pipeline.setup(
    config_path="Configs/mymodel_config.json",
    use_config=False,
    make_config=True,
    prepare=True
)
```
###### Summary

- **`config_path="Configs/mymodel_config.json"`**: Specifies the path for the configuration file.
- **`use_config=False`**: Indicates that the pipeline should not load an existing configuration file from the specified path.
- **`make_config=True`**: Signals the pipeline to create a new configuration file at the specified path with default or provided settings.
- **`prepare=True`**: Directs the pipeline to prepare the training and validation data loaders immediately after setup.

###### Overall Outcome
This code initializes the pipeline without loading previous settings, creates a new configuration file, and prepares the data loaders for training.
---
###### case2
```python
pipeline.setup(
    config_path="Configs/mymodel_config.json",
    use_config=False,
    make_config=True
)
```

###### Summary

- **`config_path="Configs/mymodel_config.json"`**: Specifies the path for the configuration file.
- **`use_config=False`**: Indicates that the pipeline should not load an existing configuration file from the specified path.
- **`make_config=True`**: Signals the pipeline to create a new configuration file at the specified path with default or provided settings.
- **`prepare=True`**: Directs the pipeline to prepare the training and validation data loaders immediately after setup.

###### Overall Outcome

This code initializes the pipeline without loading previous settings, creates a new configuration file, and prepares the data loaders for training.

#### Configuration File Path Handling

When initializing the `PipeLine` class, if the user does not specify a path for the configuration file, the pipeline will automatically create a new directory in the current working directory. This directory will be named after the pipeline name provided during initialization.

- **Automatic Directory Creation**: If the `config_path` is not given, the pipeline will create a folder with the same name as the pipeline in the current working directory. Inside this folder, it will save the configuration file, weights, and history files.

##### Example Behavior

If you initialize the `PipeLine` like this:

```python
pipeline = PipeLine(name="MyModelPipeline")
pipeline.setup(make_config=True)
```

The pipeline will create a directory named `MyModelPipeline` in the current working directory, and store the following files within it:

- Configuration file: `MyModelPipeline/MyModelPipeline.json`
- Weights file: `MyModelPipeline/MyModelPipeline_weights.pth`
- History file: `MyModelPipeline/MyModelPipeline_history.csv`

This automatic organization helps keep your project structured and ensures that all related files are easily accessible.

---


### prepare_data

Sets up the data loaders for training and validation datasets.

#### Parameters
- `dataset_loc` (str): The module location of the dataset.
- `train_folder` (str): The path to the training data folder.
- `train_batch_size` (int): The batch size for training.
- `valid_folder` (str): The path to the validation data folder.
- `valid_batch_size` (int): The batch size for validation.

#### `prepare_data(dataset_loc=None, train_folder=None, train_batch_size=None, valid_folder=None, valid_batch_size=None)`

The `prepare_data` method prepares the training and validation data loaders from the specified dataset class. Key functionalities include:

- **DataLoader Creation**: It instantiates the dataset class from the specified location and creates PyTorch DataLoaders for both training and validation datasets. This allows for efficient data loading and preprocessing during training.

- **Validation of Inputs**: The method checks if the necessary parameters (such as training and validation folders, and batch sizes) are provided. If any parameters are missing, it provides informative error messages to guide the user in correcting the configuration.

- **Random Cropping**: The method supports random cropping of audio files for variability in training, ensuring the model can generalize better by being exposed to different parts of the audio data.

##### Example Usage

```python
pipeline.prepare_data(dataset_loc="Libs.datasets.MyDataset", train_folder="data/train", valid_folder="data/valid")
```

This command will set up the DataLoaders for the specified dataset class and training/validation directories.


### update

Updates the configuration and saves the training history.

#### Parameters
- `data` (dict): A dictionary containing the training and validation metrics.

### train

Trains the model for the specified number of epochs.

#### Parameters
- `num_epochs` (int): The number of epochs to train the model.
#### `train(num_epochs=5)`

The `train` method is responsible for training the model over a specified number of epochs. Key functionalities include:

- **Weight Saving**: The method saves the model weights whenever a new best validation loss is achieved. This ensures that you can always revert to the best-performing model during training.

- **History Tracking**: At the end of each epoch, the method logs training and validation metrics, such as accuracy and loss, into a CSV file specified in the configuration. This history can be used to analyze the model's performance over time and visualize training progress.

- **Epoch Management**: The method continues training from the last completed epoch if the pipeline has been configured previously, allowing for seamless continuation of training sessions.

##### Example Usage

```python
pipeline.train(num_epochs=10)
```

This command will train the model for 10 epochs, saving weights and logging history after each epoch.

---


### validate

Evaluates the model on the validation dataset.

#### Returns
- `avg_loss` (float): The average validation loss.
- `accuracy` (float): The validation accuracy.




## Configuration File

The configuration file is used to specify the parameters for training a model with the `PipeLine` class. Below is an example of a configuration file in JSON format:

### Example: `mymodel_config.json`

```json
{
  "piLn_name": "MyModel",
  "model_loc": "Libs.models.MyModel",
  "DataSet_loc": "Libs.datasets.MyDataset",
  "accuracy_loc": "Libs.accuracies.accuracy",
  "loss_loc": "Libs.losses.CrossEntropyLoss",
  "optimizer_loc": "Libs.optimizers.Adam",
  "train_folder": "data/train",
  "valid_folder": "data/valid",
  "train_batch_size": 32,
  "valid_batch_size": 16,
  "weights_path": "Weights/mymodel_weights.pth",
  "history_path": "History/mymodel_history.csv",
  "config_path": "Configs/mymodel_config.json",
  "last": {
    "epoch": 0,
    "train_accuracy": 0,
    "train_loss": "inf"
  },
  "best": {
    "epoch": 0,
    "val_accuracy": 0,
    "val_loss": "inf"
  }
}
```

### Description of Configuration Fields:

- **piLn_name**: The name of the pipeline.
- **model_loc**: The module location of the model class (e.g., `Libs.models.MyModel`).
- **DataSet_loc**: The module location of the dataset class (e.g., `Libs.datasets.MyDataset`).
- **accuracy_loc**: The module location of the accuracy metric (e.g., `Libs.accuracies.accuracy`).
- **loss_loc**: The module location of the loss function class (e.g., `Libs.losses.CrossEntropyLoss`).
- **optimizer_loc**: The module location of the optimizer class (e.g., `Libs.optimizers.Adam`).
- **train_folder**: The directory path containing the training data.
- **valid_folder**: The directory path containing the validation data.
- **train_batch_size**: The batch size to use for training.
- **valid_batch_size**: The batch size to use for validation.
- **weights_path**: The file path where model weights will be saved.
- **history_path**: The file path for saving the training history as a CSV file.
- **config_path**: The file path for saving the configuration file.
- **last**: An object that stores the last epoch's information, including the epoch number, training accuracy, and training loss.
- **best**: An object that stores the best epoch's information, including the epoch number, validation accuracy, and validation loss.

### Usage

To load this configuration file in the `PipeLine` class, specify the `config_path` parameter when setting up the pipeline:

```python
pipeline = PipeLine(config_path="Configs/mymodel_config.json")
pipeline.setup(use_config=True)
```

This allows you to reuse the same settings for training your model without needing to redefine the parameters each time.


### Custom Components

#### Model

Define your model in `Libs/models.py`:

```python
import torch.nn as nn

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # Define model layers

    def forward(self, x):
        # Define forward pass
        return x
```

#### Dataset

Define your dataset in `Libs/datasets.py`:

```python
import torch
from torch.utils.data import Dataset

class CustomDataset1(Dataset):
    def __init__(self, data_path):
        # Load and preprocess data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return data and label
        return data, label
```

#### Optimizer

Define your optimizer in `Libs/optimizers.py`:

```python
import torch.optim as optim

def get_optimizer(model, lr, momentum):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
```

#### Loss Function

Define your loss function in `Libs/losses.py`:

```python
import torch.nn as nn

def get_loss():
    return nn.MSELoss()
```







## Customization

To effectively use the `PipeLine` class, users are encouraged to customize the `Libs` folder to suit their specific needs. You can create or modify the following files to define your models, datasets, accuracies, losses, and optimizers:

- **`Libs/models.py`**: Define your model classes here. You can create custom neural network architectures or use pre-existing ones.
  
- **`Libs/datasets.py`**: Implement your dataset classes to load and preprocess your data.
  
- **`Libs/accuracies.py`**: Create custom accuracy metrics for evaluating your models.
  
- **`Libs/losses.py`**: Define your loss functions or use existing ones from PyTorch.
  
- **`Libs/optimizers.py`**: Implement or import custom optimizers for training your models.

By following this structure, you can keep your project organized and make it easier to manage different models and datasets.



## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.

