# PyTorch Training Pipeline

This project provides a modular and configurable training pipeline for PyTorch models. It allows you to define and manage your experiments through a JSON configuration file, enabling seamless model training, validation, checkpointing, and history tracking.

## Features

- **Modular Design**: Easily load different models, datasets, optimizers, and loss functions.
- **Configuration File**: Define experiments in a JSON file for easy setup and reproducibility.
- **Training and Validation**: Separate training and validation loops to evaluate model performance.
- **Checkpointing**: Save and load model states to resume training from the last checkpoint.
- **History Tracking**: Track and save training and validation loss for each epoch.

## File Structure

```
Your_directory
  └── Libs
    └── __init__.py
    ├── accuracies.py
    ├── datasets.py
    ├── losses.py
    ├── models.py
    └── optimizers.py
  ├── History
  ├── Configs
  ├── Weights
  

├── config.json

└── README.md
```

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Ensure all required packages are installed. You can use `requirements.txt` if provided.

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/pytorch-training-pipeline.git
cd pytorch-training-pipeline
```

2. Install required packages (if `requirements.txt` is available):

```bash
pip install -r requirements.txt
```

### Configuration

Define your experiment configuration in `config.json`. Below is an example:

```json
{
  'model_loc': 'Libs.models.model1',
 'DataSet_loc': None,
 'accuracy_loc': 'Libs.accuracies.Accu1',
 'loss_loc': 'Libs.losses.loss1',
 'optimizer_loc': 'Libs.optimizers.opt1',
 'piLn_name': '1st_try',
 'last': {
          'epoch': 0,
          'train_accuracy': 0,
          'train_loss': inf
        },
 'best': {
          'epoch': 0,
          'val_accuracy': 0,
          'val_loss': inf},
 'valid_folder': '../DataSets/PREPARED_2/valid-Copy/',
 'train_folder': '../DataSets/PREPARED_2/train-Copy/',
 'valid_batch_size': 4,
 'train_batch_size': 4,
 'weights_path': '1st_try/1st_try_w.pth',
 'config_path': '1st_try/1st_try.json',
 'history_path': '1st_try/1st_try_h.csv'
}
```
### Custom Components

#### Model

Define your model in `models/model1.py`:

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

### Running the Pipeline

Run the pipeline using the following command:

```bash
python pipeline.py
```

This script will:
1. Load the configuration from `config.json`.
2. Initialize the model, datasets, optimizer, and loss function.
3. Train the model, validating at each epoch.
4. Save the best model based on validation loss.
5. Track and save the training and validation loss history.


Sure! Here is a comprehensive README file for your GitHub repository that describes the `PipeLine` class.

```markdown
# PipeLine: A Flexible and Configurable Training Pipeline for PyTorch

The `PipeLine` class provides a flexible and configurable framework for training PyTorch models. It allows you to dynamically load models, datasets, loss functions, optimizers, and metrics from specified module locations. Additionally, it supports configuration through JSON files, enabling easy reuse and sharing of training setups.

## Table of Contents

- [Installation](#installation)
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
- [Contributing](#contributing)
- [License](#license)

## Installation

To use the `PipeLine` class, you need to have Python and PyTorch installed. You can install PyTorch from the official [PyTorch website](https://pytorch.org/get-started/locally/).

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/pipeline.git
```

## Usage

### Setting Up the Pipeline

You can configure the pipeline either programmatically or by using a configuration file. The setup process involves specifying the locations of the model, loss function, optimizer, and dataset.

#### Example

```python
from pipeline import PipeLine

pipeline = PipeLine(name="MyModel", config_path="path/to/config.json")

pipeline.setup(
    name="MyModel",
    model_loc="models.MyModel",
    accuracy_loc="metrics.accuracy",
    loss_loc="losses.CrossEntropyLoss",
    optimizer_loc="torch.optim.Adam",
    dataset_loc="datasets.MyDataset",
    train_folder="data/train",
    train_batch_size=32,
    valid_folder="data/valid",
    valid_batch_size=32,
    history_path="path/to/history.csv",
    weights_path="path/to/weights.pth",
    config_path="path/to/config.json",
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
    dataset_loc="datasets.MyDataset",
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
pipeline.setup(config_path="path/to/config.json", use_config=True)
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

### prepare_data

Sets up the data loaders for training and validation datasets.

#### Parameters
- `dataset_loc` (str): The module location of the dataset.
- `train_folder` (str): The path to the training data folder.
- `train_batch_size` (int): The batch size for training.
- `valid_folder` (str): The path to the validation data folder.
- `valid_batch_size` (int): The batch size for validation.

### update

Updates the configuration and saves the training history.

#### Parameters
- `data` (dict): A dictionary containing the training and validation metrics.

### train

Trains the model for the specified number of epochs.

#### Parameters
- `num_epochs` (int): The number of epochs to train the model.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

