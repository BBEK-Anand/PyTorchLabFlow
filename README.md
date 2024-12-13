# PyTorchLabFlow
[![PyPI version](https://badge.fury.io/py/PyTorchLabFlow.svg)](https://badge.fury.io/py/PyTorchLabFlow)
[![Downloads](https://static.pepy.tech/badge/Pytorchlabflow)](https://pepy.tech/project/Pytorchlabflow)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?style=flat&logo=github)](https://github.com/BBEK-Anand/PyTorchLabFlow)

---

**PyTorchLabFlow** is your go-to offline solution for managing PyTorch experiments with ease. Run experiments securely on your local machine, no data sharing with third parties. Need more power? Seamlessly transfer your setup to a high-end system without any reconfiguration. whether you are on a laptop or a workstation, PyTorchLabFlow ensures flexibility and privacy, allowing you to experiment anywhere, anytime, without internet dependency.

For end to end use case check [Military_AirCraft_Classification](https://github.com/BBEK-Anand/Military_AirCraft_Classification)

# Table of Contents
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Workflow](#workflow)
- [Functions](#functions)
- [PipeLine](#pipeline)
- [Tutorial](https://github.com/BBEK-Anand/DeepFake-Detection/blob/main/PyTorchLabFlow.ipynb)
- [Templates](#templates)
- [Credits](#credits)
- [Contributing](#contributing)
- [License](#license)
# installation
```python
pip install PyTorchLabFlow
```

---

# Directory Structure

PyTorchLabFlow works on the following directory sructure, this will be created by using `setup_project`.

```
Your_directory
  ├── Libs
  │   ├── __init__.py
  │   ├── accuracies.py
  │   ├── datasets.py
  │   ├── losses.py
  │   ├── models.py
  │   └── optimizers.py
  ├── internal
  │   ├── Histories
  │   ├── Configs
  │   ├── Weights
  │   ├── Archived
  │   ├── config.json
  │   ├── Default_Config.json
  │   ├── Archived
  │   │   ├── Histories
  │   │   ├── Configs
  │   │   ├── Weights
  │   │   └── config.json
  │   └── Transfer
  │       ├── Histories
  │       ├── Configs
  │       ├── Weights
  │       └── config.json
  ├── Modeling.ipynb
  └── Training.ipynb
```

- `Libs`: Contains the Python scripts for models, datasets, losses, optimizers, and accuracy metrics.
- `History`: Contains CSV files with training history for different pipelines.
- `Configs`: Contains JSON configuration files for different pipelines.
- `Weights`: Contains model weights for different pipelines.
- `config.json`: A sample configuration file.


# Workflow
  - [Setup Project](#step-1-setup-project)
  - [Organize Components](#step-2-organize-components)
  - [Configure Defaults](#step-3-configure-defaults)
  - [Model Creation](#step-4-model-creation)
  - [Training](#step-5-training)
  - [Evaluate Performance](#step-6-evaluate-performance)
  - [Other Features](#step-7-other-features)

## Step 1: Setup Project
To set up your project, run the following in a Python script or Jupyter notebook:

```python
from PyTorchLabFlow import setup_project

setup_project("YourProjectName")
```

This will create the following directories:
- `Libs/` for storing models, datasets, optimizers, etc.
- `internal/` for configuration, weights, and histories
- `DataSets/` for organizing your training and validation data

Navigate to the newly created project directory and create a new Jupyter notebook file named `modelling.ipynb`.

## Step 2: Organize Components
- Place your components in the appropriate files:
  - **Loss functions**: `Libs/losses.py`
  - **Optimizer functions**: `Libs/optimizers.py`
  - **Accuracy functions**: `Libs/accuracies.py`
  - **Models**: `Libs/models.py`
  - **Datasets**: `Libs/datasets.py`
  
  For example, you can create multiple accuracy, loss, and optimizer functions inside their respective files.

## Step 3: Configure Defaults
Set up default configurations by calling the `set_default_config()` function:

```python
from PyTorchLabFlow import set_default_config

set_default_config({
    "accuracy_loc": "Libs.accuracies.testAccuracy",
    "loss_loc": "Libs.losses.testLoss",
    "optimizer_loc": "Libs.optimizers.testLoss",
    "train_data_src": "DataSets/train",
    "valid_data_src": "DataSets/valid",
    "train_batch_size": 32,
    "valid_batch_size": 32
})
```

This sets default locations and batch sizes for components.
from theabove example, values of "accuracy_loc","loss_loc" and "optimizer_loc" should be correct, i.e the functions testAccuracy, testLoss, testLoss should be inside the respective files.

## Step 4: Model Creation
In `modelling.ipynb`, define your model class and dataset class. Test compatibility by calling:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from PytorchLabFlow import test_mods
class testCNN(nn.Module):
    def __init__(self):
        super(testCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.fc1 = nn.Linear(8*64*64, 16)
        self.fc2 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc2(x))
        return x

class testDS(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
                            transforms.Resize((128,128)),  # Resize images 
                            transforms.ToTensor()        # Convert images to tensor
                        ])
        self.image_paths = []
        self.labels = []
        
        # Map directories with 'real' or 'fake' in their name to corresponding labels
        self.class_to_idx = {'real': 1, 'fake': 0}  # 1 for real, 0 for fake
        
        # Traverse through subdirectories
        for subdir in os.listdir(root_dir):
            class_name = None
            
            # Identify if subdir is 'real' or 'fake' by checking the directory name
            if 'real' in subdir.lower():
                class_name = 'real'
            elif 'fake' in subdir.lower():
                class_name = 'fake'
            
            if class_name:
                class_dir = os.path.join(root_dir, subdir)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        # Only load .jpg files
                        if img_name.lower().endswith('.jpg'):
                            self.image_paths.append(img_path)
                            self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms if provided
        image = self.transform(image)
        
        return image, label

```
```python
P = test_mods(model=testCNN(), dataset=testDS)
P.train(num_epochs=2)
```
If the dataset and model are compatible, P.train will run and it will start training, if model and dataset class are incompatible then it show error and you have too modify the classes. Otherwise paste the whole classes of model and dataset in respective files.
## Step 5: Training
Create another notebook file, `training.ipynb`, to train your model. To start a new experiment, call:

```python
from PyTorchLabFlow import train_new

train_new(
    name="exp01", 
    model_loc="Libs.models.testCNN",
    dataset_loc="Libs.datasets.testDS",
    train_data_src="DataSets/train", 
    valid_data_src="DataSets/valid"
)
```
Here the parameter name is your experiment pipeline(ppl) this should be unique for each experiment. To know about previous experiment pipelines use `get_ppls(mode='name')`.

This will initialize the training pipeline and return an instance. You can also use `re_train()` to re-train existing experiments.

## Step 6: Evaluate Performance
Once the training is complete, check the model's performance by visualizing the results:

```python
from PyTorchLabFlow import performance_plot

performance_plot(ppl="exp01")
```

This will plot the accuracy and loss over the training epochs.
## Step 7: Other Features
### Archive an experiment
If you want to remove experiment/s then use `archive` function. this will move all files releted to an experiment to `Archived` folder For unarchiving follow the documentation.

### Delete an experiment
If you want to delete (archive does not delete a pipeline) then use `delete` function after archiving the pipeline.

### Transfer an experiment
If you want to transfer experimment/s to a different system, use `transfer` function. 
    
    - By using `transfer(type='export',mode='copy')` copy all the files to `Transfer` folder. (`Transfer` folder is inside `your_project_root/internal`)
    - setup new project in your other(high-end) system, then this will create the same directory structure.
    - copy the `Transfer` folder( experiment configurations) and `Libs` folder( experiment components) from your 1st(low-end system/laptop) system and replace these in your 2nd(high-end) system's respectively their places.
    - create a new jupyter file and use `transfer(type='import',mode='copy')`, it will copy all the files from `Transfer` folder to `internal` folder.
    - then use `re_train` or `multi_train` for training.

---

<h1 style="font-size: 100px;" id="functions">Functions</h1>

Following functions are 

  - [setup_project](#setup_project)
  - [get_ppls](#get_ppls)
  - [verify](#verify)
  - [up2date](#up2date)
  - [set_default_config](#set_default_config)
  - [test_mods](#test_mods)
  - [train_new](#train_new)
  - [re_train](#re_train)
  - [use_ppl](#use_ppl)
  - [performance_plot](#performance_plot)
  - [multi_train](#multi_train)
  - [get_model](#get_model)
  - [archive](#archive)
  - [delete](#delete)
  - [setup_trasfer](#setup_trasfer)
  - [transfer](#transfer)

---

# setup_project
Create the directory structure for a new machine learning project.

    This function sets up the required directories and files for organizing datasets, 
    models, and configurations for a machine learning project. It creates a base project 
    structure with folders like 'DataSets', 'Libs', 'internal', and 'Archived', and 
    generates some basic template files.

## Parameters
    
### project_name : str, optional
        The name of the project directory to create. Defaults to 'MyProject'.
### create_root : bool, optional
        If set to False, the project will be created in the current directory without creating a root folder. Defaults to True.
## Returns
    
    None
        The function prints the status of the setup process.

    Notes
    
    - This function will create template Python files under the 'Libs' directory for models, datasets, accuracies, losses, and optimizers.
    - It will also generate configuration JSON files in the 'internal' directory.

# get_ppls
Retrieves pipeline information based on the specified mode and configuration.

    This function reads from different configuration files depending on the `config` parameter
    and returns pipeline information according to the `mode` parameter. The function supports
    three modes to retrieve different aspects of pipeline data and three configurations to specify
    which set of pipelines to retrieve.

## Parameters
    
### mode : str, optional
        Determines the type of information to return. Options include:
            - 'name' (default): Returns a list of experiment names.
            - 'epoch': Returns a list of the last trained epochs for each experiment.
            - 'all': Returns a dictionary containing the name, last epoch, and validation accuracy for each experiment.

### config : str, optional
        Specifies the configuration file to use. Options include:
            - 'internal' (default): For yor base enviroment, Uses the configuration file located at "internal/config.json".
            - 'archive': For archived experiments, Uses the configuration file located at "internal/Archived/config.json".
            - 'transfer': For the experiments which are for/from Transfer to other machine, Uses the configuration file located at "internal/Transfer/config.json".

## Returns
    
    list or dict
        Depending on the `mode`, the return type will vary:
            - If `mode` is 'name', a list of experiment names is returned.
            - If `mode` is 'epoch', a list of last trained epochs is returned.
            - If `mode` is 'all', a dictionary with detailed experiment information is returned.

## Raises
    
    FileNotFoundError
        If the configuration file specified by the `config` parameter does not exist.
    JSONDecodeError
        If there is an error decoding the JSON file.

# verify
Verifies the existence or uniqueness of a pipeline based on the given mode and configuration.

    This function checks if a pipeline or its components already exist in the specified configuration 
    based on the provided `mode`. It can verify by pipeline name, model-dataset combination, or training 
    configurations. Additionally, it can log information about the presence of duplicates.

## Parameters
  
### ppl : str, list, dict
        - If `mode` is 'name', `ppl` should be a string representing the pipeline name.
        - If `mode` is 'mod_ds', `ppl` should be a dictionary containing 'model_loc' and 'DataSet_loc' keys.
        - If `mode` is 'training', `ppl` should be a dictionary containing training configuration details:
            - 'optimizer_loc', 'train_batch_size', 'valid_batch_size', 'accuracy_loc', 'loss_loc', 'train_data_src', and 'valid_data_src'.
        - If `mode` is 'all', `ppl` should be a dictionary with 'piLn_name', 'model_loc', 'DataSet_loc', and training configuration details as described above.

### mode : str, optional
        Specifies the type of verification to perform. Options include:
            - 'name' (default): Verifies if the pipeline name exists in the configuration.
            - 'mod_ds': Verifies if the combination of model and dataset already exists in the configuration.
            - 'training': Verifies if the combination of training configurations (optimizer, batch sizes, accuracy, loss, data sources) already exists.
            - 'all': Checks for commonalities across all specified modes (name, model-dataset, training).

### config : str, optional
        Specifies the configuration file to use. Options include:
            - 'internal' (default): Uses the `internal/config.json` file.
            - 'archive': Uses the `internal/Archived/config.json` file.
            - 'transfer': Uses the `internal/Transfer/config.json` file.

### log : bool, optional
        If True, logs information about existing pipelines or combinations that match the query. Defaults to True.

## Returns
    
    bool, list, or str
        - If `mode` is 'name', returns the pipeline name if it exists, otherwise False.
        - If `mode` is 'mod_ds', returns a list of names where the model-dataset combination matches, or False if no matches are found.
        - If `mode` is 'training', returns a list of names where the training configurations match, or False if no matches are found.
        - If `mode` is 'all', returns a list of commonalities between all specified modes or a message indicating common pipelines across all modes.

## Raises
  
    FileNotFoundError
        If the specified configuration or experiment files are not found.
    JSONDecodeError
        If there is an error decoding the JSON files.

# up2date
    Updates the root configuration file with the latest information from individual experiment JSON files.

    - This function reads experiment data from JSON files in the `Configs` directory based on the specified `config`parameter and updates the corresponding root configuration file (`config.json`) with the latest epoch and validation accuracy for each experiment. 
    - If an experiment is new or has updated information, it reflects those changes in the root configuration file.

## Parameters
    
### config : str, optional
        Specifies which configuration files to update. Options include:
            - 'internal' (default): Updates the `internal/config.json` file with data from the `internal/Configs/` folder.
            - 'archive': Updates the `internal/Archived/config.json` file with data from the `internal/Archived/Configs/` folder.
            - 'transfer': Updates the `internal/Transfer/config.json` file with data from the `internal/Transfer/Configs/` folder.

### Returns
    
    None
        This function does not return any value. It updates the specified configuration file directly.

### Raises
    
    FileNotFoundError
        - If the specified configuration or experiment files are not found.
    JSONDecodeError
        - If there is an error decoding the JSON files.

# set_default_config      

    Set the default configuration for the project.

    - This function saves the provided configuration data to 'internal/Default_Config.json'. 
    - The configuration includes paths to accuracy, loss, optimizer, dataset, and batch sizes for training and validation.

## Parameters
    
### data : dict
        A dictionary containing the configuration settings. Keys should include 'accuracy_loc', 'loss_loc', 'optimizer_loc', 'train_data_src', 'valid_data_src', 'train_batch_size', and 'valid_batch_size'.

## Returns

    None
        The function writes the configuration to a JSON file.

## Examples
   
    ```python
    set_default_config({
            "accuracy_loc": "Libs.accuracies.BinAcc",
            "loss_loc": "Libs.losses.BinLoss",
            "optimizer_loc": "Libs.optimizers.AdamOpt",
            "train_data_src": "DataSets/train",
            "valid_data_src": "DataSets/valid",
            "train_batch_size": 32,
            "valid_batch_size": 32
        })
    # Saves the configuration to the default configuration file.
    ```
    

# test_mods
Configures and initializes a testing pipeline for evaluating a machine learning model.

    This function sets up a testing pipeline by configuring the model, dataset, and various paths for saving metrics, 
    optimizer state, and model weights. It uses default configuration values if some parameters are not provided.

## Parameters:
    - dataset (Type[Dataset], optional): The dataset class to be used for testing. Must be a subclass of `Dataset`.
    - model (Type[nn.Module], optional): The model to be tested. Must be a subclass of `nn.Module`.
    - model_loc (str, optional): The location of the model within the `Libs.models` module. If not fully qualified, it will be prefixed with 'Libs.models.'.
    - accuracy_loc (str, optional): The location for saving accuracy metrics within the `Libs.accuracies` module. If not provided, uses default from configuration file.
    - loss_loc (str, optional): The location for saving loss metrics within the `Libs.losses` module. If not provided, uses default from configuration file.
    - optimizer_loc (str, optional): The location for saving optimizer state within the `Libs.optimizers` module. If not provided, uses default from configuration file.
    - dataset_loc (str, optional): The location of the dataset class within the `Libs.datasets` module. If not fully qualified, it will be prefixed with 'Libs.datasets.'.
    - train_data_src (str, optional): Source of training data. If not provided, uses default from configuration file.
    - train_batch_size (int, optional): Batch size for training data. If not provided, uses default from configuration file.
    - valid_data_src (str, optional): Source of validation data. If not provided, uses default from configuration file.
    - valid_batch_size (int, optional): Batch size for validation data. If not provided, uses default from configuration file.
    - prepare (bool, optional): Whether to prepare the pipeline or not. Default is False.

## Returns:
    - PipeLine: An instance of the `PipeLine` class configured for testing with the provided or default parameters.
    

# train_new
Initializes and sets up a new pipeline for training a machine learning model.

    - This function configures a new training pipeline by specifying the model, dataset, loss function, accuracy metric, optimizer, and various training parameters. 
    - It uses default values from a configuration file(saved using save_default_config) if certain parameters are not provided. 
    - The function verifies the uniqueness of the pipeline name and initializes a `PipeLine` instance if the name is not already in use.

## Parameters
    
### name : str, optional
        The name of the pipeline. If not provided, the pipeline will not be created.
    
### model_loc : str, optional
        The location of the model module. If a simple name is provided, it is prefixed with 'Libs.models.'.

### loss_loc : str, optional
        The location of the loss function module. If a simple name is provided, it is prefixed with 'Libs.losses.'.
    
### accuracy_loc : str, optional
        The location of the accuracy metric module. If a simple name is provided, it is prefixed with 'Libs.accuracies.'.

### optimizer_loc : str, optional
        The location of the optimizer module. If a simple name is provided, it is prefixed with 'Libs.optimizers.'.

### dataset_loc : str, optional
        The location of the dataset module. If a simple name is provided, it is prefixed with 'Libs.datasets.'.
    
### train_data_src : str, optional
        The source path for training data. Defaults to the value specified in the default configuration file if not provided.
    
### valid_data_src : str, optional
        The source path for validation data. Defaults to the value specified in the default configuration file if not provided.
    
### train_batch_size : int, optional
        The batch size for training data. Defaults to the value specified in the default configuration file if not provided.
    
### valid_batch_size : int, optional
        The batch size for validation data. Defaults to the value specified in the default configuration file if not provided.
    
### prepare : callable, optional
        A function or callable to prepare the pipeline. This function is called during the setup of the pipeline.

## Returns

    PipeLine
        An instance of the `PipeLine` class configured with the specified parameters, or `None` if the pipeline name is already in use.

## Notes

    - If the `name` provided is already used by an existing pipeline, a new pipeline will not be created.
    - Default values are fetched from the "internal/Default_Config.json" configuration file.
    - Paths for weights, history, and configuration files are constructed based on the provided `name`.

# re_train

    Re-trains an existing pipeline or initializes a new pipeline with the provided configuration.

    - This function sets up and optionally trains a `PipeLine` instance based on the specified configuration or pipeline name. 
    - If a pipeline name (`ppl`) is provided, it constructs the configuration path from the pipeline name. 
    - If `num_epochs` is specified and greater than zero, it performs training for the given number of epochs.

## Parameters
  
### ppl : str, optional
        - The name of the pipeline for which to re-train or initialize. If provided, the function constructs the configuration path as "internal/Configs/{ppl}.json". 
        - If `ppl` is `None`, the `config_path` must be specified.
    
### config_path : str, optional
        - The path to the configuration file. This parameter is ignored if `ppl` is provided, as the configuration path will be constructed from the `ppl` name.
    
### train_data_src : str, optional
        - The source path for training data. This is used only if `num_epochs` is greater than zero and a new training session 
        is to be started.
    
### valid_data_src : str, optional
        - The source path for validation data. 
        - This is used only if `num_epochs` is greater than zero and a new training session is to be started.
    
### prepare : callable, optional
        A function or callable to prepare the pipeline. This function is called during the setup of the pipeline if `num_epochs` 
        is zero or if `prepare` is specified.
    
### num_epochs : int, optional
        The number of epochs for training. If greater than zero, the `PipeLine` will be trained for this many epochs. If zero, 
        only the pipeline will be set up without training.

## Returns
    
    PipeLine
        An instance of the `PipeLine` class, set up and optionally trained according to the provided parameters.

## Notes
    
    - If both `ppl` and `config_path` are provided, `ppl` takes precedence, and `config_path` will be constructed from `ppl`.
    - The function will initialize a new `PipeLine` instance if the `ppl` parameter is provided or if `config_path` is specified.
    - Training is only performed if `num_epochs` is greater than zero.
    
# use_ppl

    Configures and initializes a pipeline for an existing model or creates a new pipeline if necessary.

    - This function either sets up a pipeline based on an existing configuration and trained model weights, or trains a new model if no trained pipeline is available. 
    - It also updates and verifies the configuration before use.

## Parameters:
    - ppl (str): The name of the existing pipeline configuration to be used.
    - trained (bool, optional): If True, uses a pre-trained model. If False, trains a new model. Default is True.
    - name (str, optional): The name for the new pipeline. Used to save new configurations, weights, and histories.
    - loss_loc (str, optional): The location for saving loss metrics. If a short name is provided, it will be prefixed with 'Libs.losses.'.
    - accuracy_loc (str, optional): The location for saving accuracy metrics. If a short name is provided, it will be prefixed with 'Libs.accuracies.'.
    - optimizer_loc (str, optional): The location for saving optimizer state. If a short name is provided, it will be prefixed with 'Libs.optimizers.'.
    - train_data_src (str, optional): The source for the training data. Overrides the source in the existing configuration.
    - valid_data_src (str, optional): The source for the validation data. Overrides the source in the existing configuration.
    - train_batch_size (int, optional): Batch size for training data. Overrides the value in the existing configuration.
    - valid_batch_size (int, optional): Batch size for validation data. Overrides the value in the existing configuration.
    - prepare (bool, optional): Whether to prepare the pipeline before running. Default is None.

## Returns:
    PipeLine: 
      - An instance of the `PipeLine` class, either initialized with the provided or default configuration 
                and pre-trained weights, or trained from scratch if no pre-trained pipeline is found.

## Notes:
    - If the pipeline is not already trained, a new one is created using the provided configuration.
    - If `trained=True`, copies the existing model weights and history to the new pipeline paths.
    - The `config_path`, `weights_path`, and `history_path` for the new pipeline are generated based on the `name` parameter.
    - The configuration file is updated and saved before initializing the pipeline.

## Raises:
    - Any verification issues with the provided pipeline configuration are printed and no pipeline is returned.


# performance_plot    

    Plots performance metrics (accuracy and loss) over epochs for one or more pipelines.

    - This function generates plots for training and validation accuracy, as well as training and validation loss, using data from a CSV file or a DataFrame. 
    - It can handle single or multiple pipelines, and plots the performance metrics for each specified pipeline.

## Parameters
    
### ppl : str, list, optional
        - The name(s) of the pipeline(s) for which to plot performance metrics. 
        - If `None`, it fetches all pipeline names using `get_ppls`. If `ppl` is a list, it plots performance metrics for each pipeline in the list. If `ppl` is a string, it is treated as a single pipeline name.

### history : str, optional
        - The path to the CSV file containing performance metrics (e.g., accuracy and loss) over epochs. 
        - If `None`, it defaults to the file located in `internal/Histories/` directory with the name corresponding to the pipeline name.

### matrics : list of str, optional  
        - List of performance data to plot (e.g., `["train_accuracy", "train_loss"]`). If `None`, defaults to `["train_accuracy", "train_loss", "val_accuracy", "val_loss"]`.

### config : str, optional
        The configuration type to use when retrieving the pipeline. Options are:
        - `internal` [default]: Uses configurations from the `internal/` directory.
        - `archive`: Uses configurations from the `internal/Archived/` directory.
        - `transfer`: Uses configurations from the `internal/Transfer/` directory.

### figsize` : tuple, optional  
        Size of the plot. Default is `(6, 4)`.
## Returns
    
    fig, ax:  
        - Matplotlib `Figure` and `Axes` objects containing the generated plots. These can be used to further customize or save the plots. If no further customization is needed, the function only displays the plots and returns `None`.

## Notes

    - The function updates the configuration before attempting to load the performance history.
    - If both `history` and `df` are `None`, an error message is printed.
    - If the DataFrame is empty, an appropriate message is printed.
    

# multi_train

    - Train or re-train multiple pipelines up to the specified number of epochs.
    - This function checks the number of epochs each pipeline has already been trained for, and continues training until the specified `last_epoch` is reached. 
    - The training configuration for each pipeline is read from a corresponding JSON file.

## Parameters
    
### ppl : list of str, optional
        A list of pipeline names to train. If not provided, all pipelines will be selected.
### last_epoch : int, optional
        The maximum number of epochs to train for. Defaults to 10.

## Returns
    
    None
        - The function prints training progress and status messages.

## Notes
  
    - If the pipeline has already been trained for the specified `last_epoch`, it will be skipped.
    - Each pipeline's configuration is expected to be located at 'internal/Configs/{pipeline_name}.json'.


# get_model

    - Retrieves the model class or its name for a given pipeline or a list of pipelines.
    - This function fetches the model class or its name based on the specified pipeline(s). 
    - It can handle single pipeline names, lists of pipeline names, or return a list of models corresponding to those pipelines.

## Parameters
    
### ppl : str, list, optional
        - The name(s) of the pipeline(s) for which to retrieve the model. 
        - If `ppl` is `None`, it fetches all pipeline names using `get_ppls`. 
        - If `ppl` is a list, it returns a list of models for each pipeline in the list. 
        - If `ppl` is a string, it is treated as a single pipeline name.
    
### name : bool, optional
        - If `True`, returns the name of the model class as a string. If `False`, returns an instance of the model class. 
        - The default is `True`.
    
### config : str, optional
      The configuration type to use when retrieving the pipeline. Options are:
        - `internal` [default]: Uses configurations from the `internal/` directory.
        - `archive`: Uses configurations from the `internal/Archived/` directory.
        - `transfer`: Uses configurations from the `internal/Transfer/` directory.

## Returns
    str, object, list of str, list of objects
        - If `ppl` is a single string, returns either the model name (if `name=True`) or an instance of the model (if `name=False`).
        - If `ppl` is a list of strings, returns a list of model names or instances for each pipeline in the list.
        - If `ppl` is `None`, retrieves all pipeline names and returns them in the specified format.
    
## Notes
    
    - The function updates the configuration before attempting to load the model.
    - Model classes are imported dynamically based on the module location specified in the pipeline configuration.
    - The module location should be a valid Python module path, and the model class should be defined in that module.

# archive

    - Archive or restore a project pipeline's files.
    - This function moves the project's configuration, weights, and history files to or from the 'internal/Archived' directory based on the provided pipeline name(s). 
    - Archiving moves files to the archive, while restoring moves them back.

## Parameters
    
### ppl : str or list of str
        - The name(s) of the project pipeline to archive or restore.
### reverse : bool, optional
        - If True, restores the project from the archive to the active directory. 
        - Defaults to False.

## Returns
    
    None
        - The function prints the status of the archiving process.

## Notes
    
    - The function checks whether the project is already archived before proceeding.
    - If a list of projects is provided, the function archives or restores each project in the list.

# delete
 
    - Delete project files from the archive.
    - This function deletes project files (configuration, weights, and history) from the archive directory. 
    - It operates on files specified by the `ppl` parameter.

## Parameters
   
### ppl : str or list of str
        - The name(s) of the project to delete. If a string, it represents a single project; if a list, it represents multiple projects.

## Returns
    
    None
        - The function performs file operations but does not return any value.

## Notes
    
    - The function will attempt to remove files from the `internal/Archived` directory.
    - If the project is not found in the archive, a message will be printed.

# setup_transfer
    - Setup the directory structure for transferring pipelines.

    - This function creates the necessary directories and configuration files for transferring pipelines between environments. 
    - It creates a 'Transfer' folder inside 'internal', with subdirectories for storing histories, weights, and configurations.

## Returns
    
    None
        - The function prints a message indicating whether the setup was completed or already exists.
    
# transfer
    - Transfer pipeline files between active and transfer directories.
    - This function transfers pipeline files (configurations, weights, and histories) either from the active directory to the transfer directory or vice versa. The transfer can be done using either copy or move operations.
    - make sure `setup_trasfer` for the first time before using `transfer`.

## Parameters
    
### ppl : str or list of str
        The name(s) of the project pipeline to transfer.
### type : str, optional
        - The type of transfer, either 'export' to transfer to the transfer directory or 'import' to transfer back to the active directory. Defaults to 'export'.
### mode : str, optional
        - The method of transfer: 'copy' to duplicate files, or 'move' to transfer them. 
        - Defaults to 'copy'.

## Returns
    None
        - The function prints the status of the transfer process.

## Notes
    - If `mode` is 'move', the files are deleted from the source after being transferred.


<h1 style="font-size: 60px;" id="pipeline">PipeLine</h1>
This is the core class of the library, responsible for managing the entire lifecycle of a deep learning model pipeline: from setup, training, and validation, to saving and restoring model configurations and weights. But it is still lengthy, so we have `train_new` class to initiate a new pipeline and `re_train` to re train a pipeline.

- [Usage](#usage)
    - [Setting Up the Pipeline](#setting-up-the-pipeline)
    - [Preparing Data](#preparing-data)
    - [Training the Model](#training-the-model)
    - [Saving and Loading Configurations](#saving-and-loading-configurations)
- [Methods](#methods)
    - [load_component](#load_component)
    - [load_optimizer](#load_optimizer)
    - [save_config](#save_config)
    - [__adjust_loader_params](#__adjust_loader_paramsmode)
    - [get_desc](#get_desccnfgnone-modeall-fullfalse)
    - [setup](#setup)
    - [prepare_data](#prepare_data)
    - [update](#update)
    - [train](#train)
    - [validate](#validate)

## Setting Up the Pipeline

You can configure the pipeline either programmatically or by using a configuration file. The setup process involves specifying the locations of the model, loss function, optimizer, and dataset.

### Example

```python
from PyTorchLabFlow.pipeline import PipeLine

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
    remark="New", 
    prepare=True
)
```

## Preparing Data

The `prepare_data` method sets up data loaders for training and validation datasets. You can customize the dataset by specifying its location and parameters.

### Example

```python
pipeline.prepare_data(
    dataset_loc="Libs.datasets.MyDataset",
    train_folder="data/train",
    train_batch_size=32,
    valid_folder="data/valid",
    valid_batch_size=32
)
```

## Training the Model

To start training the model, use the `train` method and specify the number of epochs.

### Example

```python
pipeline.train(num_epochs=10)
```

## Saving and Loading Configurations

The `PipeLine` class supports saving the configuration to a JSON file and loading it for future use.

### Example

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


### `__adjust_loader_params(mode)`

This function adjusts the parameters for a data loader based on the mode of operation (either "train" or "valid"), the batch size, dataset size, system resources (memory and CPU), and other configurations.

#### Arguments:
- **mode** (`str`): The mode of operation. It can be either:
  - `"train"`: For training purposes.
  - `"valid"`: For validation purposes.

#### Returns:
- **dict**: A dictionary containing optimal settings for the data loader:
  - `'dataset'` (`Dataset` object): The dataset instance to be used.
  - `'batch_size'` (`int`): The adjusted batch size based on the dataset size.
  - `'shuffle'` (`bool`): Whether to shuffle the dataset (`True` for training, `False` for validation).
  - `'num_workers'` (`int`): The number of CPU workers to use for loading data.
  - `'collate_fn'` (`callable`): A collate function for batching, if available.
  - `'pin_memory'` (`bool`): Whether to use pinned memory, based on batch size and available system memory.

#### Functionality:
- Initializes the dataset and batch size based on the given mode.
- Adjusts the batch size if it exceeds the size of the dataset.
- Decides whether to use pinned memory (`pin_memory`) based on the batch size.
- Sets the number of workers (`num_workers`) for data loading based on the batch size and system CPU cores.
- Considers system memory availability to adjust the number of workers and whether to enable pinned memory.
- Returns a dictionary with the optimal data loader parameters.

#### Error Handling:
- Raises a `ValueError` if the mode is neither `"train"` nor `"valid"`.

#### Example:
```python
params = self.__adjust_loader_params("train")
print(params['batch_size'])  # Adjusted batch size
print(params['num_workers'])  # Optimal number of workers
```

---

### `get_desc(cnfg=None, mode="all", full=False)`

This function generates a descriptive summary of the configuration parameters based on the specified mode. It provides key information about components of the model, dataset, training, and other configurations, either in a full or abbreviated form.

#### Arguments:
- **cnfg** (`dict`, optional): A dictionary containing configuration parameters. If not provided, the function will use the instance's default configuration (`self.cnfg`).
- **mode** (`str`, optional): The mode of operation. It determines what part of the configuration is included in the description:
  - `"mods"`: Includes information about the model and dataset locations.
  - `"training"`: Includes training-related information, such as data sources, optimizer, and batch sizes.
  - `"all"` (default): Includes all available configuration details (model, dataset, training parameters).
- **full** (`bool`, optional): Whether to return the full path or just the base name of the components:
  - `True`: Returns the full path or full configuration values.
  - `False`: Returns only the base names or abbreviated configuration values.

#### Returns:
- **str**: A string that concatenates the relevant configuration information, separated by `#`. The content varies based on the chosen mode and whether the full information is requested.

#### Functionality:
- **When `mode == "mods"`**:
  - Returns the locations of the model and dataset, either in full or abbreviated form, depending on the `full` flag.
  
- **When `mode == "training"`**:
  - Returns training-related configurations, including the optimizer location, data source paths, and batch sizes, either in full or abbreviated form.

- **When `mode == "all"`** (default):
  - Returns a comprehensive summary, including model location, dataset location, accuracy and loss paths, optimizer location, data sources, and batch sizes, in full or abbreviated form.

- The returned string concatenates all selected components with a `#` separator.

#### Example:
```python
desc = self.get_desc(mode="training", full=False)
print(desc)
# Output might look like: "optimizer_class#train_data#valid_data#32#64"
```

#### Example with `full=True`:
```python
desc = self.get_desc(mode="all", full=True)
print(desc)
# Output might look like: "full/model/path/full/dataset/path/full/accuracy/path/full/loss/path/full/optimizer/path/full/train/data/src/full/valid/data/src/32/64"
```

#### Notes:
- The `full` argument allows flexibility in the level of detail returned, which can be useful for logging or debugging.
- Paths are normalized with `os.path.normpath()` for consistent formatting, and only the base name is shown when `full=False`.

---

This function helps provide a detailed or summarized overview of the configuration based on the mode and level of detail required.



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



##### Example Uses

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






## Configuration File Path Handling

When initializing the `PipeLine` class, if the user does not specify a path for the configuration file, the pipeline will automatically create a new directory in the current working directory. This directory will be named after the pipeline name provided during initialization.

- **Automatic Directory Creation**: If the `config_path` is not given, the pipeline will create a folder with the same name as the pipeline in the current working directory. Inside this folder, it will save the configuration file, weights, and history files.

### Example Behavior

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
  "loss_loc": "Libs.losses.myloss",
  "optimizer_loc": "Libs.optimizers.myoptimizer",
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

***

# Templates

Templates for different use cases

  - [CNN_LSTM_Attention_with_Mask](#cnn_lstm_attention_with_mask)
  - [Audio Spectrogram Transformer](#audio-spectrogram-transformer)
  - [MesoNet](#mesonet)
  - [CSV file](#csv-file-data)
  - [Using preTrained Model](#using-pretrained-model)
  - [Dynamic learning rate](#dynamic-learning-rate)

## CNN_LSTM_Attention_with_Mask
    Designed to handle variable length audio file that captures TTS in any portion of the audio
### dataset
    The DS01 dataset class loads .wav audio files, transforms them into log-scaled mel-spectrograms using Librosa, and returns them with corresponding binary labels. The mel-spectrograms are converted to tensors and padded to match the maximum length in the batch, with an output shape of (batch_size, 1, n_mels, max_length) for spectrograms and (batch_size, 1) for labels.
```python
class DS01(Dataset):
    def __init__(self, folder, sr=22050, n_mels=128, n_fft=2000, hop_length=512):
        self.folder = folder
        self.file_list = [f for f in os.listdir(folder) if f.endswith('.wav')]
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.folder, self.file_list[idx])
        waveform, sr = librosa.load(file_path, sr=self.sr)

        # Convert the waveform to a mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=waveform, sr=self.sr, n_fft=self.n_fft, 
            hop_length=self.hop_length, n_mels=self.n_mels
        )

        # Convert to log scale
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Convert to tensor and add a channel dimension
        log_mel_spectrogram = torch.tensor(log_mel_spectrogram).float().unsqueeze(0)

        # Extract label from the file name (e.g., last character in filename)
        label = int(os.path.basename(file_path)[-5])

        # Ensure label is a single scalar (binary label)
        label = torch.tensor(label, dtype=torch.float32)  # Shape: []

        return log_mel_spectrogram, label
    
    @staticmethod
    def collate_fn(batch):
        spectrograms, labels = zip(*batch)

        # Get the maximum length for padding
        max_length = max(s.shape[-1] for s in spectrograms)

        # Pad the spectrograms
        padded_spectrograms = [torch.nn.functional.pad(s, (0, max_length - s.shape[-1])) for s in spectrograms]

        # Convert labels to a tensor
        labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)  # Ensure labels are reshaped to [batch_size, 1]

        return torch.stack(padded_spectrograms), labels
```

### Model / Architecture
    This model combines a CNN for feature extraction, a bi-directional LSTM for sequence modeling, an attention mechanism, and fully connected layers for binary classification. It accepts input in the shape (batch_size, 1, H, W) with an optional mask for sequence padding. The output is a binary prediction with shape (batch_size, 1), making it suitable for tasks like sequence classification where spatial and temporal features are both important.
```python
class CNN_LSTMAttentionWithMask(nn.Module):
    def __init__(self):
        super(CNN_LSTMAttentionWithMask, self).__init__()
        
        # CNN layers for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (batch_size, 1, H, W) -> (batch_size, 32, H, W)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (batch_size, 32, H/2, W/2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (batch_size, 32, H/2, W/2) -> (batch_size, 64, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # (batch_size, 64, H/4, W/4)
        )
        
        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(input_size=64*32, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(128 * 2, 128 * 2),  # Input: LSTM output size (bi-directional), Output: attention size
            nn.Tanh(),
            nn.Linear(128 * 2, 1)  # Single value for each time step
        )
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(128 * 2, 100)  # Output from attention applied on LSTM hidden state
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 1)  # Ensure output is single dimension for binary classification
        
        # Batch normalization and dropout
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask=None, *args, **kwargs):  # Accept *args and **kwargs for flexibility
        x = x.to(next(self.parameters()).device) #!important
        # CNN feature extraction
        x = self.cnn(x)  # (batch_size, 64, H/4, W/4)

        # Reshape for LSTM (flatten height and width)
        batch_size, channels, height, width = x.size()

        # Reshape: (batch_size, channels, height, width) -> (batch_size, width, channels*height)
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch_size, width, channels, height)
        x = x.view(batch_size, width, -1)  # (batch_size, width, channels*height), suitable for LSTM input

        # Compact weights for LSTM for contiguous memory access
        self.lstm.flatten_parameters()

        # LSTM sequence modeling
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len=width, lstm_hidden_size*2)

        # Attention mechanism with masking
        attention_scores = self.attention(lstm_out).squeeze(-1)  # (batch_size, seq_len)

        # Apply masking to attention scores if mask is provided
        if mask is not None:
            mask = mask.unsqueeze(-1)  # Adjust shape to match the attention scores
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))  # Mask padded areas

        attention_weights = F.softmax(attention_scores, dim=1)  # Normalize attention scores

        # Apply attention weights to LSTM output
        weighted_lstm_out = attention_weights.unsqueeze(-1) * lstm_out  # (batch_size, seq_len, lstm_hidden_size*2)

        # Sum across the sequence dimension (batch_size, lstm_hidden_size*2)
        x = torch.sum(weighted_lstm_out, dim=1)

        # Fully connected layers for classification
        x = F.relu(self.bn1(self.fc1(x)))  # (batch_size, 100)
        x = F.relu(self.bn2(self.fc2(x)))  # (batch_size, 10)
        x = self.dropout(x)

        # Ensure final output shape is (batch_size, 1) for binary classification
        x = torch.sigmoid(self.fc3(x))  # (batch_size, 1)
        x = x.unsqueeze(-1)  # Reshape from [batch_size, 1] to [batch_size, 1, 1]

        return x

```
## Audio Spectrogram Transformer
    Designed for audio classification
### dataset

    The DSbbek04 dataset class processes audio files and converts them into MFCC spectrograms of size (21, 21) for input to models. It loads audio files from the specified folder, crops them to a fixed duration (based on the sample rate and crop duration), and computes MFCC features. The class includes a method for random cropping of the waveform and returns both the MFCC tensor and the label (extracted from the filename). This is useful for tasks like audio classification, with MFCCs representing the spectral characteristics of the audio.

```python
class DSbbek04(Dataset):
    '''mfcc (21,21)'''
    def __init__(self, folder, sr=22050, crop_duration=0.47):
        self.folder = folder
        self.file_list = [f for f in os.listdir(folder)]
        self.sr = sr
        self.crop_duration = crop_duration
        self.crop_size = int(self.crop_duration * self.sr)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sr = librosa.load(os.path.join(self.folder, file_path))
        waveform = torch.tensor(waveform).unsqueeze(0)
        cropped_waveform = self.random_crop(waveform, self.crop_size)
        
        mfcc = librosa.feature.mfcc(y = cropped_waveform.numpy(), sr = sr, n_mfcc=21)
        mfcc= torch.tensor(mfcc)#.unsqueeze(0)
        label = int(os.path.basename(file_path)[-5])
        return mfcc, label

    def random_crop(self, waveform, crop_size):
        num_samples = waveform.size(1)
        if num_samples <= crop_size:
            padding = crop_size - num_samples
            cropped_waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            start = random.randint(0, num_samples - crop_size)
            cropped_waveform = waveform[:, start:start + crop_size]
        return cropped_waveform

``` 
### model / architecture

    The MdLamn01 model is an Audio Spectrogram Transformer (AST) designed for processing MFCC spectrograms with input dimensions of (1, 21, 21). It first applies a patch embedding via a convolution layer, transforming the input into smaller patches. Then, the encoded patches are processed by a custom Transformer Encoder Layer (with multi-head attention and feedforward layers), followed by global average pooling. Finally, the output is passed through a fully connected layer, producing a binary classification result using a sigmoid activation.

```python
class MdLamn01(nn.Module):
    '''
    Audio Spectrogram Transformer(AST)
    takes (21,21)| mfcc
    DSbbek04
    '''
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(MdLamn01, self).__init__()
        
        self.patch_embedding = nn.Conv2d(1, d_model, kernel_size=(4, 4), stride=(4,4))
        
        self.transformer_layer = self.TfEL(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc = nn.Linear(d_model, 1)
        
    def TfEL(self,d_model,nhead,dim_feedforward,dropout=0.1):
        class TransformerEncoderLayer(nn.Module):
            def __init__(self, d_model, nhead, dim_feedforward, dropout):
                super(TransformerEncoderLayer, self).__init__()
                self.self_attn = nn.MultiheadAttention(d_model, nhead)
                self.linear1 = nn.Linear(d_model, dim_feedforward)
                self.dropout = nn.Dropout(dropout)
                self.linear2 = nn.Linear(dim_feedforward, d_model)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)

            def _sa_block(self, x, attn_mask=None, key_padding_mask=None):
                # Permute the input to (sequence_length, batch_size, embedding_dim)
                x = x.permute(1, 0, 2)  # Shape: [seq_length, batch_size, d_model]
                x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
                x = x.permute(1, 0, 2)  # Revert to [batch_size, seq_length, d_model]
                return x

            def forward(self, src, src_mask=None, src_key_padding_mask=None):
                x = self.norm1(src + self._sa_block(src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask) )
                x = self.norm2(x + self.linear2(self.dropout(F.relu(self.linear1(x)))) )
                return x
        return TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = self.patch_embedding(x)
        x = x.flatten(2).permute(0, 2, 1)  # Shape: [batch_size, seq_length, d_model]
        x = self.transformer_layer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = F.sigmoid(self.fc(x))
        return x
```
## MesoNet
    MesoNet of 4 layer  for image classification
### dataset
    The DS01 dataset class is designed for binary classification of images, where images from directories containing the keywords 'real' or 'fake' are labeled accordingly. The class loads .jpg files from subdirectories, resizes them to (128, 128), and applies transformations to convert them to tensors. Labels are mapped to 1 for 'real' and 0 for 'fake'. The __getitem__ method returns the image and its corresponding label.
```python
# Custom Dataset Class for similar 'real' and 'fake' directories
class DS01(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
                            transforms.Resize((128,128)),  # Resize images 
                            transforms.ToTensor()        # Convert images to tensor
                        ])
        self.image_paths = []
        self.labels = []
        
        # Map directories with 'real' or 'fake' in their name to corresponding labels
        self.class_to_idx = {'real': 1, 'fake': 0}  # 1 for real, 0 for fake
        
        # Traverse through subdirectories
        for subdir in os.listdir(root_dir):
            class_name = None
            
            # Identify if subdir is 'real' or 'fake' by checking the directory name
            if 'real' in subdir.lower():
                class_name = 'real'
            elif 'fake' in subdir.lower():
                class_name = 'fake'
            
            if class_name:
                class_dir = os.path.join(root_dir, subdir)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        # Only load .jpg files
                        if img_name.lower().endswith('.jpg'):
                            self.image_paths.append(img_path)
                            self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms if provided
        image = self.transform(image)
        
        return image, label
```
### model / architecture
    The Meso4_01 model takes an input size of (3, 128, 128) and passes it through four convolutional layers with batch normalization and max-pooling, followed by two fully connected layers. The output is a binary classification, with a sigmoid activation at the final layer.

```python
# Meso4 Model
class Meso4_01(nn.Module):
    def __init__(self):
        super(Meso4_01, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)  

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        
        self.fc1 = nn.Linear(16 * 4 * 4, 16)
        self.fc2 = nn.Linear(16, 1)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc2(x))
        return x

```
## CSV file data
    For tabular data. 
### dataset
    The DS01 dataset class loads data from a CSV file into a Pandas DataFrame, where the __getitem__ method extracts the features (all columns except 'Quality') and the label ('Quality') for each sample. The features are converted into a torch tensor and reshaped to include a channel dimension (features[np.newaxis, :]). The label is also converted into a tensor of type torch.long. The method returns the reshaped features and the label.
```python
class DS01(Dataset):
    def __init__(self, csv_file):
        # Load the CSV file into a DataFrame
        self.data_frame = pd.read_csv(csv_file)
    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # Get a row at the given index (iloc is used to access by index)
        row = self.data_frame.iloc[idx]
        
        features = row.drop('Quality').values.astype(float)  # Assuming 'label' column contains labels
        label = row['Quality']

        # Optionally convert features to torch tensors
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        features = features[np.newaxis,:]
        # label = label.view(1,1)
        features.shape, label.shape,label
        
        return features, label
```
### model / architecture
    The Mdl01 class is a simple feedforward neural network implemented using PyTorch's nn.Module. It takes a feature vector of length 7 and passes it through a series of linear layers with batch normalization and ReLU activations to produce a binary output.
```python
class Mdl01(nn.Module):
    def __init__(self):
        super(Mdl01, self).__init__()
        self.Seq = nn.Sequential(
            nn.Linear(7,10),
            nn.BatchNorm1d(10),
            nn.ReLU(10),
            nn.Linear(10,20),
            nn.BatchNorm1d(20),
            nn.ReLU(20),
            nn.Linear(20,15),
            nn.BatchNorm1d(15),
            nn.ReLU(15),
            nn.Linear(15,5),
            nn.BatchNorm1d(5),
            nn.ReLU(5),
            nn.Linear(5,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = x.to(next(self.parameters()).device) #!important
        x = x.view(-1,7)
        x = self.Seq(x)
        return x
```
## Using preTrained Model
### dataset
The DS_01 class defines a custom PyTorch Dataset for loading and preprocessing image data from a directory structure where each subfolder corresponds to a class. The images are resized to 224x224, converted to tensors, and normalized using ImageNet statistics. It also provides a custom collate_fn function for batching, which includes one-hot encoding the labels for classification tasks. The dataset supports indexing and returns both the transformed image and its corresponding label.
```python
class DS_01(Dataset):
    def __init__(self, root_dir):
        self.desc = "resize[224, 224] -> transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
        self.root_dir = root_dir
        
        # Define default transformations (resize, to tensor, and normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224 pixels
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Get all class names (subfolder names)
        self.class_names = sorted(os.listdir(root_dir))
        
        # Create a mapping from class name to class index
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        
        # Collect all image paths and their corresponding class labels
        self.image_paths = []
        self.labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith(('.png', '.jpg', '.jpeg')):  # You can adjust extensions
                        self.image_paths.append(os.path.join(class_dir, filename))
                        self.labels.append(self.class_to_idx[class_name])
    def collate_fn(batch):
        inputs, labels = zip(*batch)  # Unzip the batch into inputs and labels
        inputs = torch.stack(inputs, dim=0)  # Stack inputs to create a tensor
        labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to tensor of type long (integer)
        def one_hot_encode(labels, num_classes=74):
            # Convert integer labels to one-hot encoded labels
            return torch.eye(num_classes)[labels]
        # Apply one-hot encoding to the labels
        labels = one_hot_encode(labels)
        
        return inputs, labels
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")  # Convert to RGB to ensure consistent color channels
        
        # Get the label for the image
        label = self.labels[idx]
        
        # Apply transformations if provided
        img = self.transform(img)
        
        return img, label

```
### model/architecture

The ResN50_40 class defines a custom neural network using a pre-trained ResNet50 model, where the final fully connected layer is removed and replaced with custom layers. It freezes most of the ResNet50 layers and only trains the last 40 parameters, adding fully connected, batch normalization, ReLU, and dropout layers to output predictions for 74 classes.

```python
class ResN50_40(nn.Module):
    def __init__(self):
        super(ResN50_40, self).__init__()

        self.desc = "[3,224,224] -> ResNet50preTrained(-40:)[2048] -> 256 -> 74"

        # Step 1: Load ResNet50 base model (without pre-trained weights) 
        self.base_model = models.resnet50(weights=None)  # No pretrained weights at first
        self.base_model.fc = nn.Identity()  # Remove the final fully connected (fc) layer

        # Step 2: Load the pre-trained weights into the base model "pretrained" folder should be inside your project directory
        state_dict = torch.load("preTrained/resnet50_pretrained.pth", weights_only=True) 
        self.base_model.load_state_dict(state_dict, strict=False)  # Load the pre-trained weights

        # Custom layers after the base model
        self.fc1 = nn.Linear(2048, 256)  # Fully connected layer
        self.batch_norm = nn.BatchNorm1d(256)  # Apply BatchNorm after fc1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 74)  # Output layer (num_classes will be 74)

        # Step 3: Create a list of all parameters 
        self.params = list(self.base_model.named_parameters())
        # print(f"Total parameters: {len(self.params)}")

        # Identify the last 10 layers
        self.last_40_params = self.params[-40:]  # Get the last 10 parameters

        # Step 4: Freeze all layers except the last 10 layers
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False  # Freeze all layers initially

        # Step 5: Unfreeze the last 10 layers
        for name, param in self.last_40_params:
            param.requires_grad = True  # Unfreeze the last 10 layers

    def forward(self, x):
        # Forward pass through ResNet50 base model
        x = self.base_model(x)  # Get features from ResNet50
        x = self.fc1(x)          # Apply fully connected layer 1
        x = self.batch_norm(x)   # Apply batch normalization after fc1
        x = self.relu(x)         # ReLU activation
        x = self.dropout(x)      # Apply dropout
        x = self.fc2(x)          # Output layer
        return x

```

## Dynamic Learning rate
The OptAdamax_sc function creates an Adamax optimizer with a learning rate of 0.001 and weight decay of 1e-5. It wraps the optimizer in the OptimizerWithScheduler class, which provides integration with a learning rate scheduler. By default, it uses the ReduceLROnPlateau scheduler, which adjusts the learning rate based on the validation loss. This setup allows easy management of optimization and dynamic learning rate adjustments during training.
```python
## Libs/optimizers.py
class OptimizerWithScheduler:
   def __init__(self, optimizer, scheduler_type='StepLR', **kwargs):
            # Create Adamax optimizer
            self.optimizer = optimizer
            
            # Choose the scheduler
            if scheduler_type == 'StepLR':
                self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)
            elif scheduler_type == 'ReduceLROnPlateau':
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3)
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")

   def step(self, loss=None):
            """ Perform one step of optimization and adjust the learning rate using the scheduler. """
            self.optimizer.step()

            # If using ReduceLROnPlateau, pass the loss to adjust the learning rate
            if isinstance(self.scheduler, ReduceLROnPlateau) and loss is not None:
                self.scheduler.step(loss)  # This adjusts the LR based on validation loss
            elif isinstance(self.scheduler, StepLR):
                self.scheduler.step()  # This just steps based on epochs, so no need for loss

   def zero_grad(self):
            """ Zero out the gradients in the optimizer """
            self.optimizer.zero_grad()

   def get_lr(self):
            """ Get the current learning rate """
            return self.optimizer.param_groups[0]['lr']  # Assuming a single learning rate for all params
   def get_last_lr(self):
            """ Get the last learning rate from the scheduler """
            return self.scheduler.get_last_lr()[0] 
    

def OptAdamax_sc(model, **kwargs):
   optimizer = optim.Adamax(model.parameters(), lr=0.001, weight_decay=1e-5)
    # Create and return an instance of the OptimizerWithScheduler class
   return OptimizerWithScheduler(optimizer, "ReduceLROnPlateau", **kwargs)
```

use case

```python
P = train_new(name="exp12",
          dataset_loc="DS_01",
          model_loc="ResN50_40",
          optimizer_loc= "Libs.optimizers.OptAdamax_sc", #costumized optimizer 
          prepare=True
         )
P.train(num_epochs=20,patience=6)
```
# Credits
- **Author:** [BBEK-Anand](https://github.com/bbek-anand) - For developing this library.
- **Documentation:**
  - [Saumya](https://github.com/S-aumya) - For writing the documentation.
  - [BBEK-Anand](https://github.com/bbek-anand) - For maintaining the documentation.
- **Libraries Used:**
  - [PyTorch](https://pytorch.org) - For Model creation.
  - [pandas](https://pandas.pydata.org/) - For saving model history and retriving.
  - [tqdm](https://github.com/tqdm) - For progress bar during traing.
  - [matplotlib](https://matplotlib.org/) - For plotting.
# Cite this
```
  @misc{PyTorchLabFlow,
  author       = {BBEK-Anand},
  title        = {PyTorchLabFlow: A Python Library for Managing Deep Learning Experiments in PyTorch},
  year         = {2024},
  url          = {https://github.com/BBEK-Anand/PyTorchLabFlow},
  note         = {Version 0.2.5},
}```

# Contributing
Feel free to fork this project, open issues, or create pull requests. Contributions are always welcome!

# License
This project is licensed under the Apache License 2.0.


