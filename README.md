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

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

