# Copyright (c) 2024 BBEK-Anand
# Licensed under the MIT License

import torch
import shutil
from torch import nn
# import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from tqdm import tqdm
import json
import importlib
import pandas as pd
# import numpy as np
from matplotlib import pyplot as plt

class PipeLine:
    """
    Class for managing the entire machine learning pipeline, including model setup, training, validation, 
    and saving configurations and weights.
    
    Args:
        name (str): Name of the pipeline. Default is 'Default_name'.
        config_path (str): Path to the configuration file. Default is None.
    """
    
    def __init__(self, name='Default_name', config_path=None):
        """
        Initializes the PipeLine class, sets up default configurations and variables like device, paths, 
        model, loss, optimizer, and data loaders.
        
        Args:
            name (str): The name of the pipeline. Defaults to 'Default_name'.
            config_path (str): Path to the configuration file. Defaults to None.
        """
        self.name = name
        self.__best_val_loss = float('inf')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = None
        self.model = None
        self.loss = None
        self.optimizer = None

        self.trainDataLoader = None
        self.validDataLoader = None

        self.config_path = config_path
        self.weights_path = None
        self.history_path = None

        self.cnfg = None
        self.__configured = False
        self.DataSet = None

    def load_component(self, module_loc):
        """
        Loads a Python class from a given module location string.
        
        Args:
            module_loc (str): Dot-separated string representing the module and class to be loaded.
        
        Returns:
            class_: The loaded class from the module.
        """
        module = importlib.import_module('.'.join(module_loc.split('.')[:-1]))
        class_ = getattr(module, module_loc.split('.')[-1])
        return class_

    def load_optimizer(self, module_loc, **kwargs):
        """
        Loads an optimizer class from a given module location string and initializes it with the model's parameters.
        
        Args:
            module_loc (str): Dot-separated string representing the module and optimizer class to be loaded.
            **kwargs: Additional arguments to pass to the optimizer during initialization.
        
        Returns:
            optimizer: The initialized optimizer instance.
        """
        module = importlib.import_module('.'.join(module_loc.split('.')[:-1]))
        class_ = getattr(module, module_loc.split('.')[-1])
        return class_(self.model, **kwargs)

    def save_config(self):
        """
        Saves the current pipeline configuration to the specified config path in JSON format.
        """
        with open(self.cnfg['config_path'], "w") as out_file:
            json.dump(self.cnfg, out_file, indent=4)

    def setup(self, name=None, model_loc=None, accuracy_loc=None, loss_loc=None, optimizer_loc=None, dataset_loc=None,
              train_data_src=None, train_batch_size=None, valid_data_src=None, valid_batch_size=None, history_path=None,
              weights_path=None, config_path=None, use_config=False, make_config=True, prepare=False):
        """
        Sets up the pipeline by loading and configuring the model, loss, optimizer, accuracy metrics, dataset, 
        and other necessary components. Can either load from an existing configuration file or create a new one.
        
        Args:
            name (str): Name of the pipeline. If None, uses default or pre-existing configuration.
            model_loc (str): Location of the model class to be loaded.
            accuracy_loc (str): Location of the accuracy metric function to be loaded.
            loss_loc (str): Location of the loss function to be loaded.
            optimizer_loc (str): Location of the optimizer class to be loaded.
            dataset_loc (str): Location of the dataset class to be loaded.
            train_data_src (str): Path to the training data source.
            train_batch_size (int): Batch size for training data loader.
            valid_data_src (str): Path to the validation data source.
            valid_batch_size (int): Batch size for validation data loader.
            history_path (str): Path to save the training history CSV file.
            weights_path (str): Path to save the model weights.
            config_path (str): Path to the configuration JSON file.
            use_config (bool): Whether to use an existing configuration file. Default is False.
            make_config (bool): Whether to create a new configuration file if one does not exist. Default is True.
            prepare (bool): Whether to prepare data loaders after setting up the pipeline. Default is False.
        """
        cnfg = {
            'model_loc': model_loc,
            'DataSet_loc': dataset_loc,
            'accuracy_loc': accuracy_loc,
            'loss_loc': loss_loc,
            'optimizer_loc': optimizer_loc,
            "piLn_name": name if name else self.name,
            'last': {'epoch': 0, 'train_accuracy': 0, 'train_loss': float('inf')},
            'best': {'epoch': 0, 'val_accuracy': 0, 'val_loss': float('inf')},
            "valid_data_src": valid_data_src,
            'train_data_src': train_data_src,
            'valid_batch_size': valid_batch_size,
            'train_batch_size': train_batch_size,
            'weights_path': weights_path,
            'config_path': config_path,
            'history_path': history_path,
        }

        if config_path and use_config:
            cnfg1 = json.load(open(config_path))
            self.history_path = cnfg1.get('history_path')
            self.weights_path = cnfg1.get('weights_path')
            self.name = cnfg1.get('piLn_name')
            self.model_name = cnfg1['model_loc'].split('.')[-1]
            self.__best_val_loss = cnfg1['best']['val_loss']
            cnfg1["valid_data_src"] = valid_data_src if valid_data_src!=None else cnfg1["valid_data_src"]
            cnfg1['train_data_src'] = train_data_src if train_data_src!=None else cnfg1['train_data_src']
            self.cnfg = cnfg1

        elif config_path and make_config:
            if not (name and accuracy_loc and loss_loc and optimizer_loc):
                raise ValueError("Required parameters: name, accuracy_loc, loss_loc, optimizer_loc")
            if (not (model_loc or self.model)) :
                print(self.model)
                raise ValueError("Required parameter model_loc or PipeLine.model")
            root = os.path.dirname(config_path)
            os.makedirs(root, exist_ok=True)

            cnfg.update({
                'piLn_name': name if name else self.name,
                'weights_path': weights_path or os.path.join(root, f"{self.name}.pth"),
                'history_path': history_path or os.path.join(root, f"{self.name}.csv"),
                'config_path': config_path,
            })

            self.history_path = cnfg['history_path']
            self.weights_path = cnfg['weights_path']
            self.cnfg = cnfg
            self.save_config()
            
            pd.DataFrame(columns=["epoch", "train_accuracy", "train_loss", "val_accuracy", "val_loss"]).to_csv(self.history_path, index=False)
            print(f'Configuration file is saved at {cnfg["config_path"]}')
            print(f'History will be saved at {cnfg["history_path"]}')
            print(f'Weights will be saved at {cnfg["weights_path"]}')

        elif not config_path and make_config:
            if not (self.name and model_loc and accuracy_loc and loss_loc and optimizer_loc):
                raise ValueError("Required parameters: name, model_loc, accuracy_loc, loss_loc, optimizer_loc")

            os.makedirs(self.name, exist_ok=True)

            cnfg.update({
                'piLn_name': self.name,
                'weights_path': os.path.join(self.name, f"{self.name}.pth"),
                'history_path': os.path.join(self.name, f"{self.name}.csv"),
                'config_path': os.path.join(self.name, f"{self.name}.json"),
            })

            self.history_path = cnfg['history_path']
            self.weights_path = cnfg['weights_path']
            self.cnfg = cnfg
            self.save_config()
            pd.DataFrame(columns=["epoch", "train_accuracy", "train_loss", "val_accuracy", "val_loss"]).to_csv(self.history_path, index=False)
            print(f'Configuration file is saved at {cnfg["config_path"]}')
            print(f'History will be saved at {cnfg["history_path"]}')
            print(f'Weights will be saved at {cnfg["weights_path"]}')

        self.model = self.load_component(self.cnfg['model_loc'])() if(self.cnfg['model_loc']!=None) else self.model
        if self.cnfg.get('last').get('epoch')==0:
            torch.save(self.model.state_dict(), self.cnfg['weights_path'])
#         if not hasattr(self.model, 'parameters'):
#             raise ValueError(f"The model class loaded from {self.cnfg['model_loc']} does not have a 'parameters' attribute.")
        
        self.loss = self.load_component(self.cnfg['loss_loc'])()
        self.optimizer = self.load_optimizer(self.cnfg['optimizer_loc'])
        self.accuracy = self.load_component(self.cnfg['accuracy_loc'])()
        self.DataSet = self.load_component(self.cnfg['DataSet_loc']) if(self.cnfg['DataSet_loc']!=None) else self.DataSet
        
        if prepare:
            dataset_loc = self.cnfg['DataSet_loc'] if(dataset_loc==None) else dataset_loc
            self.prepare_data(dataset_loc=dataset_loc)

    def prepare_data(self, dataset_loc=None, train_data_src=None, train_batch_size=None, valid_data_src=None, valid_batch_size=None):
        """
        Prepares the data loaders for training and validation by loading the dataset and setting up batch sizes 
        and paths. Also saves the configuration after preparation.
        
        Args:
            dataset_loc (str): Location of the dataset class to be loaded.
            train_data_src (str): Path to the training data source.
            train_batch_size (int): Batch size for the training data loader.
            valid_data_src (str): Path to the validation data source.
            valid_batch_size (int): Batch size for the validation data loader.
        """
        if not (train_data_src or self.cnfg['train_data_src']):
            raise ValueError('train_data_src is not found')
        if not (valid_data_src or self.cnfg['valid_data_src']):
            raise ValueError('valid_data_src is not found')
        if not (train_batch_size or self.cnfg['train_batch_size']):
            raise ValueError('train_batch_size is not found')
        if not (valid_batch_size or self.cnfg['valid_batch_size']):
            raise ValueError('valid_batch_size is not found')

        self.cnfg.update({
            'valid_batch_size': valid_batch_size or self.cnfg['valid_batch_size'],
            'valid_data_src': valid_data_src or self.cnfg['valid_data_src'],
            'train_batch_size': train_batch_size or self.cnfg['train_batch_size'],
            'train_data_src': train_data_src or self.cnfg['train_data_src'],
        })
        self.save_config()

        self.DataSet = self.load_component(dataset_loc) if (dataset_loc!=None) else self.DataSet

        self.trainDataLoader = DataLoader(self.DataSet(self.cnfg['train_data_src']), batch_size=self.cnfg['train_batch_size'], shuffle=True)
        self.validDataLoader = DataLoader(self.DataSet(self.cnfg['valid_data_src']), batch_size=self.cnfg['valid_batch_size'], shuffle=False)
        print('Data loaders are successfully created')
        
        self.model=self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        if(self.cnfg['last']['epoch']>0):
                self.model.load_state_dict(torch.load(self.weights_path))
        self.__configured =True

    def update(self, data):
        """
        Updates the pipeline's configuration with the current epoch, training accuracy, training loss, 
        validation accuracy, and validation loss. If the validation loss improves, saves the model weights.
        
        Args:
            data (dict): Dictionary containing epoch, train_accuracy, train_loss, val_accuracy, and val_loss.
        """
        self.cnfg['last'] = {'epoch': data['epoch'], 'train_accuracy': data['train_accuracy'], 'train_loss': data['train_loss']}
        if data['val_loss'] < self.cnfg['best']['val_loss']:
            self.cnfg['best'] = {'epoch': data['epoch'], 'val_accuracy': data['val_accuracy'], 'val_loss': data['val_loss']}
            torch.save(self.model.state_dict(), self.cnfg['weights_path'])
            print(f"Best Model Weights Updated: Epoch {data['epoch']} - Val Loss: {data['val_loss']}")
        self.save_config()
        record = pd.DataFrame([[data['epoch'], data['train_accuracy'], data['train_loss'], data['val_accuracy'], data['val_loss']]],
                              columns=["epoch", "train_accuracy", "train_loss", "val_accuracy", "val_loss"])
        record.to_csv(self.cnfg['history_path'], mode='a', header=False, index=False)

    def train(self,num_epochs=5):
        """
        Trains the model for a specified number of epochs. Computes and prints the loss and accuracy 
        for both training and validation after each epoch.
        
        Args:
            num_epochs (int): Number of epochs to train the model. Default is 5.
        """
        if (not self.__configured):
            print('Preparation Error. execute prepare_data or set prepare=True in setup before training')
            return None
        start_epoch=self.cnfg['last']['epoch']
        end_epoch = start_epoch+num_epochs
        self.model.to(self.device)
        for epoch in range(start_epoch,end_epoch):
            self.model.train()
            running_loss = 0.0
            running_accuracy = 0.0
            accuracy_metric = self.accuracy.to(self.device)
            train_loader_tqdm = tqdm(self.trainDataLoader, desc=f'Epoch {epoch+1}/{end_epoch}', leave=True)
            
            for data in train_loader_tqdm:
                inputs = data[0]
                labels = data[1]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                labels = labels.float().view(-1,1)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                if labels.shape != outputs.shape:
                    labels = labels.view_as(outputs)
                loss = self.loss(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                running_accuracy += accuracy_metric(outputs, labels.int()).item()
                train_loader_tqdm.set_postfix(loss=running_loss/len(train_loader_tqdm), accuracy=running_accuracy/len(train_loader_tqdm))
            
            train_loss = running_loss / len(self.trainDataLoader)
            train_accuracy = running_accuracy / len(self.trainDataLoader)
            val_loss, val_accuracy = self.validate()
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}')
            data = {'epoch':epoch+1,'train_accuracy':train_accuracy,'train_loss':train_loss,'val_accuracy':val_accuracy,'val_loss':val_loss}
            self.update(data)
        print('Finished Training')

    def validate(self):
        """
        Validates the model on the validation dataset, computing the loss and accuracy.
        
        Returns:
            avg_loss (float): The average validation loss.
            accuracy (float): The validation accuracy.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(self.validDataLoader, desc='Validating', leave=False):
                inputs = data[0]
                labels = data[-1]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)                
                labels = labels.float().unsqueeze(1) #only for base audio file

                outputs = self.model(inputs) # 0 bcz input only has base audio array
                loss = self.loss(outputs, labels)
                running_loss += loss.item()
                correct += self.accuracy(outputs, labels.int()).item()
                total += labels.size(0)
            accuracy = correct / len(self.validDataLoader)
            avg_loss = running_loss / len(self.validDataLoader)
            return avg_loss, accuracy

def setup_project(project_name="MyProject",create_root=True):#
    """
    Create the directory structure for a new machine learning project.

    This function sets up the required directories and files for organizing datasets, 
    models, and configurations for a machine learning project. It creates a base project 
    structure with folders like 'DataSets', 'Libs', 'internal', and 'Archived', and 
    generates some basic template files.

    Parameters
    ----------
    project_name : str, optional
        The name of the project directory to create. Defaults to 'MyProject'.
    create_root : bool, optional
        If set to False, the project will be created in the current directory without creating 
        a root folder. Defaults to True.

    Returns
    -------
    None
        The function prints the status of the setup process.

    Notes
    -----
    - This function will create template Python files under the 'Libs' directory for models, datasets, 
      accuracies, losses, and optimizers.
    - It will also generate configuration JSON files in the 'internal' directory.

    Examples
    --------
    >>> setup_project("AudioClassification")
    # Creates a project directory with the necessary structure for 'AudioClassification'.
    """
    
    if(not create_root):
        project_name="./"
    if not os.path.exists(project_name):
        os.mkdir(project_name)
        os.mkdir(os.path.join(project_name,'DataSets'))
        os.mkdir(os.path.join(project_name,'DataSets','train'))
        os.mkdir(os.path.join(project_name,'DataSets','valid'))

        os.mkdir(os.path.join(project_name,'Libs'))
        with open(os.path.join(project_name,'Libs','__init__.py'), 'w') as file:
            code = '''\
# This is the __init__.py file for the package

# Import necessary modules or functions here
'''
            file.write(code)
        with open(os.path.join(project_name,'Libs','models.py'), 'w') as file:
            code = '''\
#import your nessessary libreries here
import torch
import torch.nn as nn
from torch.nn import functional as F

####    DEMO
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
#
# class DemoModel(nn.Module):
#     def __init__(self, num_classes=1):
#         super(DemoModel, self).__init__()
#         self.crop_duration = 0.47
#         self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(32 * 2590, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#        
#     def forward(self, x):
#        
#         x = x.to(next(self.parameters()).device) #!important
#        
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 32 * 2590)
#         x = F.relu(self.fc1(x))
#         x = F.sigmoid(self.fc2(x))
#         return x



'''
            file.write(code)
        with open(os.path.join(project_name,'Libs','datasets.py'), 'w') as file:
            code = '''\
#import your nessessary libreries here
from torch.utils.data import Dataset

# from torch.utils.data import Dataset
# import os
# import librosa
# import torch
# import random
# 
# class BaseDataSet(Dataset):
#     def __init__(self, folder, sr=22050, crop_duration=0.47):
#         self.folder = folder
#         self.file_list = [f for f in os.listdir(folder)]
#         self.sr = sr
#         self.crop_duration = crop_duration
#         self.crop_size = int(self.crop_duration * self.sr)
# 
#     def __len__(self):
#         return len(self.file_list)
# 
#     def __getitem__(self, idx):
#         file_path = self.file_list[idx]
#         waveform, sr = librosa.load(os.path.join(self.folder, file_path))
#         waveform = torch.tensor(waveform).unsqueeze(0)
#         cropped_waveform = self.random_crop(waveform, self.crop_size)
# 
#         label = int(os.path.basename(file_path)[-5])
#         return cropped_waveform, label
# 
#     def random_crop(self, waveform, crop_size):
#         num_samples = waveform.size(1)
#         if num_samples <= crop_size:
#             padding = crop_size - num_samples
#             cropped_waveform = torch.nn.functional.pad(waveform, (0, padding))
#         else:
#             start = random.randint(0, num_samples - crop_size)
#             cropped_waveform = waveform[:, start:start + crop_size]
#         return cropped_waveform
 

'''
            file.write(code)
        with open(os.path.join(project_name,'Libs','accuracies.py'), 'w') as file:
            code = '''\
#import your nessessary libreries here


#define your accuracy functions here
####   DEMO
# from torchmetrics.classification import BinaryAccuracy

# def BinAcc():
#     return BinaryAccuracy()
'''
            file.write(code)
        with open(os.path.join(project_name,'Libs','losses.py'), 'w') as file:
            code = '''\
#import your nessessary libreries here


#define your Loss functions Here functions here
###   DEMO
# from torch import nn

# def BCElogit():
#     return nn.BCEWithLogitsLoss()
# def BCE():
#     return nn.BCELoss()
'''
            file.write(code)
        with open(os.path.join(project_name,'Libs','optimizers.py'), 'w') as file:
            code = '''\
#import your nessessary libreries here


#define your Optimizer functions here
###DEMO
#import torch.optim as optim

#def OptAdam(model,**kwargs):
#    return optim.Adam(model.parameters(),**kwargs)
'''
            file.write(code)
        os.mkdir(os.path.join(project_name,'internal'))
        os.mkdir(os.path.join(project_name,'internal','Histories'))
        os.mkdir(os.path.join(project_name,'internal','Weights'))
        os.mkdir(os.path.join(project_name,'internal','Configs'))
        with open(os.path.join(project_name,'internal','Default_Config.json'), 'w') as file:
            data = {
                "accuracy_loc": None,
                "loss_loc":None,
                "optimizer_loc":None,
                "train_data_src": None,
                "valid_data_src": None,
                "train_batch_size": None,
                "valid_batch_size": None
            }
            json.dump(data, file, indent=4)
        with open(os.path.join(project_name,'internal','config.json'), 'w') as file:
            data = {
            }
            json.dump(data, file, indent=4)
        
        os.mkdir(os.path.join(project_name,'internal','Archived'))
        os.mkdir(os.path.join(project_name,'internal','Archived','Histories'))
        os.mkdir(os.path.join(project_name,'internal','Archived','Weights'))
        os.mkdir(os.path.join(project_name,'internal','Archived','Configs'))
        with open(os.path.join(project_name,'internal','Archived','config.json'), 'w') as file:
            data = {
            
            }
            json.dump(data, file, indent=4)
        os.mkdir(os.path.join(project_name,'internal','Test'))
        print(f"{'All directories are created'}")

def get_ppls(mode='name', config="internal"): #mode = {name}|epoch|all,#config = {internal}|archive|transfer
    """
    Retrieves pipeline information based on the specified mode and configuration.

    This function reads from different configuration files depending on the `config` parameter
    and returns pipeline information according to the `mode` parameter. The function supports
    three modes to retrieve different aspects of pipeline data and three configurations to specify
    which set of pipelines to retrieve.

    Parameters
    ----------
    mode : str, optional
        Determines the type of information to return. Options include:
            - 'name' (default): Returns a list of experiment names.
            - 'epoch': Returns a list of the last trained epochs for each experiment.
            - 'all': Returns a dictionary containing the name, last epoch, and validation accuracy for each experiment.

    config : str, optional
        Specifies the configuration file to use. Options include:
            - 'internal' (default): For yor base enviroment, Uses the configuration file located at "internal/config.json".
            - 'archive': For archived experiments, Uses the configuration file located at "internal/Archived/config.json".
            - 'transfer': For the experiments which are for/from Transfer to other machine, Uses the configuration file located at "internal/Transfer/config.json".

    Returns
    -------
    list or dict
        Depending on the `mode`, the return type will vary:
            - If `mode` is 'name', a list of experiment names is returned.
            - If `mode` is 'epoch', a list of last trained epochs is returned.
            - If `mode` is 'all', a dictionary with detailed experiment information is returned.

    Raises
    ------
    FileNotFoundError
        If the configuration file specified by the `config` parameter does not exist.
    JSONDecodeError
        If there is an error decoding the JSON file.
    """

    root = {
        "internal": "internal/",
        "archive": "internal/Archived/",
        "transfer": "internal/Transfer/"
    }.get(config, "internal/")
    up2date(config=config)
    
    with open(root+"config.json") as fl:
        cnf = json.load(fl)
        if(mode=='name'):
            return list(cnf.keys())
        elif(mode=='epoch'):
            ls = [cnf[i]["last_epoch"] for i in cnf.keys()]
            return ls
        elif(mode=='all'):
            return cnf

def verify(ppl,mode='name',config="internal",log=False):   # mode = name|mod_ds|training #config = {internal}|archive|transfer
    """
    Verifies the existence or uniqueness of a pipeline based on the given mode and configuration.

    This function checks if a pipeline or its components already exist in the specified configuration 
    based on the provided `mode`. It can verify by pipeline name, model-dataset combination, or training 
    configurations. Additionally, it can log information about the presence of duplicates.

    Parameters
    ----------
    ppl : str, list, dict
        - If `mode` is 'name', `ppl` should be a string representing the pipeline name.
        - If `mode` is 'mod_ds', `ppl` should be a dictionary containing 'model_loc' and 'DataSet_loc' keys.
        - If `mode` is 'training', `ppl` should be a dictionary containing training configuration details:
          'optimizer_loc', 'train_batch_size', 'valid_batch_size', 'accuracy_loc', 'loss_loc', 
          'train_data_src', and 'valid_data_src'.
        - If `mode` is 'all', `ppl` should be a dictionary with 'piLn_name', 'model_loc', 'DataSet_loc', 
          and training configuration details as described above.

    mode : str, optional
        Specifies the type of verification to perform. Options include:
            - 'name' (default): Verifies if the pipeline name exists in the configuration.
            - 'mod_ds': Verifies if the combination of model and dataset already exists in the configuration.
            - 'training': Verifies if the combination of training configurations (optimizer, batch sizes, accuracy, loss, data sources) already exists.
            - 'all': Checks for commonalities across all specified modes (name, model-dataset, training).

    config : str, optional
        Specifies the configuration file to use. Options include:
            - 'internal' (default): Uses the `internal/config.json` file.
            - 'archive': Uses the `internal/Archived/config.json` file.
            - 'transfer': Uses the `internal/Transfer/config.json` file.

    log : bool, optional
        If True, logs information about existing pipelines or combinations that match the query. Defaults to True.

    Returns
    -------
    bool, list, or str
        - If `mode` is 'name', returns the pipeline name if it exists, otherwise False.
        - If `mode` is 'mod_ds', returns a list of names where the model-dataset combination matches, or False if no matches are found.
        - If `mode` is 'training', returns a list of names where the training configurations match, or False if no matches are found.
        - If `mode` is 'all', returns a list of commonalities between all specified modes or a message indicating common pipelines across all modes.

    Raises
    ------
    FileNotFoundError
        If the specified configuration or experiment files are not found.
    JSONDecodeError
        If there is an error decoding the JSON files.
    """
    
    root = {
        "internal": "internal/",
        "archive": "internal/Archived/",
        "transfer": "internal/Transfer/"
    }.get(config, "internal/")
    
    up2date(config=config)
    
    if(mode=='name'):
        if(isinstance(ppl,dict)):
            ppl = ppl['piLn_name']
        with open(root+"config.json") as fl:
            cnf0 = json.load(fl)
            if(ppl in cnf0.keys()):
                if(log==True):
                    print(ppl,"is already exists. ","*last pipeLine is",list(cnf0.keys())[-1])
                return ppl
            else:
                return False
    elif(mode=='mod_ds'):
        mods = []
        ls = os.listdir(root+"Configs")
        for i in ls:
            with open(os.path.join(root,"Configs",i)) as fl:
                cnf0 = json.load(fl)
                mods.append([cnf0['piLn_name'],cnf0['model_loc'],cnf0['DataSet_loc']])
        matches = []
        for i in mods:
            if(i[1]==ppl['model_loc'] and i[2]==ppl['DataSet_loc']):
                matches.append(i[0])
        if(len(matches)>0):
            if(log==True):
                print('same combination of model & dataset is already used in ',matches)
            return matches
        else:
            return False
    elif(mode=="training"):
        mods = []
        ls = os.listdir(root+"Configs")
        for i in ls:
            with open(os.path.join(root,"Configs",i)) as fl:
                cnf0 = json.load(fl)
                mods.append([cnf0['piLn_name'],cnf0['optimizer_loc'],
                            cnf0['train_batch_size'],cnf0['valid_batch_size'],
                            cnf0["accuracy_loc"],cnf0["loss_loc"],
                            cnf0["train_data_src"],cnf0["valid_data_src"]
                            
                        ])
        matches = []
        for i in mods:
            if(i[1]==ppl['optimizer_loc'] and i[2]==ppl['train_batch_size'] and i[3]==ppl['valid_batch_size'] and ppl["accuracy_loc"]==i[4] and ppl["loss_loc"]==i[5] and  ppl["train_data_src"]==i[6] and ppl["valid_data_src"]==i[7]):
                matches.append(i[0])
        if(len(matches)>0):
            if(log==True):
                print('same combination of accuracy,loss,optimizers,batch_sizes,data_sources is already used in ',matches)
            return matches
        else:
            return False
    elif(mode=='all'):

        a1 = verify(ppl['piLn_name'], mode='name', config=config, log=log)
        a2 = verify(ppl, mode='mod_ds', config=config, log=log)
        a3 = verify(ppl, mode='training', config=config, log=log)
        
        match_flags = {
            'name': [a1],
            'mod_ds': a2,
            'training': a3
        }
        
        valid_sets = {key: set(val) for key, val in match_flags.items() if val}

        if valid_sets:
            if len(valid_sets) == 3:
                return f"Common in all: {set.intersection(*valid_sets.values())}"
            else:
                intersections = dict()
                if 'name' in valid_sets and 'mod_ds' in valid_sets:
                    intersections["name & mod_ds"]= set.intersection(valid_sets['name'], valid_sets['mod_ds'])
                if 'mod_ds' in valid_sets and 'training' in valid_sets:
                    intersections["mod_ds & training"]= set.intersection(valid_sets['mod_ds'], valid_sets['training'])
                if 'name' in valid_sets and 'training' in valid_sets:
                    intersections["name & training"]= set.intersection(valid_sets['name'], valid_sets['training'])
                return intersections
        return False

def up2date(config='internal'): #config = {internal}|archive|transfer
    """
    Updates the root configuration file with the latest information from individual experiment JSON files.

    This function reads experiment data from JSON files in the `Configs` directory based on the specified `config`
    parameter and updates the corresponding root configuration file (`config.json`) with the latest epoch and 
    validation accuracy for each experiment. If an experiment is new or has updated information, it reflects 
    those changes in the root configuration file.

    Parameters
    ----------
    config : str, optional
        Specifies which configuration files to update. Options include:
            - 'internal' (default): Updates the `internal/config.json` file with data from the `internal/Configs/` folder.
            - 'archive': Updates the `internal/Archived/config.json` file with data from the `internal/Archived/Configs/` folder.
            - 'transfer': Updates the `internal/Transfer/config.json` file with data from the `internal/Transfer/Configs/` folder.

    Returns
    -------
    None
        This function does not return any value. It updates the specified configuration file directly.

    Raises
    ------
    FileNotFoundError
        If the specified configuration or experiment files are not found.
    JSONDecodeError
        If there is an error decoding the JSON files.
    """
    root = {
        "internal": "internal/",
        "archive": "internal/Archived/",
        "transfer": "internal/Transfer/"
    }.get(config, "internal/")

    ls = os.listdir(root+"Configs")
    plns = []
    # print(root)
    for i in ls:
        with open(os.path.join(root+"Configs",i)) as fl:
            cnf0 = json.load(fl)
            plns.append([cnf0['piLn_name'],cnf0['last']['epoch'],cnf0['best']['val_accuracy']])
    with open(root+"config.json") as fl:
        cnf = json.load(fl)
        for i in plns:
            if(i[0] in cnf.keys()):
                if(cnf[i[0]]["last_epoch"]<i[1]):
                    print(f"last epoch updated from {cnf[i[0]]['last_epoch']} to {i[1]} for PipeLine:{i[0]}")
                    cnf[i[0]]["last_epoch"]=i[1]
                    
                if(cnf[i[0]]["best_val_accuracy"]!=i[2]):
                    print(f"Validation accuracy updated from {cnf[i[0]]['best_val_accuracy']} to {i[2]} for PipeLine:{i[0]}")
                    cnf[i[0]]["best_val_accuracy"]=i[2]
            else:
                cnf[i[0]]={
                            "last_epoch":i[1],
                            "best_val_accuracy":i[2]
                        }
                print(f"new Pipeline initialized : {i[0]} with last_epoch:{i[1]} and best_val_accuracy:{i[2]}")
    with open(root+"config.json","w") as fl:
            json.dump(cnf, fl, indent=4)

def set_default_config(data:dict):
    """
    Set the default configuration for the project.

    This function saves the provided configuration data to 'internal/Default_Config.json'. The
    configuration includes paths to accuracy, loss, optimizer, dataset, and batch sizes for training 
    and validation.

    Parameters
    ----------
    data : dict
        A dictionary containing the configuration settings. Keys should include 'accuracy_loc', 
        'loss_loc', 'optimizer_loc', 'train_data_src', 'valid_data_src', 'train_batch_size', and 
        'valid_batch_size'.

    Returns
    -------
    None
        The function writes the configuration to a JSON file.

    Examples
    --------
    >>> set_default_config({
            "accuracy_loc": "Libs.accuracies.BinAcc",
            "loss_loc": "Libs.losses.BinLoss",
            "optimizer_loc": "Libs.optimizers.AdamOpt",
            "train_data_src": "DataSets/train",
            "valid_data_src": "DataSets/valid",
            "train_batch_size": 32,
            "valid_batch_size": 32
        })
    # Saves the configuration to the default configuration file.
    """
    cnfg = {
                "accuracy_loc": data.get("accuracy_loc"),
                "loss_loc":data.get("loss_loc"),
                "optimizer_loc":data.get("optimizer_loc"),
                "train_data_src": data.get("train_data_src"),
                "valid_data_src": data.get("valid_data_src"),
                "train_batch_size": data.get("train_batch_size"),
                "valid_batch_size": data.get("valid_batch_size")
            }
    with open(os.path.join('internal','Default_Config.json'), 'w') as file:
            json.dump(cnfg, file, indent=4)

def test_mods(dataset=None,model=None,model_loc=None, accuracy_loc=None, loss_loc=None, optimizer_loc=None, dataset_loc=None,
              train_data_src=None, train_batch_size=None, valid_data_src=None, valid_batch_size=None,
              prepare=False):
    """
    Configures and initializes a testing pipeline for evaluating a machine learning model.

    This function sets up a testing pipeline by configuring the model, dataset, and various paths for saving metrics, 
    optimizer state, and model weights. It uses default configuration values if some parameters are not provided.

    Parameters:
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

    Returns:
    - PipeLine: An instance of the `PipeLine` class configured for testing with the provided or default parameters.
    """
    if(model_loc!=None and len(model_loc.split('.'))==1):
        model_loc = 'Libs.models.'+model_loc
    if(dataset_loc!=None and len(dataset_loc.split('.'))==1):
        dataset_loc = 'Libs.datasets.'+dataset_loc
    with open("internal/Default_Config.json") as fl:
        def_conf = json.load(fl)
    if(accuracy_loc is None):
        accuracy_loc = def_conf['accuracy_loc']
    if(loss_loc is None):
        loss_loc = def_conf['loss_loc']
    if(optimizer_loc is None):
        optimizer_loc = def_conf['optimizer_loc']
    if(train_data_src is None):
        train_data_src = def_conf['train_data_src']
    if(valid_data_src is None):
        valid_data_src =def_conf['valid_data_src']
    if(train_batch_size is None):
        train_batch_size = def_conf['train_batch_size']
    if(valid_batch_size is None):
        valid_batch_size =def_conf['valid_batch_size']
    if(accuracy_loc!=None and len(accuracy_loc.split('.'))==1):
        accuracy_loc = 'Libs.accuracies.'+accuracy_loc
    if(loss_loc!=None and len(loss_loc.split('.'))==1):
        loss_loc = 'Libs.losses.'+loss_loc
    if(optimizer_loc!=None and len(optimizer_loc.split('.'))==1):
        optimizer_loc = 'Libs.optimizers.'+optimizer_loc

    P = PipeLine()
    if(dataset and issubclass(dataset,Dataset)):
        P.DataSet = dataset
    if(model and issubclass(model.__class__,nn.Module)):
        P.model = model
    P.setup(name='test', 
            model_loc=model_loc, accuracy_loc=accuracy_loc, 
            loss_loc=loss_loc, optimizer_loc=optimizer_loc, 
            dataset_loc=dataset_loc, train_data_src=train_data_src, 
            train_batch_size=train_batch_size, valid_data_src=valid_data_src, 
            valid_batch_size=valid_batch_size, history_path="internal/Test/test_h.csv",
            weights_path="internal/Test/test_w.pth", 
            config_path="internal/Test/test_c.json", prepare=prepare)
    return P

def train_new(
                    name=None,
                    model_loc=None,
                    loss_loc=None,
                    accuracy_loc=None,
                    optimizer_loc=None,
                    dataset_loc=None,
                    train_data_src=None,
                    valid_data_src=None,
                    train_batch_size=None,
                    valid_batch_size=None,
                    prepare=None,
                ):
    """
    Initializes and sets up a new pipeline for training a machine learning model.

    This function configures a new training pipeline by specifying the model, dataset, loss function, accuracy metric, optimizer,
    and various training parameters. It uses default values from a configuration file(saved using save_default_config) if certain parameters are not provided. 
    The function verifies the uniqueness of the pipeline name and initializes a `PipeLine` instance if the name is not already in use.

    Parameters
    ----------
    name : str, optional
        The name of the pipeline. If not provided, the pipeline will not be created.
    
    model_loc : str, optional
        The location of the model module. If a simple name is provided, it is prefixed with 'Libs.models.'.

    loss_loc : str, optional
        The location of the loss function module. If a simple name is provided, it is prefixed with 'Libs.losses.'.
    
    accuracy_loc : str, optional
        The location of the accuracy metric module. If a simple name is provided, it is prefixed with 'Libs.accuracies.'.

    optimizer_loc : str, optional
        The location of the optimizer module. If a simple name is provided, it is prefixed with 'Libs.optimizers.'.

    dataset_loc : str, optional
        The location of the dataset module. If a simple name is provided, it is prefixed with 'Libs.datasets.'.
    
    train_data_src : str, optional
        The source path for training data. Defaults to the value specified in the default configuration file if not provided.
    
    valid_data_src : str, optional
        The source path for validation data. Defaults to the value specified in the default configuration file if not provided.
    
    train_batch_size : int, optional
        The batch size for training data. Defaults to the value specified in the default configuration file if not provided.
    
    valid_batch_size : int, optional
        The batch size for validation data. Defaults to the value specified in the default configuration file if not provided.
    
    prepare : callable, optional
        A function or callable to prepare the pipeline. This function is called during the setup of the pipeline.

    Returns
    -------
    PipeLine
        An instance of the `PipeLine` class configured with the specified parameters, or `None` if the pipeline name is already in use.

    Notes
    -----
    - If the `name` provided is already used by an existing pipeline, a new pipeline will not be created.
    - Default values are fetched from the "internal/Default_Config.json" configuration file.
    - Paths for weights, history, and configuration files are constructed based on the provided `name`.
    """
    P = PipeLine()

    if(model_loc!=None and len(model_loc.split('.'))==1):
        model_loc = 'Libs.models.'+model_loc
    if(dataset_loc!=None and len(dataset_loc.split('.'))==1):
        dataset_loc = 'Libs.datasets.'+dataset_loc
    with open("internal/Default_Config.json") as fl:
        def_conf = json.load(fl)
    if(accuracy_loc is None):
        accuracy_loc = def_conf['accuracy_loc']
    if(loss_loc is None):
        loss_loc = def_conf['loss_loc']
    if(optimizer_loc is None):
        optimizer_loc = def_conf['optimizer_loc']
    if(train_data_src is None):
        train_data_src = def_conf['train_data_src']
    if(valid_data_src is None):
        valid_data_src =def_conf['valid_data_src']
    if(train_batch_size is None):
        train_batch_size = def_conf['train_batch_size']
    if(valid_batch_size is None):
        valid_batch_size =def_conf['valid_batch_size']
    if(accuracy_loc!=None and len(accuracy_loc.split('.'))==1):
        accuracy_loc = 'Libs.accuracies.'+accuracy_loc
    if(loss_loc!=None and len(loss_loc.split('.'))==1):
        loss_loc = 'Libs.losses.'+loss_loc
    if(optimizer_loc!=None and len(optimizer_loc.split('.'))==1):
        optimizer_loc = 'Libs.optimizers.'+optimizer_loc
    dct={
        'model_loc': model_loc,
        'DataSet_loc': dataset_loc,
        'accuracy_loc': accuracy_loc,
        'loss_loc': loss_loc,
        'optimizer_loc': optimizer_loc,
        'piLn_name': name,
        'valid_batch_size': valid_batch_size,
        'train_batch_size': train_batch_size,
        }
    if(verify(ppl=dct['piLn_name'],mode='name') == False):
        P.setup(
            name=name,
            model_loc=model_loc,
            loss_loc=loss_loc,
            accuracy_loc=accuracy_loc,
            optimizer_loc=optimizer_loc,
            dataset_loc=dataset_loc,
            train_data_src=train_data_src,
            valid_data_src=valid_data_src,
            weights_path='internal/Weights/'+name+'.pth',
            history_path='internal/Histories/'+name+'.csv',
            config_path='internal/Configs/'+name+'.json',
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            make_config=True,
            prepare=prepare
            )
        return P
    return P
        
def re_train(ppl=None,config_path=None,train_data_src=None,valid_data_src=None,prepare=None,num_epochs=0):
    """
    Re-trains an existing pipeline or initializes a new pipeline with the provided configuration.

    This function sets up and optionally trains a `PipeLine` instance based on the specified configuration or pipeline name. 
    If a pipeline name (`ppl`) is provided, it constructs the configuration path from the pipeline name. If `num_epochs` is 
    specified and greater than zero, it performs training for the given number of epochs.

    Parameters
    ----------
    ppl : str, optional
        The name of the pipeline for which to re-train or initialize. If provided, the function constructs the configuration 
        path as "internal/Configs/{ppl}.json". If `ppl` is `None`, the `config_path` must be specified.
    
    config_path : str, optional
        The path to the configuration file. This parameter is ignored if `ppl` is provided, as the configuration path will 
        be constructed from the `ppl` name.
    
    train_data_src : str, optional
        The source path for training data. This is used only if `num_epochs` is greater than zero and a new training session 
        is to be started.
    
    valid_data_src : str, optional
        The source path for validation data. This is used only if `num_epochs` is greater than zero and a new training session 
        is to be started.
    
    prepare : callable, optional
        A function or callable to prepare the pipeline. This function is called during the setup of the pipeline if `num_epochs` 
        is zero or if `prepare` is specified.
    
    num_epochs : int, optional
        The number of epochs for training. If greater than zero, the `PipeLine` will be trained for this many epochs. If zero, 
        only the pipeline will be set up without training.

    Returns
    -------
    PipeLine
        An instance of the `PipeLine` class, set up and optionally trained according to the provided parameters.

    Notes
    -----
    - If both `ppl` and `config_path` are provided, `ppl` takes precedence, and `config_path` will be constructed from `ppl`.
    - The function will initialize a new `PipeLine` instance if the `ppl` parameter is provided or if `config_path` is specified.
    - Training is only performed if `num_epochs` is greater than zero.
    """
    
    P = PipeLine()
    if(ppl and verify(ppl,config='internal',mode='name',log=False)):
        config_path = "internal/Configs/"+ppl+".json"
    else:
        print(ppl, "not exists")
        return None
    if(num_epochs>0):
        P.setup(config_path=config_path, 
                train_data_src=train_data_src, 
                valid_data_src=valid_data_src, 
                use_config=True, prepare=True)
        P.train(num_epochs=num_epochs)
    elif(num_epochs==0):
        P.setup(config_path=config_path, 
                train_data_src=train_data_src, 
                valid_data_src=valid_data_src, 
                use_config=True, prepare=prepare)
    return P

def multi_train(ppl = None,last_epoch=10):#
    """
    Train or re-train multiple pipelines up to the specified number of epochs.

    This function checks the number of epochs each pipeline has already been trained for, and continues training
    until the specified `last_epoch` is reached. The training configuration for each pipeline is read from 
    a corresponding JSON file.

    Parameters
    ----------
    ppl : list of str, optional
        A list of pipeline names to train. If not provided, all pipelines will be selected.
    last_epoch : int, optional
        The maximum number of epochs to train for. Defaults to 10.
    steps : int, optional
        Steps parameter, currently not in use.

    Returns
    -------
    None
        The function prints training progress and status messages.

    Notes
    -----
    - If the pipeline has already been trained for the specified `last_epoch`, it will be skipped.
    - Each pipeline's configuration is expected to be located at 'internal/Configs/{pipeline_name}.json'.

    Examples
    --------
    >>> multi_train(ppl=['pipeline1', 'pipeline2'], last_epoch=15)
    # Trains 'pipeline1' and 'pipeline2' up to 15 epochs.
    """
    if(isinstance(ppl, list)):
        pa = get_ppls(mode='all')
        epoch = [pa[i]['last_epoch'] for i in ppl]
    else:
        ppl = get_ppls(mode='name')
        epoch = get_ppls(mode='epoch')
    for i in range(len(ppl)):
        if(epoch[i]<last_epoch):
            print(f"{ppl[i]:=^60}")
            re_train(config_path='internal/Configs/'+ppl[i]+'.json',prepare=True,num_epochs=last_epoch-epoch[i])
    print("All training Done")

def performance_plot(ppl=None,history=None,df=None,config="internal"):
    """
    Plots performance metrics (accuracy and loss) over epochs for one or more pipelines.

    This function generates plots for training and validation accuracy, as well as training and validation loss,
    using data from a CSV file or a DataFrame. It can handle single or multiple pipelines, and plots the performance 
    metrics for each specified pipeline.

    Parameters
    ----------
    ppl : str, list, optional
        The name(s) of the pipeline(s) for which to plot performance metrics. If `None`, it fetches all pipeline names 
        using `get_ppls`. If `ppl` is a list, it plots performance metrics for each pipeline in the list. If `ppl` is a 
        string, it is treated as a single pipeline name.

    history : str, optional
        The path to the CSV file containing performance metrics (e.g., accuracy and loss) over epochs. If `None`, it 
        defaults to the file located in `internal/Histories/` directory with the name corresponding to the pipeline 
        name.

    df : pandas.DataFrame, optional
        A DataFrame containing performance metrics (e.g., accuracy and loss) over epochs. If `df` is provided, it is used 
        directly for plotting. If `None`, it will attempt to read the DataFrame from the CSV file specified in `history`.

    config : str, optional
        The configuration type to use when retrieving the pipeline. Options are:
        - `internal` [default]: Uses configurations from the `internal/` directory.
        - `archive`: Uses configurations from the `internal/Archived/` directory.
        - `transfer`: Uses configurations from the `internal/Transfer/` directory.

    Returns
    -------
    None
        Displays the performance plots using Matplotlib. If there is an error or if the DataFrame is empty, it returns an 
        error message as a string.

    Notes
    -----
    - The function updates the configuration before attempting to load the performance history.
    - If both `history` and `df` are `None`, an error message is printed.
    - If the DataFrame is empty, an appropriate message is printed.
    """
    if(ppl is None):
        ppl = get_ppls(mode='name',config=config)
    if(isinstance(ppl,list)):
        for i in ppl:
            performance_plot(ppl=i,config=config)
        return None
    elif(isinstance(ppl,str)):

        root = {
        "internal": "internal/",
        "archive": "internal/Archived/",
        "transfer": "internal/Transfer/"
            }.get(config, "internal/")

        up2date(config=config)

        history = root+"Histories/"+ppl+".csv"
        if( not os.path.isfile(history)):
            return f"the file{history}  is not found"
    
    if( history!=None):
        df = pd.read_csv(f"{history}")
    if(df is None):
        print("It needs one of arguments history and df. df is dtaframe from the csv file and history is path to history.csv")
        return "Error"
    if(df.empty):
        print(history.split('/')[-1].split('.')[0],"Empty")
        return "Empty DataFrame given..!"
        
    fig,ax = plt.subplots(1,2)
    fig.set_size_inches(15,5)
    df.plot(x='epoch',y=['train_accuracy','val_accuracy'],ax=ax[0],title="Accuracy trade-off")
    df.plot(x='epoch',y=['train_loss','val_loss'],ax=ax[1],title="Loss trade-off")
    fig.suptitle(history.split('/')[-1].split('.')[0], fontsize=16)
    plt.show()

def get_model(ppl=None,name=True,config="internal"):
    """
    Retrieves the model class or its name for a given pipeline or a list of pipelines.

    This function fetches the model class or its name based on the specified pipeline(s). It can handle single pipeline names, 
    lists of pipeline names, or return a list of models corresponding to those pipelines.

    Parameters
    ----------
    ppl : str, list, optional
        The name(s) of the pipeline(s) for which to retrieve the model. If `ppl` is `None`, it fetches all pipeline names 
        using `get_ppls`. If `ppl` is a list, it returns a list of models for each pipeline in the list. If `ppl` is a string, 
        it is treated as a single pipeline name.
    
    name : bool, optional
        If `True`, returns the name of the model class as a string. If `False`, returns an instance of the model class. 
        The default is `True`.
    
    config : str, optional
        The configuration type to use when retrieving the pipeline. Options are:
        - `internal` [default]: Uses configurations from the `internal/` directory.
        - `archive`: Uses configurations from the `internal/Archived/` directory.
        - `transfer`: Uses configurations from the `internal/Transfer/` directory.

    Returns
    -------
    str, object, list of str, list of objects
        - If `ppl` is a single string, returns either the model name (if `name=True`) or an instance of the model (if `name=False`).
        - If `ppl` is a list of strings, returns a list of model names or instances for each pipeline in the list.
        - If `ppl` is `None`, retrieves all pipeline names and returns them in the specified format.
    
    Notes
    -----
    - The function updates the configuration before attempting to load the model.
    - Model classes are imported dynamically based on the module location specified in the pipeline configuration.
    - The module location should be a valid Python module path, and the model class should be defined in that module.
    """
    if(ppl is None):
        ppl = get_ppls(mode='name',config=config)
        return get_model(ppl=ppl,name=name,config=config)
    elif(isinstance(ppl,list)):
        models = [get_model(ppl=i, name=name,config=config) for i in ppl]
        return models
    elif(isinstance(ppl,str)):

        root = {
            "internal": "internal/",
            "archive": "internal/Archived/",
            "transfer": "internal/Transfer/"
        }.get(config, "internal/")
        
        up2date(config=config)
        file = root+"Configs/"+ppl+".json"
        with open(file) as fl:
            cnf = json.load(fl)
            module_loc = cnf['model_loc']
            if(name==True):
                return module_loc.split('.')[-1]
            else:
                module = importlib.import_module('.'.join(module_loc.split('.')[:-1]))
                class_ = getattr(module, module_loc.split('.')[-1])                
                return class_()

def use_ppl(ppl,trained=True,name=None,
            loss_loc=None,
            accuracy_loc=None,
            optimizer_loc=None,
            train_data_src=None,
            valid_data_src=None,
            train_batch_size=None,
            valid_batch_size=None,
            prepare=None):
    """
    Configures and initializes a pipeline for an existing model or creates a new pipeline if necessary.

    This function either sets up a pipeline based on an existing configuration and trained model weights, 
    or trains a new model if no trained pipeline is available. It also updates and verifies the configuration before use.

    Parameters:
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

    Returns:
    - PipeLine: An instance of the `PipeLine` class, either initialized with the provided or default configuration 
                and pre-trained weights, or trained from scratch if no pre-trained pipeline is found.

    Notes:
    - If the pipeline is not already trained, a new one is created using the provided configuration.
    - If `trained=True`, copies the existing model weights and history to the new pipeline paths.
    - The `config_path`, `weights_path`, and `history_path` for the new pipeline are generated based on the `name` parameter.
    - The configuration file is updated and saved before initializing the pipeline.

    Raises:
    - Any verification issues with the provided pipeline configuration are printed and no pipeline is returned.
    """
    if(accuracy_loc!=None and len(accuracy_loc.split('.'))==1):
        accuracy_loc = 'Libs.accuracies.'+accuracy_loc
    if(loss_loc!=None and len(loss_loc.split('.'))==1):
        loss_loc = 'Libs.losses.'+loss_loc
    if(optimizer_loc!=None and len(optimizer_loc.split('.'))==1):
        optimizer_loc = 'Libs.optimizers.'+optimizer_loc
    config_path = "internal/Configs/"+ppl+".json"
    with open(config_path) as fl:
        cnfg = json.load(fl)
    cnfg.update({
            'piLn_name': name,
            'valid_batch_size': valid_batch_size or cnfg['valid_batch_size'],
            'valid_data_src': valid_data_src or cnfg['valid_data_src'],
            'train_batch_size': train_batch_size or cnfg['train_batch_size'],
            'train_data_src': train_data_src or cnfg['train_data_src'],
            'optimizer_loc' :optimizer_loc or cnfg['optimizer_loc'],
            'accuracy_loc' : cnfg['accuracy_loc'],
            'loss_loc' : cnfg['loss_loc'],
            'history_path': "internal/Histories/"+name+".csv",
            'weights_path': "internal/Weights/"+name+".pth",
            'config_path' : "internal/Configs/"+name+".json"
        })
    vrf = verify(ppl=cnfg, mode="all")
    if((not vrf) or ("mod_ds & training" not in vrf.keys())):
        if(trained):
            with open("internal/Configs/"+name+".json",'w') as fl:
                json.dump(cnfg, fl, indent=4)
            shutil.copy2(src="internal/Configs/"+ppl+".pth",dst=cnfg['weights_path'])
            shutil.copy2(src="internal/Configs/"+ppl+".csv",dst=cnfg['history_path'])
            print("New ppl created ",name)
            P =PipeLine()
            P.setup(config_path=cnfg['config_path'],prepare=prepare)
            return P
        else:
            P = train_new(
                    name = name,
                    model_loc = cnfg['model_loc'],
                    loss_loc = cnfg['loss_loc'],
                    accuracy_loc = cnfg['accuracy_loc'],
                    optimizer_loc = cnfg['optimizer_loc'],
                    dataset_loc = cnfg['dataset_loc'],
                    train_data_src = cnfg['train_data_src'],
                    valid_data_src = cnfg['valid_data_src'],
                    train_batch_size = cnfg['train_batch_size'],
                    valid_batch_size = cnfg['valid_batch_size'],
                    prepare=prepare
                )
            return P
    else:
        print(vrf)

def archive(ppl, reverse = False):
    """
    Archive or restore a project pipeline's files.

    This function moves the project's configuration, weights, and history files to or from the 'internal/Archived' 
    directory based on the provided pipeline name(s). Archiving moves files to the archive, while restoring 
    moves them back.

    Parameters
    ----------
    ppl : str or list of str
        The name(s) of the project pipeline to archive or restore.
    reverse : bool, optional
        If True, restores the project from the archive to the active directory. Defaults to False.

    Returns
    -------
    None
        The function prints the status of the archiving process.

    Notes
    -----
    - The function checks whether the project is already archived before proceeding.
    - If a list of projects is provided, the function archives or restores each project in the list.

    Examples
    --------
    >>> archive('pipeline1')
    # Archives 'pipeline1' files.

    >>> archive(['pipeline1', 'pipeline2'], reverse=True)
    # Restores 'pipeline1' and 'pipeline2' files from the archive.
    """
    if(isinstance(ppl,str)):
        up2date(config="internal")
        source = "internal/"
        destin = "internal/Archived/"
        log = verify(ppl=ppl,mode="name",config="internal",log=False)
        if(reverse):
            up2date(config="archive")
            destin = "internal/"
            source = "internal/Archived/"
            log = verify(ppl=ppl,mode="name",config="archive",log=False)
        
            
        if(log is not False):
            shutil.move(src=source+"Weights/"+ppl+".pth",dst=destin+"Weights/"+ppl+".pth")
            shutil.move(src=source+"Configs/"+ppl+".json",dst=destin+"Configs/"+ppl+".json")
            shutil.move(src=source+"Histories/"+ppl+".csv",dst=destin+"Histories/"+ppl+".csv")
           
            with open(source+"config.json") as fl:
                cnf0 = json.load(fl)
            with open(destin+"config.json") as fl:
                cnf1 = json.load(fl)
            with open(destin+"config.json","w") as fl:
                cnf1[ppl] = cnf0[ppl]
                json.dump(cnf1, fl, indent=4)
            del cnf0[ppl]
            with open(source+"config.json","w") as fl:
                json.dump(cnf0, fl, indent=4)
            if(reverse):
                print(f"{ppl} unarchived successfully")
            else:
                print(f"{ppl} archived successfully")
        else:
            print(f"could not transfer! check {ppl}'s availability")
    
    # if(isinstance(ppl,str)):
    #     up2date(config="archive")
    #     log = verify(ppl,mode='name',config="internal",log=False)
    #     if(reverse):
    #         log = verify(ppl,mode='name',config="archive",log=False)
    #     if(log is not False):
    #         with open("internal/Configs/"+ppl+".json") as fl:
    #             cnf = json.load(fl)
    #         shutil.move(src=cnf["weights_path"],dst=os.path.join("internal","Archived","Weights",os.path.basename(cnf["weights_path"])))
    #         shutil.move(src=cnf["config_path"],dst=os.path.join("internal","Archived","Configs",os.path.basename(cnf["config_path"])))
    #         shutil.move(src=cnf["history_path"],dst=os.path.join("internal","Archived","Histories",os.path.basename(cnf["history_path"])))
    #         with open("internal/config.json") as fl:
    #             cnf0 = json.load(fl)
    #         with open("internal/Archived/config.json") as fl:
    #             cnf1 = json.load(fl)
    #         with open("internal/Archived/config.json","w") as fl:
    #             cnf1[ppl] = cnf0[ppl]
    #             json.dump(cnf1, fl, indent=4)
    #         del cnf0[ppl]
    #         with open("internal/config.json","w") as fl:
    #             json.dump(cnf0, fl, indent=4)
    #         print(f"{ppl} archived successfully")
    
    elif(isinstance(ppl,list)):
        for i in ppl:
            archive(i, reverse=reverse)

def delete(ppl):
    """
    Delete project files from the archive.

    This function deletes project files (configuration, weights, and history) from the 
    archive directory. It operates on files specified by the `ppl` parameter.

    Parameters
    ----------
    ppl : str or list of str
        The name(s) of the project to delete. If a string, it represents a single project; 
        if a list, it represents multiple projects.

    Returns
    -------
    None
        The function performs file operations but does not return any value.

    Notes
    -----
    - The function will attempt to remove files from the `internal/Archived` directory.
    - If the project is not found in the archive, a message will be printed.

    Examples
    --------
    >>> delete('my_project')
    # Deletes the files for 'my_project' if they exist in the archive.

    >>> delete(['project1', 'project2'])
    # Deletes the files for 'project1' and 'project2' if they exist in the archive.
    """
    if(isinstance(ppl,str)):
        log = verify(config="archive",log=False)
        if(not log):
            os.remove("internal/Archived/Configs/"+ppl+".json")
            os.remove("internal/Archived/Weights/"+ppl+".pth")
            os.remove("internal/Archived/Histories/"+ppl+".csv")
        else:
            print(f"{ppl} is not in archive")
    elif(isinstance(ppl,list)):
        for i in ppl:
            delete(ppl=i)

def setup_transfer():
    """
    Setup the directory structure for transferring pipelines.

    This function creates the necessary directories and configuration files for transferring pipelines
    between environments. It creates a 'Transfer' folder inside 'internal', with subdirectories for 
    storing histories, weights, and configurations.

    Returns
    -------
    None
        The function prints a message indicating whether the setup was completed or already exists.

    Examples
    --------
    >>> setup_transfer()
    # Sets up the directory structure for pipeline transfer.
    """
    if not os.path.exists(os.path.join('internal','Transfer')):
        os.mkdir(os.path.join('internal','Transfer'))
        os.mkdir(os.path.join('internal','Transfer','Histories'))
        os.mkdir(os.path.join('internal','Transfer','Weights'))
        os.mkdir(os.path.join('internal','Transfer','Configs'))
        with open(os.path.join('internal','Transfer','config.json'), 'w') as file:
            data = {
                }
            json.dump(data, file, indent=4)
        print("Setup completed for transfer...")
    else:
        print("already exists")

def transfer(ppl, type='export',mode="copy"): #type=export|import,mode=copy|move    
    """
    Transfer pipeline files between active and transfer directories.

    This function transfers pipeline files (configurations, weights, and histories) either 
    from the active directory to the transfer directory or vice versa. The transfer can be 
    done using either copy or move operations.

    Parameters
    ----------
    ppl : str or list of str
        The name(s) of the project pipeline to transfer.
    type : str, optional
        The type of transfer, either 'export' to transfer to the transfer directory or 'import' 
        to transfer back to the active directory. Defaults to 'export'.
    mode : str, optional
        The method of transfer: 'copy' to duplicate files, or 'move' to transfer them. 
        Defaults to 'copy'.

    Returns
    -------
    None
        The function prints the status of the transfer process.

    Notes
    -----
    - If `mode` is 'move', the files are deleted from the source after being transferred.

    Examples
    --------
    >>> transfer('pipeline1', type='export', mode='move')
    # Moves 'pipeline1' files to the transfer directory.

    >>> transfer(['pipeline1', 'pipeline2'], type='import', mode='copy')
    # Copies 'pipeline1' and 'pipeline2' files from the transfer directory back to the active directory.
    """

    if(isinstance(ppl,str)):
        if(type=="export"):
            up2date(config="internal")
            source = "internal/"
            destin = "internal/Transfer/"
            log = verify(ppl=ppl,mode="name",config="internal",log=False)
        elif(type=="import"):
            up2date(config="transfer")
            destin = "internal/"
            source = "internal/Transfer/"
            log = verify(ppl=ppl,mode="name",config="transfer",log=False)
        if(log is not False):
            if(mode=="move"):
                shutil.move(src=source+"Weights/"+ppl+".pth",dst=destin+"Weights/"+ppl+".pth")
                shutil.move(src=source+"Configs/"+ppl+".json",dst=destin+"Configs/"+ppl+".json")
                shutil.move(src=source+"Histories/"+ppl+".csv",dst=destin+"Histories/"+ppl+".csv")
            elif(mode=="copy"):
                shutil.copy2(src=source+"Histories/"+ppl+".csv",dst=destin+"Histories/"+ppl+".csv")
                shutil.copy2(src=source+"Weights/"+ppl+".pth",dst=destin+"Weights/"+ppl+".pth")
                shutil.copy2(src=source+"Configs/"+ppl+".json",dst=destin+"Configs/"+ppl+".json")
                
            with open(source+"config.json") as fl:
                cnf0 = json.load(fl)
            with open(destin+"config.json") as fl:
                cnf1 = json.load(fl)
            with open(destin+"config.json","w") as fl:
                cnf1[ppl] = cnf0[ppl]
                json.dump(cnf1, fl, indent=4)
            if(mode=="move"):
                del cnf0[ppl]
            with open(source+"config.json","w") as fl:
                json.dump(cnf0, fl, indent=4)
            print(f"{ppl} tranfered successfully")
        else:
            print(f"could not transfer! check {ppl}'s availability or make sure `setup_trasfer` for the first time before using `transfer`")
        
    elif(isinstance(ppl,list)):
        for i in ppl:
            transfer(ppl=i,mode=mode,type=type)
