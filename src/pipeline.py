import torch
import shutil
# import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pandas as pd
from tqdm import tqdm
import json
import importlib
import pandas as pd
# import numpy as np
from matplotlib import pyplot as plt

class PipeLine:
    def __init__(self, name='Default_name', config_path=None):
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
        module = importlib.import_module('.'.join(module_loc.split('.')[:-1]))
        class_ = getattr(module, module_loc.split('.')[-1])
        return class_

    def load_optimizer(self, module_loc, **kwargs):
        module = importlib.import_module('.'.join(module_loc.split('.')[:-1]))
        class_ = getattr(module, module_loc.split('.')[-1])
        return class_(self.model, **kwargs)

    def save_config(self):
        with open(self.cnfg['config_path'], "w") as out_file:
            json.dump(self.cnfg, out_file, indent=4)

    def setup(self, name=None, model_loc=None, accuracy_loc=None, loss_loc=None, optimizer_loc=None, dataset_loc=None,
              train_folder=None, train_batch_size=None, valid_folder=None, valid_batch_size=None, history_path=None,
              weights_path=None, config_path=None, use_config=False, make_config=True, prepare=False):

        cnfg = {
            'model_loc': model_loc,
            'DataSet_loc': dataset_loc,
            'accuracy_loc': accuracy_loc,
            'loss_loc': loss_loc,
            'optimizer_loc': optimizer_loc,
            "piLn_name": name if name else self.name,
            'last': {'epoch': 0, 'train_accuracy': 0, 'train_loss': float('inf')},
            'best': {'epoch': 0, 'val_accuracy': 0, 'val_loss': float('inf')},
            "valid_folder": valid_folder,
            'train_folder': train_folder,
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
            cnfg1["valid_folder"] = valid_folder if valid_folder!=None else cnfg1["valid_folder"]
            cnfg1['train_folder'] = train_folder if train_folder!=None else cnfg1['train_folder']
            self.cnfg = cnfg1

        elif config_path and make_config:
            if not (name and model_loc and accuracy_loc and loss_loc and optimizer_loc):
                raise ValueError("Required parameters: name, model_loc, accuracy_loc, loss_loc, optimizer_loc")

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
#         if not hasattr(self.model, 'parameters'):
#             raise ValueError(f"The model class loaded from {self.cnfg['model_loc']} does not have a 'parameters' attribute.")
        
        self.loss = self.load_component(self.cnfg['loss_loc'])()
        self.optimizer = self.load_optimizer(self.cnfg['optimizer_loc'])
        self.accuracy = self.load_component(self.cnfg['accuracy_loc'])()
        self.DataSet = self.load_component(self.cnfg['DataSet_loc'])
        
        if prepare:
            dataset_loc = self.cnfg['DataSet_loc'] if(dataset_loc==None) else dataset_loc
            self.prepare_data(dataset_loc=dataset_loc)

    def prepare_data(self, dataset_loc=None, train_folder=None, train_batch_size=None, valid_folder=None, valid_batch_size=None):
        if not (train_folder or self.cnfg['train_folder']):
            raise ValueError('train_folder is not found')
        if not (valid_folder or self.cnfg['valid_folder']):
            raise ValueError('valid_folder is not found')
        if not (train_batch_size or self.cnfg['train_batch_size']):
            raise ValueError('train_batch_size is not found')
        if not (valid_batch_size or self.cnfg['valid_batch_size']):
            raise ValueError('valid_batch_size is not found')

        self.cnfg.update({
            'valid_batch_size': valid_batch_size or self.cnfg['valid_batch_size'],
            'valid_folder': valid_folder or self.cnfg['valid_folder'],
            'train_batch_size': train_batch_size or self.cnfg['train_batch_size'],
            'train_folder': train_folder or self.cnfg['train_folder'],
        })
        self.save_config()

        self.DataSet = self.load_component(dataset_loc) if (dataset_loc!=None) else self.DataSet

        self.trainDataLoader = DataLoader(self.DataSet(self.cnfg['train_folder']), batch_size=self.cnfg['train_batch_size'], shuffle=True)
        self.validDataLoader = DataLoader(self.DataSet(self.cnfg['valid_folder']), batch_size=self.cnfg['valid_batch_size'], shuffle=False)
        print('Data loaders are successfully created')
        
        self.model=self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        if(self.cnfg['last']['epoch']>0):
                self.model.load_state_dict(torch.load(self.weights_path))
        self.__configured =True

    def update(self, data):
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
        if (not self.__configured):
            print('Preparation Error')
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

def up2date(config_folder='./internal/Configs',archive=False):
    config_file = "./internal/config.json"
    if(archive):
        config_folder = "internal/Archived/Configs"
        config_file =   "internal/Archived/config.json"
    ls = os.listdir(config_folder)
    plns = []
    for i in ls:
        with open(os.path.join(config_folder,i)) as fl:
            cnf0 = json.load(fl)
            plns.append([cnf0['piLn_name'],cnf0['last']['epoch'],cnf0['best']['val_accuracy']])
    with open(config_file) as fl:
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
    with open(config_file,"w") as fl:
            json.dump(cnf, fl, indent=4)

def get_pplns(mode='name', archive=False): #mode = name|epoch|all
    cnf = "./internal/config.json"
    if(archive):
        cnf = cnf = "./internal/Archived/config.json"
    up2date(archive=archive)
    
    with open(cnf) as fl:
        cnf = json.load(fl)
        if(mode=='name'):
            return list(cnf.keys())
        elif(mode=='epoch'):
            ls = [cnf[i]["last_epoch"] for i in cnf.keys()]
            return ls
        elif(mode=='all'):
            return cnf

def verify(cnfg,mode='name',config_folder='./internal/Configs'):   # mode = name|mod_ds|training
    if(mode=='name'):
        up2date(config_folder=config_folder)
        with open("./internal/config.json") as fl:
            cnf0 = json.load(fl)
            if(cnfg['piLn_name'] in cnf0.keys()):
                print(cnfg['piLn_name'],"is already exists. ","*last pipeLine is",list(cnf0.keys())[-1])
                return None
            else:
                return True
    elif(mode=='mod_ds'):
        mods = []
        ls = os.listdir(config_folder)
        for i in ls:
            with open(os.path.join(config_folder,i)) as fl:
                cnf0 = json.load(fl)
                mods.append([cnf0['piLn_name'],cnf0['model_loc'],cnf0['DataSet_loc']])
        for i in mods:
            if(i[1]==cnfg['model_loc'] and i[2]==cnfg['DataSet_loc']):
                print('same combination is already used in ',i[0])
                return None
        else:
            return True
    elif(mode=="training"):
        mods = []
        ls = os.listdir(config_folder)
        for i in ls:
            with open(os.path.join(config_folder,i)) as fl:
                cnf0 = json.load(fl)
                mods.append([cnf0['piLn_name'],cnf0['optimizer_loc'],cnf0['train_batch_size'],cnf0['valid_batch_size']])
        for i in mods:
            if(i[1]==cnfg['optimizer_loc'] and i[2]==cnfg['train_batch_size'] and i[3]==cnfg['valid_batch_size']):
                print('same combination is already used in ',i[0])
                return None
        else:
            return True
    elif(mode=='all'):
        a1 = verify(cnfg,mode='name')
        a2 = verify(cnfg,mode='mod_ds')
        a3 = verify(cnfg,mode='training')
        if(a1==a2==a3==True):
            return True            

def train_new(
                    name=None,
                    model_loc=None,
                    loss_loc=None,
                    accuracy_loc=None,
                    optimizer_loc=None,
                    dataset_loc=None,
                    train_folder=None,
                    valid_folder=None,
                    train_batch_size=None,
                    valid_batch_size=None,
                    prepare=None
                ):
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
    if(train_folder is None):
        train_folder = def_conf['train_folder']
    if(valid_folder is None):
        valid_folder =def_conf['valid_folder']
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
    if(verify(dct,mode='name')):
        P =PipeLine()

        P.setup(
            name=name,
            model_loc=model_loc,
            loss_loc=loss_loc,
            accuracy_loc=accuracy_loc,
            optimizer_loc=optimizer_loc,
            dataset_loc=dataset_loc,
            train_folder=train_folder,
            valid_folder=valid_folder,
            weights_path='internal/Weights/'+name+'.pth',
            history_path='internal/Histories/'+name+'.csv',
            config_path='internal/Configs/'+name+'.json',
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            make_config=True,
            prepare=prepare
               )
        return P
        
def re_train(ppl=None,config_path=None,train_folder=None,valid_folder=None,prepare=None,num_epochs=0):
    P = PipeLine()
    if(ppl is not None):
        config_path = "internal/Configs/"+ppl+".json"
    if(num_epochs>0):
        P.setup(config_path=config_path, 
                train_folder=train_folder, 
                valid_folder=valid_folder, 
                use_config=True, prepare=True)
        P.train(num_epochs=num_epochs)
    elif(num_epochs==0):
        P.setup(config_path=config_path, 
                train_folder=train_folder, 
                valid_folder=valid_folder, 
                use_config=True, prepare=prepare)
    return P
def get_model(ppl=None,name=True,archive=False):
    if(ppl is None):
        ppl = get_pplns(mode='name',archive=archive)
        return get_model(ppl=ppl,name=name,archive=archive)
    elif(isinstance(ppl,list)):
        models = [get_model(ppl=i, name=name,archive=archive) for i in ppl]
        return models
    elif(isinstance(ppl,str)):
        file = "internal/Configs/"+ppl+".json"
        if(archive):
            file = "internal/Archived/Configs/"+ppl+".json"
        with open(file) as fl:
            cnf = json.load(fl)
            module_loc = cnf['model_loc']
            if(name==True):
                return module_loc.split('.')[-1]
            else:
                module = importlib.import_module('.'.join(module_loc.split('.')[:-1]))
                class_ = getattr(module, module_loc.split('.')[-1])                
                return class_()

def performance_plot(ppl=None,history=None,df=None,archive=False):
    if(ppl is None):
        ppl = get_pplns(mode='name',archive=archive)
    if(isinstance(ppl,list)):
        for i in ppl:
            performance_plot(ppl=i,archive=archive)
        return None
    elif(isinstance(ppl,str)):
        history = "internal/Histories/"+ppl+".csv"
        if(archive):
            history = "internal/Archived/Histories/"+ppl+".csv"
        if( not os.path.isfile(history)):
            return "Error"
    
    if( history!=None):
        df = pd.read_csv(f"{history}")
    if(df is None):
        print("It needs one of arguments history and df. df is dtaframe from the csv file and history is path to history.csv")
        return "Error"
    if(df.empty):
        print(history.split('/')[-1].split('.')[0],"Empty")
        return "Empty"
        
    fig,ax = plt.subplots(1,2)
    fig.set_size_inches(15,5)
    df.plot(x='epoch',y=['train_accuracy','val_accuracy'],ax=ax[0],title="Accuracy trade-off")
    df.plot(x='epoch',y=['train_loss','val_loss'],ax=ax[1],title="Loss trade-off")
    fig.suptitle(history.split('/')[-1].split('.')[0], fontsize=16)
    plt.show()

def multi_train(ppl = None,last_epoch=10,steps=None):
    if(isinstance(ppl, list)):
        pa = get_pplns(mode='all')
        epoch = [pa[i]['last_epoch'] for i in ppl]
    else:
        ppl = get_pplns(mode='name')
        epoch = get_pplns(mode='epoch')
    for i in range(len(ppl)):
        if(epoch[i]<last_epoch):
            print(f"{ppl[i]:=^60}")
            re_train(config_path='internal/Configs/'+ppl[i]+'.json',prepare=True,num_epochs=last_epoch-epoch[i])
    print("All trainig Done")

def setup_project(project_name="MyProject",create_root=True):
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
                "train_folder": None,
                "valid_folder": None,
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

        print(f"{'All directories are created'}")

def set_default_config(data:dict):
    cnfg = {
                "accuracy_loc": data.get("accuracy_loc"),
                "loss_loc":data.get("loss_loc"),
                "optimizer_loc":data.get("optimizer_loc"),
                "train_folder": data.get("train_folder"),
                "valid_folder": data.get("valid_folder"),
                "train_batch_size": data.get("train_batch_size"),
                "valid_batch_size": data.get("valid_batch_size")
            }
    with open(os.path.join('internal','Default_Config.json'), 'w') as file:
            json.dump(cnfg, file, indent=4)
def archive(ppl, reverse = False):
    if(isinstance(ppl,str)):
        with open("internal/Configs/"+ppl+".json") as fl:
            cnf = json.load(fl)
        shutil.move(src=cnf["weights_path"],dst=os.path.join("internal","Archived","Weights",os.path.basename(cnf["weights_path"])))
        shutil.move(src=cnf["config_path"],dst=os.path.join("internal","Archived","Configs",os.path.basename(cnf["config_path"])))
        shutil.move(src=cnf["history_path"],dst=os.path.join("internal","Archived","Histories",os.path.basename(cnf["history_path"])))
        with open("internal/config.json") as fl:
            cnf0 = json.load(fl)
        with open("internal/Archived/config.json") as fl:
            cnf1 = json.load(fl)
        with open("internal/Archived/config.json","w") as fl:
            cnf1[ppl] = cnf0[ppl]
            json.dump(cnf1, fl, indent=4)
        del cnf0[ppl]
        with open("internal/config.json","w") as fl:
            json.dump(cnf0, fl, indent=4)
        print(f"{ppl} archived successfully")
    elif(isinstance(ppl,list)):
        for i in ppl:
            archive(i, reverse=reverse)

def setup_transfer():
    os.mkdir(os.path.join('internal','Transfer'))
    os.mkdir(os.path.join('internal','Transfer','Histories'))
    os.mkdir(os.path.join('internal','Transfer','Weights'))
    os.mkdir(os.path.join('internal','Transfer','Configs'))
    with open(os.path.join('internal','Transfer','config.json'), 'w') as file:
        data = {
            }
        json.dump(data, file, indent=4)
    print("Setup completed for transfer...")

def transfer(ppl, type='export',mode="copy"): #type=export|import,mode=copy|move    
    if(isinstance(ppl,str)):
        if(type=="export"):
            config = "internal/Configs/"+ppl+".json"
            source = "internal/"
            destin = "internal/Transfer/"
        elif(type=="import"):
            config = "internal/Transfer/Configs/"+ppl+".json"
            destin = "internal/"
            source = "internal/Transfer/"
        if(mode=="move"):
            shutil.move(src=source+"Weigths/"+ppl+".pth",dst=destin+"Weigths/"+ppl+".pth")
            shutil.move(src=source+"Configs/"+ppl+".json",dst=destin+"Configs/"+ppl+".json")
            shutil.move(src=source+"Histories/"+ppl+".csv",dst=destin+"Histories/"+ppl+".csv")
        elif(mode=="copy"):
            shutil.copy2(src=source+"Weigths/"+ppl+".pth",dst=destin+"Weigths/"+ppl+".pth")
            shutil.copy2(src=source+"Configs/"+ppl+".json",dst=destin+"Configs/"+ppl+".json")
            shutil.copy2(src=source+"Histories/"+ppl+".csv",dst=destin+"Histories/"+ppl+".csv")
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
        print(f"{ppl} tranfered successfully")
    elif(isinstance(ppl,list)):
        for i in ppl:
            transfer(ppl=i,mode=mode,type=type)
