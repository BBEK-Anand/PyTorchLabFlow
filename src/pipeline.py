import torch
# import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from tqdm import tqdm
import json
import importlib
import pandas as pd
import numpy as np
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
                'weights_path': weights_path or os.path.join(root, f"{self.name}_w.pth"),
                'history_path': history_path or os.path.join(root, f"{self.name}_h.csv"),
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
                'weights_path': os.path.join(self.name, f"{self.name}_w.pth"),
                'history_path': os.path.join(self.name, f"{self.name}_h.csv"),
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


def update(config_folder='./Configs'):
    ls = os.listdir(config_folder)
    plns = []
    for i in ls:
        with open(os.path.join(config_folder,i)) as fl:
            cnf0 = json.load(fl)
            plns.append([cnf0['piLn_name'],cnf0['last']['epoch'],cnf0['best']['val_accuracy']])
    with open("./internal/config.json") as fl:
        cnf = json.load(fl)
        for i in plns:
            if(i[0] in cnf.keys()):
                if(cnf[i[0]]["last_epoch"]<i[1]):
                    print(f"last epoch updated from {cnf[i[0]]['last_epoch']} to {i[1]} for PipeLine:{i[0]}")
                    cnf[i[0]]["last_epoch"]=i[1]
                    
                if(cnf[i[0]]["best_val_accuracy"]>i[2]):
                    print(f"Validation accuracy updated from {cnf[i[0]]['best_val_accuracy']} to {i[2]} for PipeLine:{i[2]}")
                    cnf[i[0]]["best_val_accuracy"]=i[2]
            else:
                cnf[i[0]]={
                            "config":config_folder+"/"+i[0]+".csv",
                            "last_epoch":i[1],
                            "best_val_accuracy":i[2]
                        }
                print(f"new Pipeline initialized : {i[0]} with last_epoch:{i[1]} and best_val_accuracy:{i[2]}")
    with open("internal/config.json","w") as fl:
            json.dump(cnf, fl, indent=4)


def get_pplns(mode='name'): #mode = name|epoch|all
    update()
    with open("internal/config.json") as fl:
        cnf = json.load(fl)
        if(mode=='name'):
            return list(cnf.keys())
        elif(mode=='epoch'):
            ls = [cnf[i]["last_epoch"] for i in cnf.keys()]
            return ls
        elif(mode=='all'):
            return cnf
def verify(cnfg,mode='name',config_folder='internal/Configs'):   # mode = name|mod_ds|training
    if(mode=='name'):
        update(config_folder=config_folder)
        with open("internal/config.json") as fl:
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
        
def re_train(config_path=None,train_folder=None,valid_folder=None,prepare=None,num_epochs=0):
    P = PipeLine()
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




def performance_plot(history=None,df=None):
    if(df!=None and history==None):
        pass
    elif(df==None and history!=None):
        df = pd.read_csv(history)
    else:
        print("It needs one of arguments history and df. df is dtaframe from the csv file and history is path to history.csv")
        return
    
    fig,ax = plt.subplots(1,2)
    fig.set_size_inches(15,5)
    df[['train_accuracy','val_accuracy']].plot(ax=ax[0],title="Accuracy trade-off")
    df[['train_loss','val_loss']].plot(ax=ax[1],title="Loss trade-off")
    plt.show()




