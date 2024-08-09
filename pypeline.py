import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import os
import random
import pandas as pd
from tqdm import tqdm
import json
import importlib



class PipeLine:
    def __init__(self,name='Default_name',config_path=None):
        self.name = name


        self.__best_val_loss = float('inf')
        self.device = torch.device("cuda")

        self.name = name
        self.model_name = None
        self.device = None #device
        self.model = None #self.build_model()
        self.loss = None #criterion #nn.BCELoss()
        self.optimizer = None #optimizer(self.model.parameters(),lr=self.learning_rate) #optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.trainDataLoader = None
        self.validDataLoader = None
        
        self.config_path = config_path
        self.weights_path = None
        self.history_path = None
        self.model_path = None
        
        self.cnfg = None
        self.__best_val_loss = None  # make this private
        
        self.__configured = False
        self.DataSet = None
         
    def load_component(self, module_loc, **kwargs):
        module = importlib.import_module('.'.join(module_loc.split('.')[:-1]))
        class_ = getattr(module, module_loc.split('.')[-1])
        return class_(**kwargs) if kwargs else class_()
    
    def load_optimizer(self,module_loc, **kwargs):
        module = importlib.import_module('.'.join(module_loc.split('.')[:-1]))
        class_ = getattr(module, module_loc.split('.')[-1])
        return class_(self.model, **kwargs)
    def save_config(self):
        with open(self.cnfg['config_path'], "w") as out_file:
            out_file.write(json.dumps(self.cnfg,indent=4))
    def setup(self,name=None, model_loc =None,
                                accuracy_loc = None,
                                loss_loc = None,
                                optimizer_loc = None,
                                dataset_loc = None,
                                train_folder=None, train_batch_size=None, valid_folder=None, valid_batch_size=None,
                                history_path=None, weights_path=None,
                                config_path=None, use_config=False, make_config=True, prepare=False):
        
        cnfg = {
                'model_loc': model_loc,
                'DataSet_loc':dataset_loc,
                'accuracy_loc':accuracy_loc,
                'loss_loc': loss_loc,
                'optimizer_loc': optimizer_loc,
                "piLn_name": None,
                'last':{
                    'epoch':0,
                    'train_accuracy':0,
                    'train_loss':float('inf')
                },
                'best':{
                    'epoch':0,
                    'val_accuracy':0,
                    'val_loss':float('inf')
                },
                
                "valid_folder":valid_folder,
                'train_folder':train_folder,
                'valid_batch_size': valid_batch_size,
                'train_batch_size':train_batch_size,
                
                'weights_path': weights_path,
                'config_path': config_path,#config_path,#os.path.join(self.name, self.name+".json"),
                'history_path': history_path,
            }
        if(config_path!=None and use_config):
            def check_file(path):
                if(os.path.isfile(path)):
                    return path
                else:
                    print(path ,"not exists")
                    return None
            cnfg = json.load(open(check_file(config_path)))
            self.history_path = check_file(cnfg['history_path'])
            self.weights_path = check_file(cnfg['weights_path'])
            
            self.name = cnfg['piLn_name']
            self.model_name = cnfg['model_loc'].split('.')[-1]
            self.__best_val_loss = cnfg['best']['val_loss']

            cnfg['weights_path'] = self.weights_path
            cnfg['history_path'] = self.history_path

            self.cnfg = cnfg
        elif(config_path!=None and make_config):
            if((name==None and self.name==None) or model_loc==None  or accuracy_loc==None or loss_loc==None or optimizer_loc==None):
                print("It needs correct parameters: PipeLine.name, model_loc, accuracy_loc, loss_loc, optimizer_loc")
                return
            root = os.path.split(config_path)[0]
            root = './' if root == '' else root
            print(config_path,'is not valid path!',root, 'does not exist') if not os.path.isdir(root) else None
            cnfg['piLn_name'] = name if name!=None else self.name 
            cnfg['DataSet_loc'] = dataset_loc
            cnfg['model_loc'] = model_loc
            cnfg['accuracy_loc'] = accuracy_loc 
            cnfg['loss_loc']= loss_loc
            cnfg['optimizer_loc'] = optimizer_loc
            cnfg['config_path'] = config_path

            
            self.weights_path = root+'/'+self.name+"_w.pth" if self.weights_path == None else self.weights_path
            self.history_path = root+'/'+self.name+"_h.csv" if self.history_path == None else self.history_path
            
            self.model = self.load_component(model_loc) if model_loc!=None else self.model
            self.DataSet = self.load_component(dataset_loc) if dataset_loc!=None else self.DataSet
            self.accuracy = self.load_component(accuracy_loc) if accuracy_loc!=None else self.accuracy
            self.loss = self.load_component(loss_loc) if loss_loc!=None else self.loss
            
            self.optimizer = self.load_optimizer(optimizer_loc) if optimizer_loc!=None else self.optimizer
            self.cnfg = cnfg
            self.config_path = cnfg['config_path']
            pd.DataFrame(columns=["epoch","train_accuracy","train_loss","val_accuracy","val_loss"]).to_csv(self.history_path,index=False)
            self.save_config()
            print('configuration file is saved at ',cnfg["config_path"])
            print('history will be saved at',cnfg["history_path"]) if history_path!= None else None
            print('weights will be saved at',cnfg["weights_path"]) if weights_path!=None else None

        elif(config_path==None and make_config):
            if(self.name==None or model_loc==None or accuracy_loc==None or loss_loc==None or optimizer_loc==None):
                print("It needs correct parameters: PipeLine.name, model_loc, accuracy_loc, loss_loc, optimizer_loc")
                return
            if(not os.path.isdir(self.name)):
                os.mkdir(self.name)

            self.weights_path = self.name+'/'+self.name+"_w.pth" if self.weights_path == None else self.weights_path
            self.history_path = self.name+'/'+self.name+"_h.csv" if self.history_path == None else self.history_path
            
            cnfg['piLn_name'] = self.name
            cnfg['DataSet_loc'] = dataset_loc
            cnfg['model_loc'] = model_loc
            cnfg['accuracy_loc'] = accuracy_loc
            cnfg['loss_loc']= loss_loc
            cnfg['optimizer_loc']=optimizer_loc
            cnfg['config_path'] = self.name+'/'+self.name+".json"

            self.__best_val_loss = cnfg['best']['val_loss']
            cnfg['weights_path'] = self.weights_path
            cnfg['history_path'] =self.history_path

            self.cnfg = cnfg
            self.config_path = cnfg['config_path']
            self.save_config()
            pd.DataFrame(columns=["epoch","train_accuracy","train_loss","val_accuracy","val_loss"]).to_csv(self.history_path,index=False)
            print('configuration file is saved at ',cnfg["config_path"])
            print('history will be saved at',cnfg["history_path"]) if history_path!= None else None
            print('weights will be saved at',cnfg["weights_path"]) if weights_path!=None else None
        
        self.model = self.load_component(self.cnfg['model_loc'])
        self.loss = self.load_component(self.cnfg['loss_loc'])
        self.optimizer = self.load_optimizer(self.cnfg['optimizer_loc'])
        self.accuracy = self.load_component(self.cnfg['accuracy_loc'])

        if(prepare):
            self.prepare_data()
            

    def prepare_data(self,dataset_loc=None, train_folder=None, train_batch_size=None, valid_folder=None, valid_batch_size=None):
        if(self.cnfg['train_folder']==None and valid_folder==None):
            print('train_folder is not found')
            return None
        if(self.cnfg['valid_folder']==None and train_folder==None):
            print('valid_folder is not found')
            return None
        if(self.cnfg['train_batch_size']==None and train_batch_size==None):
            print('train_batch_size is not found')
            return None
        if(self.cnfg['valid_batch_size']==None and valid_batch_size==None):
            print('valid_batch_size is not found')
            return None
        self.cnfg['valid_batch_size'] = valid_batch_size if valid_batch_size != None else self.cnfg['valid_batch_size']
        self.cnfg['valid_folder'] = valid_folder if valid_folder != None else self.cnfg['valid_folder']
        self.cnfg['train_batch_size'] = train_batch_size if train_batch_size != None else self.cnfg['train_batch_size']
        self.cnfg['train_folder'] = train_folder if train_folder != None else self.cnfg['train_folder']
        self.save_config()
        self.DataSet = self.load_component(dataset_loc) if dataset_loc != None else self.DataSet
        class BaseDataSet(Dataset):
            def __init__(self, folder, batch_size):
                self.folder = folder
                self.file_list = [f for f in os.listdir(folder)]
                self.crop_duration = 0.47
                self.sr = 22050
                self.crop_size = int(self.crop_duration * self.sr)
                
            def __len__(self):
                return len(self.file_list)
            
            def __getitem__(self, idx):
                file_path = self.file_list[idx]

                waveform, sr = librosa.load(os.path.join(self.folder,file_path))
                # Convert to tensor and add channel dimension
                waveform = torch.tensor(waveform).unsqueeze(0)
                
                cropped_waveform = self.random_crop(waveform, self.crop_size)
                
                label = int(os.path.basename(file_path)[-5])  # Assume label is the first character after the last underscore
                
                return cropped_waveform, label
            
            def random_crop(self, waveform, crop_size, ):
                num_samples = waveform.size(1)
                if num_samples <= crop_size:
                    padding = crop_size - num_samples
                    cropped_waveform = torch.nn.functional.pad(waveform, (0, padding))
                else:
#   
                    start = random.randint(0, num_samples - crop_size)
                    cropped_waveform = waveform[:, start:start + crop_size]
                return cropped_waveform 
   
        self.DataSet = BaseDataSet if self.DataSet == None else self.DataSet
        train_dataset = self.DataSet(self.cnfg['train_folder'] , self.cnfg['train_batch_size'])
        self.trainDataLoader = DataLoader(train_dataset, batch_size=self.cnfg['train_batch_size'], shuffle=True)
        
        valid_dataset = self.DataSet(self.cnfg['valid_folder'], self.cnfg['valid_batch_size'])
        self.validDataLoader = DataLoader(valid_dataset, batch_size=self.cnfg['valid_batch_size'], shuffle=True)
        self.__configured = True

    def update(self,data):
        self.cnfg['last'] = {'epoch':data["epoch"],"train_accuracy":data['train_accuracy'],"train_loss":data['train_loss']}
        if data['val_loss'] < self.__best_val_loss:
            self.__best_val_loss = data['val_loss']
            torch.save(self.model.state_dict(), self.weights_path)
            self.cnfg['best'] = {'epoch':data["epoch"],"val_accuracy":data['val_accuracy'],"val_loss":data['val_loss']}
            print(f"Best model saved at epoch {data['epoch']} with validation loss: {data['val_loss']}")
            
        pd.DataFrame([data]).to_csv(self.history_path, mode='a',header=False,index=False)        
        self.save_config()

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
                inputs = data[:-1]
                labels = data[-1]
                inputs = [i.to(self.device) for i in inputs]
                labels = labels.to(self.device)
                labels = labels.float().unsqueeze(1)
                self.optimizer.zero_grad()
                outputs = self.model(inputs[0])
                loss = self.loss(outputs, labels)
                
                
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                running_accuracy += accuracy_metric(torch.sigmoid(outputs), labels.int()).item()
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
                
                inputs = data[:-1]
                labels = data[-1]
                inputs = [i.to(self.device) for i in inputs]
                labels = labels.to(self.device)                
                labels = labels.float().unsqueeze(1) #only for base audio file

                outputs = self.model(inputs[0]) # 0 bcz input only has base audio array
                loss = self.loss(outputs, labels)
                running_loss += loss.item()
                predictions = torch.sigmoid(outputs)
                correct += self.accuracy(predictions, labels.int()).item()
                total += labels.size(0)
            accuracy = correct / len(self.validDataLoader)
            avg_loss = running_loss / len(self.validDataLoader)
            return avg_loss, accuracy