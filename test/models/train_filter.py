import pickle
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader
from test.models.models import FilterModel

import random
import yaml,sys
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

class transform(object):
    def __init__(self, nBits=[16384,2048],radius=2):
        self.nBits=nBits
        self.radius=radius
    def __call__(self,sample):
        """reactants
        product
        template_idx"""
        r,p,l = sample['reactants'], sample['product'], sample['template_idx']
        fp_p = self.fp(p,self.nBits[0])        
        fp_p_= self.fp(p,self.nBits[1])
        fp_r = sum([self.fp(_,nBits[1]) for _ in r])
        
        label = l
        return {'fp_r':torch.Tensor(fp_r),
                'fp_p':torch.Tensor(fp_p),
                'fp_p_':torch.Tensor(fp_p_)}
    
    def fp(self,smi,nBits):
        arr = np.array((1,))
        bit = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),
                                                    radius = self.radius,
                                                    nBits=nBits)
        rdkit.DataStructs.ConvertToNumpyArray(bit,arr)
        return arr

class MyDataset(Dataset):
    def __init__(self, file_name,radius=2,nBits=[16384,2048]):
        print(file_name)
        with open(file_name,'rb') as f:
            self.data = pickle.load(f)
        self.data = self.data[:int(len(self.data))]
        self.transform = transform(nBits=nBits,radius=radius)
    def __len__(self,):
        return len(self.data)
    def __getitem__(self,idx):
        return self.transform(self.data[idx])

def collate_test(batch):
    fp_rxn = []
    fp_prod = []
    label = []
    def _get(batch,idx,cri=0):
        i = batch[idx]
        j = batch[idx+cri]
        r, p_, p, label = i['fp_r'], i['fp_p_'], j['fp_p'], cri == 0
        return p_-r, p, label

    ran = random.uniform(0.2,0.8)    
    for i in range(len(batch)-1):
        if ran < random.uniform(0,1):
            rxn_,prod_,label_ = _get(batch,i,cri=0)
            fp_rxn.append(rxn_); fp_prod.append(prod_); label.append(label_)
        else:
            rxn_,prod_,label_ = _get(batch,i,cri=1)
            fp_rxn.append(rxn_); fp_prod.append(prod_); label.append(label_)
    if ran < random.uniform(0,1):
        rxn_,prod_,label_ = _get(batch,-1,cri=0)
        fp_rxn.append(rxn_); fp_prod.append(prod_); label.append(label_)
    else:
        rxn_,prod_,label_ = _get(batch,-1,cri=1)
        fp_rxn.append(rxn_); fp_prod.append(prod_); label.append(label_)
    
    out = {'fp_rxn':torch.stack(fp_rxn), 
           'fp_prod':torch.stack(fp_prod), 
           'label':torch.Tensor(label).unsqueeze(-1)}
    return out 

def read_config(config_file):
    with open(config_file,'r') as f:
        config = yaml.load(f.read(),Loader=yaml.SafeLoader)
    
    keys = ['train_option_filter', 'filter_model']
    name = 'filter'+'_'.join([str(v) for k,v in config['filter_model'].items()])

    config_ = config['train_option_filter']
    default = {}
    key1 = 'data_file num_workers batch_size device epochs lr train_from save save_every'.split()
    val = [{'train':'/home/ksh/CASP/data/USPTO_MIT/lowe_1976_2013_train.pck', 
            'val':'/home/ksh/CASP/data/USPTO_MIT/lowe_1976_2013_val.pck'},
            14,
            64,
            'cpu',
            50, 
            0.01,
            '',
            f'/home/ksh/CASP/test/models/model_save/{name}',
            10]

    for i,k in enumerate(key1):
        default[k] = val[i]
    
    default.update(config_)
    return default, config['filter_model']


config_file = sys.argv[1]
config, model_config = read_config(config_file)
data_file = config['data_file']
num_workers = config['num_workers']
batch_size = config['batch_size']
device = config['device']
epochs = config['epochs']
lr = config['lr']
train_from = config['train_from']
save = config['save']
save_every = config['save_every']

nBits = model_config['fp_dim']
hidden_dim = model_config['hidden_dim']
num_layers = model_config['num_layers']
dropout = model_config['dropout']

print('dataset preparing')
tr_ds = MyDataset(data_file['train'],radius=2,nBits=nBits)
val_ds = MyDataset(data_file['val'],radius=2,nBits=nBits)
tr_dataloader = DataLoader(tr_ds, batch_size=batch_size, 
                           shuffle=True, num_workers=num_workers, 
                           collate_fn=collate_test)
val_dataloader = DataLoader(val_ds, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers,
                            collate_fn=collate_test)
print('dataset prepared')

print('model preparing')
model = FilterModel.from_config(config_file)
epoch_from = 0
model = model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 0.1)
scheduling_step = [0,1,2,3,70]
scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones= [(i+1)*len(tr_dataloader) for i in scheduling_step],
                        gamma=0.25)
if train_from:
    print(f'model load from {train_from}')
    ckpt = torch.load(train_from,map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    #epoch_from = train_from.split('epoch')[-1]
    epoch_from = ckpt['epoch']
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
print('model prepared')


print('epoch starts!')
per_n = len(tr_dataloader)//3
st = time.time()
for epoch in range(epochs):
    epoch += epoch_from
    print(f'-------------------------{epoch}th-------------------------')
    running_loss = 0.0
    model.train()
    for i, data in enumerate(tr_dataloader):
        fp_rxn,fp_prod,label = data['fp_rxn'], data['fp_prod'], data['label']
        fp_rxn = fp_rxn.to(device); fp_prod = fp_prod.to(device); label = label.to(device); 
        optimizer.zero_grad()
        
        outputs = model([fp_prod,fp_rxn])
        loss = criterion(outputs,label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        if i % per_n == per_n -1:
            for param_group in optimizer.param_groups:
                lr_ = param_group['lr']
            print(f'{i+1}/{len(tr_dataloader)}th batch running loss : {running_loss/per_n} \t {lr_}')
            running_loss = 0.0
    
    model.eval()
    running_loss = 0.0
    for i, data in enumerate(val_dataloader):
        fp_rxn,fp_prod,label = data['fp_rxn'], data['fp_prod'], data['label']
        fp_rxn = fp_rxn.to(device); fp_prod = fp_prod.to(device); label = label.to(device); 
        outputs = model([fp_prod,fp_rxn])
        loss = criterion(outputs,label)
        running_loss += loss.item()
    print(f'{epoch} val loss : {running_loss/len(val_dataloader)}')
    if epoch % 5 == 4:
        print(f'running time : {(time.time() - st)/60:0.3f}min')
    if epoch % save_every == save_every -1:
        torch.save({'model_state_dict':model.state_dict(),
                    'epoch':epoch+1,
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict()}
                    , f'{save}.epoch{epoch+1}.pt')
