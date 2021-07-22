import torch
import torch.nn as nn
import torch.nn.functional as F
from test.models.models import FilterModel,ExpansionModel
import yaml

def load_torch_model(config_file, model_type, device='cpu'):
   
    print(config_file)
    print('------------------------------------')
    if 'expansion' in model_type or 'Expansion' in model_type:
        model = ExpansionModel.from_config(config_file)
        with open(config_file,'r') as f:
            load_file = yaml.load(f.read(), Loader=yaml.SafeLoader)
            load_file = load_file['load_expansion_model']
        ckpt = torch.load(load_file, map_location = device)
        model.load_state_dict(ckpt['model_state_dict'])

    elif 'filter' in model_type or 'Filter' in model_type:
        model = FilterModel.from_config(config_file)
        with open(config_file,'r') as f:
            load_file = yaml.load(f.read(), Loader=yaml.SafeLoader)
            load_file = load_file['load_filter_model']
        ckpt = torch.load(load_file, map_location = device)
        model.load_state_dict(ckpt['model_state_dict'])
        
    return model
