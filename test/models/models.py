import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

class KeywordMissingError(Exception):
    pass


class ExpansionModel(nn.Module):
    
    def __init__(self,
                 fp_dim=16384,
                 hidden_dim=1024, 
                 num_templates=121, 
                 num_layers=5,
                 dropout=0.1
                 ):

        super(ExpansionModel,self).__init__()
        self.fp_dim = fp_dim
        self.num_templates = num_templates
        self.torchmodel = True
        
        self.embedding = DenseReLU(inp_dim=fp_dim, 
                                   out_dim=hidden_dim, 
                                   dropout=0.3)
        self.layers = [HighwayReLU(dim=hidden_dim,dropout=dropout) 
                                                    for _ in range(num_layers)]
        self.layers = nn.ModuleList(self.layers)
        self.last_layer = nn.Linear(hidden_dim, num_templates)
    
    @classmethod
    def from_config(cls,config_file):
        with open(config_file,'r') as f:
            config = yaml.load(f.read(), Loader=yaml.SafeLoader)
        config = config['expansion_model']
        keys = ['fp_dim','hidden_dim','num_templates','num_layers','dropout']
        if not set(config.keys()).issubset(set(keys)):
            msg = f"config keys : {[i for i in config.keys()]}\n" +\
                  f"model __init__ needs : {keys}"
            raise KeywordMissingError(msg)
        return cls(**config)

    def __len__(self):
        return self.fp_dim
    
    def predict(self,x):
        return self.forward(x)

    def forward(self,x):
        h = self.embedding(x)

        for layer in self.layers:
            h = layer(h)
        pred = self.last_layer(h)
        return pred

#class ExpansionModel(nn.Module):
#    def __init__(self,
#                 inp_dim=16384,
#                 out_dim=121, 
#                 hidden_dim=0, 
#                 num_layers=5,
#                 ):
#
#        super(ExpansionModel,self).__init__()
#        self._inp_dim = inp_dim
#        self.output_size = out_dim
#        self.torchmodel = True
#
#        self.first_layer = ReLUDenseLayer(inp_dim=inp_dim,
#                                            hidden_dim=hidden_dim,
#                                            out_dim=1024)
#        self.linears = [ReLUDenseLayer(inp_dim=1024,
#                                       hidden_dim=hidden_dim,
#                                       out_dim=300)] +\
#                       [ReLUDenseLayer(inp_dim=300, 
#                                       hidden_dim=hidden_dim,
#                                       out_dim=300) 
#                               for i in range(num_layers)]
#        self.linears = nn.ModuleList(self.linears)
#        self.last_layer = ReLUDenseLayer(inp_dim=300,
#                                         hidden_dim=hidden_dim,
#                                         out_dim=out_dim)
#
#    def __len__(self):
#        return self._inp_dim
#    
#    def predict(self,x):
#        return self.forward(x)
#
#    def forward(self,x):
#        pred = self.first_layer(x)
#        for l in self.linears:
#            pred = l(pred)
#        pred = self.last_layer(pred)
#        return pred

class FilterModel(nn.Module):
    def __init__(self,
                 fp_dim=[16384,2048],
                 hidden_dim=1024, 
                 num_layers=5,
                 dropout=0.1,
                 ):

        super(FilterModel,self).__init__()
        self.fp_dim = fp_dim
        self.torchmodel = True

        self.product_embedding = DenseReLU(inp_dim=self.fp_dim[0],
                                            out_dim=hidden_dim,
                                            dropout=0.3)
        self.reaction_embedding = DenseReLU(inp_dim=self.fp_dim[1],
                                            out_dim=hidden_dim,
                                            dropout=dropout)
        self.layers = [HighwayReLU(dim=hidden_dim,dropout=dropout) 
                                        for _ in range(num_layers)]
        self.layers = nn.ModuleList(self.layers)
        
        self.cosine = nn.CosineSimilarity()
        self.post_cosine = nn.Sequential(nn.Linear(1,hidden_dim),
                                         nn.ELU(),
                                         nn.Linear(hidden_dim,1))
        self.sigmoid = nn.Sigmoid()

    @classmethod
    def from_config(cls,config_file):
        with open(config_file,'r') as f:
            config = yaml.load(f.read(), Loader=yaml.SafeLoader)
        config = config['filter_model']
        keys = ['fp_dim','hidden_dim','num_layers','dropout']
        if not set(config.keys()).issubset(set(keys)):
            msg = f"config keys : {[i for i in config.keys()]}\n" +\
                  f"model __init__ needs : {keys}"
            raise KeywordMissingError(msg)
        return cls(**config)

    def __len__(self):
        return tuple(self.fp_dim)
    
    def predict(self,fp_list):
        return self.forward(fp_list)

    def forward(self,fp_list):
        fp_prod, fp_rxn = fp_list
        h_rxn = self.reaction_embedding(fp_rxn)
        h_prod = self.product_embedding(fp_prod)
        for layer in self.layers:
            h_prod = layer(h_prod)
        cos = self.cosine(h_rxn,h_prod).unsqueeze(1)
        pred = self.sigmoid(self.post_cosine(cos))
        return pred

#class FilterModel(nn.Module):
#    def __init__(self,
#                 inp_dim=16384,
#                 out_dim=1024, 
#                 hidden_dim=0, 
#                 num_layers=5,
#                 ):
#
#        super(FilterModel,self).__init__()
#        self._inp_dim = inp_dim
#        self.torchmodel = True
#
#        self._gen_product_embedding(inp_dim, out_dim, hidden_dim, num_layers)
#        self.reaction_embedding = ReLUDenseLayer(inp_dim=inp_dim,
#                                                hidden_dim=hidden_dim,
#                                                out_dim=out_dim)
#        self.cosine = torch.nn.CosineSimilarity()
#        self.sigmoid = nn.Sigmoid()
#
#    def __len__(self):
#        return self._inp_dim
#    
#    def _gen_product_embedding(self,
#                               inp_dim,
#                               out_dim, 
#                               hidden_dim, 
#                               num_layers,
#                               ):
#
#        first_layer = ReLUDenseLayer(inp_dim=inp_dim,
#                                            hidden_dim=hidden_dim,
#                                            out_dim=1024)
#        linears = [ReLUDenseLayer(inp_dim=1024,
#                                  hidden_dim=hidden_dim,
#                                  out_dim=1024)] +\
#                  [ReLUDenseLayer(inp_dim=1024, 
#                                  hidden_dim=hidden_dim,
#                                  out_dim=1024) 
#                          for i in range(num_layers)]
#        last_layer = ReLUDenseLayer(inp_dim=1024,
#                                    hidden_dim=hidden_dim,
#                                    out_dim=out_dim)
#        self.product_embedding = nn.ModuleList([first_layer]+\
#                                               linears+\
#                                               [last_layer])
#        return
#
#    def predict(self,fp_list):
#        return self.forward(fp_list)
#
#    def forward(self,fp_list):
#        fp_prod, fp_rxn = fp_list
#        rxn = self.reaction_embedding(fp_rxn)
#        for l in self.product_embedding:
#            fp_prod = l(fp_prod)
#        #prod = self.product_embedding(fp_prod)
#        prod = fp_prod
#        pred = self.cosine(rxn,prod)
#        return self.sigmoid(pred)

class HighwayReLU(nn.Module):
    def __init__(self,
                 dim = 1024,
                 dropout = 0.1,):
        super(HighwayReLU,self).__init__()
        self.layer = nn.Sequential(nn.Linear(dim,dim), nn.ELU())
        self.gate = nn.Sequential(nn.Linear(dim,1), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        h = self.layer(x)
        gate = self.gate(x)
        out = h*gate + x*(1-gate)
        return self.dropout(out)

class DenseReLU(nn.Module):
    def __init__(self,
                 inp_dim = 512,
                 out_dim = 512,
                 dropout = 0.1):
        super(DenseReLU,self).__init__()
        self.layer = nn.Sequential(nn.Linear(inp_dim,out_dim),nn.ELU())
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        h = self.layer(x)
        return self.dropout(h)

class ReLUDenseLayer(nn.Module):
    def __init__(self, 
            inp_dim=300, hidden_dim=0, 
            out_dim=300):
        super(ReLUDenseLayer,self).__init__()
        self.inp_dim = inp_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        if hidden_dim:
            self.l1 = nn.Sequential(nn.Linear(inp_dim,hidden_dim),nn.ReLU())
            self.l2 = nn.Sequential(nn.Linear(hidden_dim,out_dim),nn.ReLU())
            #self.bn = nn.BatchNorm1d(out_dim)
        else:
            self.l1 = nn.Sequential(nn.Linear(inp_dim,out_dim),nn.ReLU())
            #self.bn = nn.BatchNorm1d(out_dim)

    def forward(self,x):
        if self.hidden_dim:
            #return self.bn(self.l2(self.l1(x)))
            return self.l2(self.l1(x))
        else:
            #return self.bn(self.l1(x))
            return self.l1(x)


