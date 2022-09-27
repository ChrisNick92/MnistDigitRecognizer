import enum
from typing import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MLP(nn.Module):
    def __init__(self, in_features = 28*28, out_features = 10,
                 hidden_layers = [512,256,128]):
        super(MLP,self).__init__()
        self.structure = OrderedDict()
        for i,layer in enumerate(hidden_layers):
            if i == 0:
                self.structure["Linear " + str(i+1)] = nn.Linear(in_features=in_features,
                                                                    out_features=layer, bias=False)
                self.structure["BatchNorm1D " + str(i+1)] = nn.BatchNorm1d(layer)
                self.structure["Relu " + str(i+1)] = nn.ReLU()
                self.structure["Dropout " + str(i+1)] = nn.Dropout(p = 0.2)
            else:
                self.structure["Linear " + str(i+1)] = nn.Linear(in_features = hidden_layers[i-1],
                                                    out_features=layer, bias=False)
                self.structure["BatchNorm1D " + str(i+1)] = nn.BatchNorm1d(layer)
                self.structure["Relu " + str(i+1)] = nn.ReLU()
                self.structure["Dropout " + str(i+1)] = nn.Dropout(p = 0.2)
        self.structure["Out Linear"] = nn.Linear(in_features=hidden_layers[-1],out_features=out_features)
        
        self.linear_relu_stack = nn.Sequential(self.structure)
        
    def forward(self,x):
        return self.linear_relu_stack(x)
    

class mlp_dataset(Dataset):
    def __init__(self, X, y):
        self.X = X; self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return torch.tensor(self.X[idx,:], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.int64)

                
                