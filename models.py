import enum
from typing import OrderedDict
import torch
import torch.nn as nn
import numpy as np
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

                

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size = 5, stride = 1, padding = 0, bias = True):
        super(conv_block,self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels =  out_channels,
                      kernel_size = kernel_size, stride = stride,
                      padding = padding, bias = bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )
        
    def forward(self,x): 
        return self.conv(x)

class CNN(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(CNN,self).__init__()
        self.backbone = nn.Sequential(
            conv_block(in_channels=in_channels,
                       out_channels=6, bias = False),
            nn.AdaptiveMaxPool2d(output_size = 12),
            conv_block(in_channels = 6, out_channels = 16,
                       bias = False),
            nn.AdaptiveMaxPool2d(output_size = 4),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4*4*16, out_features=256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 128, bias = False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 10)
        )
    
    def forward(self,x):
        x = self.backbone(x)
        return self.classifier(x)
    

class cnn_dataset(Dataset):
    def __init__(self,X,y, transforms = None):
        self.X = np.expand_dims(X, axis = 0); self.y = y
        self.transforms = transforms
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        if self.transforms is not None:
            img = np.resize(self.X[:,idx,:], (28,28,1))
            augmentations = self.transforms
            features = augmentations(image = img)["image"]
            features = torch.tensor(np.transpose(features, (2,0,1)), dtype=torch.float32)
            labels = torch.tensor(self.y[idx], dtype = torch.int64)
            return features, labels
        else:
            features = torch.tensor(self.X[:,idx,:], dtype = torch.float32).view(-1,28,28)
            label = torch.tensor(self.y[idx], dtype = torch.int64)
            return features, label