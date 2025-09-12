import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class ProteinTmDataset(Dataset):
    def __init__(self, embeddings, tm_values):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(tm_values, dtype=torch.float32).unsqueeze(1)  # regression target

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class thermalMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=[256, 128], activation=nn.ReLU, dropout=0.0):
        super(thermalMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())   # flexible activation
            if dropout > 0:
                layers.append(nn.Dropout(dropout))  # optional dropout
            prev_dim = h
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))  # regression output
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)