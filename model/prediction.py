import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import itertools


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)

    def forward(self, x):
        out = self.linear(x)
        return out

class PredDomainLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(PredDomainLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, input):
        out = self.linear(input)
        pred_label = torch.sigmoid(out)
       
        return pred_label