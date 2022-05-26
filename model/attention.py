import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import itertools

class AttentionLayer(nn.Module):
    """ Attention Layer"""
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio, self.training),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        out = self.linear(x)
        return out
        # weight = F.softmax(out.view(1, -1), dim=1)
        # return weight

class SelfAttentionLayer_tuser(nn.Module):
    """ Self attention Layer"""
    def __init__(self, embedding_dim, drop_ratio=0.1):
        super(SelfAttentionLayer_tuser, self).__init__()
        self.embedding_dim = embedding_dim

        self.query_linear = nn.Sequential()
        self.query_linear.add_module('fc_ise1_query', nn.Linear(embedding_dim, embedding_dim//2))
        self.query_linear.add_module('ac_ise1_query', nn.ReLU(True))
        self.query_linear.add_module('dropout_query', nn.Dropout(drop_ratio))

        self.key_linear = nn.Sequential()
        self.key_linear.add_module('fc_ise1_key', nn.Linear(embedding_dim, embedding_dim//2))
        self.key_linear.add_module('ac_ise1_key', nn.ReLU(True))
        self.key_linear.add_module('dropout_key', nn.Dropout(drop_ratio))

        self.value_linear = nn.Sequential()
        self.value_linear.add_module('fc_ise1_value', nn.Linear(embedding_dim, embedding_dim))
        self.value_linear.add_module('ac_ise1_value', nn.ReLU(True))
        self.value_linear.add_module('value_query', nn.Dropout(drop_ratio))

    def forward(self, x):
        """ 
            Inputs :
                x  : a group members' embeddings cat item embeddings [B, N, 2C]
            Returns :
                out : out : self attention value + input feature         
        """
        proj_query = self.query_linear(x) # [B, N , C//2]
        proj_key = self.key_linear(x) # [B, N , C//2]
        proj_value = self.value_linear(x) # [B, N , C]
        
        return proj_query, proj_key, proj_value