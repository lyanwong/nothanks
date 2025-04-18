from torch import nn
import torch
from torch.optim import Adam

class nothank_model(nn.Module):
    def __init__(self, n_player):
        super().__init__()
        self.n_action = 2
        self.n_player = n_player
        self.n_param_per_player = 34 # 33 cards + 1 number of chips
        self.n_state_param = 37 # 3 for whose turn, 1 for total chips, 33 for turned card
        self.input_dim = self.n_player*self.n_param_per_player + self.n_state_param
                
        self.linear_1 = nn.Linear(self.input_dim, 256)
        self.linear_2 = nn.Linear(256, 128)
        self.linear_3 = nn.Linear(128, 64)

        self.relu = nn.ReLU()

        self.policy_output = nn.Linear(64, 1)
        self.value_output = nn.Linear(64, self.n_player)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid() # CrossEntropyLoss does the softmax

    
    def forward(self, X):
        X = self.linear_1(X)
        X = self.relu(X)
        X = self.linear_2(X)
        X = self.relu(X)
        X = self.linear_3(X)
        X = self.relu(X)

        policy_raw = self.policy_output(X)
        value_raw = self.value_output(X)

        policy = self.sigmoid(policy_raw)
        value = self.tanh(value_raw)
        
        return policy, value
    