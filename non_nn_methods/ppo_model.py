from utils import *
import copy

from torch import nn
import torch
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from tqdm import tqdm

class ppo_gen_2(nn.Module):
    def __init__(self, n_player, n_card = 33):
        super().__init__()
        self.n_action = 2
        self.n_card = n_card
        self.n_player = n_player
        self.n_param_per_player = self.n_card + 1 # 33 cards + 1 number of chips
        self.n_state_param = self.n_card*2 + 3 # 33 for flipped card, 33 for remain card, 1 for chip in pot, 1 for number of cards remained, 1 for good card
        self.input_dim = self.n_player*self.n_param_per_player + self.n_state_param
        self.gen = 2

        # self.policy = nn.Sequential(
        #     nn.Linear(self.input_dim, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, self.n_action)
        #     #FIX: NEED A MASK IN HERE FOR LEGAL ACTIONS
        #     # no need to run through softmax, use logit instead for numerical stability
        #     )
        
        # self.value = nn.Sequential(
        #     nn.Linear(self.input_dim, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 1)
        #     )
        
        self.policy = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, self.n_action)
            #FIX: NEED A MASK IN HERE FOR LEGAL ACTIONS
            # no need to run through softmax, use logit instead for numerical stability
            )
        
        self.value = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 1)
            )
    
    def get_policy(self, X, legal_move_mask):
        """Mask the legal output
        legal_move_mask: boolean tensor, True for masked"""
        logit = self.policy(X)
        logit_masked = logit.masked_fill(legal_move_mask, float('-inf'))
        return logit_masked

    def forward(self, X, legal_move_mask, action = None):
        """Get value, probability
        legal_move_mask: boolean tensor
        action: tensor(1) Integer. This is the old sampled action. If none will do sampling
        """
        logit = self.get_policy(X, legal_move_mask)
        prob = Categorical(logits = logit)
        if action == None:
            action = prob.sample() # sample the action
        log_prob = prob.log_prob(action) # this will be used for surrogate loss (log(a) - log(b) = log(a/b))
        value = self.value(X)

        return action, log_prob, prob.entropy(), value # sampled action, log probability of it, its entropy,value from value network
    

class ppo_gen_3(ppo_gen_2):
    def __init__(self, n_player, n_card = 33):
        super().__init__(n_player, n_card)
        # self.n_action = 2
        # self.n_card = n_card
        # self.n_player = n_player
        self.n_param_per_player = self.n_card + 1 # 33 cards + 1 number of chips
        self.n_state_param = self.n_card*2 + 5 # 33 for flipped card, 33 for remain card, 1 for chip in pot, 1 for number of cards remained, 1 for good card
        self.input_dim = self.n_player*self.n_param_per_player + self.n_state_param
        self.gen = 3
        
        self.policy = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, self.n_action)
            )
        
        self.value = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 1)
            )