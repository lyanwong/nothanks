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
        
class ppo_gen_4(ppo_gen_2):
    """Collapse"""
    def __init__(self, n_player, n_card = 33):
        super().__init__(n_player, n_card)

        self.n_param_per_player = self.n_card + 1 # 33 cards + 1 number of chips
        self.n_state_param = self.n_card*2 + 5 # 33 for flipped card, 33 for remain card, 1 for chip in pot, 1 for number of cards remained, 1 for good card self, 1 for good card other, 1 for chipinpot/current
        self.input_dim = 2*self.n_param_per_player + self.n_state_param # 2 because 1 for self, 1 for opponents
        self.gen = 4
        
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
        

# Gen 5 - CNN
class Flatten_custom(nn.Module):
    def __init__(self, start_dim_batch: int = 1, start_dim_unbatch: int = 0, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim_batch = start_dim_batch
        self.start_dim_unbatch = start_dim_unbatch
        self.end_dim = end_dim
    def forward(self, x):
        if len(x.shape) == 4:
            return x.flatten(self.start_dim_batch, self.end_dim)
        else:
            return x.flatten(self.start_dim_unbatch, self.end_dim)

flatten_custom  = Flatten_custom()

# Gen 5 - CNN

class ppo_gen_5(ppo_gen_2):
    def __init__(self, n_player, n_card = 33):
        super().__init__(n_player, n_card)
        self.in_channel = n_player + 2 # 1 for flipped card, 1 for remaining cards
        self.out_channel = 16
        self.n_state_param = n_player + 5 # 1 for chip in pot, 1 for number of cards remained, 1 for good card self, 1 for good card other, 1 for chipinpot/current
        self.flatten_dimension = 512 # hard code
        self.gen = 5
        self.flatten_custom = Flatten_custom()
        self.policy = nn.Sequential(
            nn.Linear(173, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, self.n_action)
            #FIX: NEED A MASK IN HERE FOR LEGAL ACTIONS
            # no need to run through softmax, use logit instead for numerical stability
            )
        
        self.value = nn.Sequential(
            nn.Linear(173, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 1)
            )
        
        self.cnn_policy = nn.Sequential(
            nn.Conv2d(
            in_channels = self.in_channel,
            out_channels = self.out_channel,
            kernel_size = (1, 3)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            Flatten_custom()
        )

        self.linear_state_policy = nn.Sequential(
            nn.Linear(self.n_state_param, 16),
            nn.LeakyReLU(negative_slope=0.01),
        )
        
        self.ff_policy = nn.Sequential(
            nn.Linear(self.flatten_dimension, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, self.n_action)
            )
        
        self.cnn_value = nn.Sequential(
            nn.Conv2d(
            in_channels = self.in_channel,
            out_channels = self.out_channel,
            kernel_size = (1, 3)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            Flatten_custom()
        )

        self.linear_state_value = nn.Linear(self.n_state_param, 16)

        self.ff_value = nn.Sequential(
            nn.Linear(self.flatten_dimension, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 1)
            )
        

    def forward_concat(self, x_card, x_state, cnn_layer, linear_layer):
        x_card_flat = cnn_layer(x_card)
        x_state_flat = linear_layer(x_state)
        if len(x_card_flat.shape) == 1:
            dim = 0
        else:
            dim = 1
        return torch.cat([x_card_flat, x_state_flat], dim = dim)

    def get_policy(self, x_card, x_state, legal_move_mask):
        """Mask the legal output
        legal_move_mask: boolean tensor, True for masked"""
        policy_concat = self.forward_concat(x_card, x_state, self.cnn_policy, self.linear_state_policy) # flattened + concat
        logit = self.ff_policy(policy_concat)
        logit_masked = logit.masked_fill(legal_move_mask, float('-inf'))
        return logit_masked

    def get_value(self, x_card, x_state):
        value_concat = self.forward_concat(x_card, x_state, self.cnn_value, self.linear_state_value) # flattened + concat
        value = self.ff_value(value_concat)
        return value

    def forward(self, x_card, x_state, legal_move_mask, action = None):
        """Get value, probability
        legal_move_mask: boolean tensor
        action: tensor(1) Integer. This is the old sampled action. If none will do sampling
        """
        logit = self.get_policy(x_card, x_state, legal_move_mask)
        prob = Categorical(logits = logit)
        if action == None:
            action = prob.sample() # sample the action
        log_prob = prob.log_prob(action) # this will be used for surrogate loss (log(a) - log(b) = log(a/b))

        value = self.get_value(x_card, x_state)
        
        return action, log_prob, prob.entropy(), value # sampled action, log probability of it, its entropy,value from value network
        
