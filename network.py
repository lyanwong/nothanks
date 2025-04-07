import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class PolicyValueNet(nn.Module):
    def __init__(self, n_players, hidden_dim):
        super(PolicyValueNet, self).__init__()

        # CNN Branch for processing matrix M
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),  # Batch Norm after Conv
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  # Batch Norm after Conv
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Downsampling to a fixed size
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Batch Norm after Linear
            nn.ReLU()
        )

        # Fully-connected Branch for processing vector b
        self.fc_branch = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Batch Norm after Linear
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Batch Norm after Linear
            nn.ReLU()
        )

        # Main Branch
        self.main_branch = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),  # Batch Norm after Linear
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Batch Norm after Linear
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Batch Norm after Linear
            nn.ReLU()
        )

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in range [0, 1]
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()   # Output n-dim vector in range [0, 1]
        )

    def forward(self, M, b):
        M = self.cnn_branch(M)
        b = self.fc_branch(b)
        x = torch.cat((M, b), dim=-1)
        x = self.main_branch(x)
        p = self.policy_head(x)
        value = self.value_head(x)
        # print(f"Debug: Policy Head Output (p): {p}")  # Debugging output
        return p, value

def train_nn(model, batch_size, n_players, hidden_dim, input_state, target_policy, target_value):

    M,b = input_state
    M = torch.tensor(M, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    target_policy = torch.tensor(target_policy, dtype=torch.float32).view(-1, 1)
    target_value = torch.tensor(target_value, dtype=torch.float32).view(-1, 1)
    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    policy_loss_fn = nn.BCELoss()
    value_loss_fn = nn.MSELoss()

    # Training loop (few steps)
    for step in range(10):
        optimizer.zero_grad()

        # Forward pass
        # print(f"Debug: M = {M}")
        # print(f"Debug: b = {b}")
        # print(f"Debug: target_policy = {target_policy}")
        # print(f"Debug: target_value = {target_value}")
        pred_policy, pred_value = model(M, b)

        # Compute L2 regularization term (sum of all parameters' squared values)
        l2_lambda = 1e-4
        l2_penalty = sum(w.pow(2.0).sum() for w in model.parameters())


        # Compute loss
        # print("Debug policy:", pred_policy, target_policy)
        policy_loss = policy_loss_fn(pred_policy, target_policy)
        value_loss = value_loss_fn(pred_value, target_value)
        loss = policy_loss + value_loss + l2_lambda * l2_penalty

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
        optimizer.step()

        if step == 9:
            print(f"Loss = {loss.item():.4f}, "
                f"Policy Loss = {policy_loss.item():.4f}, "
                f"Value Loss = {value_loss.item():.4f}")
            
        # Save model
        # torch.save(model.state_dict(), 'policy_value_net.pth')
    
    return model


if __name__ == "__main__":
    batch_size = 16
    n_players = 5
    hidden_dim = 64

    # Create dummy inputs
    M = np.random.rand(batch_size, 3, 5, 33)  # Shape = (batch_size, channels, height, width)
    b = np.random.rand(batch_size, 3)  # Shape = (batch_size, vector_dim)

    # Target values for training
    target_policy = np.random.rand(batch_size, 1)  # Policy target in [0, 1]
    target_value = np.random.rand(batch_size, 1)  # Value target in [0, 1]
    
    # Create network
    model = PolicyValueNet(n_players, hidden_dim)
    model.train()
    model = train_nn(model, batch_size, n_players, hidden_dim, (M, b), target_policy, target_value)

    # Set to evaluation mode if you're using it for inference
    model.eval()  
    
    # Create sample test input
    M_test = torch.randn(1, 3, 5, 33)  # Batch size = 1, Channels = 2, Height = 5, Width = 33
    b_test = torch.randn(1, 3)  # Batch size = 1, Vector size = vector_dim

    # Forward pass
    with torch.no_grad():  # Disable gradient calculation for inference
        p, value = model(M_test, b_test)

    # Display output
    print(f"Policy output: {p.item():.4f}")  
    print(f"Value output: {value.squeeze().tolist()}")  
