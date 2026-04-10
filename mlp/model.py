"""
mlp/model.py - Shared MLP architecture used for routing.
"""

import torch.nn as nn

HIDDEN_DIM = 256


class MLP(nn.Module):
    """Two-layer MLP for binary routing classification.

    Input:  SAE sparse feature vector (d_in = d_sae)
    Output: scalar logit (positive → route to weak, negative → route to strong)
    """

    def __init__(self, d_in: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
