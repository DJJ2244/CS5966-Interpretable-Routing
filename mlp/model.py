import torch.nn as nn

HIDDEN_DIM = 256


class MLP(nn.Module):
    """A Simple two layer MLP that outputs the logit
    which then is used to route to either weak or strong models.

    Input:  The models sparse feature vectors per query (d_in = d_sae)
    Output: logit (positive → weak, negative → strong)
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
