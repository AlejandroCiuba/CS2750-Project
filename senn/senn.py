# SENN class
from torch import nn

import torch


class SENN(nn.Module):

    def __init__(self, concept_dim: int, emb_size: int,
                 hidden: int = 64, pos_theta: bool = True):

        super().__init__()
        self.K = concept_dim
        self.emb_size = emb_size
        self.hidden_size = hidden
        self.pos_theta = pos_theta

        self.theta_net = nn.Sequential(
            nn.Linear(self.emb_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.K),
        )

        self.bias = nn.Parameter(torch.zeros(1))

    def theta(self, emb):
        th = self.theta_net(emb)
        if self.pos_theta:
            return torch.softmax(th)    
        return th

    def forward(self, cvec, emb):

        th = self.theta(emb)
        out = cvec * th
        logit = out.sum(dim=1, keepdim=True) + self.bias

        return logit, cvec, th