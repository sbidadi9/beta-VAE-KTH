import torch
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm
from nns.embedding import SineActivation, CosineActivation, PosEncoding
import torch.nn as nn 

class MLP(nn.Module):

    def __init__(self, nmode, hidden_size, num_layer):
        """
        Args:
            latent_dim (int): dimension of latent variable z
            hidden_dim (int): number of neurons per hidden layer
            num_layer  (int): number of hidden layers
        """
        super().__init__()

        assert num_layer >= 1, "num_layer must be >= 1"

        layers = []

        # ---- input layer ----
        layers.append(nn.Linear(nmode, hidden_size))
        layers.append(nn.ReLU())

        # ---- hidden layers ----
        for _ in range(num_layer - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # ---- output layer ----
        layers.append(nn.Linear(hidden_size, nmode))

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        """
        Args:
            z: Tensor of shape (batch_size, latent_dim)
               or (batch_size, seqLen, latent_dim) with seqLen=1
        Returns:
            z_next: Tensor of same shape as input latent_dim
        """
        
        z = z[:, -1, :]
        z_next = self.net(z)
        return z_next.unsqueeze(1)
