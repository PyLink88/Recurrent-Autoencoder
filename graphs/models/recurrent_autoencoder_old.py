"""
Recurrent Autoencoder pytorch implementation
"""

import torch
import torch.nn as nn
import json
from easydict import EasyDict as edict
import numpy as np
import random


class RecurrentEncoder(nn.Module):
    def __init__(self, n_features, latent_dim):
        super().__init__()

        self.rec_enc1 = nn.GRU(n_features, latent_dim, batch_first=True)

    def forward(self, x):
        _, h_n = self.rec_enc1(x)

        return h_n


class RecurrentDecoder(nn.Module):
    def __init__(self, latent_dim, n_features):
        super().__init__()

        self.rec_dec1 = nn.GRUCell(n_features, latent_dim)
        self.dense_dec1 = nn.Linear(latent_dim, n_features)

    def forward(self, h_0, seq_len):
        # Initialize output
        x = torch.tensor([])

        # Final hidden from encoder
        h_i = h_0.squeeze()

        # Reconstruct first element with encoder output
        x_i = self.dense_dec1(h_i)
        # x = torch.cat([x, x_i])

        # Reconstruct remaining elements
        for i in range(0, seq_len):
            h_i = self.rec_dec1(x_i, h_i)
            x_i = self.dense_dec1(h_i)
            x = torch.cat([x, x_i], axis=1)

        return x


class RecurrentAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = self.config.latent_dim
        self.n_features = self.config.n_features
        self.encoder = RecurrentEncoder(self.n_features, self.latent_dim)
        self.decoder = RecurrentDecoder(self.latent_dim, self.n_features)

    def forward(self, x):
        seq_len = x.shape[1]
        h_n = self.encoder(x)
        out = self.decoder(h_n, seq_len)
        # NOTE: you need to squeeze and flipping
        return torch.flip(out.unsqueeze(2), [1])
        # return out.unsqueeze(2)


if __name__ == '__main__':
    # Configuration
    config = {}
    config['n_features'] = 1
    config['latent_dim'] = 4
    config = edict(config)

    # Random data
    X = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.float32).unsqueeze(
        2)

    # Encoder
    model1 = RecurrentEncoder(config.n_features, config.latent_dim)
    out1 = model1(X)

    # Decoder
    model2 = RecurrentDecoder(config.latent_dim, config.n_features)
    seq_len = X.shape[1]
    out2 = model2(out1, seq_len)

    # All together
    model = RecurrentAE(config)
    out = model(X)

    loss = nn.L1Loss(reduce='mean')
    print(loss(X, out))
