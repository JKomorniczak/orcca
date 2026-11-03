import torch.nn as nn


def EncDec_mlp(n_features, n_embed=10, hidden=(100, 10)):
    """Dense autoencoder; `hidden` are encoder widths, the decoder mirrors them."""
    widths = [n_features, *hidden, n_embed]
    layers = []
    for a, b in zip(widths, widths[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    for a, b in zip(widths[::-1], widths[::-1][1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers.pop()
    return nn.Sequential(*layers)
