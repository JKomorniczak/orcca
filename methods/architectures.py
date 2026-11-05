import torch.nn as nn


def EncDec_mlp(n_features, n_embed=10, hidden=(100, 10)):
    """Dense autoencoder; `hidden` are encoder widths, the decoder mirrors them."""
    encoder_widths = [n_features] + list(hidden) + [n_embed]
    decoder_widths = encoder_widths[::-1]

    layers = []
    for i in range(len(encoder_widths) - 1):
        layers.append(nn.Linear(encoder_widths[i], encoder_widths[i + 1]))
        layers.append(nn.ReLU())
    for i in range(len(decoder_widths) - 1):
        layers.append(nn.Linear(decoder_widths[i], decoder_widths[i + 1]))
        if i < len(decoder_widths) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)
