"""CIFAR-10 embeddings: ImageNet ResNet-18 without the head (512-d), inputs upscaled to 224, L2-normalized, cached in CACHE. """

import pathlib

import numpy as np
import torch
import torchvision
from torchvision.models import ResNet18_Weights, resnet18

CACHE = pathlib.Path("files/embeddings/cifar_resnet18_l2.npz")


def extract(X, batch_size=256):
    """Images (N, 3, 32, 32) in [0, 1] -> ResNet-18 features (N, 512)."""
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.eval().to(device)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)   # ImageNet
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

    out = []
    with torch.no_grad():
        for batch in torch.as_tensor(X).split(batch_size):
            batch = torch.nn.functional.interpolate(batch.to(device), size=224, mode="bilinear", align_corners=False)
            out.append(model((batch - mean) / std).float().cpu().numpy())
    return np.concatenate(out)


def load_embeddings():
    """CIFAR-10 embedding pool (X, y), L2-normalized, from cache if available."""
    if CACHE.exists():
        with np.load(CACHE) as f:
            return f["X"], f["y"]

    data = torchvision.datasets.CIFAR10("./files/", train=True, download=True)
    X = (data.data.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    y = np.asarray(data.targets, dtype=int)
    feats = extract(X)
    feats = (feats / np.linalg.norm(feats, axis=1, keepdims=True)).astype(np.float32)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=feats, y=y)
    return feats, y
