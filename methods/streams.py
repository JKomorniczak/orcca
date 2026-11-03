"""Synthetic streams (OWDSG, smooth drift, novel classes) and their startup phase."""

from functools import lru_cache

import numpy as np
import torch

from methods.OWDSG import OWDSG
from methods.architectures import EncDec_mlp

N_CHUNKS = 100
CHUNK_SIZE = 50

def config(n_known, n_novel):
    """Startup phase: 125 samples per known class, ending 50 samples before the first novelty."""
    first_novel = (N_CHUNKS // n_novel // 2) * CHUNK_SIZE if n_novel else N_CHUNKS * CHUNK_SIZE
    return dict(n_known=n_known, true_n_classes=n_known + n_novel,
                offline_size=min(125 * n_known, first_novel - 50))


def weights_for(n_known):
    """Known-class proportions from linspace(0.4, 0.6), normalized."""
    w = np.linspace(0.4, 0.6, n_known)
    return list(w / w.sum())


@lru_cache(maxsize=None)
def _pool(dataset):
    if dataset == "cifar":
        from methods.embeddings import load_embeddings
        return load_embeddings()
    from methods.real_data import load_real
    return load_real(dataset)


def make_architecture(dataset):
    """Autoencoder n_features -> 64 -> 10 -> 64 -> n_features."""
    n_features = 100 if dataset == "tabular" else _pool(dataset)[0].shape[1]
    return EncDec_mlp(n_features, 10, hidden=(64,))



def make_stream(dataset, n_drifts, random_state, n_known, n_novel, class_sep=None,
                class_order=None, chunk_size=CHUNK_SIZE):
    """-> (X, y, drift chunks, novelty chunks); y: 0..n_known-1 known, then novel."""
    common = dict(n_drifts=n_drifts, n_novel=n_novel, n_chunks=N_CHUNKS, n_classes=n_known,
                  weights=weights_for(n_known), chunk_size=chunk_size, percentage_novel=0.3,
                  random_state=random_state, smooth_drift=True, drift_magnitude=1.0)
    if dataset == "tabular":
        stream = OWDSG(n_features=100, n_informative=50, class_sep=class_sep, **common)
    else:
        X, y = _pool(dataset)
        stream = OWDSG(X=X, y=y, class_order=class_order, **common)
    return (torch.as_tensor(stream.X, dtype=torch.float), stream.y.astype(int),
            np.asarray(stream.drift_gt, dtype=int), np.asarray(stream.novel_gt, dtype=int))
