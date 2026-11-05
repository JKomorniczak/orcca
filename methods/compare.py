"""Shared setup of the comparison experiments: stream grid, methods and a single run."""

import numpy as np
import torch

from methods import streams as S
from methods.ORCCA import ORCCA
from methods.minas import Minas
from methods.ecsminerwf import ECSMinerWF
from methods.echo import Echo
from methods.stream_clustering import CluStreamDetector, DenStreamDetector
from methods.baselines import OCND, CND
from methods.evaluation import evaluate_stream, mask_undefined

# grid of the synthetic streams (e_compare_syn, e_ablation)
SPLITS = ((2, 8), (4, 6), (6, 2)) # (known, novel) classes
N_DRIFT_VALUES = (0, 1, 3, 5, 7)
RANDOM_STATES = tuple(range(10))


def stream_setup(dataset):
    """(class_sep, {seed: class_order}) for a dataset."""
    if dataset == "tabular":
        return 2.5, {}
    # CIFAR: seed-dependent label permutation, the first n_known classes are known
    return None, {rs: np.random.RandomState(rs).permutation(10) for rs in RANDOM_STATES}


# (CluStream, DenStream) kwargs, depend on the feature scale; real datasets (standardized) as tabular
CLUSTERING_KWARGS = {
    "tabular": (dict(max_micro_clusters=100, micro_cluster_r_factor=10, time_window=1000),
                dict(epsilon=2.0, mu=3.0, beta=0.5)),
    "cifar":   (dict(max_micro_clusters=100, micro_cluster_r_factor=2, time_window=1000,
                     init_mu=0.0273, init_sigma=0.0348, halflife=0.3),
                dict(epsilon=0.7, mu=3.0, beta=0.5)),
}


def build_methods(dataset, offline_size):
    """(name, kind, builder(random_state, n_true_classes)); kind: "stream" | "batch"."""
    clustream_kw, denstream_kw = CLUSTERING_KWARGS["cifar" if dataset == "cifar" else "tabular"]
    return (
        ("ORCCA", "stream", lambda rs, k: ORCCA(
            architecture=S.make_architecture(dataset), buffer_size=offline_size, random_state=rs)),
        ("ORCCA_noadapt", "stream", lambda rs, k: ORCCA(
            architecture=S.make_architecture(dataset), buffer_size=offline_size, random_state=rs,
            adapt_prob=0.0)),
        ("MINAS", "batch", lambda rs, k: Minas(
            kini=3, random_state=rs, min_short_mem_trigger=10, min_examples_cluster=10,
            unknown_as_novelty_id=False)),
        ("ECSMinerWF", "batch", lambda rs, k: ECSMinerWF(
            K=5, min_examples_cluster=20, ensemble_size=3, random_state=rs,
            unknown_as_novelty_id=False)),
        ("ECHO", "batch", lambda rs, k: Echo(
            K=5, min_examples_cluster=20, ensemble_size=3, W=500, random_state=rs,
            unknown_as_novelty_id=False)),
        # CluStream gets the true number of classes
        ("CluStream", "stream", lambda rs, k: CluStreamDetector(
            n_macro_clusters=k, random_state=rs, **clustream_kw)),
        ("DenStream", "stream", lambda rs, k: DenStreamDetector(
            n_samples_init=offline_size, **denstream_kw)),
        ("OCND", "batch", lambda rs, k: OCND(nu=0.5)),
        ("CND", "batch", lambda rs, k: CND(threshold=0.9, random_state=rs)),
    )


def run_single(stream, kind, build, random_state, cfg):
    """One run -> dict of the three criteria on the online phase."""
    X, y = stream[0], stream[1]
    n_known, offline = cfg["n_known"], cfg["offline_size"]
    torch.manual_seed(random_state)
    detector = build(random_state, cfg["true_n_classes"])

    if kind == "stream": # ORCCA, ORCCA_noadapt, CluStream, DenStream
        offline_preds = np.array([detector.process(X[i].reshape(1, -1), y[i])[1]
                                  for i in range(offline)])
        predictions = np.array([detector.process(X[i].reshape(1, -1))[1]
                                for i in range(offline, X.shape[0])])
    else: # MINAS, ECSMinerWF, ECHO, OCND, CND
        offline_preds = None
        X_flat = X.numpy().astype(np.float64)
        detector.fit(X_flat[:offline], y[:offline])
        predictions = np.array([detector.predict(X_flat[i:i + 1])[0]
                                for i in range(offline, X_flat.shape[0])])

    # unsupervised methods: criterion 1 with optimal label matching
    match_labels = False
    if hasattr(detector, "n_known"): # ORCCA, ORCCA_ablation
        known_ids = set(range(detector.n_known))
    elif kind == "batch":
        known_ids = set(np.unique(y[:offline]).tolist())
    else:
        known_ids = set(np.unique(offline_preds).tolist()) # clusters of the offline phase
        match_labels = True

    res = evaluate_stream(y[offline:], predictions, n_known, known_ids, match_labels)
    return mask_undefined(res, getattr(detector, "undefined_criteria", ()))
