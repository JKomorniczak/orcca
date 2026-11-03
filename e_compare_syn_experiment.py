"""e_compare_syn - ORCCA vs reference methods on synthetic streams with drift and novelties.

DATASET: "tabular" (make_classification, class_sep=2.5) | "cifar" (CIFAR-10 embeddings, class order permuted per seed). 

Drift is smooth translation (methods/streams.py).
Grid (methods/compare.py): splits (known + novel) 2+8, 4+6, 6+2 x n_drifts 0, 1, 3, 5, 7
x 10 seeds. Methods: ORCCA, ORCCA_noadapt (adapt_prob=0), MINAS, ECSMinerWF, ECHO, CluStream,
DenStream, OCND, CND.

Saved: results/e_compare_syn_{DATASET}.npz
"""

import os

# single-thread BLAS
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import warnings

import numpy as np
from scipy.stats import ConstantInputWarning

from methods import streams as S
from methods.compare import (SPLITS, N_DRIFT_VALUES, RANDOM_STATES, stream_setup, build_methods, run_single)
from methods.evaluation import METRIC_KEYS

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)

DATASET = "tabular" # "tabular" | "cifar"

class_sep, class_orders = stream_setup(DATASET)
names = np.array([m[0] for m in build_methods(DATASET, 1)])
metrics = {k: np.full((len(names), len(SPLITS), len(N_DRIFT_VALUES), len(RANDOM_STATES)), np.nan)
           for k in METRIC_KEYS}

for i_p, (n_known, n_novel) in enumerate(SPLITS):
    cfg = S.config(n_known, n_novel)
    methods = build_methods(DATASET, cfg["offline_size"])
    for i_d, n_drifts in enumerate(N_DRIFT_VALUES):
        streams = [S.make_stream(DATASET, n_drifts, rs, class_sep=class_sep, n_known=n_known,
                                 n_novel=n_novel, class_order=class_orders.get(rs))
                   for rs in RANDOM_STATES]
        for i_m, (name, kind, build) in enumerate(methods):
            for i_r, (stream, rs) in enumerate(zip(streams, RANDOM_STATES)):
                try:
                    res = run_single(stream, kind, build, rs, cfg)
                    for k in METRIC_KEYS:
                        metrics[k][i_m, i_p, i_d, i_r] = res[k]
                except Exception as exception:
                    print(exception)
            c = (i_m, i_p, i_d)
           
        np.savez_compressed(
            f"results/e_compare_syn_{DATASET}.npz", **metrics, names=names,
            splits=np.asarray(SPLITS), n_drift_values=np.asarray(N_DRIFT_VALUES),
            random_states=np.asarray(RANDOM_STATES),
            class_sep=np.nan if class_sep is None else class_sep,
            # (seed, class), original labels in role order; empty on tabular
            class_orders=np.asarray([class_orders[rs] for rs in RANDOM_STATES]
                                    if class_orders else np.empty((0, 0), int)),
        )
