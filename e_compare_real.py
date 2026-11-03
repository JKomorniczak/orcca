"""e_compare_real - comparison of methods on real datasets: Forest Covertype, KDD Cup '99, PAMAP2.

DATASET: "covertype" | "kdd" | "pamap2" (methods/real_data.py), in the original sample order.
The first 1500 / 2000 / 2000 samples are labeled; classes with >= 50 samples there are known,
other samples in that phase are removed.

Saved: results/e_compare_real_{DATASET}.npz
"""

import os

# single-thread BLAS
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import warnings

import numpy as np
from methods.compare import build_methods, run_single
from methods.evaluation import METRIC_KEYS
from methods.real_data import make_real_stream

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATASET = "pamap2" # "covertype" | "kdd" | "pamap2"

stream, cfg = make_real_stream(DATASET)

methods = build_methods(DATASET, cfg["offline_size"])
names = np.array([m[0] for m in methods])
metrics = {k: np.full(len(names), np.nan) for k in METRIC_KEYS}

for i_m, (name, kind, build) in enumerate(methods):
    try:
        res = run_single(stream, kind, build, 0, cfg)
        for k in METRIC_KEYS:
            metrics[k][i_m] = res[k]
    except Exception as exception:
        print(exception)

    np.savez_compressed(f"results/e_compare_real_{DATASET}.npz", **metrics, names=names)
