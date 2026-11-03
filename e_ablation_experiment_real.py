"""e_ablation_real - ORCCA ablations of e_ablation on Forest Covertype.

Stream and settings of e_compare_real (methods/real_data.py), single seed.
Variants as in e_ablation_experiment.py, all computed here.
Saved: results/e_ablation_real_covertype.npz - three criteria per variant.
"""

import os

# single-thread BLAS
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import warnings

import numpy as np

from methods import streams as S
from methods.ORCCA_ablation import ORCCA_ablation
from methods.compare import run_single
from methods.evaluation import METRIC_KEYS
from methods.real_data import make_real_stream

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATASET = "covertype"

VARIANTS = (
    ("ORCCA", {}),
    ("no_adaptation", dict(adapt_prob=0.0)),
    ("adaptation_no_weighting", dict(adapt_by_error=False)),
    ("no_split", dict(bimodality_threshold=float("inf"))),
    ("strategy_contrastive", dict(known_strategy="contrastive", novelty_threshold=0.275)),
    ("strategy_kde", dict(known_strategy="kde", novelty_threshold=0.2)),
)

stream, cfg = make_real_stream(DATASET)
names = np.array([v[0] for v in VARIANTS])
results = {k: np.full(len(VARIANTS), np.nan) for k in METRIC_KEYS}

for i_v, (name, overrides) in enumerate(VARIANTS):
    build = lambda rs, k: ORCCA_ablation(architecture=S.make_architecture(DATASET),
                                         buffer_size=cfg["offline_size"], random_state=rs, **overrides)
    try:
        res = run_single(stream, "stream", build, 0, cfg)
        for k in METRIC_KEYS:
            results[k][i_v] = res[k]
    except Exception as exception:
        print(exception)

    np.savez_compressed(f"results/e_ablation_real_{DATASET}.npz", **results, names=names)
