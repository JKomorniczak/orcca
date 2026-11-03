"""e_ablation - ORCCA ablations, each variant disables or changes one mechanism.

Variants (methods/ORCCA_ablation.py): ORCCA (full), no_adaptation (adapt_prob=0),
adaptation_no_weighting (adapt_by_error=False), no_split (bimodality_threshold=inf),
strategy_<s> for each other known_strategy. 
Streams of e_compare_syn;
Saved: results/e_ablation_{DATASET}.npz
"""

import re
import warnings

import numpy as np

from methods import streams as S
from methods.ORCCA_ablation import ORCCA_ablation
from methods.compare import SPLITS, N_DRIFT_VALUES, RANDOM_STATES, stream_setup, run_single
from methods.evaluation import METRIC_KEYS

warnings.filterwarnings("ignore", message="Mean of empty slice")

DATASET = "tabular" # "tabular" | "cifar"

# (name, ORCCA_ablation overrides, name in e_compare_syn to copy from)
VARIANTS = (
    ("ORCCA", {}, "ORCCA"),
    ("no_adaptation", dict(adapt_prob=0.0), "ORCCA_noadapt"),
    ("adaptation_no_weighting", dict(adapt_by_error=False), None),
    ("no_split", dict(bimodality_threshold=float("inf")), None),
    # tau best for the strategy in the hyperparameter search
    ("strategy_contrastive", dict(known_strategy="contrastive", novelty_threshold=0.275), None),
    ("strategy_kde", dict(known_strategy="kde", novelty_threshold=0.2), None),
)


def builder(overrides, offline_size):
    return lambda rs, k: ORCCA_ablation(architecture=S.make_architecture(DATASET),
                                        buffer_size=offline_size, random_state=rs, **overrides)


class_sep, class_orders = stream_setup(DATASET)
syn = np.load(f"results/e_compare_syn_{DATASET}.npz", allow_pickle=True)
syn_names = [re.sub("^ORCA", "ORCCA", str(n)) for n in syn["names"]]    # results saved under the old name

# e_compare_syn must match the current code: same grid and one reproduced ORCCA run
assert np.array_equal(syn["splits"], SPLITS) and np.array_equal(syn["n_drift_values"], N_DRIFT_VALUES) \
    and np.array_equal(syn["random_states"], RANDOM_STATES), "different grid in e_compare_syn"
    
n_known, n_novel = SPLITS[0]
cfg = S.config(n_known, n_novel)
stream = S.make_stream(DATASET, N_DRIFT_VALUES[-1], RANDOM_STATES[0], class_sep=class_sep,
                       n_known=n_known, n_novel=n_novel, class_order=class_orders.get(RANDOM_STATES[0]))
res = run_single(stream, "stream", builder({}, cfg["offline_size"]), RANDOM_STATES[0], cfg)
stored = float(syn["known_vs_novel_balanced_accuracy"][syn_names.index("ORCCA"), 0, -1, 0])
assert np.isclose(res["known_vs_novel_balanced_accuracy"], stored), \
    f"reproduced ORCCA {res['known_vs_novel_balanced_accuracy']:.6f} != stored {stored:.6f}"

names = np.array([v[0] for v in VARIANTS])
results = {k: np.full((len(VARIANTS), len(SPLITS), len(N_DRIFT_VALUES), len(RANDOM_STATES)), np.nan)
           for k in METRIC_KEYS}
for i_v, (_, _, from_syn) in enumerate(VARIANTS):
    if from_syn:
        for k in METRIC_KEYS:
            results[k][i_v] = syn[k][syn_names.index(from_syn)]

for i_p, (n_known, n_novel) in enumerate(SPLITS):
    cfg = S.config(n_known, n_novel)
    for i_d, n_drifts in enumerate(N_DRIFT_VALUES):
        streams = [S.make_stream(DATASET, n_drifts, rs, class_sep=class_sep, n_known=n_known,
                                 n_novel=n_novel, class_order=class_orders.get(rs))
                   for rs in RANDOM_STATES]
        for i_v, (name, overrides, from_syn) in enumerate(VARIANTS):
            if from_syn:
                continue
            build = builder(overrides, cfg["offline_size"])
            for i_r, (stream, rs) in enumerate(zip(streams, RANDOM_STATES)):
                try:
                    res = run_single(stream, "stream", build, rs, cfg)
                    for k in METRIC_KEYS:
                        results[k][i_v, i_p, i_d, i_r] = res[k]
                except Exception as exception:
                    print(exception)
            

        np.savez_compressed(f"results/e_ablation_{DATASET}.npz", **results, names=names,
                            splits=np.asarray(SPLITS), n_drift_values=np.asarray(N_DRIFT_VALUES))
