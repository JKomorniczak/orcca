"""e_resources - time per sample and state size of the methods on long synthetic streams.

Streams: tabular, 3 known classes, class_sep=2.0, 100 chunks x 1000 = 100 000 samples.
Grid: novel classes 0, 3, 5 x drifts 0, 2, 4 x seeds 100-104.

Methods: methods/compare.py; CluStream on the first seed only.
Cost measured with methods/cost.py in blocks of online samples, after one unmeasured
warm-up run per method. A rerun skips cells already in the results file.

Saved: results/e_resources_tabular.npz
"""

import os

# single-thread
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import re
import warnings

import numpy as np
import torch

from methods import cost as C
from methods import streams as S
from methods.compare import build_methods

warnings.filterwarnings("ignore", message="Mean of empty slice")

N_NOVEL_VALUES = (0, 3, 5)
N_DRIFT_VALUES = (0, 2, 4)
RANDOM_STATES = (100, 101, 102, 103, 104)
BLOCK = 2500
OUT_PATH = "results/e_resources_tabular.npz"

torch.set_num_threads(1)
n_samples = S.N_CHUNKS * 1000
offline = S.config(3, 0)["offline_size"]
methods = [m for m in build_methods("tabular", offline) if m[0] != "ORCCA_noadapt"]
names = [m[0] for m in methods]
block_end = np.minimum(np.arange(offline + BLOCK, n_samples + BLOCK, BLOCK), n_samples)
block_sizes = np.diff(np.concatenate([[offline], block_end]))
grid = (len(names), len(N_NOVEL_VALUES), len(N_DRIFT_VALUES), len(RANDOM_STATES))

if os.path.exists(OUT_PATH):
    data = np.load(OUT_PATH)
    assert [re.sub("^ORCA", "ORCCA", str(n)) for n in data["names"]] == names and data["block_times"].shape[:4] == grid
    curves = {k: np.array(data[k], dtype=float) for k in ("block_times", "block_bytes")}
else:
    curves = {k: np.full(grid + (len(block_sizes),), np.nan) for k in ("block_times", "block_bytes")}

warmed_up = False
for i_n, n_novel in enumerate(N_NOVEL_VALUES):
    cfg = S.config(3, n_novel)
    for i_d, n_drifts in enumerate(N_DRIFT_VALUES):
        for i_r, rs in enumerate(RANDOM_STATES):
            cells = [i_m for i_m, name in enumerate(names)
                     if not (name == "CluStream" and i_r > 0)
                     and not np.isfinite(curves["block_times"][i_m, i_n, i_d, i_r]).all()]
            if not cells:
                continue
            X, y, _, _ = S.make_stream("tabular", n_drifts, rs, class_sep=2.0, n_known=3, n_novel=n_novel, chunk_size=1000)

            if not warmed_up:
                n_warm = offline + 10000
                for name, kind, build in methods:
                    C.measure(kind, build, X[:n_warm], y[:n_warm], rs, offline, cfg["true_n_classes"], BLOCK)
                warmed_up = True

            for i_m in cells:
                name, kind, build = methods[i_m]
                idx = (i_m, i_n, i_d, i_r)
                try:
                    res = C.measure(kind, build, X, y, rs, offline, cfg["true_n_classes"], BLOCK)
                    for k in ("block_times", "block_bytes"):
                        curves[k][idx] = np.nan
                        curves[k][idx][:len(res[k])] = res[k]
                except Exception as exception:
                    print(exception)
                
            np.savez_compressed(OUT_PATH, **curves, block_sizes=block_sizes, names=np.array(names),
                                n_novel_values=np.asarray(N_NOVEL_VALUES),
                                n_drift_values=np.asarray(N_DRIFT_VALUES),
                                random_states=np.asarray(RANDOM_STATES))
