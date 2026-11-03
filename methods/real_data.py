"""Real datasets for e_compare_real, in the original sample order.

covertype   sklearn fetch_covtype; 581 012 samples, 54 features, 7 classes.
kdd         sklearn fetch_kddcup99 (10%, duplicates kept); 3 categorical features one-hot; 23 classes.
pamap2      UCI 231, Protocol part, subjects 101-109; activityID 0 removed, NaNs ffill/bfill
            per subject, decimated to 10 Hz; heart rate + 3 IMUs without orientation; 12 classes.
            
All: zero-variance features removed.
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_covtype, fetch_kddcup99

# https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
PAMAP2_DIR = "files/pamap2/PAMAP2_Dataset/Protocol"
PAMAP2_CACHE = "files/pamap2/pamap2_protocol.npz"
OFFLINE_SIZE = {"covertype": 1500, "kdd": 2000, "pamap2": 2000}   # before the first novel class
PAMAP2_ACTIVITIES = {1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
                     6: "cycling", 7: "nordic_walking", 12: "ascending_stairs",
                     13: "descending_stairs", 16: "vacuum_cleaning", 17: "ironing",
                     24: "rope_jumping"}


def _covertype():
    data = fetch_covtype()
    names = np.array(["spruce_fir", "lodgepole_pine", "ponderosa_pine", "cottonwood_willow",
                      "aspen", "douglas_fir", "krummholz"])
    return data.data, names[data.target.astype(int) - 1]


def _kdd():
    categorical = ["protocol_type", "service", "flag"]
    df = fetch_kddcup99(percent10=True, as_frame=True).frame
    labels = df.pop("labels").str.decode("ascii").str.rstrip(".").to_numpy()
    for col in df.columns:
        df[col] = df[col].str.decode("ascii") if col in categorical else df[col].astype(float)
    return pd.get_dummies(df, columns=categorical, dtype=float).to_numpy(), labels


def _pamap2():
    if os.path.exists(PAMAP2_CACHE):
        cache = np.load(PAMAP2_CACHE)
        return cache["X"], cache["y"]
    
    # .dat row: timestamp, activityID, heart rate, 3 IMUs x 17 columns (last 4: orientation)
    features = [2] + [start + j for start in (3, 20, 37) for j in range(13)]
    X_parts, y_parts = [], []
    for subject in range(101, 110):
        df = pd.read_csv(os.path.join(PAMAP2_DIR, f"subject{subject}.dat"), sep=" ", header=None)
        df = df[df[1] != 0]
        X_parts.append(df[features].ffill().bfill().to_numpy()[::10])      # 100 Hz -> 10 Hz
        y_parts.append(df[1].to_numpy()[::10])
    X = np.concatenate(X_parts).astype(np.float64)
    y = np.array([PAMAP2_ACTIVITIES[int(a)] for a in np.concatenate(y_parts)])
    np.savez_compressed(PAMAP2_CACHE, X=X, y=y)
    return X, y


def load_real(dataset):
    """(X float32 standardized, y class names) in the original dataset order."""
    X, y = {"covertype": _covertype, "kdd": _kdd, "pamap2": _pamap2}[dataset]()
    X = np.asarray(X, dtype=np.float64)
    std = X.std(axis=0)
    X = ((X[:, std > 0] - X[:, std > 0].mean(axis=0)) / std[std > 0]).astype(np.float32)
    return X, np.asarray(y, dtype=str)


def make_real_stream(dataset):
    """(stream, cfg) of e_compare_real: the first OFFLINE_SIZE[dataset] samples are labeled,
    classes with >= 50 samples there are known, other startup samples removed;
    labels renumbered: known, then novel by first appearance."""
    offline_size = OFFLINE_SIZE[dataset]
    X, labels = load_real(dataset)

    # known classes: >= 50 samples in the startup phase; the other startup samples are removed
    start = labels[:offline_size]
    classes, counts = np.unique(start, return_counts=True)
    known = set(classes[counts >= 50])
    keep = np.ones(len(labels), dtype=bool)
    keep[:offline_size] = np.isin(start, list(known))
    X, labels = X[keep], labels[keep]

    # roles: known, then novel, each in order of first appearance
    uniq, first = np.unique(labels, return_index=True)
    by_first = uniq[np.argsort(first)]
    order = [c for c in by_first if c in known] + [c for c in by_first if c not in known]
    role = {c: i for i, c in enumerate(order)}
    y = np.array([role[c] for c in labels], dtype=int)

    cfg = dict(n_known=len(known), offline_size=int(keep[:offline_size].sum()), true_n_classes=len(order))
    stream = (torch.as_tensor(X, dtype=torch.float), y)
    return stream, cfg
