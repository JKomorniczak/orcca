"""Computational cost of a stream method per online block: processing time and state size (e_resources)."""

import gc
import sys
import time

import numpy as np
import torch

def state_bytes(obj, seen=None):
    """Recursive state size; tensors by logical size, shared objects counted once."""
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))

    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, (str, bytes, int, float, bool, type(None))):
        return sys.getsizeof(obj)

    total = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            total += state_bytes(k, seen) + state_bytes(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            total += state_bytes(v, seen)
    elif hasattr(obj, "__dict__"):
        total += state_bytes(vars(obj), seen)
    elif hasattr(obj, "__slots__"):
        for slot in obj.__slots__:
            if hasattr(obj, slot):
                total += state_bytes(getattr(obj, slot), seen)
    else:
        total += sys.getsizeof(obj)
    return total



def measure(kind, build, X, y, random_state, offline_size, n_true_classes, block):
    """One run -> block_times and block_bytes, one value per block of the online phase."""
    torch.manual_seed(random_state)
    gc.collect()

    detector = build(random_state, n_true_classes)

    X_flat = X.numpy().astype(np.float64) if kind == "batch" else None

    # startup phase, not measured
    if kind == "stream":
        for i in range(offline_size):
            detector.process(X[i].reshape(1, -1), y[i])
    else:
        detector.fit(X_flat[:offline_size], y[:offline_size])

    n = X.shape[0]
    block_times, block_bytes = [], []

    for start in range(offline_size, n, block):
        stop = min(start + block, n)
        t0 = time.perf_counter()
        if kind == "stream":
            for i in range(start, stop):
                detector.process(X[i].reshape(1, -1))
        else:
            for i in range(start, stop):
                detector.predict(X_flat[i:i + 1])
        block_times.append(time.perf_counter() - t0)
        # outside the timing
        block_bytes.append(float(state_bytes(detector)))

    return {"block_times": np.asarray(block_times), "block_bytes": np.asarray(block_bytes)}
