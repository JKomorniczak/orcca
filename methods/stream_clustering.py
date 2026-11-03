"""Unsupervised stream clustering baselines (river) with the ORCCA process() interface.

process(sample) -> (is_novelty, cluster_id): cluster_id is the river cluster id (-1 before
initialization), is_novelty is True when the id appears for the first time.

CluStream gets the true number of clusters (n_macro_clusters).
"""

from typing import Optional, Tuple

import numpy as np

from river import cluster


class _RiverAdapter:
    def __init__(self, model) -> None:
        self.model = model
        self._seen_ids = set()

    def process(self, sample, y=None) -> Tuple[bool, int]:
        """y is ignored."""
        x = {i: float(v) for i, v in enumerate(np.asarray(sample, dtype=float).ravel())}

        self.model.learn_one(x)
        try:
            cluster_id = int(self.model.predict_one(x))
        except Exception:
            # not initialized yet
            return (False, -1)

        is_novelty = cluster_id not in self._seen_ids
        self._seen_ids.add(cluster_id)
        return (is_novelty, cluster_id)


class CluStreamDetector(_RiverAdapter):
    """CluStream (Aggarwal et al., 2003).

    init_mu/init_sigma: init of river's k-means centers; set to the feature mean/std,
    otherwise high-dimensional data collapse into a single macro-cluster.
    """

    def __init__(
        self,
        n_macro_clusters: int,
        max_micro_clusters: int = 100,
        micro_cluster_r_factor: int = 2,
        time_window: int = 1000,
        time_gap: int = 100,
        init_mu: float = 0.0,
        init_sigma: float = 1.0,
        halflife: float = 0.5,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(cluster.CluStream(
            n_macro_clusters=n_macro_clusters,
            max_micro_clusters=max_micro_clusters,
            micro_cluster_r_factor=micro_cluster_r_factor,
            time_window=time_window,
            time_gap=time_gap,
            seed=random_state,
            mu=init_mu,
            sigma=init_sigma,
            halflife=halflife,
        ))


class DenStreamDetector(_RiverAdapter):
    """DenStream (Cao et al., 2006). Requires mu * beta > 1 (river divides by zero otherwise)."""

    def __init__(
        self,
        epsilon: float = 0.5,
        mu: float = 3.0,
        beta: float = 0.5,
        decaying_factor: float = 0.25,
        n_samples_init: int = 100,
        stream_speed: int = 100,
    ) -> None:
        super().__init__(cluster.DenStream(
            decaying_factor=decaying_factor,
            beta=beta,
            mu=mu,
            epsilon=epsilon,
            n_samples_init=n_samples_init,
            stream_speed=stream_speed,
        ))
