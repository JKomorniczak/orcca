# source: https://github.com/jgaud/streamndr/blob/main/streamndr/model/echo.py

from collections import deque

import numpy as np
from scipy.stats import pointbiserialr, beta
from sklearn.preprocessing import MinMaxScaler

from methods.utils import (
    MicroCluster,
    ShortMem,
    ShortMemInstance,
    ClusterModel,
    generate_microclusters,
    check_f_outlier,
    get_closest_clusters,
    get_most_occurring_by_column,
    qnsc,
)

class Echo:
    """ECHO novelty detection. [1]

    [1] Haque, Ahsanul, et al. "Efficient handling of concept drift and concept evolution over stream data."
    2016 IEEE 32nd international conference on data engineering (ICDE). IEEE, 2016.

    """

    def __init__(self,
                 K=50,
                 min_examples_cluster=50,
                 ensemble_size=6,
                 W=1000,
                 verbose=0,
                 random_state=None,
                 unknown_as_novelty_id=True):
        """unknown_as_novelty_id: label unexplained samples with the id of the next
        created cluster (max label + 1) instead of -1."""

        self.unknown_as_novelty_id = unknown_as_novelty_id
        self.K = K
        self.min_examples_cluster = min_examples_cluster
        self.ensemble_size = ensemble_size
        self.W = W
        self.verbose = verbose
        self.random_state = random_state

        self.models = []
        self.short_mem = ShortMem()
        self.association_coefficients = []
        self.purity_coefficients = []
        self.confidence_window = deque(maxlen=self.W)
        self.window = deque(maxlen=self.W)

        self.sample_counter = 0
        self.before_offline_phase = True

    def fit(self, X, y):
        """Offline phase: ensemble_size models on the whole X/y (different seeds) + confidence coefficients."""
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        timestamp = len(X)

        for i in range(self.ensemble_size):
            state = None if self.random_state is None else self.random_state + i
            microclusters = generate_microclusters(X, y, timestamp, self.K, min_samples=0,
                                                     random_state=state)
            if len(microclusters) > 0:
                self.models.append(ClusterModel(microclusters, list(np.unique(y))))

        for model in self.models:
            closest_clusters_model, dist = get_closest_clusters(X, [mc.centroid for mc in model.microclusters])
            model_label = [model.microclusters[c].label for c in closest_clusters_model]

            associations = np.array([model.microclusters[c].max_distance for c in closest_clusters_model]) - dist
            purities = (np.array([model.microclusters[c].n_label_instances for c in closest_clusters_model]) /
                        np.array([model.microclusters[c].n for c in closest_clusters_model]))

            correct = [1 if y1 == y2 else 0 for y1, y2 in zip(y, model_label)]

            self.association_coefficients.append(self._correlation(associations, correct))
            self.purity_coefficients.append(self._correlation(purities, correct))

        self.before_offline_phase = False
        return self

    @staticmethod
    def _correlation(heuristic, correct):
        """Point-biserial correlation; 1.0 when undefined (constant input)."""
        r = pointbiserialr(heuristic, correct).statistic
        return 1.0 if not np.isfinite(r) else float(r)

    def _novelty_label(self):
        labels = [label for model in self.models for label in model.labels]
        return max(labels) + 1 if labels else 0

    def _unknown_label(self):
        return self._novelty_label() if self.unknown_as_novelty_id else -1

    def predict(self, X):
        if self.before_offline_phase:
            raise Exception("Model must be fitted first")

        f_outliers = check_f_outlier(X, self.models)
        closest_model_cluster, average_confidences, y_preds = self._majority_voting(X, True)

        pred_label = []
        for i in range(len(X)):
            self.sample_counter += 1

            closest_cluster = self.models[closest_model_cluster[i][0]].microclusters[closest_model_cluster[i][1]]

            if not f_outliers[i]:
                pred_label.append(y_preds[i])
                closest_cluster.update_cluster(X[i], self.sample_counter, False)

            else:
                pred_label.append(self._unknown_label())

                self.short_mem.append(ShortMemInstance(X[i], self.sample_counter))

                if len(self.short_mem) > self.min_examples_cluster:
                    novel_cluster = self._novelty_detect()
                    if novel_cluster is not None:
                        if self.verbose > 1:
                            print("Novel cluster: ", novel_cluster)
                        elif self.verbose > 0:
                            print("Novel cluster: ", novel_cluster.small_str())

                        pred_label[-1] = novel_cluster.label

                        for model in self.models:
                            model.microclusters.append(novel_cluster)
                            model.labels.append(novel_cluster.label)

                    # buffer cleared regardless of the outcome (paper)
                    for _ in range(len(self.short_mem)):
                        self._remove_sample_from_short_mem(0)

            self.window.append(ShortMemInstance(X[i], self.sample_counter, y_pred=pred_label[-1]))
            self.confidence_window.append(average_confidences[i])

            change_point = self._detect_change()
            if change_point != -1:
                if self.verbose > 1:
                    print("Change detected at point: ", change_point)
                self._update_classifier(change_point)

        return np.array(pred_label)

    def _majority_voting(self, X, return_labels=True):
        closest_clusters = []
        labels = []
        dists = []
        confidences = []

        for i, model in enumerate(self.models):
            closest_clusters_model, dist = get_closest_clusters(X, [mc.centroid for mc in model.microclusters])
            closest_clusters.append(closest_clusters_model)
            model_label = [model.microclusters[c].label for c in closest_clusters_model]
            labels.append(model_label)
            dists.append(dist)

            associations = np.array([model.microclusters[c].max_distance for c in closest_clusters_model]) - dist
            purities = (np.array([model.microclusters[c].n_label_instances for c in closest_clusters_model]) /
                        np.array([model.microclusters[c].n for c in closest_clusters_model]))

            confidences.append(np.dot(associations, self.association_coefficients[i]) +
                                np.dot(purities, self.purity_coefficients[i]))

        scaler = MinMaxScaler()
        average_confidences = np.mean(scaler.fit_transform(confidences), axis=0)

        best_models = np.argmin(dists, axis=0)
        closest_model_cluster = [
            (best_models[i], closest_clusters[best_models[i]][i]) for i in range(len(X))
        ]

        if return_labels:
            return closest_model_cluster, average_confidences, get_most_occurring_by_column(labels)
        return closest_model_cluster, average_confidences

    def _novelty_detect(self):
        if self.verbose > 1:
            print("Novelty detection started")

        X = self.short_mem.get_all_points()
        new_class_vote = 0
        potential_novel_points_idx = {index: 0 for index in range(len(X))}

        for model in self.models:
            qnscs = qnsc(X, model.microclusters, self.min_examples_cluster)
            nb_potential_novel_points = 0

            for i, qnsc_value in enumerate(qnscs):
                if qnsc_value > 0:
                    potential_novel_points_idx[i] += 1
                    nb_potential_novel_points += 1

            if nb_potential_novel_points > self.min_examples_cluster:
                new_class_vote += 1

        if new_class_vote == len(self.models):
            novel_points_idx = [idx for idx, vote in potential_novel_points_idx.items() if vote == len(self.models)]

            if len(novel_points_idx) >= self.min_examples_cluster:
                novel_points = [X[i] for i in novel_points_idx]
                label = max(label for model in self.models for label in model.labels) + 1
                return MicroCluster(label, instances=np.array(novel_points), timestamp=self.sample_counter,
                                     n_label_instances=len(novel_points))

        return None

    def _detect_change(self, alpha=0.05, gamma=100):
        """Change point index, -1 if none; n (forced boundary) if mean confidence <= 0.3 or window > W.

        gamma: cushion period (points ignored at each end of the window).
        """
        Th = -np.log(alpha)
        n = len(self.confidence_window)
        omega_n = 0
        k_max = -1

        confidence_list = list(self.confidence_window)

        if n <= self.W and np.mean(self.confidence_window) > 0.3:
            for k in range(gamma, n - gamma):
                pre_data = confidence_list[:k]
                post_data = confidence_list[k:]

                alpha0, beta0 = self._estimate_beta_params(pre_data)
                alpha1, beta1 = self._estimate_beta_params(post_data)

                S_k_n = self._calculate_S_k_n(post_data, alpha0, beta0, alpha1, beta1)

                if S_k_n > omega_n:
                    omega_n = S_k_n
                    k_max = k

            return k_max if omega_n >= Th else -1
        return n

    def _estimate_beta_params(self, data):
        """Estimate Beta distribution parameters using the method of moments."""
        mean = np.mean(data)
        var = np.var(data)
        if var == 0:
            return 1, 1
        alpha = mean * (mean * (1 - mean) / var - 1)
        beta_param = (1 - mean) * (mean * (1 - mean) / var - 1)
        return max(alpha, 0.01), max(beta_param, 0.01)

    def _calculate_S_k_n(self, data, alpha0, beta0, alpha1, beta1):
        """Log-likelihood ratio of the post-change vs pre-change Beta fits."""
        try:
            pdf1 = beta.pdf(data, alpha1, beta1)
            pdf0 = beta.pdf(data, alpha0, beta0)
        except OverflowError:
            return 0.0

        ratio = np.divide(pdf1, pdf0, out=np.ones_like(pdf1), where=pdf0 != 0)
        log_ratio = np.log(ratio, out=np.zeros_like(ratio), where=ratio > 0)

        return np.sum(log_ratio)

    def _update_classifier(self, change_point):
        # unsupervised: trained on own predictions (paper: true labels below tau)
        if len(self.window) == 0:
            return
        X_train = np.array([inst.point for inst in self.window])
        y_train = np.array([inst.y_pred for inst in self.window])

        new_model = self._train_new_model(X_train, y_train)

        if len(self.models) < self.ensemble_size:
            self.models.append(new_model)
        else:
            oldest_model_index = int(np.argmin([model.microclusters[0].timestamp for model in self.models]))
            self.models[oldest_model_index] = new_model

        self.window = deque(list(self.window)[change_point:], maxlen=self.W)
        self.confidence_window = deque(list(self.confidence_window)[change_point:], maxlen=self.W)

    def _train_new_model(self, X, y):
        K0 = min(self.K, len(X))
        microclusters = generate_microclusters(X, y, self.sample_counter, K0, min_samples=0,
                                                 random_state=self.random_state)
        return ClusterModel(microclusters, list(np.unique(y)))

    def _remove_sample_from_short_mem(self, index):
        if self.short_mem.get_instance(index) is None:
            return
        self.short_mem.remove(index)

