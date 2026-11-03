# source: https://github.com/jgaud/streamndr/blob/main/streamndr/model/ecsminer.py
#         https://github.com/jgaud/streamndr/blob/main/streamndr/model/ecsminerwf.py


from collections import Counter

import numpy as np

from methods.utils import (
    ShortMem,
    ShortMemInstance,
    ClusterModel,
    generate_microclusters,
    check_f_outlier,
    get_closest_clusters,
    get_most_occurring_by_column,
    qnsc,
)


class ECSMinerWF:
    """ECSMinerWF (ECSMiner without feedback) novelty detection. [1]

    [1] de Faria, Elaine Ribeiro, André Carlos Ponce de Leon Ferreira Carvalho, and Joao Gama. "MINAS: multiclass learning algorithm for novelty detection in data streams."
    Data mining and knowledge discovery 30 (2016): 640-680.
    """

    def __init__(self,
                 K=50,
                 min_examples_cluster=50,
                 ensemble_size=6,
                 verbose=0,
                 random_state=None,
                 unknown_as_novelty_id=True):
        """unknown_as_novelty_id: label unexplained samples with the id of the next
        created cluster (max label + 1) instead of -1."""

        self.unknown_as_novelty_id = unknown_as_novelty_id
        self.K = K
        self.min_examples_cluster = min_examples_cluster
        self.ensemble_size = ensemble_size
        self.verbose = verbose
        self.random_state = random_state

        self.models = []
        self.novel_models = []
        self.short_mem = ShortMem()
        self.sample_counter = 0
        self.last_nd = -self.min_examples_cluster
        self.chunk_size = None
        self.before_offline_phase = True

    def fit(self, X, y):
        """Offline phase: one chunk-classifier (K micro-clusters) per chunk."""
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        self.chunk_size = int(np.ceil(len(X) / self.ensemble_size))

        timestamp = len(X)

        for i in range(self.ensemble_size):
            start = i * self.chunk_size
            end = start + self.chunk_size
            X_chunk = X[start:end]
            y_chunk = y[start:end]
            if len(X_chunk) == 0:
                break

            # micro-clusters with < 3 instances are discarded (ECSMiner)
            microclusters = generate_microclusters(X_chunk, y_chunk, timestamp, self.K, min_samples=3,
                                                     random_state=self.random_state)
            if len(microclusters) > 0:
                self.models.append(ClusterModel(microclusters, list(np.unique(y_chunk))))

        self.before_offline_phase = False
        return self

    def _novelty_label(self):
        labels = [label for model in self.models for label in model.labels]
        if not labels:
            return 0
        max_novel = max((label for model in self.novel_models
                         for label in model.labels), default=-1)
        return max(max(labels), max_novel) + 1

    def _unknown_label(self):
        return self._novelty_label() if self.unknown_as_novelty_id else -1

    def predict(self, X):
        if self.before_offline_phase:
            raise Exception("Model must be fitted first")

        f_outliers = check_f_outlier(X, self.models)
        closest_model_cluster, y_preds = self._majority_voting(X)

        pred_label = []
        for i in range(len(X)):
            self.sample_counter += 1

            closest_cluster = self.models[closest_model_cluster[i][0]].microclusters[closest_model_cluster[i][1]]

            self._filter_buffer()

            if not f_outliers[i]:
                pred_label.append(y_preds[i])
                closest_cluster.update_cluster(X[i], self.sample_counter, False)

            else:
                # F-outlier; novel models checked per sample, they can grow mid-batch
                explained_by_novel = None
                if self.novel_models:
                    novel_microclusters = [mc for model in self.novel_models for mc in model.microclusters]
                    closest_novel_idx, novel_dist = get_closest_clusters(X[i:i + 1], [mc.centroid for mc in novel_microclusters])
                    if novel_dist[0] <= novel_microclusters[closest_novel_idx[0]].max_distance:
                        explained_by_novel = novel_microclusters[closest_novel_idx[0]]

                if explained_by_novel is not None:
                    pred_label.append(explained_by_novel.label)
                    explained_by_novel.update_cluster(X[i], self.sample_counter, False)

                else:
                    pred_label.append(self._unknown_label())

                    self.short_mem.append(ShortMemInstance(X[i], self.sample_counter))

                    if (len(self.short_mem) > self.min_examples_cluster) and \
                       ((self.last_nd + self.min_examples_cluster) <= self.sample_counter):
                        self.last_nd = self.sample_counter

                        novel_clusters = self._novelty_detect()

                        if novel_clusters is not None:
                            max_label_ensemble = max(label for model in self.models for label in model.labels)
                            max_label_novel = max((label for model in self.novel_models for label in model.labels), default=-1)
                            new_label = max(max_label_ensemble, max_label_novel) + 1

                            for novel_cluster in novel_clusters:
                                novel_cluster.label = new_label
                                if self.verbose > 0:
                                    print("Novel cluster detected: ", novel_cluster.small_str())

                                for instance in novel_cluster.instances:
                                    self._remove_sample_from_short_mem(self.short_mem.index(np.array(instance)))
                                # drop examples so the state does not grow (streamndr keeps them)
                                novel_cluster.instances = None

                            pred_label[-1] = new_label
                            self.novel_models.append(ClusterModel(novel_clusters, [new_label]))

        return np.array(pred_label)

    def _majority_voting(self, X, return_labels=True):
        closest_clusters = []
        labels = []
        dists = []

        for model in self.models:
            closest_clusters_model, dist = get_closest_clusters(X, [mc.centroid for mc in model.microclusters])
            closest_clusters.append(closest_clusters_model)
            labels.append([model.microclusters[c].label for c in closest_clusters_model])
            dists.append(dist)

        best_models = np.argmin(dists, axis=0)

        # ignore an older model's vote if it was not trained on the class predicted by a younger one
        # and the point is its outlier
        for k in range(len(X)):
            for i in range(len(self.models) - 1, 0, -1):
                for j in range(0, i):
                    if labels[i][k] not in self.models[j].labels:
                        outlier = True
                        for mc in self.models[j].microclusters:
                            if mc.distance_to_centroid(X[k]) <= mc.max_distance:
                                outlier = False
                                break
                        if outlier:
                            labels[j][k] = -1

        closest_model_cluster = [
            (best_models[i], closest_clusters[best_models[i]][i]) for i in range(len(X))
        ]

        if return_labels:
            return closest_model_cluster, get_most_occurring_by_column(labels)
        return closest_model_cluster

    def _novelty_detect(self):
        if self.verbose > 1:
            print("Novelty detection started")
        X = self.short_mem.get_all_points()
        new_class_vote = 0

        K0 = round(self.K * (len(X) / self.chunk_size))
        K0 = max(K0, self.K)
        K0 = min(K0, len(X))

        f_microclusters = generate_microclusters(X, np.array([-1] * len(X)), self.sample_counter, K0,
                                                   keep_instances=True, min_samples=0,
                                                   random_state=self.random_state)
        f_centroids = np.array([mc.centroid for mc in f_microclusters])

        potential_novel_clusters_idx = []
        for model in self.models:
            qnscs = qnsc(f_centroids, model.microclusters, self.min_examples_cluster)
            total_instances = 0
            for i, f_microcluster in enumerate(f_microclusters):
                if qnscs[i] > 0:
                    total_instances += f_microcluster.n
                    potential_novel_clusters_idx.append(i)

            if total_instances > self.min_examples_cluster:
                new_class_vote += 1

        if new_class_vote == len(self.models):
            novel_clusters_idx = [idx for idx, count in Counter(potential_novel_clusters_idx).items() if count == len(self.models)]
            return [f_microclusters[i] for i in novel_clusters_idx]

        return None

    def _filter_buffer(self):
        if len(self.short_mem) == 0:
            return
        closest_model_cluster = self._majority_voting(self.short_mem.get_all_points(), False)

        for i, instance in enumerate(self.short_mem.get_all_instances()):
            closest_cluster = self.models[closest_model_cluster[i][0]].microclusters[closest_model_cluster[i][1]]

            if (self.sample_counter - instance.timestamp > self.chunk_size) or \
               (closest_cluster.distance_to_centroid(instance.point) <= closest_cluster.max_distance):
                self._remove_sample_from_short_mem(self.short_mem.index(instance))

    def _remove_sample_from_short_mem(self, index):
        if self.short_mem.get_instance(index) is None:
            return
        self.short_mem.remove(index)

