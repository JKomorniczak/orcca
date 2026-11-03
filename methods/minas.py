# source: https://github.com/jgaud/streamndr/blob/main/streamndr/model/minas.py

import numpy as np
from sklearn.cluster import KMeans
from methods.utils import MicroCluster, ShortMem, ShortMemInstance, get_closest_clusters

class Minas():
    """MINAS novelty detection. [1]

    [1] de Faria, Elaine Ribeiro, André Carlos Ponce de Leon Ferreira Carvalho, and Joao Gama. "MINAS: multiclass learning algorithm for novelty detection in data streams." 
    Data mining and knowledge discovery 30 (2016): 640-680.

    Differences from streamndr:
      * micro-clusters keep only the summary (N, LS, SS, T, label), not the examples (as in the paper);
      * radius = radius_factor * sqrt(SS/N - |LS/N|^2) (CluStream definition), fixed at creation;
      * dist_std (std of distances to centroid) stored for the TV1 threshold;
      * explained when distance <= radius (Algorithm 2; streamndr: <).
    """
    MAX_MEMORY_SIZE = 50000

    def __init__(self,
                 kini=3,
                 random_state=None,
                 min_short_mem_trigger=10,
                 min_examples_cluster=10,
                 threshold_strategy=1,
                 threshold_factor=1.1,
                 window_size=1000,
                 unknown_as_novelty_id=True,
                 radius_factor=1.5):
        """unknown_as_novelty_id: label unexplained samples with the id of the next
        created cluster (max label + 1) instead of -1."""
        self.kini = kini
        self.random_state = random_state
        self.radius_factor = radius_factor

        self.microclusters = []

        self.short_mem = ShortMem()
        self.sleep_mem = []
        self.min_short_mem_trigger = min_short_mem_trigger
        self.min_examples_cluster = min_examples_cluster
        self.threshold_strategy = threshold_strategy
        self.threshold_factor = threshold_factor
        self.window_size = window_size
        self.sample_counter = 0
        self.unknown_as_novelty_id = unknown_as_novelty_id

    def fit(self, X, y):
        self.microclusters = self._offline(X, y)
        return self
    
    def predict(self, X):
        closest_clusters, _ = get_closest_clusters(X, [microcluster.centroid for microcluster in self.microclusters])        
        pred_label = []
        
        for i in range(len(closest_clusters)):
            self.sample_counter += 1

            if closest_clusters[i] != -1:
                closest_cluster = self.microclusters[closest_clusters[i]]

                if closest_cluster.distance_to_centroid(X[i]) <= closest_cluster.radius:
                    pred_label.append(closest_cluster.label)

                    closest_cluster.update_cluster(X[i], self.sample_counter)

                else:
                    # label before _label_as_unknown, which may create clusters
                    pred_label.append(self._unknown_label())
                    self._label_as_unknown(X[i])

            else:
                pred_label.append(self._unknown_label())
                self._label_as_unknown(X[i])

   
        if self.sample_counter % self.window_size == 0:
            self._trigger_forget()

        return np.array(pred_label)
    
    def _novelty_label(self):
        if not self.microclusters:
            return 0
        return max(cluster.label for cluster in self.microclusters) + 1

    def _unknown_label(self):
        return self._novelty_label() if self.unknown_as_novelty_id else -1

    def _label_as_unknown(self, X):
        self.short_mem.append(ShortMemInstance(X, self.sample_counter))
        
        if len(self.short_mem) >= self.min_short_mem_trigger:
            self._novelty_detect()

    def _summarize(self, cluster):
        points = np.asarray(cluster.instances)
        cluster.dist_std = float(np.std(cluster.distance_to_centroid(points)))
        diff = cluster.squared_sum / cluster.n - np.dot(cluster.centroid, cluster.centroid)
        cluster.radius = self.radius_factor * np.sqrt(diff) if diff > 1e-15 else 0.0
        cluster.instances = None
        return cluster

    def _offline(self, X_train, y_train):
        microclusters = []
        timestamp = len(X_train)
        
        for y_class in np.unique(y_train):
            X_class = X_train[y_train == y_class]

            class_cluster_clf = KMeans(n_clusters=self.kini, n_init='auto',
                                        random_state=self.random_state)
            class_cluster_clf.fit(X_class)
            labels = class_cluster_clf.labels_

            for class_cluster in np.unique(labels):
                cluster_instances = X_class[labels == class_cluster]

                microclusters.append(
                    self._summarize(MicroCluster(y_class, cluster_instances, timestamp))
                )

        return microclusters

    def _novelty_detect(self):
        possible_clusters = []
        X = self.short_mem.get_all_points()
        K0 = min(self.kini, len(X))

        cluster_clf = KMeans(n_clusters=K0, n_init='auto',
                                random_state=self.random_state)
        cluster_clf.fit(X)
        labels = cluster_clf.labels_

        for cluster_label in np.unique(labels):
            cluster_instances = X[labels == cluster_label]
            possible_clusters.append(
                MicroCluster(-1, cluster_instances, self.sample_counter))
        
        
        for cluster in possible_clusters:
            if cluster.is_cohesive(self.microclusters) and cluster.is_representative(self.min_examples_cluster):
                closest_cluster = cluster.find_closest_cluster(self.microclusters)
                closest_distance = cluster.distance_to_centroid(closest_cluster.centroid)

                threshold = self._best_threshold(closest_cluster, self.threshold_strategy)

                if closest_distance <= threshold:  # extension
                    cluster.label = closest_cluster.label
                    
                elif self.sleep_mem:
                    closest_cluster = cluster.find_closest_cluster(self.sleep_mem)
                    closest_distance = cluster.distance_to_centroid(closest_cluster.centroid)
                    
                    if closest_distance <= threshold:  # reactivate from sleep memory
                        cluster.label = closest_cluster.label
                        self.sleep_mem.remove(closest_cluster)
                        closest_cluster.timestamp = self.sample_counter
                        self.microclusters.append(closest_cluster)
                        
                    else:  # novelty
                        cluster.label = max([cluster.label for cluster in self.microclusters]) + 1   
                else:  # novelty
                    cluster.label = max([cluster.label for cluster in self.microclusters]) + 1
                    
                for point in cluster.instances:
                    index = self.short_mem.index(np.array(point))
                    self.short_mem.remove(index)

                self.microclusters.append(self._summarize(cluster))
                    

    def _best_threshold(self, closest_cluster, strategy):
        def run_strategy_1():
            factor_1 = self.threshold_factor
            return factor_1 * closest_cluster.dist_std

        if strategy == 1:
            return run_strategy_1()
        else:
            factor_2 = factor_3 = self.threshold_factor
            clusters_same_class = self._get_clusters_in_class(closest_cluster.label)
            if len(clusters_same_class) == 1:
                return run_strategy_1()
            else:
                class_centroids = np.array([cluster.centroid for cluster in clusters_same_class])
                distances = closest_cluster.distance_to_centroid(class_centroids)
                if strategy == 2:
                    return factor_2 * np.max(distances)
                elif strategy == 3:
                    return factor_3 * np.mean(distances)

    def _get_clusters_in_class(self, label):
        return [cluster for cluster in self.microclusters if cluster.label == label]

    def _trigger_forget(self):
        for cluster in list(self.microclusters):
            if (cluster.timestamp < self.sample_counter - self.window_size) and (len(self.microclusters) > 1):
    
                self.sleep_mem.append(cluster)
                self.microclusters.remove(cluster)

        for instance in self.short_mem.get_all_instances():
            if instance.timestamp < self.sample_counter - self.window_size:
                self.short_mem.remove(self.short_mem.index(instance))