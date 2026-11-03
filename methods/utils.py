# source: https://github.com/jgaud/streamndr/blob/main/streamndr/utils/data_structure.py
#         https://github.com/jgaud/streamndr/blob/main/streamndr/utils/cluster_utils.py

import numpy as np
import math
import hashlib
from collections import Counter

from sklearn.cluster import KMeans

class MicroCluster(object):
    """Micro-cluster summary (n, linear_sum, squared_sum, centroid, max_distance)."""

    def __init__(self,
                 label,
                 instances=None,
                 timestamp=0,
                 keep_instances=True,
                 n_label_instances=None  # count of the majority label (ECHO purity)
                 ):

        super().__init__()
        self.label = label

        if instances is not None:
            if not isinstance(instances, list):
                instances_list = instances.tolist()
            else:
                instances_list = instances
                instances = np.array(instances)
            self.instances = instances_list
            self.n = len(instances)
            self.linear_sum = instances.sum(axis=0)

            self.squared_sum = np.square(np.linalg.norm(self.instances, axis=1)).sum()
            self.centroid = self.linear_sum / self.n
            self.max_distance = np.max(self.distance_to_centroid(instances))
            self.mean_distance = np.mean(self.distance_to_centroid(instances))
            self.update_properties()

        else:
            self.instances = None
            self.n = 0
            self.linear_sum = 0
            self.squared_sum = 0
            self.max_distance = 0
            self.mean_distance = 0

        self.timestamp = timestamp
        self.n_label_instances = n_label_instances

        if not keep_instances:
            self.instances = None

    def get_radius(self):
        factor = 1.5
        diff = (self.squared_sum / self.n) - np.dot(self.centroid, self.centroid)
        if diff > 1e-15:
            return factor * np.sqrt(diff)
        else:
            return 0

    def distance_to_centroid(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if len(X.shape) == 1:
            return np.linalg.norm(X - self.centroid)
        else:
            return np.linalg.norm(X - self.centroid, axis=1)

    def find_closest_cluster(self, clusters):
        return min(clusters, key=lambda cl: cl.distance_to_centroid(self.centroid))

    def update_cluster(self, X, timestamp, update_summary=False):
        """update_summary=False: only the timestamp (and stored instances) change."""

        self.timestamp = timestamp

        if self.instances is not None:
            self.instances.append(X)

        if update_summary:
            self.mean_distance = (self.n * self.mean_distance + self.distance_to_centroid(X)) / (self.n + 1)
            self.n += 1
            self.linear_sum = np.sum([self.linear_sum, X], axis=0)
            self.squared_sum = np.sum([self.squared_sum, np.square(X).sum()], axis=0)
            self.update_properties()

    def update_properties(self):
        self.centroid = self.linear_sum / self.n

        if self.instances is not None:
            self.radius = self.get_radius()
            if np.max(self.distance_to_centroid(self.instances)) > self.max_distance:
                self.max_distance = np.max(self.distance_to_centroid(self.instances))

    def is_cohesive(self, clusters):
        """Silhouette > 0; b = distance to the closest centroid, a = std of distances to own centroid."""
        b = self.distance_to_centroid(self.find_closest_cluster(clusters).centroid)
        a = np.std(self.distance_to_centroid(self.instances))
        silhouette = (b - a) / max(a, b)
        return silhouette > 0

    def is_representative(self, min_examples):
        return self.n >= min_examples


class ShortMemInstance:
    """Buffered point with timestamp and optional true/predicted label."""
    def __init__(self, point, timestamp, y_true=None, y_pred=None):
        self.point = point
        self.timestamp = timestamp
        self.y_true = y_true
        self.y_pred = y_pred

    def __eq__(self, other):
        """Compares points only (timestamp ignored)."""
        if isinstance(other, np.ndarray):
            return np.all(self.point == other)
        if isinstance(other, ShortMemInstance):
            return np.all(self.point == other.point)
        return NotImplemented

class ClusterModel:
    """Ensemble member: micro-clusters and the labels it was trained on."""
    def __init__(self, microclusters, labels):
        self.microclusters = microclusters
        self.labels = labels


class ShortMem:
    """Buffer of ShortMemInstances indexed by the hash of the point."""
    def __init__(self):
        self.list = []
        self.dictionary = {}

    def append(self, instance):
        h = hashlib.sha256(instance.point.tobytes()).hexdigest()

        if h in self.dictionary:
            self.list[self.dictionary[h]][1].append(instance)
        else:
            index = len(self.list)
            self.list.append((h, [instance]))
            self.dictionary[h] = index

    def remove(self, index):
        if 0 <= index < len(self.list):
            instance_list = self.list[index][1]

            if len(instance_list) == 1:
                del self.dictionary[self.list[index][0]]
                self.list.pop(index)

                for i in range(index, len(self.list)):
                    self.dictionary[self.list[i][0]] = i
            else:
                instance_list.pop(-1)

    def index(self, instance):
        """Index of the hash bucket, -1 if not found."""
        if isinstance(instance, np.ndarray):
            return self.dictionary.get(hashlib.sha256(instance.tobytes()).hexdigest(), -1)
        elif isinstance(instance, ShortMemInstance):
            return self.dictionary.get(hashlib.sha256(instance.point.tobytes()).hexdigest(), -1)
        return -1

    def get_all_instances(self):
        return [instance for _, instances_list in self.list for instance in instances_list]
    
    def get_instance(self, index):
        if 0 <= index < len(self.list):
            return self.list[index][1][0]
    
    def get_all_points(self):
        return np.array([instance.point for _, instances_list in self.list for instance in instances_list])
    
    def __len__(self):
        # instances, not hash buckets
        return sum(len(instances_list) for _, instances_list in self.list)

# max rows per batch in get_closest_clusters
MAX_MEMORY_SIZE = 50000

def get_closest_clusters(X, centroids):
    """Index of and distance to the closest centroid for each point."""
    if len(centroids) == 0:
        return np.full(len(X), -1), -1
        
    centroids = np.array(centroids)
    norm_dists = np.zeros((X.shape[0],centroids.shape[0]))

    for idx in range(math.ceil(X.shape[0]/MAX_MEMORY_SIZE)):
        sl = slice(idx*MAX_MEMORY_SIZE, (idx+1)*MAX_MEMORY_SIZE)
        norm_dists[sl] = np.linalg.norm(np.subtract(X[sl, :, None], np.transpose(centroids)), axis=1)

    return np.argmin(norm_dists, axis=1), np.amin(norm_dists, axis=1)

def qnsc(pseudopoints, model, q_p=5):
    """q-neighborhood silhouette coefficient. [1]

    [1] Masud, Mohammad, et al. "Classification and novel class detection in concept-drifting data streams under time constraints." 
    IEEE Transactions on knowledge and data engineering 23.6 (2010): 859-874.
    """
    qnscs = []
    cluster_by_label = {}

    for cluster in model:
        if cluster.label not in cluster_by_label:
            cluster_by_label[cluster.label] = []

        cluster_by_label[cluster.label].append(cluster.centroid)

    
    distances = np.linalg.norm(pseudopoints[:, np.newaxis] - pseudopoints, axis=2)

    np.fill_diagonal(distances, np.inf)
    q = min(q_p, len(pseudopoints)-2)
    indices = np.argpartition(distances, q, axis=1)[:, :q] 

    min_distances = np.take_along_axis(distances, indices, axis=1)
    dc_outs = np.mean(min_distances, axis=1)
    
    for i, point in enumerate(pseudopoints):
        dc_out = dc_outs[i]
        dc_q = []
        for _, clusters in cluster_by_label.items():
            dists = []
            for centroid in clusters:
                dists.append(np.linalg.norm(point-centroid))
            
            q = min(q_p, len(dists))
            q_closest = np.partition(dists, q-1)[:q]
            dc_q.append(np.sum(q_closest)/q)

        dcmin_q = np.min(dc_q)
        qnsc = (dcmin_q - dc_out) / max(dcmin_q, dc_out)

        qnscs.append(qnsc)
    return qnscs


def generate_microclusters(X, y, timestamp, K, keep_instances=False, min_samples=0, random_state=None):
    """K micro-clusters (k-means) labelled by majority class; clusters with < min_samples instances are dropped."""
    if K == 0:
        return []
    y = np.asarray(y)

    cluster_labels = KMeans(n_clusters=K, n_init="auto", random_state=random_state).fit(X).labels_

    microclusters = []
    for microcluster in np.unique(cluster_labels):
        cluster_instances = X[cluster_labels == microcluster]
        y_cluster_instances = y[cluster_labels == microcluster]

        values, counts = np.unique(y_cluster_instances, return_counts=True)
        most_common_y = values[np.argmax(counts)]

        if len(cluster_instances) >= min_samples:
            mc = MicroCluster(most_common_y, instances=cluster_instances, timestamp=timestamp,
                               keep_instances=keep_instances, n_label_instances=np.max(counts))
            microclusters.append(mc)

    return microclusters


def check_f_outlier(X, models):
    """F-outlier: outside the decision boundary (max_distance) of every micro-cluster of every model."""
    f_outliers = []
    for point in X:
        f_outlier = True
        for model in models:
            for microcluster in model.microclusters:
                if microcluster.distance_to_centroid(point) <= microcluster.max_distance:
                    f_outlier = False
                    break
            else:
                continue
            break

        f_outliers.append(f_outlier)

    return f_outliers


def get_most_occurring_by_column(l):
    """Majority vote per sample across models, ignoring -1; -1 if no known vote."""
    most_common_values = {}
    for col in zip(*l):
        counts = Counter(val for val in col if val != -1)

        most_common_value = counts.most_common(1)

        if not most_common_value:
            most_common_value = -1
        else:
            most_common_value = most_common_value[0][0]

        most_common_values[len(most_common_values)] = most_common_value

    return [most_common_values[i] for i in range(len(most_common_values))]