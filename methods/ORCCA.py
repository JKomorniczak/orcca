"""ORCCA - Open-world Reconstruction-based Classifier with drift Adaptation."""

import copy
from collections import deque

import numpy as np
import torch
from scipy.stats import kurtosis, skew
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity


class ORCCA:
    def __init__(self, architecture, n_epochs=150, novelty_threshold=0.275, fallback=2.5,
                 buffer_size=150, bimodality_threshold=5.0/9.0, random_state=42,
                 adapt_prob=0.6, adapt_window=200):
        self.n_epochs = n_epochs
        self.novelty_threshold = novelty_threshold
        self.fallback = fallback
        self.buffer_size = buffer_size
        self.bimodality_threshold = bimodality_threshold
        self.random_state = random_state
        self.adapt_prob = adapt_prob
        self.adapt_window = adapt_window

        self.ae = architecture # novelty models continue from its weights
        # per model: ae, kde, std, mean (scaled error), threshold, opt, hist (normalized errors)
        self.models = []
        self.n_known = 0  # the first n_known models are the known classes
        self.buffer, self.buffer_errors, self.labels = [], [], []
        self.rng = np.random.RandomState(random_state)

    def _errors(self, ae, X):
        """Relative reconstruction error: L1 / L1 of the input."""
        with torch.no_grad():
            l1_error = (X - ae(X)).abs().sum(1)
        return l1_error / X.abs().sum(1)

    def _density(self, m, e):
        """One-sided KDE density: errors below the class mean are clipped to the mean."""
        e = torch.maximum(e, m["mean"]).numpy().reshape(-1, 1)
        return torch.as_tensor(m["kde"].score_samples(e), dtype=torch.float32).exp()

    def _add_model(self, X, negatives=None):
        """Trains self.ae further on X and adds a calibrated copy to the ensemble."""
        X_tr, X_val = train_test_split(X, test_size=0.3, random_state=self.random_state)
        opt = torch.optim.Adam(self.ae.parameters(), lr=1e-3)
        for _ in range(self.n_epochs):
            opt.zero_grad()
            torch.nn.MSELoss()(self.ae(X_tr), X_tr).backward()
            opt.step()
        ae = copy.deepcopy(self.ae)

        e = self._errors(ae, X_val)
        std = e.std(unbiased=False)
        e = e / std
        mean = e.mean()

        # contrastive threshold between own and other known classes errors
        threshold = mean + self.fallback
        if negatives is not None and len(negatives) >= 2:
            neg = self._errors(ae, negatives) / std
            if neg.mean() > mean:
                s_own, s_neg = e.std(unbiased=False), neg.std(unbiased=False)
                threshold = (mean * s_neg + neg.mean() * s_own) / (s_own + s_neg)

        self.models.append(dict(ae=ae, kde=KernelDensity(bandwidth="scott").fit(e.numpy().reshape(-1, 1)),
                                std=std, mean=mean, threshold=threshold, opt=None,
                                hist=deque(maxlen=self.adapt_window)))

    def _startup(self, x, y):
        """Collects the labeled startup phase, then one model per known class."""
        self.buffer.append(x.clone())
        self.labels.append(y)
        if len(self.buffer) == self.buffer_size:
            X, labels = torch.cat(self.buffer), torch.as_tensor(self.labels)
            untrained = copy.deepcopy(self.ae)
            for c in torch.unique(labels):
                self.ae = copy.deepcopy(untrained)
                self._add_model(X[labels == c], X[labels != c])
            self.n_known = len(self.models)
            self.buffer, self.labels = [], []
        return (False, 0)

    def _buffer_novelty(self, x, errors):
        """Unrecognized sample -> buffer; a full buffer becomes a new model."""
        self.buffer.append(x.clone())
        self.buffer_errors.append(errors)
        novelty_id = len(self.models)
        if len(self.buffer) >= self.buffer_size:
            self._learn_novelty()
        return (True, novelty_id)

    def process(self, x, y=None):
        """-> (is_novelty, model index); `y` required only in the startup phase."""
        x = torch.as_tensor(x, dtype=torch.float32).reshape(1, -1)
        if self.n_known == 0:
            return self._startup(x, y)

        # known classes: lowest normalized error among those below threshold
        known = self.models[:self.n_known]
        errors = torch.stack([self._errors(m["ae"], x)[0] / m["std"] for m in known])
        norm_err = errors - torch.stack([m["mean"] for m in known])
        passing = (errors < torch.stack([m["threshold"] for m in known])).nonzero(as_tuple=True)[0]
        if len(passing) > 0:
            best = int(passing[int(torch.argmin(norm_err[passing]))])
            self._adapt(self.models[best], x, float(norm_err[best]))
            return (False, best)

        # novelty models: cascade
        for i in range(self.n_known, len(self.models)):
            m = self.models[i]
            e = self._errors(m["ae"], x)[0] / m["std"]
            errors = torch.cat([errors, e.reshape(1)])
            if bool(self._density(m, e.reshape(1))[0] > self.novelty_threshold):
                return (False, i)
        return self._buffer_novelty(x, errors)

    def _adapt(self, m, x, norm_err):
        """Random fine-tuning step of a known-class model, favouring a low normalized error."""
        m["hist"].append(norm_err)
        p = self.adapt_prob
        if len(m["hist"]) >= 20:
            h = np.asarray(m["hist"])
            g = np.sum(h > norm_err) / (len(h) - 1)     # fraction of recent samples with a larger error
            p = min(1.0, 2.0 * g * self.adapt_prob)     # mean p stays adapt_prob
        if self.rng.rand() < p:
            self._finetune(m, x)

    def _finetune(self, m, x):
        """One Adam step on x; calibration stays frozen."""
        if m["opt"] is None:
            m["opt"] = torch.optim.Adam(m["ae"].parameters(), lr=1e-4)
        m["opt"].zero_grad()
        torch.nn.MSELoss()(m["ae"](x), x).backward()
        m["opt"].step()

    def _learn_novelty(self):
        """New model from the buffer, or from its larger mode when the errors are bimodal."""
        X, E = torch.cat(self.buffer), torch.stack(self.buffer_errors) # E: (n, n_models)

        # Sarle's bimodality coefficient per model
        n, Ea = len(E), E.numpy()
        S, K = skew(Ea, axis=0, bias=True), kurtosis(Ea, axis=0, bias=True)
        bc = torch.as_tensor((S * S + 1.0) / (K + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))), dtype=torch.float32)
        major = None
        if float(bc.max()) > self.bimodality_threshold:
            try:
                labels = GaussianMixture(2, random_state=self.random_state).fit_predict(
                    Ea[:, (bc > self.bimodality_threshold).numpy()])
                major = torch.as_tensor(labels == np.bincount(labels, minlength=2).argmax())
            except ValueError: # identical errors
                pass
        if major is None or bool(major.all()):
            self._add_model(X)
            self.buffer, self.buffer_errors = [], []
            return

        self._add_model(X[major])
        rest = X[~major]
        E = torch.stack([self._errors(m["ae"], rest) / m["std"] for m in self.models], dim=1)
        unknown = self._density(self.models[-1], E[:, -1]) <= self.novelty_threshold
        self.buffer = list(rest[unknown].split(1))
        self.buffer_errors = list(E[unknown])
