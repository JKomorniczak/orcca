"""ORCCA with switchable mechanisms for the ablation study (e_ablation_experiment.py).

known_strategy: "contrastive_argmin" (ORCCA), "contrastive" (cascade with the contrastive
thresholds) or "kde" (cascade with KDE density for the known classes too).
adapt_by_error=False: fine-tuning with a constant probability adapt_prob.
"""

import torch

from methods.ORCCA import ORCCA


class ORCCA_ablation(ORCCA):
    def __init__(self, architecture, known_strategy="contrastive_argmin", adapt_by_error=True, **kwargs):
        super().__init__(architecture, **kwargs)
        self.known_strategy = known_strategy
        self.adapt_by_error = adapt_by_error

    def process(self, x, y=None):
        if self.known_strategy == "contrastive_argmin" or self.n_known == 0:
            return super().process(x, y)

        # cascade over all models
        # the first one that recognizes the sample wins
        x = torch.as_tensor(x, dtype=torch.float32).reshape(1, -1)
        errors = []
        for i, m in enumerate(self.models):
            e = self._errors(m["ae"], x)[0] / m["std"]            # scaled
            errors.append(e)
            if i < self.n_known and self.known_strategy == "contrastive":
                recognized = bool(e < m["threshold"])
            else:
                recognized = bool(self._density(m, e.reshape(1))[0] > self.novelty_threshold)
            if recognized:
                if i < self.n_known:
                    self._adapt(m, x, float(e - m["mean"]))
                return (False, i)
        return self._buffer_novelty(x, torch.stack(errors))

    def _adapt(self, m, x, norm_err):
        if self.adapt_by_error:
            return super()._adapt(m, x, norm_err)
        if self.rng.rand() < self.adapt_prob:
            self._finetune(m, x)
