"""Example figure: how ORCCA separates known classes. """

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from methods import streams as S
from methods.ORCCA import ORCCA

plt.rcParams['font.family'] = 'monospace'
col = plt.cm.coolwarm([0., 1.])                    # class 0, class 1

S.weights_for = lambda n: [1.0 / n] * n            # equal class proportions
stream_x, stream_y, _, _ = S.make_stream('tabular', 0, class_sep=2.0, n_known=2, n_novel=0, random_state=887)
cfg = S.config(2, 0)
offline, chunk = cfg['offline_size'], 50
orcca = ORCCA(architecture=S.make_architecture('tabular'), buffer_size=offline)
xy = PCA(n_components=2).fit_transform(stream_x.numpy())

# label -1 = unknown
end = 21 * chunk
labels = np.full(len(stream_y), -1)
for i in range(end):
    is_novel, model_id = orcca.process(stream_x[i].reshape(1, -1), stream_y[i] if i < offline else None)
    if i >= offline and not is_novel:
        labels[i] = model_id

# normalized errors of the last 3 chunks in both known models
idx = np.arange(end - 3 * chunk, end)
X, y = stream_x[idx], stream_y[idx]
known = orcca.models[:orcca.n_known]
Z = torch.stack([orcca._errors(m['ae'], X) / m['std'] - m['mean'] for m in known], dim=1).numpy()
thresholds = [float(m['threshold'] - m['mean']) for m in known]
grid = np.linspace(-6, 30, 800)
kde = [np.exp(m['kde'].score_samples((grid + float(m['mean'])).reshape(-1, 1))) for m in known]
z_lo, z_hi = Z.min() - 0.2, Z.max() + 0.2

fig, ax = plt.subplots(1, 4, figsize=(14, 4))

for k in range(2):
    a = ax[k]
    bins = np.linspace(z_lo, z_hi, 35)
    a.hist(Z[y == k, k], bins=bins, color='k', label='class %i' % k)
    a.hist(Z[y != k, k], bins=bins, color=col[0], alpha=0.8, label='class %i' % (1 - k))
    t = a.twinx()
    t.plot(grid, kde[k], color=col[1], label='KDE')
    t.axvline(thresholds[k], color='k', ls='--', lw=1, label='threshold')
    a.set_ylim(top=a.get_ylim()[1] * 1.4)
    t.set_ylim(0, t.get_ylim()[1] * 1.4)
    t.spines[:].set_visible(False)
    a.set_xlim(z_lo, z_hi)
    a.set_title('class %i' % k)
    a.set_xlabel('normalized reconstruction error')
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = t.get_legend_handles_labels()
    t.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc='upper left')

a = ax[2]
for c in range(2):
    a.scatter(Z[y == c, 0], Z[y == c, 1], s=10, color=col[c], label='class %i' % c)
    
a.axvline(thresholds[0], color='k', ls='--', lw=1, label='threshold')
a.axhline(thresholds[1], color='k', ls='--', lw=1)

a.set_xlim(z_lo, z_hi)
a.set_ylim(z_lo, z_hi)
a.set_title('error in both models')
a.set_xlabel('error in model 0')
a.set_ylabel('error in model 1')
a.legend(frameon=False, fontsize=7, loc='lower left')

a = ax[3]
a.scatter(xy[:, 0], xy[:, 1], c='0.92', s=3)
for c in range(2):
    a.scatter(xy[idx[labels[idx] == c], 0], xy[idx[labels[idx] == c], 1], s=10, color=col[c], label='class %i' % c)
    
a.scatter(xy[idx[labels[idx] == -1], 0], xy[idx[labels[idx] == -1], 1], s=10, color='k', label='unknown (buffer)')
a.set_ylim(top=a.get_ylim()[1] + 0.3 * np.ptp(a.get_ylim()))

a.set_title('ORCCA label')
a.set_xlabel('PC 1')
a.set_ylabel('PC 2')
a.legend(frameon=False, fontsize=7, loc='upper right')

for a in ax:
    a.grid(ls=':')
    a.set_axisbelow(True)
    a.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('figs/example_knowns.png')
plt.savefig('figs/example_knowns.pdf')
