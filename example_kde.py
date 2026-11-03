"""Example figure: normalized errors, KDE responses and ORCCA decisions."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from methods import streams as S
from methods.ORCCA import ORCCA

plt.rcParams['font.family'] = 'monospace'
col = plt.cm.coolwarm([0., 1.])
SHOW_CHUNKS = (20, 32)

stream_x, stream_y, _, _ = S.make_stream('tabular', 0, class_sep=2.0, n_known=2, n_novel=2, random_state=887)

cfg = S.config(2, 2)
offline, chunk = cfg['offline_size'], 50
orcca = ORCCA(architecture=S.make_architecture('tabular'), buffer_size=offline)
xy = PCA(n_components=2).fit_transform(stream_x.numpy())

grid = np.linspace(-6, 30, 800)
snapshots = {}
decisions = np.full(len(stream_y), -1) # 1 unknown

for i in range(len(stream_y)):
    is_novel, _ = orcca.process(stream_x[i].reshape(1, -1), stream_y[i] if i < offline else None)
    if i >= offline:
        decisions[i] = int(is_novel)
    c = (i + 1) // chunk - 1
    if (i + 1) % chunk == 0 and c in SHOW_CHUNKS:
        idx = np.arange((c - 2) * chunk, (c + 1) * chunk)
        E = torch.stack([orcca._errors(m['ae'], stream_x[idx]) / m['std'] for m in orcca.models], dim=1)
        known = orcca.models[:orcca.n_known]
        Z = E[:, :orcca.n_known] - torch.stack([m['mean'] for m in known])
        density = torch.stack([orcca._density(m, E[:, j]) for j, m in enumerate(orcca.models)], dim=1)
        snapshots[c] = dict(
            idx=idx,
            z=Z[torch.arange(len(idx)), torch.argmin(Z, dim=1)].numpy(),   # best known model
            max_density=density.max(dim=1).values.numpy(),
            kde=[np.exp(m['kde'].score_samples((grid + float(m['mean'])).reshape(-1, 1))) for m in known],
        )

fig, ax = plt.subplots(2, 3, figsize=(11, 6.6))
z_lo = min(np.percentile(s['z'], 1) for s in snapshots.values())
z_hi = max(s['z'].max() for s in snapshots.values()) + 0.2
vmax = max(1e-6, max(s['max_density'].max() for s in snapshots.values()))

for row, (c, title) in enumerate(zip(SHOW_CHUNKS, ('before novelty', 'novel class appears'))):
    s = snapshots[c]
    known = stream_y[s['idx']] < 2

    a = ax[row, 0]
    bins = np.linspace(z_lo, z_hi, 35)
    a.hist(s['z'][known], bins=bins, color='k', label='known classes')
    a.hist(s['z'][~known], bins=bins, color=col[0], alpha=0.8, label='novel class')
    a.set_ylabel(title)
    t = a.twinx()
    
    for k, (r, color) in enumerate(zip(s['kde'], (col[1], 'tomato'))):
        t.plot(grid, r, color=color, label='KDE class %i' % k)
    t.set_ylim(bottom=0)
    t.set_yticks([])
    t.spines[:].set_visible(False)
    a.spines['right'].set_visible(False)
    a.set_xlim(z_lo, z_hi)
    
    if row == 0:
        a.set_title('error and KDE response')
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = t.get_legend_handles_labels()
        t.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc='upper right')
    else:
        a.set_xlabel('normalized reconstruction error')

    a = ax[row, 1]
    a.scatter(xy[:, 0], xy[:, 1], c='0.92', s=3)
    a.scatter(xy[s['idx'], 0], xy[s['idx'], 1], c=s['max_density'], s=12, cmap='coolwarm',
              vmin=0, vmax=vmax)
    if row == 0:
        a.set_title('highest KDE density')

    a = ax[row, 2]
    a.scatter(xy[:, 0], xy[:, 1], c='0.92', s=3)
    d = decisions[s['idx']]
    for k, (label, color) in enumerate((('recognized as known', col[1]), ('unknown (buffer)', col[0]))):
        a.scatter(xy[s['idx'][d == k], 0], xy[s['idx'][d == k], 1], s=12, color=color,
                  label=label if row == 0 else None)
    if row == 0:
        a.set_title("ORCCA label")
        a.legend(frameon=False, fontsize=8, loc='upper right')

for a in ax.ravel():
    a.grid(ls=':')
    a.set_axisbelow(True)
    a.spines['top'].set_visible(False)
for a in ax[:, 1:].ravel():
    a.spines['right'].set_visible(False)
    a.set_ylabel('PC 2')
for a in ax[-1, 1:]:
    a.set_xlabel('PC 1')

plt.tight_layout()
plt.savefig('figs/example_kde.png', dpi=150)
plt.savefig('figs/example_kde.pdf')
