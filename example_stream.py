"""Example figure: a 2D stream with 2 known classes, 4 novel ones and 2 drifts."""

import numpy as np
import matplotlib.pyplot as plt

from methods.OWDSG import OWDSG
from methods import streams as S

plt.rcParams['font.family'] = 'monospace'
col = plt.cm.coolwarm([0., 1., 0.58])
SHOWN_CHUNKS = (10, 30, 50, 70, 90)

stream = OWDSG(n_drifts=2, n_novel=4, n_features=2, n_informative=2, n_classes=2,
               weights=S.weights_for(2), n_chunks=100, chunk_size=400, n_clusters_per_class=1,
               class_sep=2.0, percentage_novel=0.3, random_state=626,
               even_gt=True, hide_label=True, smooth_drift=True, drift_magnitude=1)
drifts = [int(d) for d in stream.get_drift_gt()]
novelties = [int(n) for n in stream.get_novel_gt()]
chunks = [tuple(np.asarray(a) for a in stream.get_chunk(i)) for i in range(100)]

# group: 0, 1 known classes, 2 any novel class
counts = np.array([[np.sum(np.minimum(y, 2) == k) for _, y in chunks] for k in range(3)])

fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(2, len(SHOWN_CHUNKS), height_ratios=[1.3, 1])
top = [fig.add_subplot(gs[0, j]) for j in range(len(SHOWN_CHUNKS))]
for ax in top[1:]:
    ax.sharex(top[0])
    ax.sharey(top[0])

for j, (ax, ch) in enumerate(zip(top, SHOWN_CHUNKS)):
    X, y = chunks[ch]
    g = np.minimum(y, 2)
    for k in (2, 0, 1):
        ax.scatter(X[g == k, 0], X[g == k, 1], s=4, color=col[k])
    ax.set_title('chunk %i' % ch)
    ax.grid(ls=':')
    ax.spines[['top', 'right']].set_visible(False)
    if j > 0:
        ax.tick_params(labelleft=False)
for k, label in enumerate(('0', '1', 'N')):
    top[0].scatter([], [], s=12, color=col[k], label=label)
top[0].legend(frameon=False, loc='upper left', ncol=1)

x0, x1 = top[0].get_xlim()
y0, y1 = top[0].get_ylim()
top[0].set_xlim(x0 - 2, x1)
top[0].set_ylim(y0, y1 + 1.5)

ax = fig.add_subplot(gs[1, :])
ax.stackplot(np.arange(100), counts, colors=col, step='post')

for d in drifts:
    ax.axvline(d, color='k', ls=':', lw=1.2)
    ax.text(d + 0.5, 400 * 1.02, 'drift', fontsize=9, va='bottom')
ax.set_xlim(0, 99)
ax.set_ylim(0, 400)
ax.set_xticks(novelties + drifts)
ax.set_xlabel('chunk index')
ax.set_ylabel('# samples')
ax.grid(ls=':')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('figs/stream_vis.png')
plt.savefig('figs/stream_vis.pdf')
