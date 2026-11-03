import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

Q_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
             8: 3.031, 9: 3.102, 10: 3.164}


plt.rcParams['font.family'] = 'monospace'
datasets = ['results/e_compare_syn_tabular.npz',
            'results/e_compare_syn_cifar.npz']

fig, ax = plt.subplots(2,2,figsize=(8,8), sharex=1, sharey=1)

methods_mask = np.array([1,0,1,1,1,1,1,1,1]).astype(bool)

cols = ['r',
        'cornflowerblue','cornflowerblue','cornflowerblue',
        'brown','brown',
        'forestgreen', 'forestgreen']

markers=['x',
         '^', 'o', 's',
         '^', 'o',
         '^', 'o',]

def mean_ranks(values):
    v = values.reshape(values.shape[0], -1)
    ranks = []
    for s_id in range(v.shape[1]):
        defined = ~np.isnan(v[:, s_id])
        if defined.any():
            r = np.full(v.shape[0], np.nan)
            r[defined] = rankdata(-v[defined, s_id])
            ranks.append(r)
    mean_rank = np.nanmean(ranks, axis=0)
    
    k, n = int(np.sum(~np.isnan(mean_rank))), len(ranks)
    cd = Q_NEMENYI[k] * np.sqrt(k * (k + 1) / (6.0 * n)) if k >= 2 else np.nan
    return mean_rank, cd

for ds_id, ds in enumerate(datasets):
    data = np.load(ds)
    methods_names = np.array([re.sub('^ORCA', 'ORCCA', str(n)) for n in data['names'][methods_mask]])   # results saved under the old name
    print(methods_names)

    mean_c1 = np.mean(data['known_balanced_accuracy'], axis=-1)[methods_mask] # 8 x 3 x 5
    mean_c2 = np.mean(data['known_vs_novel_balanced_accuracy'], axis=-1)[methods_mask]
    mean_c3 = np.mean(data['novel_ari'], axis=-1)[methods_mask]

    rank_c1, cd_c1 = mean_ranks(mean_c1)
    rank_c2, cd_c2 = mean_ranks(mean_c2)
    rank_c3, cd_c3 = mean_ranks(mean_c3)

    for m_id in range(8):
        if ds_id == 0:
            ax[0, 0].scatter([], [], label=methods_names[m_id], color=cols[m_id], s=40, marker=markers[m_id])

        # c1 x c2
        ax[0, ds_id].scatter(rank_c1[m_id], rank_c2[m_id], color=cols[m_id], s=40, marker=markers[m_id])
        # c2 x c3
        ax[1, ds_id].scatter(rank_c2[m_id], rank_c3[m_id], color=cols[m_id], s=40, marker=markers[m_id])
        # ranges CD
        for row, (x, y, cdx, cdy) in enumerate(((rank_c1[m_id], rank_c2[m_id], cd_c1, cd_c2),
                                                (rank_c2[m_id], rank_c3[m_id], cd_c2, cd_c3))):
            if np.isnan(x) or np.isnan(y):
                continue
            ax[row, ds_id].plot([x - cdx, x + cdx], [y, y], ls='--', lw=1, color=cols[m_id], alpha=0.6)
            ax[row, ds_id].plot([x, x], [y - cdy, y + cdy], ls='--', lw=1, color=cols[m_id], alpha=0.6)

    ax[0, ds_id].set_title('%s data streams' % ds.split('_')[-1].split('.')[0])

    ax[0, ds_id].set_xlabel('Mean rank Criterion 1')
    ax[1, ds_id].set_xlabel('Mean rank Criterion 2')
ax[0, 0].set_ylabel('Mean rank Criterion 2')
ax[1, 0].set_ylabel('Mean rank Criterion 3')

for aa in ax.ravel():
    aa.grid(ls=':')
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
ax[0, 0].set_xlim(8.5, 0.5)
ax[0, 0].set_ylim(8.5, 0.5)

handles, labels = ax[0, 0].get_legend_handles_labels()
blank = plt.Line2D([], [], linestyle='')
groups = [[0], [1, 2, 3], [4, 5], [6, 7]]
n_rows = max(len(g) for g in groups)
handles = [h for g in groups for h in [handles[i] for i in g] + [blank] * (n_rows - len(g))]
labels = [l for g in groups for l in [labels[i] for i in g] + [''] * (n_rows - len(g))]
fig.legend(handles, labels, loc='lower center', ncols=len(groups), frameon=False)

plt.tight_layout(rect=(0, 0.09, 1, 1))
plt.savefig('foo.png')
plt.savefig('figs/compare-syn-vis_rank.png')
plt.savefig('figs/compare-syn-vis_rank.pdf')