import re
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'monospace'

ds = 'results/e_resources_tabular.npz'
data = np.load(ds)

methods_names = np.array([re.sub('^ORCA', 'ORCCA', str(n)) for n in data['names']])   # results saved under the old name
print(methods_names)
# ['ORCCA' 'MINAS' 'ECSMinerWF' 'ECHO' 'CluStream' 'DenStream' 'OCND' 'CND']

times = np.nanmean(data['block_times'], axis=3) # (8, 3, 3, 40) methods x novel x drifts x blocks
per_sample = times * 1e6 / data['block_sizes']
memory = np.nanmean(data['block_bytes'], axis=3) / 1024

parts = {'start': slice(0, 10),
         'end': slice(29, 40)}

def growth(values):
    start = np.nanmean(values[..., parts['start']], axis=-1)
    end = np.nanmean(values[..., parts['end']], axis=-1)
    return 100 * (end - start) / start

time_growth = growth(per_sample)
memory_growth = growth(memory)
print(time_growth.shape)  # (8, 3, 3)

cols = ['r',
        'cornflowerblue','cornflowerblue','cornflowerblue',
        'brown','brown',
        'forestgreen', 'forestgreen']

markers=['x',
         '^', 'o', 's',
         '^', 'o',
         '^', 'o',]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# one point per method and stream
for m_id in range(len(methods_names)):
    ax.scatter(time_growth[m_id].ravel(), memory_growth[m_id].ravel(), color=cols[m_id],
               marker=markers[m_id], s=30, label=methods_names[m_id])

ax.axhline(0, color='k', lw=0.8, ls=':')
ax.axvline(0, color='k', lw=0.8, ls=':')
ax.set_xlabel('Growth of time per sample [%]')
ax.set_ylabel('Growth of memory [%]')
ax.grid(ls=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
blank = plt.Line2D([], [], linestyle='')
groups = [[0], [1, 2, 3], [4, 5], [6, 7]]
n_rows = max(len(g) for g in groups)
handles = [h for g in groups for h in [handles[i] for i in g] + [blank] * (n_rows - len(g))]
labels = [l for g in groups for l in [labels[i] for i in g] + [''] * (n_rows - len(g))]
fig.legend(handles, labels, loc='lower center', ncols=len(groups), frameon=False)
plt.tight_layout(rect=(0, 0.13, 1, 1))
plt.savefig('foo.png')
plt.savefig('figs/time_memo.png')
plt.savefig('figs/time_memo.pdf')
