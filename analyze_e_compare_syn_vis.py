import re
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'monospace'

datasets = ['results/e_compare_syn_tabular.npz', 
            'results/e_compare_syn_cifar.npz']
# ['known_balanced_accuracy', 'known_vs_novel_balanced_accuracy', 'novel_ari',
#  'names', 'splits', 'n_drift_values', 'random_states', 'class_sep', 'class_orders']

fig, ax = plt.subplots(2,2,figsize=(8,8), sharex=1, sharey=1)

methods_mask = np.array([1,0,1,1,1,1,1,1,1]).astype(bool)
for ds_id, ds in enumerate(datasets):
    data = np.load(ds)
    print(data.files)
    methods_names = np.array([re.sub('^ORCA', 'ORCCA', str(n)) for n in data['names'][methods_mask]])   # results saved under the old name
    print(methods_names)
    print(data['known_balanced_accuracy'].shape) # 10 reps
    
    mean_c1 = np.mean(data['known_balanced_accuracy'], axis=-1)
    print(mean_c1.shape) # 9 (methods) x 3 (splits (2/8, 4/6, 6/2)) x 5 (drifts: data['n_drift_values'])
    mean_c1 = mean_c1[methods_mask]
    print(mean_c1.shape) 
    
    mean_c2 = np.mean(data['known_vs_novel_balanced_accuracy'], axis=-1)  # 9 (methods) x 3 (splits (2/8, 4/6, 6/2)) x 5 (drifts: data['n_drift_values'])
    mean_c2 = mean_c2[methods_mask]
    print(mean_c2.shape) 
    
    mean_c3 = np.mean(data['novel_ari'], axis=-1)  # 9 (methods) x 3 (splits (2/8, 4/6, 6/2)) x 5 (drifts: data['n_drift_values'])
    mean_c3 = mean_c3[methods_mask]
    print(mean_c3.shape) 
    
    cols = ['r', 
            'cornflowerblue','cornflowerblue','cornflowerblue',
            'brown','brown',
            'forestgreen', 'forestgreen']
    
    markers=['x',
             '^', 'o', 's',
             '^', 'o',
             '^', 'o',]
    
    for n_splist_id in range(len(data['splits'])):
        n_known, n_novel = data['splits'][n_splist_id]
        for n_drift_id, n_drifts in enumerate(data['n_drift_values']):
            
            for m_id in range(8):  
                if n_splist_id+n_drift_id+ds_id == 0: 
                    ax[0, ds_id].scatter(mean_c1[m_id, n_splist_id, n_drift_id], mean_c2[m_id, n_splist_id, n_drift_id], color=cols[m_id], label=methods_names[m_id], s=25, marker=markers[m_id])
              
                # c1 x c2
                ax[0, ds_id].scatter(mean_c1[m_id, n_splist_id, n_drift_id], mean_c2[m_id, n_splist_id, n_drift_id], color=cols[m_id], s=25, marker=markers[m_id])
                # c2 x c3
                ax[1, ds_id].scatter(mean_c2[m_id, n_splist_id, n_drift_id], mean_c3[m_id, n_splist_id, n_drift_id], color=cols[m_id], s=25,  marker=markers[m_id])

    ax[0, ds_id].set_title(ds.split('_')[-1].split('.')[0])

    ax[0, ds_id].set_xlabel('Criterion 1 (KC BAC)')
    ax[1, ds_id].set_xlabel('Criterion 2 (KC/UC BAC)')
ax[0, 0].set_ylabel('Criterion 2 (KC/UC BAC)')
ax[1, 0].set_ylabel('Criterion 3 (UC ARI)')

for aa in ax.ravel():
    aa.grid(ls=':')
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)

handles, labels = ax[0, 0].get_legend_handles_labels()
blank = plt.Line2D([], [], linestyle='')
groups = [[0], [1, 2, 3], [4, 5], [6, 7]]
n_rows = max(len(g) for g in groups)
handles = [h for g in groups for h in [handles[i] for i in g] + [blank] * (n_rows - len(g))]
labels = [l for g in groups for l in [labels[i] for i in g] + [''] * (n_rows - len(g))]
fig.legend(handles, labels, loc='lower center', ncols=len(groups), frameon=False)

plt.tight_layout(rect=(0, 0.09, 1, 1))
plt.savefig('foo.png')
plt.savefig('figs/compare-syn-vis.png')
plt.savefig('figs/compare-syn-vis.pdf')