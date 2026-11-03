import re
import numpy as np
from tabulate import tabulate
from scipy.stats import rankdata

Q_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
             8: 3.031, 9: 3.102, 10: 3.164}

datasets = ['results/e_compare_real_covertype.npz',
            'results/e_compare_real_kdd.npz',
            'results/e_compare_real_pamap2.npz']

metrics = ['known_balanced_accuracy',
           'known_vs_novel_balanced_accuracy',
           'novel_ari']

methods_mask = np.array([1,0,1,1,1,1,1,1,1]).astype(bool)

data = [np.load(ds) for ds in datasets]
methods_names = np.array([re.sub('^ORCA', 'ORCCA', str(n)) for n in data[0]['names'][methods_mask]])   # results saved under the old name
print(methods_names)

for metric_id in range(3):

    rows = []
    ranks = []
    for ds_id, ds in enumerate(datasets):
        row = [ds.split('_')[-1].split('.')[0]]

        values = data[ds_id][metrics[metric_id]][methods_mask]
        print(values.shape)  # 8 (methods)
        for m_id in range(len(methods_names)):
            v = values[m_id]
            row.append('---' if np.isnan(v) else np.round(v, 3))
        rows.append(row)

        mask = np.isnan(values)==False
        r = np.full(len(values), np.nan)
        r[mask] = rankdata(-values[mask])
        ranks.append(r)

    mean_rank = np.nanmean(ranks, axis=0)
    # CD = q_alpha * sqrt(k (k + 1) / (6 N)); k - ranked methods, N - ranked streams
    k, n = int(np.sum(~np.isnan(mean_rank))), len(ranks)
    cd = Q_NEMENYI[k] * np.sqrt(k * (k + 1) / (6.0 * n)) if k >= 2 else np.nan
    rows.append(['mean rank (CD=%.2f)' % cd] + ['---' if np.isnan(x) else np.round(x, 2) for x in mean_rank])

    tab = tabulate(rows, headers=['stream'] + list(methods_names), tablefmt='latex')
    print(tab)
    with open('tables/real_metric_%i.tex' % metric_id, 'w') as file:
        file.write(tab)
