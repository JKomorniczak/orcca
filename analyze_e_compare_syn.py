import re
import numpy as np
from tabulate import tabulate
from scipy.stats import rankdata

Q_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
             8: 3.031, 9: 3.102, 10: 3.164}

datasets = ['results/e_compare_syn_tabular.npz', 
            'results/e_compare_syn_cifar.npz']
# ['known_balanced_accuracy', 'known_vs_novel_balanced_accuracy', 'novel_ari',
#  'names', 'splits', 'n_drift_values', 'random_states', 'class_sep', 'class_orders']

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
    
    for metric_id in range(3):
        
        rows = []
        ranks = []
        for n_splist_id in range(len(data['splits'])):
            n_known, n_novel = data['splits'][n_splist_id]
            for n_drift_id, n_drifts in enumerate(data['n_drift_values']):
                row = ["K%i | N%i | D%i" % (n_known, n_novel, n_drifts)]
                
                for m_id in range(8):
                    aa = [mean_c1, mean_c2, mean_c3][metric_id]
                    print(aa.shape)
                    aaa = aa[m_id, n_splist_id, n_drift_id]
                    print(aaa.shape)
                    row.append('---' if np.isnan(aaa) else np.round(aaa,3))
                rows.append(row)

                vals = [mean_c1, mean_c2, mean_c3][metric_id][:, n_splist_id, n_drift_id]
                mask = np.isnan(vals)==False
                
                r = np.full(len(vals), np.nan)
                r[mask] = rankdata(-vals[mask])
                ranks.append(r)

        mean_rank = np.nanmean(ranks, axis=0)
        # CD = q_alpha * sqrt(k (k + 1) / (6 N)); k - ranked methods, N - ranked streams
        k, n = int(np.sum(~np.isnan(mean_rank))), len(ranks)
        cd = Q_NEMENYI[k] * np.sqrt(k * (k + 1) / (6.0 * n)) if k >= 2 else np.nan
        rows.append(['mean rank (CD=%.2f)' % cd] + ['---' if np.isnan(x) else np.round(x, 2) for x in mean_rank])

        tab = tabulate(rows, headers=['stream'] + list(methods_names), tablefmt='latex')
        with open('tables/data_%s_metric_%i.tex' % (ds.split('_')[-1].split('.')[0], metric_id), 'w') as file:
            file.write(tab)
    
    
   