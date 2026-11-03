import re
import numpy as np
from tabulate import tabulate

datasets = ['results/e_ablation_tabular.npz']

# ['known_balanced_accuracy', 'known_vs_novel_balanced_accuracy', 'novel_ari',
#  'names', 'splits', 'n_drift_values']

metrics = ['known_balanced_accuracy',
           'known_vs_novel_balanced_accuracy',
           'novel_ari']

for ds in datasets:
    data = np.load(ds)
    variants_names = np.array([re.sub('^ORCA', 'ORCCA', str(n)) for n in data['names']])   # results saved under the old name
    # ['ORCCA' 'no_adaptation' 'adaptation_no_weighting' 'no_split' 'strategy_contrastive' 'strategy_kde']
    print(data['known_balanced_accuracy'].shape) #(6, 3, 5, 10) = variants x splits x drifts x reps

    for metric_id in range(3):
        mean = np.mean(data[metrics[metric_id]], axis=-1) # (6, 3, 5) mean over reps

        rows = []
        ranks = []
        for n_splist_id in range(len(data['splits'])):
            n_known, n_novel = data['splits'][n_splist_id]
            for n_drift_id, n_drifts in enumerate(data['n_drift_values']):
                row = ["K%i | N%i | D%i" % (n_known, n_novel, n_drifts)]
                vals = mean[:, n_splist_id, n_drift_id]
                # variant - ORCCA: 
                # positive = the variant is better, 
                # negative = worse
                for v_id in range(len(variants_names)):
                    if np.isnan(vals[v_id]):
                        row.append('---')
                    elif v_id == 0:
                        row.append('%.3f' % vals[0])
                    else:
                        row.append('%+.3f' % (vals[v_id] - vals[0]))
                rows.append(row)

        tab = tabulate(rows, headers=['stream'] + list(variants_names), tablefmt='latex', disable_numparse=True)
        print(tab)
        with open('tables/ablation_%s_metric_%i.tex' % (ds.split('_')[-1].split('.')[0], metric_id), 'w') as file:
            file.write(tab)

