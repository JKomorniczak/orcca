import re
import numpy as np
from tabulate import tabulate

datasets = ['results/e_ablation_real_covertype.npz']

# ['known_balanced_accuracy', 'known_vs_novel_balanced_accuracy', 'novel_ari', 'names']

metrics = ['known_balanced_accuracy',
           'known_vs_novel_balanced_accuracy',
           'novel_ari']

for ds in datasets:
    data = np.load(ds)

    variants_names = np.array([re.sub('^ORCA', 'ORCCA', str(n)) for n in data['names']])
    # ['ORCCA' 'no_adaptation' 'adaptation_no_weighting' 'no_split' 'strategy_contrastive' 'strategy_kde']
    print(data['known_balanced_accuracy'].shape) #(6) = variants
    
    rows = []
    for metric_id in range(3):
        row = ["covertype metric %i" % metric_id]
        vals = data[metrics[metric_id]]
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
    with open('tables/ablation_%s.tex' % (ds.split('_')[-1].split('.')[0]), 'w') as file:
        file.write(tab)

