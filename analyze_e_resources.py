import re
import numpy as np
from tabulate import tabulate

ds = 'results/e_resources_tabular.npz'

metrics = ['known_balanced_accuracy',
           'known_vs_novel_balanced_accuracy',
           'novel_ari']

# ['block_times', 'block_bytes', 'block_sizes', 'names', 'n_novel_values', 'n_drift_values', 'random_states']

data = np.load(ds)
methods_names = np.array([re.sub('^ORCA', 'ORCCA', str(n)) for n in data['names']])   # results saved under the old name
print(methods_names)
times = np.nanmean(data['block_times'], axis=3)

print(times.shape)
# (8, 3, 3, 40) = methods x novel x drifts x blocks

# time per sample [us] in every block
per_sample = times * 1e6 / data['block_sizes']
print(per_sample.shape)

n_blocks = per_sample.shape[-1]
parts = {'start': slice(0, 10),
         'middle': slice(14, 25),
         'end': slice(29, 40)}

rows = []
for n_novel_id, n_novel in enumerate(data['n_novel_values']):
    for n_drift_id, n_drifts in enumerate(data['n_drift_values']):
        stream = 'N%i | D%i' % (n_novel, n_drifts)
        ps = per_sample[:, n_novel_id, n_drift_id] # (methods, blocks)

        for p, part in parts.items():
            row = ['%s | %s' % (stream, p)]
            for method_id in range(len(methods_names)):
                v = ps[method_id, part]
                row.append('---' if np.isnan(v).all() else np.round(np.nanmean(v), 1))
            rows.append(row)

        # growth from start to end
        row = ['%s growth [%%]' % stream]
        for method_id in range(len(methods_names)):
            start = np.nanmean(ps[method_id, parts['start']])
            end = np.nanmean(ps[method_id, parts['end']])
            row.append('---' if np.isnan(start) or np.isnan(end) else np.round(100 * (end - start) / start, 1))
        rows.append(row)

name = ds.split('_')[-1].split('.')[0]
tab = tabulate(rows, headers=['part'] + list(methods_names), tablefmt='latex')

with open('tables/time_%s.tex' % name, 'w') as file:
    file.write(tab)

#  memory
memory = np.nanmean(data['block_bytes'], axis=3) / 1024
print(memory.shape)
# (8, 3, 3, 40) = methods x novel x drifts x blocks

rows = []
for n_novel_id, n_novel in enumerate(data['n_novel_values']):
    for n_drift_id, n_drifts in enumerate(data['n_drift_values']):
        stream = 'N%i | D%i' % (n_novel, n_drifts)
        mem = memory[:, n_novel_id, n_drift_id]

        for p, part in parts.items():
            row = ['%s | %s' % (stream, p)]
            for method_id in range(len(methods_names)):
                v = mem[method_id, part]
                row.append('---' if np.isnan(v).all() else np.round(np.nanmean(v), 1))
            rows.append(row)

        # growth from start to end
        row = ['%s growth [%%]' % stream]
        for method_id in range(len(methods_names)):
            start = np.nanmean(mem[method_id, parts['start']])
            end = np.nanmean(mem[method_id, parts['end']])
            row.append('---' if np.isnan(start) or np.isnan(end) else np.round(100 * (end - start) / start, 1))
        rows.append(row)

tab = tabulate(rows, headers=['part'] + list(methods_names), tablefmt='latex')

with open('tables/memory_%s.tex' % name, 'w') as file:
    file.write(tab)

