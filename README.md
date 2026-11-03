# ORCCA — Open-world Reconstruction-based Classifier with Concept Drift Adaptation

### Abstract
Data stream processing has become a landmark in modern machine learning applications, with concept drifts and novel class appearances posing the primary challenges faced by recognition methods. Adapting to concept drifts while also being able to detect novel classes and distinguish groups of unknown observations can be seen as a step towards effective method employment in an open-world setting. This work proposes a classification method that can detect and learn the distributions of new, unknown classes, as well as adapt to concept drifts once the distribution of known classes shifts. The proposed method uses the autoencoders' reconstruction errors and the KDE that models their distribution. The autoencoders incrementally adapt to the changing distributions of known classes via weight fine-tuning, resulting in continuous adjustment to evolving concepts. Conducted experiments used a diverse set of synthetic data streams, embeddings of a real-world static CIFAR dataset arranged into a stream to simulate concept drifts and the emergence of unknowns, as well as real-world datasets where novelty is observed. The methodology was compared with state-of-the-art approaches with three-criteria recognition quality assessment and resource consumption evaluation. The results indicate that the proposed method is competitive with reference approaches in stationary environments and shows as the best when concept drifts are involved.

### Repository

| path | content |
|---|---|
| `methods/ORCCA.py` | ORCCA method |
| `methods/ORCCA_ablation.py` | ORCCA with switchable mechanisms, used only in ablation experiments |
| `methods/OWDSG.py`, `methods/streams.py` | stream generator |
| `methods/evaluation.py` | three evaluation criteria |
| `methods/compare.py` | stream grid, method settings |
| `methods/cost.py` | time and state-size measurement |
| `methods/embeddings.py`, `methods/real_data.py` | tools for generating CIFAR-10 ResNet-18 embeddings, loading Covertype, KDD '99, PAMAP2 |
| `methods/minas.py`, `ecsminerwf.py`, `echo.py`, `stream_clustering.py`, `baselines.py` | reference methods, based on `streamndr` and `river` libraries |
| `e_*.py` | experiments |
| `analyze_*.py` | analysis of the results |
| `results` | results of experiments as `.npz` |
| `figs`, `tables` | figures and tables based on results|
