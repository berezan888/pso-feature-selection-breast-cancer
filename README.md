# PSO feature selection for breast cancer diagnosis: replication study

Replication of Kazerani, *Improving Breast Cancer Diagnosis Accuracy by Particle Swarm Optimization
Feature Selection*, Int. J. Comput. Intell. Syst. 17:44 (2024), doi:10.1007/s44196-024-00428-5.

Optimisation problem: choose a subset of features (binary vector, d = 9 / 30 / 33) that maximises
`F(x) = 0.7 * Acc_train + 0.3 * Acc_val` (4-fold CV inside the training part). Solver: PSO
(w = 0.9, c1 = 1, c2 = 2, 20 particles). Datasets: Coimbra, WDBC, WPBC (UCI). Ten classifiers.

## Setup and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests                      # unit tests
python main.py --config configs/default.yaml   # main experiment (resumable, ~30 min on 2 cores)
python main.py --config configs/default.yaml --report-only   # rebuild tables/figures only
```

| config | purpose |
|---|---|
| `configs/default.yaml` | main run: leak-free preprocessing, reduced PSO budget (15 iterations) |
| `configs/paper_budget.yaml` | paper budget: 100 iterations (slow) |
| `configs/paper_order.yaml` | preprocessing fitted on all data before the split, as in the paper's flowchart |
| `configs/smoke.yaml` | 10-second check that everything runs |

## Outputs (`results*/`)

`raw_runs.csv` one row per run and method · `results.csv` mean/median/std/min/max · `paired_tests.csv`
Wilcoxon tests over seeds · `comparison_with_paper.csv` · `selected_features.csv` · `convergence.csv` · `figures/`.

Methods compared on the **same split and the same evaluation budget**: `all_features`, `pso`,
`random_search` (random masks, same number of fitness evaluations as PSO).

## Assumptions and deviations from the paper (state them in the report)

1. Position in [0,1]^d, feature selected if position > 0.5 (the paper does not specify the decoding).
2. Velocity clipped to ±0.5; constant inertia weight 0.9 (Table 9).
3. Outliers = values outside 1.5·IQR, replaced by the mean; missing values replaced by the mean.
4. Preprocessing statistics from the training part only (default) vs. all data (`paper_order.yaml`).
5. Reduced PSO iterations (15 instead of 100) and lighter RF/AdaBoost/MLP to fit the compute budget.
6. WPBC keeps `recurrence_time` as in the paper (33 features), although it is only known after the outcome.
7. Classifier `LR` is linear regression on 0/1 labels with threshold 0.5.
8. Positive class: malignant (WDBC), recurrence (WPBC), patient (Coimbra).
9. Paper tables were typed from the PDF (`paper_results/paper_tables.csv`); verify against the article.

## Tools

Libraries: scikit-learn, NumPy, pandas, SciPy, matplotlib. The authors of the paper published no code.

Educational project, not for clinical use. Data: UCI Machine Learning Repository.
