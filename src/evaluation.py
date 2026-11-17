"""Test-set metrics, aggregation, paired tests, comparison with the paper, figures."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, roc_auc_score

from .classifiers import make_classifier, score_function

METRICS = ["accuracy", "sensitivity", "specificity", "precision_pos", "precision_neg", "auc"]


def evaluate_subset(mask, Xtr, ytr, Xte, yte, clf_name, seed, clf_params=None):
    """Train on the training part with the selected features, evaluate on the held-out test part."""
    mask = np.asarray(mask, bool)
    clf = make_classifier(clf_name, seed, clf_params).fit(Xtr[:, mask], ytr)
    pred = clf.predict(Xte[:, mask])
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    s = score_function(clf)(Xte[:, mask])
    s = s[:, 1] if s.ndim == 2 else s
    div = lambda a, b: a / b if b else 0.0
    return {
        "accuracy": (tp + tn) / len(yte), "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp), "precision_pos": div(tp, tp + fp),
        "precision_neg": div(tn, tn + fn), "auc": roc_auc_score(yte, s),
        "n_features": int(mask.sum()),
    }


def summarize(raw):
    """mean / median / std / min / max over runs for every dataset x classifier x method."""
    cols = METRICS + ["n_features", "fit_seconds"]
    g = raw.groupby(["dataset", "classifier", "method"])[cols]
    out = g.agg(["mean", "median", "std", "min", "max"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_runs"] = g.size()
    return out.reset_index()


def paired_tests(raw, metric="accuracy"):
    """Paired Wilcoxon signed-rank test over seeds (same split for all methods)."""
    rows = []
    for (ds, clf), d in raw.groupby(["dataset", "classifier"]):
        piv = d.pivot(index="seed", columns="method", values=metric)
        for a, b in [("pso", "all_features"), ("pso", "random_search")]:
            diff = (piv[a] - piv[b]).dropna()
            try:
                p = wilcoxon(diff).pvalue if (diff != 0).any() and len(diff) >= 6 else np.nan
            except ValueError:
                p = np.nan
            rows.append({"dataset": ds, "classifier": clf, "comparison": f"{a} vs {b}",
                         "metric": metric, "mean_diff": diff.mean(), "wilcoxon_p": p, "n": len(diff)})
    return pd.DataFrame(rows)


def compare_with_paper(summary, paper):
    """Join our mean metrics with the paper's tables (methods: all_features, pso)."""
    ours = summary[summary.method.isin(["all_features", "pso"])]
    keep = ["dataset", "classifier", "method"] + [f"{m}_mean" for m in METRICS]
    ours = ours[keep].rename(columns={f"{m}_mean": m for m in METRICS})
    m = ours.merge(paper, on=["dataset", "classifier", "method"], suffixes=("_ours", "_paper"))
    for metric in METRICS:
        m[f"{metric}_diff"] = m[f"{metric}_ours"] - m[f"{metric}_paper"]
    return m


def feature_frequency(selected_rows, feature_names_by_dataset):
    """How often every feature was selected by PSO (per dataset x classifier)."""
    recs = []
    for (ds, clf), d in selected_rows.groupby(["dataset", "classifier"]):
        masks = np.array([[c == "1" for c in s] for s in d["mask"]])
        for j, name in enumerate(feature_names_by_dataset[ds]):
            recs.append({"dataset": ds, "classifier": clf, "feature": name,
                         "selection_frequency": masks[:, j].mean()})
    return pd.DataFrame(recs)


def make_figures(raw, summary, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    for ds in raw.dataset.unique():
        s = summary[summary.dataset == ds]
        clfs = sorted(s.classifier.unique())
        x = np.arange(len(clfs))
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for k, (m, lab) in enumerate([("all_features", "all features"), ("pso", "PSO"),
                                      ("random_search", "random search")]):
            t = s[s.method == m].set_index("classifier").reindex(clfs)
            ax[0].bar(x + (k - 1) * 0.27, t.accuracy_mean, 0.27, yerr=t.accuracy_std, label=lab, capsize=2)
            ax[1].bar(x + (k - 1) * 0.27, t.n_features_mean, 0.27, label=lab)
        for a, yl in zip(ax, ["test accuracy (mean ± std over seeds)", "number of selected features"]):
            a.set_xticks(x); a.set_xticklabels(clfs); a.set_ylabel(yl)
            a.legend(loc="upper center", ncol=3, fontsize=8, frameon=False)
        lo = max(0.4, s.accuracy_mean.min() - 0.1)
        ax[0].set_ylim(lo, 1.0 + 0.12 * (1.0 - lo))
        ax[1].set_ylim(0, 1.25 * s.n_features_mean.max())
        fig.suptitle(f"{ds}: effect of feature selection"); fig.tight_layout()
        fig.savefig(outdir / f"accuracy_features_{ds}.png", dpi=150); plt.close(fig)
