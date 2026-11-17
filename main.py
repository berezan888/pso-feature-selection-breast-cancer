"""Run the PSO feature-selection replication.

    python main.py --config configs/default.yaml           # run (resumable) + report
    python main.py --config configs/default.yaml --report-only
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from src.data_preprocessing import load_dataset, split_and_preprocess
from src.evaluation import (compare_with_paper, evaluate_subset, feature_frequency, make_figures,
                            paired_tests, summarize)
from src.fitness import FitnessFunction
from src.pso import pso_feature_selection, random_search_feature_selection

ROOT = Path(__file__).resolve().parent


def run_job(dataset, clf, seed, cfg):
    """One (dataset, classifier, seed): same split + same budget for all three methods."""
    X, y, names = load_dataset(dataset, cfg.get("drop_features", {}).get(dataset, []))
    pre = cfg["preprocess"]
    Xtr, Xte, ytr, yte = split_and_preprocess(X, y, seed, pre["test_size"], pre["fit_on"], pre["outliers"])
    params = cfg.get("clf_params", {}).get(clf)
    ev = lambda mask: evaluate_subset(mask, Xtr, ytr, Xte, yte, clf, seed, params)
    fit = lambda: FitnessFunction(Xtr, ytr, clf, seed, params, **cfg["fitness"])
    d = Xtr.shape[1]
    rows, conv = [], []

    def add(method, mask, secs, best_f, n_evals, hist):
        rows.append({"dataset": dataset, "classifier": clf, "seed": seed, "method": method,
                     **ev(mask), "fit_seconds": secs, "best_fitness": best_f, "n_evals": n_evals,
                     "mask": "".join("1" if b else "0" for b in mask)})
        step = max(1, len(hist) // 200)
        conv.extend({"dataset": dataset, "classifier": clf, "seed": seed, "method": method,
                     "step": i, "best_fitness": h} for i, h in enumerate(hist) if i % step == 0)

    add("all_features", np.ones(d, bool), 0.0, np.nan, 0, [])
    t = time.time()
    m, f, n, h = pso_feature_selection(fit(), d, seed, **cfg["pso"])
    add("pso", m, time.time() - t, f, n, h)
    t = time.time()
    m, f, n, h = random_search_feature_selection(fit(), d, seed, n)
    add("random_search", m, time.time() - t, f, n, h[cfg["pso"]["n_particles"] - 1::cfg["pso"]["n_particles"]])
    return rows, conv


def report(cfg, outdir):
    raw = pd.read_csv(outdir / "raw_runs.csv", dtype={"mask": str})
    summary = summarize(raw)
    summary.to_csv(outdir / "results.csv", index=False)
    paired_tests(raw).to_csv(outdir / "paired_tests.csv", index=False)
    paper = pd.read_csv(ROOT / "paper_results" / "paper_tables.csv")
    compare_with_paper(summary, paper).to_csv(outdir / "comparison_with_paper.csv", index=False)
    names = {ds: load_dataset(ds, cfg.get("drop_features", {}).get(ds, []))[2] for ds in raw.dataset.unique()}
    feature_frequency(raw[raw.method == "pso"], names).to_csv(outdir / "selected_features.csv", index=False)
    make_figures(raw, summary, outdir / "figures")
    print(f"report written to {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    outdir = ROOT / cfg["results_dir"]
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path, conv_path = outdir / "raw_runs.csv", outdir / "convergence.csv"

    if not a.report_only:
        done = set()
        if raw_path.exists():
            r = pd.read_csv(raw_path, dtype={"mask": str})
            done = set(zip(r.dataset, r.classifier, r.seed))
        by_clf = cfg.get("seeds_by_classifier", {})
        jobs = [(d, c, s) for d in cfg["datasets"] for c in cfg["classifiers"]
                for s in by_clf.get(c, cfg["seeds"]) if (d, c, s) not in done]
        print(f"{len(jobs)} jobs to run ({len(done)} already done)", flush=True)
        t0 = time.time()
        gen = Parallel(n_jobs=cfg["n_jobs"], return_as="generator_unordered")(
            delayed(run_job)(d, c, s, cfg) for d, c, s in jobs)
        for k, (rows, conv) in enumerate(gen, 1):  # checkpoint after every finished job
            pd.DataFrame(rows).to_csv(raw_path, mode="a", header=not raw_path.exists(), index=False)
            pd.DataFrame(conv).to_csv(conv_path, mode="a", header=not conv_path.exists(), index=False)
            print(f"[{k}/{len(jobs)}] {rows[0]['dataset']} {rows[0]['classifier']} seed={rows[0]['seed']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    report(cfg, outdir)


if __name__ == "__main__":
    main()
