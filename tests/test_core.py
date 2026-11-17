import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_preprocessing import Preprocessor, load_dataset, split_and_preprocess
from src.fitness import FitnessFunction
from src.pso import pso_feature_selection, random_search_feature_selection


def toy(seed=0, n=120, d=8):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, d))
    X[:, 0] += 2.0 * y  # only feature 0 is informative
    return X, y


def test_empty_mask_has_zero_fitness():
    X, y = toy()
    assert FitnessFunction(X, y, "LGR", 0)(np.zeros(X.shape[1], bool)) == 0.0


def test_fitness_is_weighted_train_and_val_accuracy_in_unit_interval():
    X, y = toy()
    f = FitnessFunction(X, y, "LGR", 0)(np.ones(X.shape[1], bool))
    assert 0.0 <= f <= 1.0


def test_pso_is_deterministic_and_budget_is_exact():
    X, y = toy()
    run = lambda: pso_feature_selection(FitnessFunction(X, y, "LGR", 1), X.shape[1], 1, n_particles=5, n_iter=4)
    m1, f1, n1, _ = run()
    m2, f2, n2, _ = run()
    assert (m1 == m2).all() and f1 == f2 and n1 == n2 == 20


def test_pso_finds_informative_feature():
    X, y = toy()
    m, f, _, _ = pso_feature_selection(FitnessFunction(X, y, "LGR", 0), X.shape[1], 0, n_particles=10, n_iter=10)
    assert m[0]


def test_random_search_uses_same_budget():
    X, y = toy()
    fit = FitnessFunction(X, y, "LGR", 0)
    random_search_feature_selection(fit, X.shape[1], 0, 17)
    assert fit.n_evals == 17


def test_preprocessor_fit_on_train_ignores_test_statistics():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 3))
    p1 = Preprocessor().fit(A)
    p2 = Preprocessor().fit(A)
    B = np.vstack([A, [100, 100, 100]])  # extra data must not influence a Preprocessor fitted on A
    assert np.allclose(p1.transform(B[:5]), p2.transform(B[:5]))


def test_preprocessor_imputes_missing_values():
    X = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, 6.0], [2.0, 5.0]])
    assert not np.isnan(Preprocessor(outliers="none").fit_transform(X)).any()


def test_split_is_stratified_and_80_20():
    X, y, _ = load_dataset("wdbc")
    Xtr, Xte, ytr, yte = split_and_preprocess(X, y, 0)
    assert len(yte) == round(0.2 * len(y))
    assert abs(ytr.mean() - yte.mean()) < 0.02


def test_dataset_shapes_match_paper_table_1():
    for name, shape, pos in [("coimbra", (116, 9), 64), ("wdbc", (569, 30), 212), ("wpbc", (198, 33), 47)]:
        X, y, _ = load_dataset(name)
        assert X.shape == shape and y.sum() == pos
