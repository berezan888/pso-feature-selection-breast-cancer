"""PSO feature selection (paper Eqs. 1-2) and a random-search baseline with the same budget."""
import numpy as np


def pso_feature_selection(fitness, n_features, seed, n_particles=20, n_iter=100,
                          w=0.9, c1=1.0, c2=2.0, vmax=0.5, threshold=0.5):
    """Continuous PSO in [0,1]^d; feature j selected iff position_j > threshold.

    v <- w*v + c1*u1*(pbest - x) + c2*u2*(gbest - x);  x <- x + v   (Eqs. 1-2)
    Returns (best_mask, best_fitness, n_evals, history_of_gbest_fitness).
    """
    rng = np.random.default_rng(seed)
    pos = rng.random((n_particles, n_features))
    vel = rng.uniform(-vmax, vmax, (n_particles, n_features))
    pbest, pbest_f = pos.copy(), np.full(n_particles, -np.inf)
    gbest, gbest_f = pos[0].copy(), -np.inf
    history = []
    for _ in range(n_iter):
        for i in range(n_particles):
            f = fitness(pos[i] > threshold)
            if f > pbest_f[i]:
                pbest_f[i], pbest[i] = f, pos[i].copy()
            if f > gbest_f:
                gbest_f, gbest = f, pos[i].copy()
        history.append(gbest_f)
        u1, u2 = rng.random(pos.shape), rng.random(pos.shape)
        vel = np.clip(w * vel + c1 * u1 * (pbest - pos) + c2 * u2 * (gbest - pos), -vmax, vmax)
        pos = np.clip(pos + vel, 0.0, 1.0)
    return gbest > threshold, gbest_f, n_particles * n_iter, history


def random_search_feature_selection(fitness, n_features, seed, n_evals):
    """Fair baseline: n_evals random feature masks, keep the best (same evaluation budget)."""
    rng = np.random.default_rng(seed + 10_000)
    best_m, best_f, history = None, -np.inf, []
    for _ in range(n_evals):
        m = rng.random(n_features) > 0.5
        f = fitness(m)
        if f > best_f:
            best_f, best_m = f, m
        history.append(best_f)
    return best_m, best_f, n_evals, history
