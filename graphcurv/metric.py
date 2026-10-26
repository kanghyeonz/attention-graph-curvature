"""ProcessBench metric: accuracy on correct solutions, exact first-error accuracy, harmonic mean"""
from fractions import Fraction
import numpy as np

def solution_counts(sols, scores, tau):
    """(ok_correct, n_correct, ok_erroneous, n_erroneous) at threshold tau"""
    ok_c = n_c = ok_e = n_e = 0
    for s in sols:
        sc = np.asarray(scores[s["sid"]], dtype=np.float64)
        flags = np.where(np.nan_to_num(sc, nan=np.inf) < tau)[0]
        if s["label"] < 0:
            n_c += 1
            ok_c += len(flags) == 0
        else:
            n_e += 1
            ok_e += len(flags) > 0 and flags[0] == s["label"]
    return ok_c, n_c, ok_e, n_e


def choose_threshold(rows, scores):
    # best of 50 quantiles of the fitting scores, smallest tau on ties
    if not rows or len({s['label'] < 0 for s in rows}) < 2:
        return .5
    pool = np.concatenate([scores[s['sid']] for s in rows])
    pool = pool[np.isfinite(pool)]
    if not pool.size:
        return .5
    best = None
    for tau in np.unique(np.quantile(pool, np.linspace(0, 1, 50))):
        c, nc, e, ne = (int(x) for x in solution_counts(rows, scores, tau))
        numerator, denominator = 2*c*e, c*ne+e*nc
        denominator = denominator or 1
        # A = 2ce/(c*ne+e*nc)
        if best is None or numerator*best[1] > best[0]*denominator:
            best = numerator, denominator, float(tau)
    return best[2]


def pooled_A(counts):
    # A over the counts of all five folds
    ok_c, n_c, ok_e, n_e = map(int, counts)
    den = ok_c*n_e + ok_e*n_c
    return float(Fraction(2*ok_c*ok_e, den)) if den else 0.
