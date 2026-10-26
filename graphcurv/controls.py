"""attention controls on the edges used for q.
neglog = quantiles of -log attention, stats3 = entropy, concentration and first-token mass
"""
import numpy as np

LEVELS = np.array([0., .25, .5, .75, 1.])
EPS = 1e-9


def quantiles(values):
    if len(values) == 0:
        return np.zeros(5, np.float32)
    return np.quantile(values, LEVELS, method='linear').astype(np.float32)


def layer_controls(attention, spans):
    a = np.asarray(attention)
    T = len(a)
    neg = np.zeros((T, 5), np.float32)
    stats = np.zeros((T, 3), np.float32)
    step_neg = np.zeros((len(spans), 5), np.float32)
    step_stats = np.zeros((len(spans), 3), np.float32)
    # problem tokens stay 0
    for j, (b, e) in enumerate(spans):
        if b == e:
            continue
        pooled = []
        for t in range(b, e):
            row = a[t, :t]  # keys before t
            mask = row > 0
            weights = np.maximum(-np.log(row[mask].astype(np.float64)), EPS)
            neg[t] = quantiles(weights)
            pooled.extend(weights.tolist())
            n = int(mask.sum())
            if not n:
                continue
            p = row[mask].astype(np.float64)
            p /= p.sum()
            entropy = float(-np.sum(p * np.log(p)) / np.log(n)) if n > 1 else 0.
            concentration = float((np.sum(p*p) - 1/n) / (1 - 1/n)) if n > 1 else 1.
            sink = float(p[0]) if mask[0] else 0.
            stats[t] = np.clip([entropy, concentration, sink], 0., 1.)
        step_neg[j] = quantiles(pooled)  # pool the step's edges first, then quantiles
        step_stats[j] = stats[b:e].mean(axis=0)
    return dict(neglog_tok=neg, neglog_step=step_neg, stats3_tok=stats, stats3_step=step_stats)


def stack_layers(attentions, spans):
    values = [layer_controls(a, spans) for a in attentions]
    return {key: np.stack([v[key] for v in values], axis=1) for key in values[0]}
