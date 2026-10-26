import numpy as np
from . import evaluation as V
from .feature_table import table_for as baseline_table
from .metric import choose_threshold, solution_counts
from .reproducibility import configure

METHODS = (
    'spec-logreg', 'spectral+neglog', 'spectral+stats3', 'spec+curvstepq5-logreg',
    'hidden-logreg', 'linear+neglog', 'linear+stats3', 'hidden+curvq5-logreg',
    'probe-mlp', 'mlp+neglog', 'mlp+stats3', 'probe-mlp+curvq5',
    'probe-reprobe', 'reprobe+neglog', 'reprobe+stats3', 'probe-reprobe+curvq5',
)
TRAINERS = {'spectral': 'spec-logreg', 'linear': 'hidden-logreg',
            'mlp': 'probe-mlp', 'reprobe': 'probe-reprobe'}


def table_for(method, rows, features):
    assert method in METHODS, method
    if method in V.METHODS:
        return baseline_table(method, rows, features)
    base, extra = method.split('+')
    result = {}
    for row in rows:
        f = features[row['sid']]
        token = f[extra + '_tok'].reshape(row['T'], -1).astype(np.float32)
        step = f[extra + '_step'].reshape(row['n_steps'], -1).astype(np.float32)
        if base == 'spectral':
            x = np.concatenate((f['spec'].reshape(row['n_steps'], -1), step), axis=-1)
        else:
            seq = np.concatenate((f['hid_seq'].astype(np.float32), token), axis=-1)
            if base == 'reprobe':
                result[row['sid']] = seq
                continue
            x = np.full((row['n_steps'], seq.shape[-1]), np.nan, np.float32)
            for j, (b, e) in enumerate(row['spans']):
                if b < e:
                    x[j] = seq[e - 1]
        result[row['sid']] = x.astype(np.float32)
    return result


def evaluate(method, rows, inputs, test, validation, seed, device='cpu', smoke=False):
    trainer = method if method in V.METHODS else TRAINERS[method.split('+')[0]]
    configure(device)
    parts = V.partition(rows, test, validation)
    if trainer.startswith('probe-reprobe'):
        scores, info = V.fit_reprobe(parts, inputs, seed, device, smoke)
    else:
        scores, info = V.fit_tabular(trainer, parts, inputs, seed, device, smoke)
    tau = choose_threshold(parts['fit'], scores)
    return dict(tau=tau, counts=list(map(int, solution_counts(parts['test'], scores, tau))),
                parts={k: [r['sid'] for r in v] for k, v in parts.items()},
                fit_info=info), scores
