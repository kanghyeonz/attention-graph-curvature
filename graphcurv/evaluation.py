"""grouped cv with 3 fit folds, 1 validation fold and 1 test fold"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data as D
from . import probes as P

METHODS = ('spec-logreg', 'spec+curvstepq5-logreg', 'hidden-logreg',
           'hidden+curvq5-logreg', 'probe-mlp', 'probe-mlp+curvq5',
           'probe-reprobe', 'probe-reprobe+curvq5')


def groups(rows):
    out = {}
    for row in rows:
        out.setdefault(row['group'], []).append(row)
    return out


def partition(rows, test_indices, validation_indices):
    test_ids, validation_ids = set(test_indices), set(validation_indices)
    test = [rows[i] for i in test_indices]
    validation = [rows[i] for i in validation_indices]
    fit = [s for i, s in enumerate(rows) if i not in test_ids | validation_ids]
    parts = dict(fit=fit, validation=validation, test=test)
    sets = [set(groups(v)) for v in parts.values()]
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
        raise ValueError('the same problem is in two of fit/validation/test')
    return parts


def arrays(rows, table):
    xs, ys = [], []
    for s in rows:
        for j, y in enumerate(D.step_labels(s['label'], s['n_steps'])):
            b, e = s['spans'][j]
            if y >= 0 and e > b:
                xs.append(table[s['sid']][j])
                ys.append(y)
    assert xs, 'no labeled steps'
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


def predict_rows(rows, table, predict):
    result = {}
    for s in rows:
        score = np.full(s['n_steps'], np.nan)
        valid = [j for j, (b, e) in enumerate(s['spans']) if e > b]
        if valid:
            score[valid] = predict(table[s['sid']][valid])
        result[s['sid']] = score
    return result


def fit_tabular(method, parts, table, seed, device='cpu', smoke=False):
    x, y = arrays(parts['fit'], table)
    inference = parts['fit'] + parts['test']
    if len(np.unique(y)) < 2:
        value = float(y[0])
        return predict_rows(inference, table, lambda x: np.full(len(x), value)), {'constant_class': value}
    if method.endswith('logreg'):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
        model.fit(x.astype(np.float64), y)
        return predict_rows(inference, table, lambda x: model.predict_proba(x.astype(np.float64))[:, 1]), {'classifier': 'logistic'}

    x = torch.tensor(x)
    mu, sd = x.mean(0), x.std(0, unbiased=True).clamp(min=1e-3)
    x, y = (x - mu) / sd, torch.tensor(y)
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))
    def batches(x, y):
        return [(x[i:i + 64], y[i:i + 64]) for i in range(0, len(y), 64)]
    tb = batches(x[perm], y[perm])
    vb = None
    if parts['validation']:
        xv, yv = arrays(parts['validation'], table)
        xv, yv = (torch.tensor(xv) - mu) / sd, torch.tensor(yv)
        vb = batches(xv, yv)
    configs = [(lr, wd, alpha) for lr in (1e-3, 1e-4, 1e-5)
               for wd in (.001, .01, .1) for alpha in (.3, .5, .7, .9, 1., 1.5, 2., 3.)]
    widths, epochs = (0, 16, 32), 200
    if smoke:
        widths, configs, epochs = (0, 16), [(1e-3, .001, 1.)], 2
    trials = []
    if vb:
        for h in widths:
            acc = P.grid_batched(x.shape[1], h, configs, tb, vb, yv, seed,
                                 device=device, epochs=epochs, patience=10)
            trials.extend((a, h, *c) for a, c in zip(acc, configs))
        top = sorted(trials, key=lambda t: -t[0])[:10]
        _, h, lr, wd, alpha = min(top, key=lambda t: (t[1], -t[0]))
    else:
        h, lr, wd, alpha = 0, 1e-3, .001, 1.
    torch.manual_seed(seed)
    model = P.MLPProbe(x.shape[1], h=h)
    P.train_probe(model, tb, epochs=epochs, lr=lr, seed=seed, device=device,
                  alpha=alpha, weight_decay=wd, val_batches=vb, patience=10)
    def predict(a):
        xt = (torch.tensor(a, dtype=torch.float32) - mu) / sd
        return P.score_probe(model, batches(xt, torch.zeros(len(xt))), device=device).numpy()
    return predict_rows(inference, table, predict), {'h': h, 'lr': lr, 'wd': wd, 'alpha': alpha,
                                                     'checkpoint': model.best}


def reprobe_microbatches(items, max_items=8, token_square_budget=8_000_000):
    # cut a batch of up to 128 steps into pieces that fit in memory
    start = 0
    while start < len(items):
        stop, width = start, 0
        while stop < len(items) and stop - start < max_items:
            proposed = max(width, len(items[stop][2]))
            if stop > start and (stop - start + 1) * proposed ** 2 > token_square_budget:
                break
            width = proposed
            stop += 1
        chunk = items[start:stop]
        x = torch.zeros(len(chunk), width, chunk[0][2].shape[1])
        pad = torch.ones(len(chunk), width, dtype=torch.bool)
        mask = torch.zeros(len(chunk), 1, width, dtype=torch.long)
        y = torch.zeros(len(chunk), 1)
        for i, (_, _, step, label) in enumerate(chunk):
            n = len(step)
            x[i, :n], pad[i, :n], mask[i, 0, :n], y[i, 0] = step, False, 1, float(label)
        yield x, pad, mask, y, [(sid, j) for sid, j, _, _ in chunk]
        start = stop


def fit_reprobe(parts, sequences, seed, device='cpu', smoke=False):
    torch.manual_seed(seed)
    # mean and sd over all tokens of the first 32 fit solutions
    sample = np.concatenate([sequences[s['sid']] for s in parts['fit'][:32]])
    mu, sd = sample.mean(0), np.maximum(sample.std(0), 1e-3)
    def pack(rows, labeled):
        items = []
        for s in rows:
            labels = D.step_labels(s['label'], s['n_steps']) if labeled else None
            for j, (b, e) in enumerate(s['spans']):
                if e <= b or (labeled and labels[j] < 0):
                    continue
                x = (sequences[s['sid']][b:e] - mu) / sd
                # the head's logit is for the erroneous class, hence 1 - y here and 1 - sigmoid when scoring
                items.append((s['sid'], j, torch.tensor(x), 1 - labels[j] if labeled else 0))
        for start in range(0, len(items), 128):
            yield items[start:start + 128]
    tb, vb = list(pack(parts['fit'], True)), list(pack(parts['validation'], True))
    assert tb, 'no fitting steps'
    model = P.ReProbeClaim(len(mu), head_dim=32 if smoke else 512,
                          n_heads=4 if smoke else 16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.1)
    epochs = 1 if smoke else 5
    total, warm = epochs * len(tb), max(1, int(.05 * epochs * len(tb)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: (st + 1) / warm if st < warm else max(0., (total - st) / max(total - warm, 1)))
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3., device=device))
    rng, best, snapshot = np.random.default_rng(seed + 5000), float('inf'), None
    best_epoch = None
    for epoch in range(epochs):
        model.train()
        for i in rng.permutation(len(tb)):
            opt.zero_grad()
            for batch in reprobe_microbatches(tb[i]):
                x, pad, mask, y = [t.to(device) for t in batch[:4]]
                batch_loss = loss(model(x, pad, mask), y)
                (batch_loss * len(y) / len(tb[i])).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            sched.step()
        model.eval()
        if vb:
            with torch.no_grad():
                values = []
                for logical in vb:
                    value = 0.
                    for batch in reprobe_microbatches(logical):
                        value += float(loss(model(*[t.to(device) for t in batch[:3]]),
                                            batch[3].to(device))) * len(batch[3]) / len(logical)
                    values.append(value)
                value = np.average(values, weights=[len(logical) for logical in vb])
            if value < best:
                best = value
                best_epoch = epoch
                snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if snapshot is not None:
        model.load_state_dict(snapshot)
    result = {s['sid']: np.full(s['n_steps'], np.nan) for s in parts['fit'] + parts['test']}
    model.eval()
    with torch.no_grad():
        for role in ('fit', 'test'):
            for logical in pack(parts[role], False):
                for x, pad, mask, _, meta in reprobe_microbatches(logical):
                    p = (1 - torch.sigmoid(model(x.to(device), pad.to(device), mask.to(device)))).cpu().numpy()
                    for i, (sid, j) in enumerate(meta):
                        result[sid][j] = float(p[i, 0])
    info = {'epochs': epochs, 'best_epoch': best_epoch, 'best_validation_loss': float(best)}
    return result, info
