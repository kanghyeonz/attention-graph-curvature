"""train and score all methods on one cell, fold by fold"""
import argparse
from pathlib import Path
import numpy as np
import torch
from .conditions import METHODS, table_for, evaluate
from .io import ROOT, read_json, write_json, versions


def load_features(folder, rows):
    missing = [r['sid'] for r in rows if not (folder / 'feat' / (r['sid'] + '.npz')).exists()]
    if missing:
        raise ValueError(f'{len(missing)} solutions of {folder.name} are not extracted yet')
    features = {}
    for row in rows:
        with np.load(folder / 'feat' / (row['sid'] + '.npz')) as z:
            features[row['sid']] = dict(z)
    return features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cell', required=True)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    p.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true', help='tiny settings for testing, written to <out>/smoke')
    a = p.parse_args()
    torch.set_num_threads(4)
    manifest = read_json(ROOT / 'experiments/manifest.json')
    plan = read_json(ROOT / 'experiments/fold_plan.json')
    rows = manifest[a.cell]
    features = load_features(a.features / a.cell, rows)
    out = a.out / ('smoke' if a.smoke else 'evaluation') / a.cell
    write_json(out / 'run.json', dict(cell=a.cell, smoke=a.smoke, versions=versions(), device=a.device))
    index = {row['sid']: i for i, row in enumerate(rows)}
    for method in a.methods:
        table = table_for(method, rows, features)
        for seed in a.seeds:
            for fold in a.folds:
                target = out / method / f'seed{seed}' / f'fold{fold}'
                if (target / 'decision.json').exists():
                    continue  # done before
                parts = plan['cells'][a.cell][str(seed)]['rounds'][fold]['parts']
                report, scores = evaluate(method, rows, table,
                    [index[s] for s in parts['test']],
                    [index[s] for s in parts['validation']], seed, a.device, a.smoke)
                report.update(cell=a.cell, method=method, seed=seed, fold=fold, smoke=a.smoke)
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / 'scores.npz', **scores)
                write_json(target / 'decision.json', report)
                print(f'{a.cell} {method} seed={seed} fold={fold} {report["counts"]}', flush=True)


if __name__ == '__main__':
    main()
