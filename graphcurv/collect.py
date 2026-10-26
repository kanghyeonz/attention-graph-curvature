"""collect the fold results under --runs into one json (ten seed scores per cell and method).
pairs that have not run are left out"""
import argparse
from pathlib import Path
import numpy as np
from .conditions import METHODS
from .io import ROOT, read_json, write_json
from .metric import pooled_A


def collect(root):
    manifest = read_json(ROOT / 'experiments/manifest.json')
    plan = read_json(ROOT / 'experiments/fold_plan.json')
    cells = {}
    for cell, rows in manifest.items():
        if not (root / 'evaluation' / cell).is_dir():
            continue
        results = {}
        for method in METHODS:
            values = []
            for seed in plan['seeds']:
                total = np.zeros(4, dtype=np.int64)
                for fold in range(5):
                    total += read_json(root / 'evaluation' / cell / method / f'seed{seed}' / f'fold{fold}' / 'decision.json')['counts']
                values.append(pooled_A(total))
            results[method] = dict(A=float(np.mean(values)), A_sd=float(np.std(values, ddof=1)), A_seeds=values)
        cells[cell] = dict(n_solutions=len(rows), results=results)
    return dict(seeds=plan['seeds'], methods=list(METHODS), cells=cells)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = collect(a.runs)
    write_json(a.out, result)
    paper = read_json(ROOT / 'results/paper.json')['cells']
    for cell, c in result['cells'].items():
        if cell not in paper:
            continue
        d = max(abs(r['A'] - paper[cell]['results'][m]['A']) for m, r in c['results'].items())
        print(f'{cell}: largest difference in A from the paper {d:.4f}')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
