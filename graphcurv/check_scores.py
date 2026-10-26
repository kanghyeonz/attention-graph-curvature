"""recompute the seed scores in results/paper.json from the saved fold predictions"""
import argparse
from pathlib import Path
import numpy as np
from .io import ROOT, read_json
from .metric import choose_threshold, solution_counts, pooled_A


def check_scores(folder):
    manifest = read_json(ROOT / 'experiments/manifest.json')
    plan = read_json(ROOT / 'experiments/fold_plan.json')
    paper = read_json(ROOT / 'results/paper.json')
    for cell, rows in manifest.items():
        lookup = {r['sid']: r for r in rows}
        # scores are [method, seed, fold, steps of all solutions one after another]
        ends = np.cumsum([0] + [r['n_steps'] for r in rows])
        where = {r['sid']: slice(ends[i], ends[i+1]) for i, r in enumerate(rows)}
        with np.load(folder / (cell + '.npz')) as z:
            methods, seeds, S = z['methods'].tolist(), z['seeds'].tolist(), z['scores']
        for mi, method in enumerate(methods):
            values = []
            for si, seed in enumerate(seeds):
                total = np.zeros(4, dtype=np.int64)
                for fold in range(5):
                    parts = plan['cells'][cell][str(seed)]['rounds'][fold]['parts']
                    scores = {sid: S[mi, si, fold, where[sid]] for sid in parts['fit'] + parts['test']}
                    tau = choose_threshold([lookup[s] for s in parts['fit']], scores)
                    total += solution_counts([lookup[s] for s in parts['test']], scores, tau)
                values.append(pooled_A(total))
            if values != paper['cells'][cell]['results'][method]['A_seeds']:
                raise ValueError(f'{cell} {method}: got {values}')
        print(cell, 'ok', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores', type=Path, default=ROOT / 'results/fold_scores')
    a = p.parse_args()
    check_scores(a.scores)


if __name__ == '__main__':
    main()
