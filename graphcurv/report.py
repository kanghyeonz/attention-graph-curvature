"""tables and figure 2 from a results file"""
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from .io import ROOT, read_json, write_json

SUBSETS = ('gsm8k', 'math', 'olympiadbench', 'omnimath')
GROUPS = {
    'Spectral, logistic': ('spec-logreg', 'spectral+neglog', 'spectral+stats3', 'spec+curvstepq5-logreg'),
    'Hidden states, linear': ('hidden-logreg', 'linear+neglog', 'linear+stats3', 'hidden+curvq5-logreg'),
    'MLP probe': ('probe-mlp', 'mlp+neglog', 'mlp+stats3', 'probe-mlp+curvq5'),
    'ReProbe': ('probe-reprobe', 'reprobe+neglog', 'reprobe+stats3', 'probe-reprobe+curvq5'),
}


def aggregate(cells, method, subset=None):
    selected = [c for name, c in cells.items() if subset is None or name.startswith(subset + '_')]
    if not selected:
        return float('nan'), float('nan')
    a = np.array([c['results'][method]['A'] for c in selected])
    w = np.array([c['n_solutions'] for c in selected], dtype=float)
    w /= w.sum()
    mean = float(w @ a)
    w2 = float(w @ w)
    se = float(np.sqrt((w @ ((a - mean) ** 2)) / (1 - w2) * w2)) if w2 < 1 else float('nan')
    return mean, se


def csv_write(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def timings(grid, manifest):
    # curvature is timed at the median step length on a grid of prefix lengths.
    # None = out of memory
    def interp(T, table):
        xs = [t for t in grid['grid'] if table.get(str(t)) is not None]
        return float(np.interp(T, xs, [table[str(t)] for t in xs])) if xs and T <= max(xs) else None
    model_cfg = read_json(ROOT / 'experiments/models.json')
    cells = {}
    for name, solutions in manifest.items():
        generator = name.split('_', 1)[1]
        model = model_cfg[generator]['id']
        layers = '32' if 'Llama' in generator else '28'
        ends = [e for s in solutions for b, e in s['spans'] if b < e]
        fw = grid['forward'][model]
        q = [interp(e, {t: v['ms'] for t, v in grid['q'][layers].items()}) for e in ends]
        tables = [{t: v[tag][key] for t, v in fw.items() if t != 'dtype'}
                  for tag, key in [('plain', 'ms'), ('attn', 'ms'), ('attn', 'headmean_ms')]]
        plain, attn, head = [[interp(e, table) for e in ends] for table in tables]
        ok = [i for i in range(len(ends)) if all(v[i] is not None for v in (plain, attn, q))]
        assert ok, name
        cells[name] = dict(n=len(solutions), steps=len(ends), within=len(ok),
            plain=float(np.mean([plain[i] for i in ok])),
            attention=float(np.mean([attn[i] - plain[i] for i in ok])),
            head_mean=float(np.mean([head[i] for i in ok])),
            curvature=float(np.mean([v for v in q if v is not None])))
    result = []
    for subset in (*SUBSETS, 'all'):
        selected = [v for k, v in cells.items() if subset == 'all' or k.startswith(subset + '_')]
        weights = np.array([v['n'] for v in selected], dtype=float)
        row = dict(subset=subset)
        row.update({key: float(np.average([v[key] for v in selected], weights=weights))
                    for key in ('plain', 'attention', 'head_mean', 'curvature')})
        row['within_memory'] = sum(v['within'] for v in selected) / sum(v['steps'] for v in selected)
        result.append(row)
    return result


def report(result, out, plot=True, grid=None):
    grid = grid if grid is not None else read_json(ROOT / 'results/grid_cost.json')
    try:
        cost = timings(grid, read_json(ROOT / 'experiments/manifest.json'))
    except KeyError:
        cost = None  # a generator we did not time
    out.mkdir(parents=True, exist_ok=True)
    cells = result['cells']
    pairs, subsets, detailed, controls, differences = [], [], [], [], []
    for label, group in GROUPS.items():
        means = [aggregate(cells, method)[0] for method in group]
        delta = np.array([c['results'][group[3]]['A'] - c['results'][group[0]]['A'] for c in cells.values()])
        p = float(wilcoxon(delta[delta != 0], alternative='two-sided', method='exact').pvalue) if np.any(delta) else 1.
        pairs.append(dict(method=label, base=means[0], neglog=means[1], stats=means[2], curvature=means[3],
                          delta=means[3]-means[0], positive_cells=int((delta > 0).sum()), p=p))
        for subset in (*SUBSETS, 'all'):
            for variant, method in zip(('Base', 'Neg-log', 'Stats', 'Curvature'), group):
                mean, se = aggregate(cells, method, None if subset == 'all' else subset)
                row = dict(method=label, variant=variant, subset=subset, A=mean, se=se)
                (subsets if variant in ('Base', 'Curvature') else controls).append(row)
        for name, cell in cells.items():
            for variant, method in zip(('Base', 'Curvature'), (group[0], group[3])):
                r = cell['results'][method]
                detailed.append(dict(cell=name, n=cell['n_solutions'], method=label, variant=variant, A=r['A'], sd=r['A_sd']))
            d = np.asarray(cell['results'][group[3]]['A_seeds']) - cell['results'][group[0]]['A_seeds']
            differences.append(dict(cell=name, n=cell['n_solutions'], method=label, delta=float(d.mean()), sd=float(d.std(ddof=1))))
    csv_write(out / 'table2.csv', list(pairs[0]), pairs)
    csv_write(out / 'table4.csv', list(subsets[0]), subsets)
    for i, subset in enumerate(SUBSETS, 5):
        csv_write(out / f'table{i}.csv', list(detailed[0]), [r for r in detailed if r['cell'].startswith(subset+'_')])
    csv_write(out / 'table9.csv', list(controls[0]), controls)
    csv_write(out / 'figure2.csv', list(differences[0]), differences)
    if cost:
        csv_write(out / 'table3.csv', list(cost[0]), cost)
    write_json(out / 'summary.json', dict(table2=pairs, table3=cost))
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(15, 8), sharey=True, sharex=True)
        order = [name for subset in SUBSETS for name in sorted(
            (name for name in cells if name.startswith(subset+'_')), key=lambda n: -cells[n]['n_solutions'])]
        for ax, label in zip(axes, GROUPS):
            lookup = {r['cell']: r for r in differences if r['method'] == label}
            for y, name in enumerate(order):
                r = lookup[name]
                color = '#2463a6' if r['delta'] > 0 else '#b33b35' if r['delta'] < 0 else '#777777'
                ax.errorbar(r['delta'], y, xerr=r['sd'], fmt='o', color=color,
                            markerfacecolor=color if abs(r['delta']) > r['sd'] else 'white', markersize=4, linewidth=1)
            ax.axvline(0, color='#999999', linewidth=.7)
            ax.set_xlim(-.4, .4)
            for boundary in (6.5, 13.5, 17.5):
                ax.axhline(boundary, color='#dddddd', linewidth=.7)
            ax.set_title(label)
            ax.set_xlabel('Paired difference in A')
            ax.grid(axis='x', alpha=.2)
        axes[0].set_yticks(range(len(order)), [f'{n.replace("-Instruct", "")} (n={cells[n]["n_solutions"]})' for n in order], fontsize=7)
        axes[0].invert_yaxis()
        fig.tight_layout()
        fig.savefig(out / 'figure2.pdf')
        fig.savefig(out / 'figure2.png', dpi=180)
        plt.close(fig)
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT / 'results/paper.json')
    p.add_argument('--out', type=Path, default=Path('outputs/paper'))
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--timings', type=Path, default=ROOT / 'results/grid_cost.json')
    a = p.parse_args()
    for row in report(read_json(a.results), a.out, not a.no_plot, read_json(a.timings)):
        print(f'{row["method"]}: {row["base"]:.4f} -> {row["curvature"]:.4f}; delta={row["delta"]:+.4f}, p={row["p"]:.6g}')


if __name__ == '__main__':
    main()
