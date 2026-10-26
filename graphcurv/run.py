"""extraction and training, one cell after the other"""
import argparse
from pathlib import Path
import subprocess
import sys
from .io import ROOT, read_json


def main():
    cells = list(read_json(ROOT / 'experiments/manifest.json'))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', action='store_true')
    select = p.add_mutually_exclusive_group()
    select.add_argument('--cells', nargs='+', choices=cells)
    select.add_argument('--cell-index', type=int)
    p.add_argument('--stage', choices=['extract', 'train', 'all'], default='all')
    p.add_argument('--data', type=Path, default=Path('data/dataset.json'))
    p.add_argument('--features', type=Path, default=Path('data/features'))
    p.add_argument('--out', type=Path, default=Path('outputs/runs'))
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.list:
        for i, cell in enumerate(cells):
            print(i, cell)
        return
    selected = [cells[a.cell_index]] if a.cell_index is not None else (a.cells or cells)
    if a.stage in ('extract', 'all') and not a.data.exists():
        p.error('run python -m graphcurv.prepare first')
    for cell in selected:
        if a.stage in ('extract', 'all'):
            subprocess.run([sys.executable, '-m', 'graphcurv.extract', '--cell', cell,
                '--data', str(a.data), '--out', str(a.features), '--device', a.device], check=True)
        if a.stage in ('train', 'all'):
            subprocess.run([sys.executable, '-m', 'graphcurv.train', '--cell', cell,
                '--features', str(a.features), '--out', str(a.out), '--device', a.device], check=True)


if __name__ == '__main__':
    main()
