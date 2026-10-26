"""download ProcessBench (and the seven models with --models).
--plan makes experiments/manifest.json and fold_plan.json from data/dataset.json"""
import argparse
import hashlib
import json
from pathlib import Path
from . import data
from .io import ROOT, read_json, write_json


def group_id(subset, problem):
    return hashlib.sha256((subset + '\n' + problem.strip()).encode()).hexdigest()


def validate_dataset(dataset):
    manifest = read_json(ROOT / 'experiments/manifest.json')
    if set(dataset) != set(manifest):
        raise ValueError('cells differ from experiments/manifest.json')
    for cell, rows in manifest.items():
        raw = {r['sid']: r for r in dataset[cell]}
        if set(raw) != {r['sid'] for r in rows}:
            raise ValueError(f'{cell}: not the solutions in the manifest')
        for row in rows:
            r = raw[row['sid']]
            if group_id(cell.split('_')[0], r['problem']) != row['group'] or r['label'] != row['label'] or len(r['steps']) != row['n_steps']:
                raise ValueError(f'{cell}/{row["sid"]}: problem, label or step count differ from the manifest')


def manifest_rows(cell, rows, tokenizer):
    subset, generator = cell.split('_', 1)
    out = []
    for r in rows:
        text, char_spans = data.assemble(r['problem'], r['steps'])
        enc = tokenizer(text, truncation=True, max_length=2048, return_offsets_mapping=True)
        out.append(dict(sid=r['sid'], generator=generator, group=group_id(subset, r['problem']),
                        label=r['label'], n_steps=len(r['steps']), T=len(enc['input_ids']),
                        spans=data.token_spans(enc['offset_mapping'], char_spans)))
    return out


def fold_plan(manifest, seeds=range(10)):
    # stratified on correct/erroneous, grouped by problem. test fold k, validation fold k+1
    from sklearn.model_selection import StratifiedGroupKFold
    cells = {}
    for cell, rows in manifest.items():
        sids = [r['sid'] for r in rows]
        cells[cell] = {}
        for seed in seeds:
            split = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
                sids, [int(r['label'] >= 0) for r in rows], [r['group'] for r in rows])
            folds = [[sids[i] for i in test] for _, test in split]
            rounds = []
            for k in range(5):
                v = (k + 1) % 5
                rounds.append(dict(test_fold=k, validation_fold=v, fit_folds=[j for j in range(5) if j not in (k, v)],
                                   parts=dict(fit=[s for s in sids if s not in folds[k] + folds[v]],
                                              validation=folds[v], test=folds[k])))
            cells[cell][str(seed)] = dict(folds=folds, rounds=rounds)
    return dict(seeds=list(seeds), cells=cells)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/dataset.json'))
    p.add_argument('--models', action='store_true')
    p.add_argument('--plan', action='store_true')
    a = p.parse_args()
    models = read_json(ROOT / 'experiments/models.json')
    if a.plan:
        from transformers import AutoTokenizer
        manifest = {}
        for cell, rows in read_json(a.data).items():
            cfg = models[cell.split('_', 1)[1]]
            tokenizer = AutoTokenizer.from_pretrained(cfg['id'], revision=cfg['revision'])
            manifest[cell] = manifest_rows(cell, rows, tokenizer)
        (ROOT / 'experiments/manifest.json').write_text(json.dumps(manifest, sort_keys=True) + '\n')
        (ROOT / 'experiments/fold_plan.json').write_text(json.dumps(fold_plan(manifest), separators=(',', ':')) + '\n')
        return
    if a.models:
        from .extract import model_snapshot
        for generator in models:
            print(generator, model_snapshot(generator)[0], flush=True)
    if a.data.exists():
        print(a.data, 'exists')
        return
    from datasets import load_dataset
    source = read_json(ROOT / 'experiments/processbench.json')
    manifest = read_json(ROOT / 'experiments/manifest.json')
    dataset = {}
    for subset in ('gsm8k', 'math', 'olympiadbench', 'omnimath'):
        ds = load_dataset(source['id'], revision=source['revision'], split=subset)
        for cell in manifest:
            split, generator = cell.split('_', 1)
            if split != subset:
                continue
            dataset[cell] = [dict(sid=f'{subset}{i}', problem=r['problem'], steps=r['steps'],
                               label=int(r['label']), generator=r['generator'])
                          for i, r in enumerate(ds) if r['generator'] == generator]
    validate_dataset(dataset)
    write_json(a.data, dataset)
    print(f'wrote {a.data}: {len(dataset)} cells, {sum(map(len, dataset.values()))} solutions')


if __name__ == '__main__':
    main()
