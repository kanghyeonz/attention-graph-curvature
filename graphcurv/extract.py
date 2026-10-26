"""features from one forward pass of each solution through its own generator"""
import argparse
from pathlib import Path
import numpy as np
import torch
from . import spectral, data
from .capture import CaptureMatrices, layer_curvature
from .controls import stack_layers
from .io import ROOT, read_json, write_json, versions


def model_snapshot(generator, model_dir=None):
    cfg = read_json(ROOT / 'experiments/models.json')[generator]
    if model_dir is None:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(cfg['id'], revision=cfg['revision'], allow_patterns=cfg.get('files'))
    return Path(model_dir), cfg


def forward(model, enc, device):
    with CaptureMatrices(model) as cap, torch.inference_mode():
        output = model.model(**{k: v.to(device) for k, v in enc.items()},
            output_attentions=False, output_hidden_states=True, use_cache=False)
        hidden = [h[0].cpu().float() for h in output.hidden_states]
    assert len(cap.attentions) == model.config.num_hidden_layers
    return cap, hidden


def features_from_forward(cap, hidden, enc, spans, device):
    T, L, n = enc['input_ids'].shape[1], len(cap.attentions), len(spans)
    qt = np.zeros((T, L, 5), np.float32)
    qs = np.zeros((n, L, 5), np.float32)
    spec = np.full((n, L, 5), np.nan, np.float32)
    with torch.inference_mode():
        for layer in range(L):
            qt[:, layer], qs[:, layer] = layer_curvature(cap.attentions[layer].to(device), spans)
            spec[:, layer] = spectral.prefix_feats(cap.symmetric[layer].to(device),
                                                   hidden[layer + 1].to(device), spans)
    arrays = dict(q5_tok=qt, q5_step=qs, spec=spec, hid_seq=hidden[-1].half().numpy(),
                  input_ids=enc['input_ids'][0].numpy(),
                  step_beg=np.array([b if e > b else -1 for b, e in spans]),
                  step_end=np.array([e - 1 if e > b else -1 for b, e in spans]))
    arrays.update(stack_layers([a.numpy() for a in cap.attentions], spans))
    for name, array in arrays.items():
        if name == 'spec':
            array = array[[j for j, (b, e) in enumerate(spans) if b < e]]
        if not np.isfinite(array).all():
            raise ValueError(f'nan/inf in {name}')
    return arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/dataset.json'))
    p.add_argument('--out', type=Path, default=Path('data/features'))
    p.add_argument('--cell', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--model-dir', type=Path)
    p.add_argument('--sids', nargs='+', help='only these solutions')
    a = p.parse_args()
    dataset = read_json(a.data)
    rows = read_json(ROOT / 'experiments/manifest.json')[a.cell]
    if a.sids and not set(a.sids) <= {r['sid'] for r in rows}:
        p.error('unknown solution id')
    snapshot, cfg = model_snapshot(a.cell.split('_', 1)[1], a.model_dir)
    folder = a.out / a.cell
    (folder / 'feat').mkdir(parents=True, exist_ok=True)
    write_json(folder / 'config.json', dict(cell=a.cell, model=cfg['id'], revision=cfg['revision'],
        dtype=cfg['dtype'], max_length=2048, versions=versions(), device=a.device))
    # skip what is already extracted
    pending = [r for r in rows if not (folder / 'feat' / (r['sid'] + '.npz')).exists()
               and (not a.sids or r['sid'] in a.sids)]
    if not pending:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=getattr(torch, cfg['dtype']),
        attn_implementation='eager', local_files_only=True).to(a.device).eval()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    raw = {r['sid']: r for r in dataset[a.cell]}
    for row in pending:
        source = raw[row['sid']]
        text, cspans = data.assemble(source['problem'], source['steps'])
        enc = tokenizer(text, truncation=True, max_length=2048,
                        return_offsets_mapping=True, return_tensors='pt')
        spans = data.token_spans(enc.pop('offset_mapping')[0].tolist(), cspans)
        if spans != row['spans'] or enc['input_ids'].shape[1] != row['T']:
            raise ValueError(f'{row["sid"]}: tokenization differs from the manifest')
        cap, hidden = forward(model, enc, a.device)
        arrays = features_from_forward(cap, hidden, enc, spans, a.device)
        path = folder / 'feat' / (row['sid'] + '.npz')
        tmp = path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
        print(f'{a.cell} {row["sid"]}', flush=True)
        del cap, hidden, arrays
        if a.device.startswith('cuda'):
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
