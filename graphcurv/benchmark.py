"""timings for Table 3 and the appendix, on random inputs"""
import argparse
import gc
import time
from pathlib import Path
import numpy as np
import torch
from .batched import curv_batched, q_batched
from .io import ROOT, read_json, write_json, versions

GRID = [64, 128, 192, 256, 384, 512, 640, 768, 1024, 1280, 1536, 1792, 2048]


def timed_q(T, layers, step, n=5):
    b, e = max(T - step, 1), T
    try:
        a = torch.tril(torch.rand(layers, T, T, device='cuda'), -1)
        for _ in range(2):
            f, edges = curv_batched(a)
            q_batched(f, edges, b, e)
            del f, edges
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(n):
            f, edges = curv_batched(a)
            q_batched(f, edges, b, e)
            del f, edges
        torch.cuda.synchronize()
        return dict(ms=1000 * (time.perf_counter() - start) / n, mib=torch.cuda.max_memory_allocated() / 2**20)
    except torch.cuda.OutOfMemoryError:
        return dict(ms=None, mib=None)


def timed_forward(model, T, attention, n=5):
    ids = torch.randint(100, 20000, (1, T), device='cuda')
    def run():
        with torch.no_grad():
            return model(input_ids=ids, output_attentions=attention, output_hidden_states=True)
    try:
        for _ in range(2):
            output = run()
            del output
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for i in range(n):
            output = run()
            if i + 1 < n:
                del output  # keep the last one for the head mean
        torch.cuda.synchronize()
        ms, mib = 1000 * (time.perf_counter() - start) / n, torch.cuda.max_memory_allocated() / 2**20
        head_ms = None
        if attention:
            tensors = output.attentions
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n):
                for a in tensors:
                    a[0].mean(0)
            torch.cuda.synchronize()
            head_ms = 1000 * (time.perf_counter() - start) / n
        return dict(ms=ms, mib=mib, headmean_ms=head_ms)
    except torch.cuda.OutOfMemoryError:
        return dict(ms=None, mib=None, headmean_ms=None)


def time_cpu(fn, n=200):
    for _ in range(20):
        fn()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return 1000 * (time.perf_counter() - start) / n


def time_gpu(fn, n=50):
    with torch.no_grad():
        for _ in range(10):
            fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / n, torch.cuda.max_memory_allocated() / 2**20


def scoring(rows, hidden_dim, layers):
    from sklearn.linear_model import LogisticRegression
    from .probes import MLPProbe, ReProbeClaim
    lengths = [e - b for r in rows for b, e in r['spans'] if b < e]
    median, longest = int(np.median(lengths)), max(lengths)
    m, q = hidden_dim, 5 * layers
    out = dict(m=m, L=layers, median_step=median, longest_step=longest)
    for tag, dim in [('spec', q), ('spec+q', 2 * q), ('hidden', m), ('hidden+q', m + q)]:
        x = np.random.randn(64, dim)
        y = np.r_[np.zeros(32), np.ones(32)]
        clf = LogisticRegression(max_iter=200).fit(x, y)
        row = np.random.randn(1, dim)
        out[f'logreg_{tag}_ms'] = time_cpu(lambda: clf.predict_proba(row))
    for tag, dim in [('mlp', m), ('mlp+q', m + q)]:
        model = MLPProbe(dim, 32).to('cuda').eval()
        x = torch.randn(1, dim, device='cuda')
        out[tag + '_ms'], out[tag + '_mib'] = time_gpu(lambda: model(x))
        model = None
        torch.cuda.empty_cache()
    for tag, dim in [('reprobe', m), ('reprobe+q', m + q)]:
        model = ReProbeClaim(dim).to('cuda').eval()
        for label, length in [('median', median), ('longest', longest)]:
            x = torch.randn(1, length, dim, device='cuda')
            pad = torch.zeros(1, length, dtype=torch.bool, device='cuda')
            mask = torch.ones(1, 1, length, dtype=torch.long, device='cuda')
            out[f'{tag}_{label}_ms'], out[f'{tag}_{label}_mib'] = time_gpu(lambda: model(x, pad, mask))
        model = None
        torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['grid', 'forward', 'scoring', 'curvature'], default='grid')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--generators', nargs='+', help='default: all seven')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if not torch.cuda.is_available():
        p.error('needs a CUDA GPU')
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    manifest = read_json(ROOT / 'experiments/manifest.json')
    models = read_json(ROOT / 'experiments/models.json')
    selected = a.generators or list(models)
    write_json(a.out.with_suffix('.environment.json'), dict(versions=versions(), gpu=torch.cuda.get_device_name(), seed=a.seed))
    if a.kind == 'grid':
        from transformers import AutoModelForCausalLM
        from .extract import model_snapshot
        lengths = [e - b for rows in manifest.values() for r in rows for b, e in r['spans'] if b < e]
        step = int(np.median(lengths))
        out = dict(grid=GRID, step_len=step, q={}, forward={})
        for layers in (28, 32):
            out['q'][str(layers)] = {}
            for T in GRID:
                out['q'][str(layers)][str(T)] = timed_q(T, layers, step)
                gc.collect()
                torch.cuda.empty_cache()
                write_json(a.out, out)
        for generator in selected:
            path, cfg = model_snapshot(generator)
            values = {'dtype': cfg['dtype']}
            out['forward'][cfg['id']] = values
            for impl, attention, tag in [('sdpa', False, 'plain'), ('eager', True, 'attn')]:
                model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, cfg['dtype']),
                    attn_implementation=impl, local_files_only=True).to('cuda').eval()
                for T in GRID:
                    values.setdefault(str(T), {})[tag] = timed_forward(model, T, attention)
                    gc.collect()
                    torch.cuda.empty_cache()
                    write_json(a.out, out)
                    print(generator, tag, T, values[str(T)][tag], flush=True)
                del model
                gc.collect()
                torch.cuda.empty_cache()
    elif a.kind == 'forward':
        from transformers import AutoModelForCausalLM
        from .extract import model_snapshot
        out = {}
        for generator in selected:
            path, cfg = model_snapshot(generator)
            selected_cells = {k: v for k, v in manifest.items() if k.split('_', 1)[1] == generator}
            for impl, attention, tag in [('sdpa', False, 'plain'), ('eager', True, 'attn')]:
                model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, cfg['dtype']),
                    attn_implementation=impl, local_files_only=True).to('cuda').eval()
                for cell, rows in selected_cells.items():
                    ends = [e for r in rows for b, e in r['spans'] if b < e]
                    row = out.setdefault(cell, dict(model=cfg['id'], dtype=cfg['dtype'],
                        median_prefix=int(np.median(ends)), longest_prefix=max(ends)))
                    for label, T in [('median', int(np.median(ends))), ('longest', max(ends))]:
                        timing = timed_forward(model, T, attention)
                        row[f'{tag}_{label}_ms'] = timing['ms']
                        row[f'{tag}_{label}_mib'] = timing['mib']
                        if attention:
                            row[f'headmean_{label}_ms'] = timing['headmean_ms']
                        gc.collect()
                        torch.cuda.empty_cache()
                    write_json(a.out, out)
                    print(cell, tag, flush=True)
                del model
                gc.collect()
                torch.cuda.empty_cache()
    else:
        # m, L and median_step from results/single_step_cost.json
        dimensions = read_json(ROOT / 'results/single_step_cost.json')
        out = {}
        for cell, rows in manifest.items():
            if cell.split('_', 1)[1] not in selected:
                continue
            d = dimensions[cell]
            if a.kind == 'scoring':
                out[cell] = scoring(rows, d['m'], d['L'])
            else:
                ends = [e for r in rows for b, e in r['spans'] if b < e]
                step = d['median_step']
                out[cell] = {}
                for label, T in [('median', int(np.median(ends))), ('longest', max(ends))]:
                    out[cell][label] = timed_q(T, d['L'], min(step, T - 1))
                    gc.collect()
                    torch.cuda.empty_cache()
            write_json(a.out, out)
            print(cell, out[cell], flush=True)


if __name__ == '__main__':
    main()
