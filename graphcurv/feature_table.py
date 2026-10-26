import numpy as np

def table_for(method, rows, features):
    result = {}
    for s in rows:
        f = features[s['sid']]
        q = f['q5_tok'].reshape(s['T'], -1).astype(np.float32)
        h = f['hid_seq'].astype(np.float32) if not method.startswith('spec') else None
        with_q = '+' in method
        if method.startswith('probe-reprobe'):
            result[s['sid']] = np.concatenate((h, q), axis=1) if with_q else h
            continue
        if method.startswith('spec'):
            x = f['spec'].reshape(s['n_steps'], -1)
            if with_q:
                x = np.concatenate((x, f['q5_step'].reshape(s['n_steps'], -1)), axis=1)
        else:
            seq = np.concatenate((h, q), axis=1) if with_q else h
            x = np.full((s['n_steps'], seq.shape[1]), np.nan, dtype=np.float32)
            for j, (b, e) in enumerate(s['spans']):
                if e > b:
                    x[j] = seq[e - 1]
        result[s['sid']] = x
    return result
