import unittest
import numpy as np
import torch
from graphcurv.io import ROOT, read_json
from graphcurv.conditions import METHODS, table_for, evaluate


def fixture():
    rng = np.random.default_rng(0)
    rows, features = [], {}
    for i in range(10):
        row = dict(sid=str(i), group=str(i), label=-1 if i % 2 == 0 else 1,
                   n_steps=2, T=5, spans=[[1, 3], [3, 5]])
        rows.append(row)
        f = dict(hid_seq=rng.normal(size=(5, 4)).astype(np.float32),
                 spec=rng.normal(size=(2, 2, 5)).astype(np.float32))
        for name, width in [('q5', 5), ('neglog', 5), ('stats3', 3)]:
            f[name + '_tok'] = rng.normal(size=(5, 2, width)).astype(np.float32)
            f[name + '_step'] = rng.normal(size=(2, 2, width)).astype(np.float32)
        features[str(i)] = f
    return rows, features


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_saved_partitions(self):
        manifest = read_json(ROOT / 'experiments/manifest.json')
        plan = read_json(ROOT / 'experiments/fold_plan.json')
        for cell, rows in manifest.items():
            group = {r['sid']: r['group'] for r in rows}
            for seed in map(str, range(10)):
                seen = []
                for i, r in enumerate(plan['cells'][cell][seed]['rounds']):
                    p = r['parts']
                    self.assertEqual(r['validation_fold'], (i + 1) % 5)
                    sets = [set(p[k]) for k in ['fit', 'validation', 'test']]
                    self.assertEqual(set.union(*sets), set(group))
                    gs = [{group[s] for s in v} for v in sets]
                    for a in range(3):
                        for b in range(a):
                            self.assertFalse(gs[a] & gs[b])
                    seen += p['test']
                self.assertEqual(sorted(seen), sorted(group))

    def test_all_methods_run(self):
        rows, features = fixture()
        for method in METHODS:
            with self.subTest(method=method):
                report, scores = evaluate(method, rows, table_for(method, rows, features), [0, 1], [2, 3], 0, 'cpu', True)
                self.assertEqual(len(report['counts']), 4)
                self.assertEqual(set(scores), {r['sid'] for r in rows[4:]} | {'0', '1'})


if __name__ == '__main__':
    unittest.main()
