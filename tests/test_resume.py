from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from graphcurv import collect, train
from graphcurv.io import write_json


class ResumeTests(unittest.TestCase):
    # one tiny cell with 6 solutions, only hidden-logreg and seed 0
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cell = 'gsm8k_Test'
        rows = [dict(sid=str(i), group=str(i), label=-1 if i % 2 == 0 else 0,
                     n_steps=1, T=2, spans=[[1, 2]]) for i in range(6)]
        parts = dict(fit=['0', '1'], validation=['2', '3'], test=['4', '5'])
        write_json(self.root / 'experiments/manifest.json', {self.cell: rows})
        write_json(self.root / 'experiments/fold_plan.json', dict(seeds=[0], cells={
            self.cell: {'0': {'rounds': [dict(parts=parts) for _ in range(5)]}}}))
        feat = self.root / 'features' / self.cell / 'feat'
        feat.mkdir(parents=True)
        for row in rows:
            np.savez_compressed(feat / (row['sid'] + '.npz'), hid_seq=np.full((2, 2), int(row['sid']), np.float32),
                                spec=np.zeros((1, 1, 5), np.float32), q5_tok=np.zeros((2, 1, 5), np.float32))

    def fit(self):
        args = ['train', '--cell', self.cell, '--features', str(self.root / 'features'),
                '--out', str(self.root / 'runs'), '--device', 'cpu', '--methods', 'hidden-logreg', '--seeds', '0']
        with patch.object(train, 'ROOT', self.root), patch('sys.argv', args):
            train.main()

    def test_rerun_skips_finished_folds(self):
        self.fit()
        with patch.object(train, 'evaluate', wraps=train.evaluate) as fits:
            self.fit()
        self.assertEqual(fits.call_count, 0)

    def test_collect(self):
        self.fit()
        with patch.object(collect, 'ROOT', self.root), patch.object(collect, 'METHODS', ('hidden-logreg',)):
            result = collect.collect(self.root / 'runs')
        self.assertEqual(len(result['cells'][self.cell]['results']['hidden-logreg']['A_seeds']), 1)


if __name__ == '__main__':
    unittest.main()
