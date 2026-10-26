import tempfile
import unittest
from pathlib import Path
from graphcurv.io import ROOT, read_json
from graphcurv.report import aggregate, report


class ReportTests(unittest.TestCase):
    def test_table2(self):
        cells = read_json(ROOT / 'results/paper.json')['cells']
        table2 = {'spec-logreg': .3261, 'spec+curvstepq5-logreg': .3474, 'hidden-logreg': .1828,
                  'hidden+curvq5-logreg': .1907, 'probe-mlp': .2991, 'probe-mlp+curvq5': .3037,
                  'probe-reprobe': .2932, 'probe-reprobe+curvq5': .3472}
        for method, value in table2.items():
            self.assertEqual(round(aggregate(cells, method)[0], 4), value)

    def test_report_files(self):
        with tempfile.TemporaryDirectory() as d:
            report(read_json(ROOT / 'results/paper.json'), Path(d), plot=False)
            names = sorted(p.name for p in Path(d).iterdir())
        self.assertEqual(names, ['figure2.csv', 'summary.json'] + [f'table{i}.csv' for i in range(2, 10)])


if __name__ == '__main__':
    unittest.main()
