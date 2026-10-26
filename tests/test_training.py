import unittest
import torch
from graphcurv.probes import MLPProbe, grid_batched


class TrainingTests(unittest.TestCase):
    def test_grid_matches_separate_fits(self):
        torch.set_num_threads(2)
        rng = torch.Generator().manual_seed(0)
        x = torch.randn(17, 5, generator=rng)
        y = torch.tensor([0., 1., 1., 0., 1., 0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1.])
        xv = torch.randn(9, 5, generator=rng)
        yv = torch.tensor([0., 1., 0., 1., 1., 0., 0., 1., 1.])
        tb = [(x[:7], y[:7]), (x[7:], y[7:])]
        vb = [(xv[:4], yv[:4]), (xv[4:], yv[4:])]
        configs = [(0., .01, 1.), (.01, .1, .3), (.1, .001, 3.)]
        for width in (0, 16, 32):
            actual = grid_batched(5, width, configs, tb, vb, yv, 1,
                                  device='cpu', epochs=12, patience=2)
            for j, (lr, wd, alpha) in enumerate(configs):
                with self.subTest(width=width, config=j):
                    torch.manual_seed(1)
                    model = MLPProbe(5, width)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                    loss = torch.nn.BCEWithLogitsLoss(pos_weight=(y == 0).sum() / (y == 1).sum() * alpha)
                    best, bad = float('inf'), 0
                    for epoch in range(12):
                        for xx, yy in tb:
                            optimizer.zero_grad()
                            loss(model(xx), yy).backward()
                            optimizer.step()
                        with torch.no_grad():
                            value = sum(float(loss(model(xx), yy)) * len(yy) for xx, yy in vb) / len(yv)
                        if value < best - 1e-6:
                            best, bad = value, 0
                            snapshot = {k: v.clone() for k, v in model.state_dict().items()}
                        else:
                            bad += 1
                        if bad >= 2:
                            break
                    model.load_state_dict(snapshot)
                    with torch.no_grad():
                        accuracy = float(((torch.sigmoid(model(xv)) > .5) == yv).float().mean())
                    self.assertEqual(actual[j], accuracy)


if __name__ == '__main__':
    unittest.main()
