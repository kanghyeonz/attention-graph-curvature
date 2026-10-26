import unittest
import numpy as np
import torch
from graphcurv import curvature as C
from graphcurv.controls import layer_controls
from graphcurv.capture import layer_curvature
from graphcurv.metric import solution_counts, choose_threshold, pooled_A
from graphcurv.batched import curv_batched, q_batched


class MathTests(unittest.TestCase):
    def test_timing_kernels_match_extraction(self):
        torch.manual_seed(0)
        a = torch.tril(torch.rand(3, 11, 11), -1)
        f, edges = curv_batched(a)
        qt, qs = q_batched(f, edges, 3, 11)
        for layer in range(3):
            reference, mask = C.edge_forman_directed(C.curv_matrix(a[layer]))
            torch.testing.assert_close(reference, f[layer], rtol=0, atol=0)
            torch.testing.assert_close(C.q5_per_token(reference, mask, 3, 11), qt[layer], rtol=0, atol=0)
            torch.testing.assert_close(C.q5_per_step(reference, mask, 3, 11), qs[layer], rtol=0, atol=0)

    def test_directed_formula(self):
        w = torch.tensor([[0., 0., 0., 0.], [2., 0., 0., 0.],
                          [3., 5., 0., 0.], [7., 11., 13., 0.]], dtype=torch.float64)
        actual, edges = C.edge_forman_directed(w)
        for u, v in edges.nonzero().tolist():
            incoming = sum(float(w[a, u])**-.5 for a in range(4) if w[a, u] > 0)
            outgoing = sum(float(w[v, b])**-.5 for b in range(4) if w[v, b] > 0)
            self.assertAlmostEqual(float(actual[u, v]), 2 - float(w[u, v])**.5 * (incoming + outgoing))
        self.assertEqual(float(actual[3, 0]), 2.)
        self.assertEqual(float(actual[3].max()), 2.)

    def test_quantiles_against_loops(self):
        rng = np.random.default_rng(0)
        for size in (3, 7, 11):
            a = np.tril(rng.random((size, size))).astype(np.float32)
            a[rng.random(a.shape) < .6] = 0
            for end in (size - 1, size):
                weights = np.zeros((end, end))
                for u in range(end):
                    for v in range(u):
                        if a[u, v] > 0:
                            weights[u, v] = max(-np.log(float(a[u, v])), 1e-9)
                expected = [[] for _ in range(end)]
                for u in range(end):
                    for v in range(end):
                        if weights[u, v] == 0:
                            continue
                        neighbors = [weights[k, u] for k in range(end) if weights[k, u] > 0]
                        neighbors += [weights[v, k] for k in range(end) if weights[v, k] > 0]
                        value = 2 - weights[u, v] ** .5 * sum(w ** -.5 for w in neighbors)
                        expected[u].append(float(np.float32(value)))
                f, edges = C.edge_forman_directed(C.curv_matrix(torch.from_numpy(a[:end, :end])))
                for begin in (0, end - 1):
                    actual = C.q5_per_token(f, edges, begin, end).numpy()
                    want = [np.quantile(v, [0, .25, .5, .75, 1]) if v else np.zeros(5)
                            for v in expected[begin:]]
                    np.testing.assert_allclose(actual, want, rtol=2e-7, atol=1e-6)
                    pooled = [v for row in expected[begin:] for v in row]
                    want = np.quantile(pooled, [0, .25, .5, .75, 1]) if pooled else np.zeros(5)
                    np.testing.assert_allclose(C.q5_per_step(f, edges, begin, end).numpy(),
                                               want, rtol=2e-7, atol=1e-6)

    def test_prefix_restriction(self):
        a = torch.tensor([[1., 0., 0., 0.], [.5, .5, 0., 0.],
                          [.2, .3, .5, 0.], [.1, .2, .3, .4]])
        q, _ = layer_curvature(a, [[1, 3], [3, 4]])
        changed = a.clone()
        changed[3] = torch.tensor([.7, .1, .1, .1])
        q2, _ = layer_curvature(changed, [[1, 3], [3, 4]])
        np.testing.assert_array_equal(q[:3], q2[:3])
        f, e = C.edge_forman_directed(C.curv_matrix(a))
        full = C.q5_per_token(f, e, 1, 3).numpy()
        self.assertFalse(np.array_equal(q[1:3], full))

    def test_controls_and_eps(self):
        a = np.array([[1, 0, 0], [.25, .75, 0], [.25, .25, .5]], dtype=np.float32)
        f = layer_controls(a, [[0, 1], [1, 3]])
        np.testing.assert_array_equal(f['stats3_tok'], [[0, 0, 0], [0, 1, 1], [1, 0, .5]])
        np.testing.assert_allclose(f['neglog_step'][1], np.log(4))
        np.testing.assert_array_equal(f['neglog_tok'][0], np.zeros(5))
        self.assertEqual(float(C.curv_matrix(torch.tensor([[1., 0.], [1., 0.]]))[1, 0]), 1e-9)

    def test_metric(self):
        rows = [dict(sid='a', label=-1), dict(sid='b', label=1), dict(sid='c', label=1)]
        scores = {'a': np.array([.5, .6]), 'b': np.array([.8, .4]), 'c': np.array([.7, np.nan])}
        self.assertEqual(solution_counts(rows, scores, .5), (1, 1, 1, 2))
        self.assertEqual(pooled_A((1, 1, 1, 2)), 2 / 3)
        self.assertEqual(pooled_A((0, 1, 0, 2)), 0.)
        self.assertEqual(choose_threshold(rows[:1], scores), .5)


if __name__ == '__main__':
    unittest.main()
