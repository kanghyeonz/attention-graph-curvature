import numpy as np
from . import curvature as F

def layer_curvature(attention, spans):
    T = attention.shape[0]
    qt = np.zeros((T, 5), dtype=np.float32)
    qs = np.zeros((len(spans), 5), dtype=np.float32)
    for j, (b, e) in enumerate(spans):
        if e <= b:
            continue
        W = F.curv_matrix(attention[:e, :e])
        curvature, edges = F.edge_forman_directed(W)
        qt[b:e] = F.q5_per_token(curvature, edges, b, e).float().cpu().numpy()
        qs[j] = F.q5_per_step(curvature, edges, b, e).float().cpu().numpy()
    return qt, qs


class CaptureMatrices:
    """hooks on every self_attn, keeps the eager weights without output_attentions=True"""
    def __init__(self, model):
        self.modules = [m for name, m in model.named_modules() if name.endswith('self_attn')]
        self.attentions, self.symmetric, self.handles = [], [], []

    def hook(self, _module, _args, output):
        if output[1] is None:
            raise RuntimeError('no attention weights, load the model with eager attention')
        heads = output[1].detach().cpu().float()
        mean_attention = heads[0].mean(0)
        self.attentions.append(mean_attention)
        # plain mean over heads, then symmetrize
        self.symmetric.append(0.5 * (mean_attention + mean_attention.T))
        return (output[0], None) + tuple(output[2:])

    def __enter__(self):
        self.handles = [m.register_forward_hook(self.hook) for m in self.modules]
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
