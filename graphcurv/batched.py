import torch

EPS = 1e-9
_LEVELS = {}

def curv_batched(A):
    # A is [L, T, T], the head-mean attention of each layer
    W = A.double().clone()
    idx = torch.arange(A.shape[-1], device=A.device)
    W[:, idx, idx] = 0.0
    E = W > 0
    Wt = torch.zeros_like(W)
    Wt[E] = (-W[E].log()).clamp(min=EPS)
    R = torch.zeros_like(Wt)
    R[E] = Wt[E].rsqrt()
    s_in, s_out = R.sum(-2), R.sum(-1)
    F = 2.0 - Wt.sqrt() * (s_in[:, :, None] + s_out[:, None, :])
    return torch.where(E, F, torch.zeros_like(F)), E

def q5_rows_batched(V, cnt):
    # rows of V sorted with the +inf padding at the end, cnt = real entries per row
    if V.device not in _LEVELS:
        _LEVELS[V.device] = torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64, device=V.device)
    P = _LEVELS[V.device]
    c = cnt.clamp(min=1).double()
    pos = P[None, :] * (c[:, None] - 1.0)
    lo = pos.floor()
    hi = torch.minimum(lo + 1.0, (c - 1.0)[:, None])
    vlo = V.gather(1, lo.long())
    vhi = V.gather(1, hi.long())
    out = vlo + (pos - lo) * (vhi - vlo)
    out[cnt == 0] = 0.0
    return out

def q_batched(F, E, b, e):
    # token q [L,n,5] and step q [L,5] for the tokens b:e
    L, T, _ = F.shape
    blk = torch.where(E[:, b:e], F[:, b:e].float().double(), torch.full_like(F[:, b:e], float("inf")))
    Vt, _ = blk.reshape(L * (e - b), T).sort(dim=1)
    qt = q5_rows_batched(Vt, E[:, b:e].sum(-1).reshape(-1)).reshape(L, e - b, 5)
    Vs, _ = blk.reshape(L, (e - b) * T).sort(dim=1)
    qs = q5_rows_batched(Vs, E[:, b:e].sum((-1, -2)))
    return qt, qs
