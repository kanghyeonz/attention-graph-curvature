"""directed Forman curvature on each step's prefix graph, and the quantile vector q"""
import torch

CURV_EPS = 1e-9
Q5_P = (0., .25, .5, .75, 1.)

def curv_matrix(sub):
    """edge weight max(-log a, CURV_EPS) for every attention a > 0, self loops removed"""
    W = sub.double().clone()
    W.fill_diagonal_(0.0)
    E = W > 0
    out = torch.zeros_like(W)
    out[E] = (-W[E].log()).clamp(min=CURV_EPS)
    return out


def _q5_rows(V, cnt):
    """min, quartiles and max of each row (linear interpolation). rows are sorted
    with the padding at the end, cnt = entries per row, empty rows give 0"""
    out = V.new_zeros((V.shape[0], 5))
    c = cnt.clamp(min=1).double()
    for pi, pq in enumerate(Q5_P):
        pos = pq * (c - 1.0)
        lo = pos.floor()
        hi = torch.minimum(lo + 1.0, c - 1.0)
        vlo = V.gather(1, lo.long().unsqueeze(1)).squeeze(1)
        vhi = V.gather(1, hi.long().unsqueeze(1)).squeeze(1)
        out[:, pi] = vlo + (pos - lo) * (vhi - vlo)
    out[cnt == 0] = 0.0
    return out


def _inf_padded(F, E):
    """float32 rounding, non-edges set to +inf"""
    return torch.where(E, F.float().double(), torch.full_like(F, float("inf")))


def q5_per_token(F, E, b, e):
    """for each token in b:e, the 5 quantiles of its outgoing edge curvatures"""
    V, _ = _inf_padded(F[b:e], E[b:e]).sort(dim=1)
    return _q5_rows(V, E[b:e].sum(1))


def q5_per_step(F, E, b, e):
    """the same 5 quantiles, over the outgoing edges of all tokens of the step together.
    the spectral method uses this one"""
    V, _ = _inf_padded(F[b:e], E[b:e]).reshape(1, -1).sort(dim=1)
    return _q5_rows(V, E[b:e].sum().reshape(1))[0]

def edge_forman_directed(W: torch.Tensor):
    """directed Forman curvature (Sreejith et al.) with unit node weights.
    W[i,j] is the weight of edge i->j, the diagonal is ignored. for e = u->v
        F(e) = 2 - sqrt(w_e) * (S_in(u) + S_out(v))
        S_in(u) = sum_{a->u} w^-1/2, S_out(v) = sum_{v->b} w^-1/2"""
    W = W.double().clone()
    W.fill_diagonal_(0.0)
    E = W > 0
    R = torch.zeros_like(W)
    R[E] = W[E].rsqrt()
    s_in, s_out = R.sum(0), R.sum(1)
    F = 2.0 - W.sqrt() * (s_in.unsqueeze(1) + s_out.unsqueeze(0))
    return torch.where(E, F, torch.zeros_like(F)), E
