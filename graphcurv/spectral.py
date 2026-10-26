"""spectral features of Noël, Geometry of Reason (2026), ported from spectral_trust 0.2.2 to the gpu"""

import numpy as np
import torch

N_SPEC = 5  # fiedler, energy, smoothness, entropy, hfer


def _laplacian(A_sym):
    """L = D - W on the clamped float32 adjacency, self loops kept"""
    A = torch.clamp(A_sym, min=0).to(torch.float32)
    return torch.diag_embed(A.sum(-1)) - A


def prefix_feats(A_sym, hid, spans, cutoff_ratio=0.5):
    """A_sym: symmetrized head-mean attention (T x T), hid: hidden states (T x d).
    one row per step: fiedler, energy, smoothness, entropy, hfer"""
    out = np.full((len(spans), N_SPEC), np.nan, dtype=np.float32)
    for i, (b, e) in enumerate(spans):
        if e <= b:
            continue
        L = _laplacian(A_sym[:e, :e])
        X = hid[:e].to(torch.float32)
        # float32 eigh on cuda often doesn't converge here (many repeated
        # eigenvalues), float64 does. if that fails too, numpy on the cpu
        try:
            evals, evecs = torch.linalg.eigh(L.double())
            evals, evecs = evals.float(), evecs.float()
        except Exception:
            ev, evec = np.linalg.eigh(L.detach().cpu().numpy().astype(np.float64))
            evals = torch.tensor(ev, dtype=torch.float32, device=L.device)
            evecs = torch.tensor(evec, dtype=torch.float32, device=L.device)
        evals = evals.clamp(min=0)

        energy = torch.einsum("ij,ik,jk->", L, X, X)  # Tr(X^T L X)
        signal_norm = (X * X).sum()
        smooth = energy / signal_norm if signal_norm >= 1e-8 else torch.zeros((), device=X.device)

        Xh = evecs.T @ X  # graph fourier transform
        se = (Xh * Xh).sum(1)
        tot = se.sum()
        if tot < 1e-8:
            ent = torch.zeros((), device=X.device)
            hfer = torch.zeros((), device=X.device)
        else:
            p = (se / tot).clamp(min=1e-12)
            ent = -(p * p.log()).sum()
            cut = int((1 - cutoff_ratio) * se.shape[0])
            hfer = se[cut:].sum() / tot
        fiedler = evals[1] if evals.shape[0] > 1 else torch.zeros((), device=X.device)
        out[i] = [float(fiedler), float(energy), float(smooth), float(ent), float(hfer)]
    return out
