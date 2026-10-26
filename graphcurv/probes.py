"""hidden-state probes: last-token MLP (COLM'25 style) and ReProbe-style encoder.
mlp loss: weighted BCE, weight w*alpha on the correct class"""
import numpy as np
import torch
import torch.nn as nn

class MLPProbe(nn.Module):
    """last-token probe of Zhang et al. (COLM 2025), h=0 is linear"""

    def __init__(self, d, h=32):
        super().__init__()
        self.net = (nn.Linear(d, 1) if h == 0 else
                    nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1)))

    def forward(self, x):
        return self.net(x).squeeze(-1)


# adapted from luh/heads/uncertainty_head_claim.py in IINemo/llm-uncertainty-head (MIT, see licenses/)
class ReProbeClaim(nn.Module):
    """ReProbe claim head (Ni et al.), one step at a time: the step's tokens get
    the marker and the score is the mean of the encoder outputs over them"""

    def __init__(self, d, head_dim=512, n_layers=1, n_heads=16, dropout=0.1,
                 mask_future_tokens=True):
        super().__init__()
        self.mask_future_tokens = mask_future_tokens
        self.proj = nn.Sequential(
            nn.Linear(d, head_dim * 2), nn.LayerNorm(head_dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim * 2, head_dim), nn.LayerNorm(head_dim), nn.GELU(),
        )
        self.entity_embedding = nn.Embedding(2, head_dim)
        layer = nn.TransformerEncoderLayer(head_dim, n_heads, dropout=dropout,
                                           activation="gelu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            layer, n_layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.Linear(head_dim, head_dim), nn.LayerNorm(head_dim), nn.GELU(),
            nn.Dropout(p=dropout), nn.Linear(head_dim, 1),
        )

    def forward(self, x, pad_mask, step_masks):
        """x [B,T,d], pad_mask [B,T] True=pad, step_masks [B,S,T] 1 on the tokens of
        each step (all-zero row for a padded step slot). returns [B,S] logits"""
        B, S, T = step_masks.shape
        h = self.proj(x)  # [B,T,H]
        m = step_masks.reshape(B * S, T)
        inp = h.unsqueeze(1).expand(B, S, T, h.shape[-1]).reshape(B * S, T, -1) \
            + self.entity_embedding(m)
        pd = pad_mask.unsqueeze(1).expand(B, S, T).reshape(B * S, T).clone()
        if self.mask_future_tokens:
            # upstream has pd &= cum, which drops the padding mask (a step's score then
            # depends on the other steps in its batch)
            cum = torch.flip(torch.cummax(torch.flip(m.int(), dims=[1]), dim=1)[0],
                             dims=[1]).bool()
            pd |= ~cum
            pd[:, 0] = False  # never mask all keys of a row
        z = self.transformer_encoder(inp, src_key_padding_mask=pd)
        w = m.unsqueeze(-1).to(z.dtype)
        pooled = (z * w).sum(1) / w.sum(1).clamp(min=1)
        return self.classifier(pooled).reshape(B, S)


def doc_loss(logits, y, w, alpha=1.0):
    pos_weight = torch.as_tensor(w * alpha, dtype=logits.dtype, device=logits.device)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)


def train_probe(model, batches, epochs=10, lr=1e-3, seed=0, device="cpu", alpha=1.0,
                weight_decay=0.0, val_batches=None, patience=0):
    torch.manual_seed(seed)
    model.to(device)
    ys = torch.cat([b[-1] for b in batches]).float()
    w = ((ys == 0).sum() / (ys == 1).sum().clamp(min=1)).item()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    batches = [[t.to(device) for t in b] for b in batches]
    # early stopping on the validation loss
    best, best_state, bad, best_epoch = None, None, 0, None
    for epoch in range(epochs):
        model.train()
        for b in batches:
            y = b[-1].float()
            loss = doc_loss(model(*b[:-1]), y, w, alpha)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val_batches:
            model.eval()
            with torch.no_grad():
                vl = float(np.average([
                    doc_loss(model(*[t.to(device) for t in vb[:-1]]),
                             vb[-1].float().to(device), w, alpha).item()
                    for vb in val_batches], weights=[len(vb[-1]) for vb in val_batches]))
            if best is None or vl < best - 1e-6:
                best, bad = vl, 0
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if patience and bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.best = dict(epoch=best_epoch, validation_loss=best, epochs_run=epoch+1)
    return model


@torch.no_grad()
def score_probe(model, batches, device="cpu"):
    model.eval()
    out = []
    for b in batches:
        b = [t.to(device) for t in b]
        out.append(torch.sigmoid(model(*b[:-1])).cpu())
    return torch.cat(out)


def _adam_step(p, g, m, v, t, lr, wd, b1=0.9, b2=0.999, eps=1e-8):
    """adam step for K configs at once"""
    g = g + wd * p
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    bias1 = 1 - b1 ** t
    bias2 = 1 - b2 ** t
    denom = (v.sqrt() / (bias2 ** 0.5)).add_(eps)
    p.sub_((lr / bias1) * (m / denom))


def grid_batched(d, h, configs, tb, vb, yv, seed, device="cpu", epochs=200, patience=10):
    """train all (lr, wd, alpha) configs of width h together on the same batches.
    returns the validation accuracy of each config, yv = validation labels in vb order"""
    K = len(configs)
    lr = torch.tensor([c[0] for c in configs], dtype=torch.float32, device=device)
    wd = torch.tensor([c[1] for c in configs], dtype=torch.float32, device=device)
    al = torch.tensor([c[2] for c in configs], dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    ref = MLPProbe(d, h=h)  # same init for every config
    if h == 0:
        P = {"W": ref.net.weight.detach().t().contiguous(), "b": ref.net.bias.detach()}
    else:
        P = {"W1": ref.net[0].weight.detach().t().contiguous(), "b1": ref.net[0].bias.detach(),
             "W2": ref.net[2].weight.detach().t().contiguous(), "b2": ref.net[2].bias.detach()}
    params = {k: v.to(device).unsqueeze(0).repeat(K, *([1] * v.dim())).clone().requires_grad_(True) for k, v in P.items()}
    state = {k: (torch.zeros_like(v), torch.zeros_like(v)) for k, v in params.items()}
    # [K] vector -> broadcastable against a [K, ...] parameter
    def shp(x, p):
        return x.view(x.numel(), *([1] * (p.dim() - 1)))
    def forward(X):
        if h == 0:
            return torch.einsum("bd,kdo->kbo", X, params["W"]).squeeze(-1) + params["b"]
        hid = torch.relu(torch.einsum("bd,kdh->kbh", X, params["W1"]) + params["b1"].unsqueeze(1))
        return torch.einsum("kbh,kho->kbo", hid, params["W2"]).squeeze(-1) + params["b2"]
    def loss_per_config(logits, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y.unsqueeze(0).expand_as(logits),
            pos_weight=(al * w).unsqueeze(1), reduction="none").mean(1)
    tb = [(X.to(device), y.to(device).float()) for X, y in tb]
    vb = [(X.to(device), y.to(device).float()) for X, y in vb]
    ys = torch.cat([y for _, y in tb])
    w = ((ys == 0).sum() / (ys == 1).sum().clamp(min=1)).item()
    best = torch.full((K,), float("inf"), device=device)
    bad = torch.zeros(K, dtype=torch.long, device=device)
    snap = {k: v.detach().clone() for k, v in params.items()}
    t = 0
    active = torch.arange(K, device=device)  # configs still training
    for _ in range(epochs):
        for X, y in tb:
            t += 1
            L = loss_per_config(forward(X), y).sum()
            grads = torch.autograd.grad(L, list(params.values()))
            with torch.no_grad():
                for (k, p), g in zip(params.items(), grads):
                    m, v = state[k]
                    _adam_step(p, g, m, v, t, shp(lr, p), shp(wd, p))
        with torch.no_grad():
            vl = sum(loss_per_config(forward(X), y) * len(y) for X, y in vb) / sum(len(y) for _, y in vb)
            improved = vl < best[active] - 1e-6
            best[active] = torch.where(improved, vl, best[active])
            bad[active] = torch.where(improved, torch.zeros_like(bad[active]), bad[active] + 1)
            for k, p in params.items():
                snap[k][active] = torch.where(shp(improved, p), p, snap[k][active])
            keep = bad[active] < patience
            if not bool(keep.all()):
                active = active[keep]
                if active.numel() == 0:
                    break
                # drop the stopped configs from the batch
                for k in list(params):
                    params[k] = params[k].detach()[keep].clone().requires_grad_(True)
                    state[k] = (state[k][0][keep].clone(), state[k][1][keep].clone())
                lr, wd, al = lr[keep], wd[keep], al[keep]
    with torch.no_grad():
        params = {k: snap[k].clone() for k in snap}
        pv = torch.cat([torch.sigmoid(forward(X)) for X, _ in vb], 1)  # [K, n_val]
        acc = ((pv > 0.5).float() == yv.to(device).float().unsqueeze(0)).float().mean(1)
    return [float(a) for a in acc.cpu()]
