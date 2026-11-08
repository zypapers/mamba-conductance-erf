"""
Closed forms for the frozen-selectivity ERF of a single Mamba block
(derived in App. L.5).

Hold Delta_t, B_t, C_t fixed. For tau >= 1, the gradient of
l = ||logits_T||^2 with respect to the embedding u_{T-tau} is 2 W_x^T v, with

    v_j     = a_{T,j} sum_{k=0}^{3} w_{j,3-k} silu'(p_{s,j}) K_{T,s,j},   s = T - tau + k <= T,
    K_{T,s,j} = sum_i C_{T,i} Bbar_{s,j,i} prod_{r=s+1}^{T} (1 - g_{r,j,i}) + D_j [s = T],
    a_{T,j} = silu(z_{T,j}) [W_out^T W_lm^T logits_T]_j,

where W_x is the x half of in_proj, w the depthwise conv taps, p the conv
pre-activation and Bbar = Delta B (the trained recurrence). The ERF is
4 E[v^T W_x W_x^T v]. The scan probe of Eq. (15) is 4 h_pred =
4 sum_j y_{T,j}^2 K_{T,T-tau,j}^2. Products are formed in log space,
log(1 - g) = A Delta.
"""

import torch
import torch.nn.functional as F

VARIANTS = ['S', 'Su', 'R', 'P', 'X', 'Xu']


@torch.no_grad()
def block_erf_curves(m, ids, targets, max_lag=1024):
    """
    Per-sequence curves, mean over targets: dict variant -> [batch, max_lag]
    (float64). Variants:
      S   scan probe 4 h_pred (Eq. 15)
      Su  S with C_{T,i} Bbar_{s,j,i} -> |C_{T,i} Bbar_{s,j,i}|
      R   S with readout a_T instead of y_T
      P   block form without cross-channel terms: 4 sum_j v_j^2 ||W_x[j]||^2
      X   block closed form ||2 W_x^T v||^2
      Xu  X with C_{T,i} Bbar_{s,j,i} -> |C_{T,i} Bbar_{s,j,i}|
    Every target needs T >= max_lag - 1. Uses the reference scan (extract mode).
    """
    extract = m.extract_mode
    m.extract_mode = True
    emb = m.embedding(ids)
    logits, inter = m(embeddings=emb)
    m.extract_mode = extract
    L = ids.shape[1]
    xs, z = m.in_proj(emb).split(m.d_inner, dim=-1)
    pre = m.conv1d(xs.transpose(1, 2))[:, :, :L].transpose(1, 2)
    sig = torch.sigmoid(pre)
    dsilu = sig * (1 + pre * (1 - sig))
    y, dt, C_t, b_bar = inter['y'], inter['delta_t'], inter['C_t'], inter['B_bar']
    cum = torch.cumsum((dt[..., None] * m.A).double(), dim=1)
    W_x = m.in_proj.weight[:m.d_inner]
    w = m.conv1d.weight[:, 0, :]
    wnorm = (W_x ** 2).sum(-1)
    taus = torch.arange(max_lag, device=ids.device)
    acc = {v: torch.zeros(ids.shape[0], max_lag, device=ids.device, dtype=torch.float64)
           for v in VARIANTS}
    for T in targets:
        starts = T - taus
        ret = torch.exp(cum[:, T][:, None] - cum[:, starts]).float()
        cb = C_t[:, T][:, None, None, :] * b_bar[:, starts]
        a = F.silu(z[:, T]) * ((logits[:, T] @ m.lm_head.weight) @ m.out_proj.weight)
        ysq = y[:, T] ** 2
        for signed, (sv, bv) in ((True, ('S', 'X')), (False, ('Su', 'Xu'))):
            K = ((cb if signed else cb.abs()) * ret).sum(-1)
            K[:, 0] += m.D
            acc[sv] += 4 * (ysq[:, None, :] * K ** 2).sum(-1).double()
            if signed:
                acc['R'] += 4 * ((a ** 2)[:, None, :] * K ** 2).sum(-1).double()
            Kw = dsilu[:, starts] * K
            G = w[:, 3] * Kw
            for k in range(1, 4):
                G[:, k:] += w[:, 3 - k] * Kw[:, :-k]
            v = a[:, None, :] * G
            acc[bv] += ((2 * v @ W_x) ** 2).sum(-1).double()
            if signed:
                acc['P'] += 4 * (v ** 2 * wnorm).sum(-1).double()
        del ret, cb
    return {k: acc[k] / len(targets) for k in VARIANTS}
