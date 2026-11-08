"""
Exact decomposition of the single-block ERF gradient into the frozen-selectivity term and
the selectivity pathways (derived in App. L.6).

For tau >= 1, u_{T-tau} reaches l = ||logits_T||^2 only through the scan inputs xi_s of
the convolution window, s = T - tau + k (k = 0..3, s <= T). With nothing held fixed, the
gradient with respect to xi_s is 2 gamma_s, where gamma_s is the sum of five parts:

  frozen    a_T * K_s,  K_{s,j} = sum_i C_{T,i} P_{s,j,i} Delta_{s,j} B_{s,i} + D_j [s = T]
  dt_ret    (a_T * R_s * sigmoid(q_s)) @ W_dt,
            R_{s,j} = sum_i C_{T,i} P_{s,j,i} A_{ji} e^{A_{ji} Delta_{s,j}} h_{s-1,j,i}
  dt_write  (a_T * V_s * sigmoid(q_s)) @ W_dt,  V_{s,j} = xi_{s,j} sum_i C_{T,i} P_{s,j,i} B_{s,i}
  B         Q_s @ W_B,  Q_{s,i} = C_{T,i} sum_j a_{T,j} P_{s,j,i} Delta_{s,j} xi_{s,j}
  C         [s = T] H @ W_C,  H_i = sum_j a_{T,j} h_{T,j,i}

Here P_{s,j,i} = prod_{r=s+1}^{T} e^{A_{ji} Delta_{r,j}}, q_s = W_dt xi_s + b_dt (so
Delta_s = softplus(q_s)), and a_{T,j} = silu(z_{T,j}) [W_out^T W_lm^T logits_T]_j.
dt_ret is the derivative of the decay at the source step (it multiplies the stored state
h_{s-1}); dt_write is the derivative of the write Delta_s B_s xi_s through Delta_s.
Every part passes through the same linear input map (silu'(p_s), the convolution taps,
W_x^T), so the gradient with respect to u_{T-tau} splits exactly into five vectors whose
sum is the full gradient.
"""

import torch
import torch.nn.functional as F

PARTS = ['frozen', 'dt_ret', 'dt_write', 'B', 'C']


def _inputs(m, ids):
    extract = m.extract_mode
    m.extract_mode = True
    emb = m.embedding(ids)
    logits, inter = m(embeddings=emb)
    m.extract_mode = extract
    L = ids.shape[1]
    xs, z = m.in_proj(emb).split(m.d_inner, dim=-1)
    pre = m.conv1d(xs.transpose(1, 2))[:, :, :L].transpose(1, 2)
    xi = F.silu(pre)
    q = m.dt_proj(xi)
    sig = torch.sigmoid(pre)
    return dict(logits=logits, z=z, pre=pre, xi=xi, q=q, dsilu=sig * (1 + pre * (1 - sig)),
                dt=inter['delta_t'], B=inter['B_t'], C=inter['C_t'], h=inter['x'])


def part_gradients(m, ids, T, max_lag, inp=None, dl_dlogits=None):
    """Gradient of a probe at T w.r.t. u_{T-tau}, tau = 0..max_lag-1, split into PARTS.
    The probe defaults to ||logits_T||^2. dl_dlogits ([batch, vocab]) gives any other probe
    whose derivative with respect to logits_T is fixed, such as a task margin.
    Returns part -> [batch, max_lag, d_model]. Lag 0 omits the residual and gate paths."""
    inp = inp or _inputs(m, ids)
    dev, A = ids.device, m.A
    te = torch.arange(max_lag, device=dev)
    s = T - te
    dt, Bt, Ct, xi, h = inp['dt'], inp['B'], inp['C'], inp['xi'], inp['h']
    cum = torch.cumsum((dt[..., None] * A).double(), dim=1)            # accumulate in float64
    P = torch.exp(cum[:, T][:, None] - cum[:, s]).to(dt.dtype)            # [b, n, j, i]
    h_prev = torch.where((s > 0)[None, :, None, None], h[:, (s - 1).clamp(min=0)],
                         torch.zeros((), dtype=h.dtype, device=dev))
    CT = Ct[:, T][:, None, None, :]
    half = inp['logits'][:, T] if dl_dlogits is None else dl_dlogits / 2   # (dl/dlogits_T) / 2
    a = F.silu(inp['z'][:, T]) * ((half @ m.lm_head.weight) @ m.out_proj.weight)
    sig_q = torch.sigmoid(inp['q'][:, s])
    dts, xis, Bs = dt[:, s], xi[:, s], Bt[:, s]
    K = (CT * P * dts[..., None] * Bs[:, :, None, :]).sum(-1)
    K[:, 0] += m.D
    R = (CT * P * A * torch.exp(dts[..., None] * A) * h_prev).sum(-1)
    V = xis * (CT * P * Bs[:, :, None, :]).sum(-1)
    Q = Ct[:, T][:, None, :] * (a[:, None, :, None] * P * (dts * xis)[..., None]).sum(2)
    gam = {'frozen': a[:, None, :] * K,
           'dt_ret': (a[:, None, :] * R * sig_q) @ m.dt_proj.weight,
           'dt_write': (a[:, None, :] * V * sig_q) @ m.dt_proj.weight,
           'B': Q @ m.B_proj.weight,
           'C': torch.zeros_like(K)}
    gam['C'][:, 0] = (a[:, :, None] * h[:, T]).sum(1) @ m.C_proj.weight
    w, W_x, ds = m.conv1d.weight[:, 0, :], m.in_proj.weight[:m.d_inner], inp['dsilu'][:, s]
    out = {}
    for k_, g in gam.items():
        gp = 2 * g * ds
        gx = w[:, 3] * gp
        for k in range(1, 4):
            gx[:, k:] += w[:, 3 - k] * gp[:, :-k]
        out[k_] = gx @ W_x
    return out


@torch.no_grad()
def decomposition_curves(m, ids, targets, max_lag=1024):
    """Per-sequence means over targets of every part's energy ||g_c||^2, every pairwise
    term 2<g_c, g_d>, and the total ||sum_c g_c||^2: dict -> [batch, max_lag] float64."""
    inp = _inputs(m, ids)
    keys = ([f'E_{c}' for c in PARTS] + [f'X_{c}_{d}' for i, c in enumerate(PARTS)
                                          for d in PARTS[i + 1:]] + ['total'])
    acc = {k: torch.zeros(ids.shape[0], max_lag, dtype=torch.float64, device=ids.device) for k in keys}
    for T in targets:
        g = part_gradients(m, ids, T, max_lag, inp)
        for i, c in enumerate(PARTS):
            acc[f'E_{c}'] += (g[c] ** 2).sum(-1).double()
            for d in PARTS[i + 1:]:
                acc[f'X_{c}_{d}'] += 2 * (g[c] * g[d]).sum(-1).double()
        acc['total'] += (sum(g.values()) ** 2).sum(-1).double()
    return {k: v / len(targets) for k, v in acc.items()}
