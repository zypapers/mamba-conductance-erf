"""
Pure PyTorch sequential scan for selective SSMs.

No fused CUDA kernels: every intermediate (A_bar, B_bar, x_t)
is a standard PyTorch tensor with gradients flowing through it.

The scan implements the recurrence:
    x_{t+1} = A_bar_t @ x_t + B_bar_t * u_t
    y_t     = C_t @ x_t + D * u_t

where A_bar_t = exp(A * delta_t) [d_inner, d_state], and the input weight is
    'euler' (default): B_bar_t = delta_t * B_t, which is what the mamba_ssm CUDA
        kernel used for training and evaluation computes (deltaB_u = delta * B * u
        in mamba_ssm's own selective_scan_ref);
    'zoh': B_bar_t = (exp(A*delta_t) - I) * A^{-1} * B_t, the exact ZOH input
        weight. No checkpoint was trained with it; it is kept for comparison
        (scripts/check_discretization.py).
"""

import torch
from torch import Tensor


def selective_scan_ref(
    A: Tensor,          # [d_inner, d_state] — diagonal entries (negative)
    B_t: Tensor,        # [batch, L, d_state] — input-dependent
    C_t: Tensor,        # [batch, L, d_state] — input-dependent
    delta_t: Tensor,    # [batch, L, d_inner] — input-dependent step sizes
    u: Tensor,          # [batch, L, d_inner] — input after conv + SiLU
    D: Tensor | None = None,  # [d_inner] — skip connection
    input_weight: str = 'euler',  # 'euler' (trained recurrence) or 'zoh'
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Pure PyTorch sequential scan.

    Returns:
        y: [batch, L, d_inner] — output sequence
        intermediates: dict with keys:
            'A_bar': [batch, L, d_inner, d_state] — discretized A
            'B_bar': [batch, L, d_inner, d_state] — discretized B
            'G':     [batch, L, d_inner, d_state] — shunt conductances
            'x':     [batch, L, d_inner, d_state] — hidden states
            'y':     [batch, L, d_inner] — output, same tensor as the first return
    """
    batch, L, d_inner = u.shape
    d_state = A.shape[1]

    # --- Discretization: A_bar is ZOH in both modes ---
    # A is [d_inner, d_state], delta_t is [batch, L, d_inner]
    # A_delta: [batch, L, d_inner, d_state]
    A_delta = delta_t.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
    A_bar = torch.exp(A_delta)  # [batch, L, d_inner, d_state]

    # Shunt conductance: G = I - A_bar = I - exp(A * delta)
    G = 1.0 - A_bar  # [batch, L, d_inner, d_state]

    # B_t is [batch, L, d_state] — broadcast over d_inner
    # B_bar: [batch, L, d_inner, d_state]
    if input_weight == 'euler':
        # B_bar = delta * B_t, as in the mamba_ssm CUDA kernel
        B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(2)
    elif input_weight == 'zoh':
        # B_bar = (exp(A*delta) - I) * A^{-1} * B_t
        # = -(I - exp(A*delta)) * A^{-1} * B_t
        # = -G / A * B_t
        # Since A < 0, -1/A = 1/|A| > 0
        A_inv = 1.0 / A  # [d_inner, d_state] — safe because A < 0
        B_bar = -G * A_inv.unsqueeze(0).unsqueeze(0) * B_t.unsqueeze(2)
    else:
        raise ValueError(f"input_weight must be 'euler' or 'zoh', got {input_weight!r}")

    # --- Sequential scan ---
    x = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
    xs = []
    ys = []

    for t in range(L):
        # x_{t+1} = A_bar_t * x_t + B_bar_t * u_t
        # A_bar_t: [batch, d_inner, d_state]
        # B_bar_t: [batch, d_inner, d_state]
        # u_t:     [batch, d_inner]
        x = A_bar[:, t] * x + B_bar[:, t] * u[:, t].unsqueeze(-1)
        xs.append(x)

        # y_t = C_t @ x_t  (summed over d_state)
        # C_t[:, t]: [batch, d_state]
        # x: [batch, d_inner, d_state]
        y_t = (C_t[:, t].unsqueeze(1) * x).sum(dim=-1)  # [batch, d_inner]
        ys.append(y_t)

    y = torch.stack(ys, dim=1)   # [batch, L, d_inner]
    x_all = torch.stack(xs, dim=1)  # [batch, L, d_inner, d_state]

    # Skip connection: y = y + D * u
    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(0)

    intermediates = {
        'A_bar': A_bar,
        'B_bar': B_bar,
        'G': G,
        'x': x_all,
        'y': y,
    }

    return y, intermediates
