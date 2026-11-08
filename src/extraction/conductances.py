"""
Conductance computation from extracted SSM parameters.

Computes g_{i,j,t} = 1 - exp(a_{j,i} * delta_{j,t}) for each state dim i,
channel j, time step t. This is the exact ZOH conductance — no small-Δ
approximation.

Uses the exact exponential form throughout (App. L.1).
"""

import torch
from torch import Tensor


def compute_conductances(
    A: Tensor,
    delta_t: Tensor,
) -> Tensor:
    """
    Compute shunt conductances from A and delta_t.

    Args:
        A: [d_inner, d_state] — diagonal A matrix (strictly negative entries).
           This is the actual A = -exp(A_log), NOT the log.
        delta_t: [batch, L, d_inner] — step sizes after softplus.

    Returns:
        g: [batch, L, d_inner, d_state] — conductances in (0, 1).
    """
    # A is [d_inner, d_state], delta_t is [batch, L, d_inner]
    # a_{j,i} * delta_{j,t} -> [batch, L, d_inner, d_state]
    # delta_t[..., None] is [batch, L, d_inner, 1]
    # A[None, None, :, :] is [1, 1, d_inner, d_state]
    exponent = A[None, None, :, :] * delta_t[..., None]  # [batch, L, d_inner, d_state]

    # g = 1 - exp(a_i * delta_t). Since a_i < 0 and delta_t > 0,
    # exponent < 0, so exp(exponent) in (0, 1), and g in (0, 1).
    g = 1.0 - torch.exp(exponent)

    return g


def compute_conductances_from_intermediates(
    intermediates: dict[str, Tensor],
) -> Tensor:
    """
    Convenience wrapper: extract A and delta_t from model intermediates dict
    and compute conductances.

    Args:
        intermediates: dict from SingleLayerMamba.forward(), must contain
                       'A' and 'delta_t'.

    Returns:
        g: [batch, L, d_inner, d_state] — conductances in (0, 1).
    """
    A = intermediates['A']          # [d_inner, d_state]
    delta_t = intermediates['delta_t']  # [batch, L, d_inner]
    return compute_conductances(A, delta_t)


def validate_conductances(g: Tensor, warn: bool = True) -> dict[str, float]:
    """
    Sanity-check conductance values.

    Returns dict with statistics. Warns if any values are out of expected range.
    """
    stats = {
        'min': g.min().item(),
        'max': g.max().item(),
        'mean': g.mean().item(),
        'std': g.std().item(),
        'frac_near_zero': (g < 0.01).float().mean().item(),
        'frac_near_one': (g > 0.99).float().mean().item(),
        'frac_negative': (g < 0.0).float().mean().item(),
        'frac_geq_one': (g >= 1.0).float().mean().item(),
    }

    if warn:
        if stats['frac_negative'] > 0:
            print(f"WARNING: {stats['frac_negative']*100:.2f}% of conductances are negative!")
        if stats['frac_geq_one'] > 0:
            print(f"WARNING: {stats['frac_geq_one']*100:.2f}% of conductances >= 1 (unstable)!")
        if stats['frac_near_one'] > 0.5:
            print(f"NOTE: {stats['frac_near_one']*100:.1f}% of conductances > 0.99 (strong reset regime)")

    return stats
