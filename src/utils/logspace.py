"""
Log-space arithmetic for running products.

Computes ∏_{s=t1}^{t2} (1 - g_s) in log space to avoid overflow/underflow.

Key identity:
    ∏_{s=t1}^{t2} (1 - g_s) = exp( Σ_{s=t1}^{t2} log(1 - g_s) )

Using cumulative sums, we can compute all lags simultaneously.
"""

import torch
from torch import Tensor


def log_retention(g: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Compute log(1 - g) with numerical stability.

    Uses log1p(-g) which is accurate for small g (near zero).

    Args:
        g: Conductances in (0, 1). Shape: arbitrary.
        eps: Clamp g to [0, 1-eps] to avoid log(0).
            Must be large enough to survive float32 rounding (1e-6 works, 1e-8 does not).

    Returns:
        log(1 - g), same shape as g.
    """
    g_safe = g.clamp(min=0.0, max=1.0 - eps)
    return torch.log1p(-g_safe)


def cumulative_log_retention(g: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    """
    Cumulative sum of log(1 - g) along a dimension.

    cumret[..., t] = Σ_{s=0}^{t} log(1 - g[..., s])

    So exp(cumret[..., t2] - cumret[..., t1-1]) = ∏_{s=t1}^{t2} (1 - g_s).

    Args:
        g: Conductances. The dimension `dim` is the time dimension.
        dim: Dimension along which to accumulate.
        eps: Stability clamp.

    Returns:
        Cumulative log retention, same shape as g.
    """
    lr = log_retention(g, eps=eps)
    return torch.cumsum(lr, dim=dim)


def running_product_for_lags(
    g: Tensor,
    max_lag: int,
    time_dim: int = 1,
    target_positions: list[int] | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """
    Compute ∏_{s=T-τ}^{T-1} (1 - g_s) for lags τ = 0, 1, ..., max_lag-1.

    This is the retention (signal attenuation) from position T-τ to T.

    Args:
        g: Conductances. Shape should have time along `time_dim`.
           e.g. [batch, L, d_inner, d_state] with time_dim=1.
        max_lag: Maximum lag τ to compute.
        time_dim: Which dimension is time.
        target_positions: List of target positions T to average over.
            If None, uses the last position.
        eps: Stability clamp.

    Returns:
        retention: [len(target_positions), max_lag, ...remaining dims...]
            retention[p, τ] = ∏_{s=T_p-τ}^{T_p-1} (1 - g_s)
            At τ=0, the product is empty → retention = 1.
    """
    L = g.shape[time_dim]

    if target_positions is None:
        target_positions = [L - 1]

    # Compute cumulative log retention along time
    # cumret[..., t] = Σ_{s=0}^{t} log(1 - g_s)
    cumret = cumulative_log_retention(g, dim=time_dim, eps=eps)

    # We need ∏_{s=T-τ+1}^{T} (1-g_s), which has τ factors.
    # Update-then-read convention: y_T uses x after processing u_T.
    # For τ=0: empty product = 1
    # For τ=1: (1-g_T), one factor
    # For τ=k: exp(cumret[T] - cumret[T-k]), k factors

    results = []
    for T in target_positions:
        lags = []
        for tau in range(max_lag):
            if tau == 0:
                shape = list(g.shape)
                shape.pop(time_dim)
                ones = torch.ones(shape, device=g.device, dtype=g.dtype)
                lags.append(ones)
            else:
                start = T - tau  # input position
                if start < 0:
                    shape = list(g.shape)
                    shape.pop(time_dim)
                    lags.append(torch.zeros(shape, device=g.device, dtype=g.dtype))
                    continue

                log_end = cumret.select(time_dim, T)
                # Product from start+1 to T = exp(cumret[T] - cumret[start])
                log_start = cumret.select(time_dim, start)
                log_prod = log_end - log_start

                lags.append(torch.exp(log_prod))

        results.append(torch.stack(lags, dim=0))  # [max_lag, ...remaining dims]

    return torch.stack(results, dim=0)  # [n_targets, max_lag, ...]


# ============================================================
# Tests
# ============================================================

def _test_small_example():
    """Verify log-space matches direct computation on a small case."""
    torch.manual_seed(42)
    g = torch.tensor([0.1, 0.2, 0.3, 0.15, 0.25])  # 5 time steps

    # Direct product for lag τ from target T=4
    # Product from T-tau+1 to T (tau factors, update-then-read convention)
    T = 4
    for tau in range(5):
        if tau == 0:
            direct = 1.0
        else:
            direct = 1.0
            for s in range(T - tau + 1, T + 1):
                direct *= (1.0 - g[s].item())

        # Log-space
        g_5d = g.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, 5, 1, 1]
        retention = running_product_for_lags(g_5d, max_lag=5, time_dim=1, target_positions=[T])
        logspace_val = retention[0, tau, 0, 0, 0].item()

        assert abs(direct - logspace_val) < 1e-6, \
            f"τ={tau}: direct={direct:.6f}, logspace={logspace_val:.6f}"

    print("  ✓ Small example: log-space matches direct product")


def _test_numerical_stability():
    """Verify stability for long sequences with small g."""
    torch.manual_seed(42)
    L = 2048
    # Small g values → products decay slowly, need many terms
    g = torch.rand(1, L, 1, 1) * 0.01  # g ~ Uniform(0, 0.01)

    retention = running_product_for_lags(g, max_lag=L, time_dim=1, target_positions=[L-1])

    # Should be monotonically decreasing with τ
    ret_vals = retention[0, :, 0, 0, 0]
    assert ret_vals[0] == 1.0, "τ=0 should give 1.0"
    assert (ret_vals[1:] <= ret_vals[:-1] + 1e-7).all(), "Should be monotonically decreasing"
    assert ret_vals[-1] > 0, "Should not underflow to zero for small g"
    assert not torch.isnan(ret_vals).any(), "No NaN values"
    assert not torch.isinf(ret_vals).any(), "No Inf values"

    print(f"  ✓ Stability: L={L}, g~U(0,0.01), final retention={ret_vals[-1]:.6e} (no underflow)")


def _test_moderate_g():
    """Verify with moderate g values that cause faster decay."""
    torch.manual_seed(42)
    L = 2048
    g = torch.rand(1, L, 1, 1) * 0.1 + 0.05  # g ~ Uniform(0.05, 0.15)

    retention = running_product_for_lags(g, max_lag=L, time_dim=1, target_positions=[L-1])
    ret_vals = retention[0, :, 0, 0, 0]

    assert ret_vals[0] == 1.0
    assert not torch.isnan(ret_vals).any()
    # With g~0.1, product decays as ~(0.9)^τ, so at τ=2047: ~10^{-94}
    # This would underflow with direct products but log-space handles it
    # (the exp at the end may give 0.0 for very large τ)

    print(f"  ✓ Moderate g: L={L}, g~U(0.05,0.15), τ=100 retention={ret_vals[100]:.6e}")


def run_tests():
    print("Running logspace arithmetic tests...")
    _test_small_example()
    _test_numerical_stability()
    _test_moderate_g()
    print("All tests passed!")


if __name__ == '__main__':
    run_tests()
