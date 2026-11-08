"""
Single-layer Mamba block.

Implements the full Mamba architecture for a single layer:
  input projection -> conv1d -> SiLU -> SSM scan -> output projection

Key design choices:
  - Fast CUDA scan for training (via mamba_ssm)
  - Pure PyTorch sequential scan for extraction (exposes all intermediates)
  - Gradient flows end-to-end from output to input embeddings
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

from .scan_ref import selective_scan_ref

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_CUDA_SCAN = True
except ImportError:
    HAS_CUDA_SCAN = False


class SingleLayerMamba(nn.Module):
    """
    Single Mamba block: embed -> mamba -> unembed.

    Architecture follows Gu & Dao (2023) with explicit parameter exposure.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_inner: int = 32,
        d_state: int = 16,
        d_conv: int = 4,
        vocab_size: int = 50257,  # GPT-2 tokenizer
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv

        # --- Token embedding ---
        self.embedding = nn.Embedding(vocab_size, d_model)

        # --- Input projection: d_model -> 2 * d_inner ---
        # First d_inner channels go through conv+SSM, second d_inner are the gate
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # --- Causal conv1d (depthwise) ---
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )

        # --- SSM parameters ---
        # A = -exp(A_log), [d_inner, d_state]: initialized to -(1, 2, ..., d_state) in
        # every channel, so strictly negative
        A_log = torch.log(
            torch.linspace(1, d_state, d_state)
            .unsqueeze(0)
            .expand(d_inner, -1)
            .clone()
        )
        self.A_log = nn.Parameter(A_log)  # store log for numerical stability

        # delta_t projections: u -> delta_t via linear + softplus
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self._init_dt(dt_min, dt_max, dt_init_floor)

        # B and C projections: input-dependent
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)

        # D: skip connection
        self.D = nn.Parameter(torch.ones(d_inner))

        # --- Output projection: d_inner -> d_model ---
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # --- LM head ---
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embedding weights
        self.lm_head.weight = self.embedding.weight

        # Store last forward's intermediates for extraction
        self._last_intermediates: dict[str, Tensor] | None = None
        # When True, use reference scan to expose intermediates
        self.extract_mode: bool = False
        # Reference-scan input weight: 'euler' matches the CUDA kernel the
        # checkpoints were trained with; 'zoh' is the exact ZOH variant, for comparison
        self.input_weight: str = 'euler'

    def _init_dt(self, dt_min: float, dt_max: float, dt_init_floor: float):
        """Initialize dt_proj bias so that softplus(bias) is log-uniform on [dt_min, dt_max], clamped below at dt_init_floor."""
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus_dt)

    @property
    def A(self) -> Tensor:
        """Diagonal A matrix (strictly negative). Shape: [d_inner, d_state]."""
        return -torch.exp(self.A_log)

    def forward(
        self,
        input_ids: Tensor | None = None,
        embeddings: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: [batch, L] — token IDs (mutually exclusive with embeddings)
            embeddings: [batch, L, d_model] — continuous embeddings (for constant input test)

        Returns:
            logits: [batch, L, vocab_size]
            intermediates: dict with SSM internals for extraction
        """
        if embeddings is not None:
            x = embeddings
        elif input_ids is not None:
            x = self.embedding(input_ids)
        else:
            raise ValueError("Provide either input_ids or embeddings")

        residual = x

        # --- Input projection ---
        xz = self.in_proj(x)  # [batch, L, 2*d_inner]
        x_ssm, z = xz.split(self.d_inner, dim=-1)

        # --- Causal conv1d ---
        # Conv expects [batch, channels, length]
        x_conv = x_ssm.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :x_ssm.shape[1]]  # causal: trim future
        x_conv = x_conv.transpose(1, 2)

        # --- SiLU activation ---
        x_conv = torch.nn.functional.silu(x_conv)

        # --- Compute selective parameters ---
        delta_t = torch.nn.functional.softplus(
            self.dt_proj(x_conv)
        )  # [batch, L, d_inner]
        B_t = self.B_proj(x_conv)  # [batch, L, d_state]
        C_t = self.C_proj(x_conv)  # [batch, L, d_state]

        # --- Sequential scan ---
        if self.extract_mode or not HAS_CUDA_SCAN:
            # Reference scan: exposes all intermediates for extraction
            y_ssm, scan_intermediates = selective_scan_ref(
                A=self.A,
                B_t=B_t,
                C_t=C_t,
                delta_t=delta_t,
                u=x_conv,
                D=self.D,
                input_weight=self.input_weight,
            )
        else:
            # Fast CUDA scan for training
            # mamba_ssm expects: A [d_inner, d_state], u [batch, L, d_inner],
            # delta [batch, L, d_inner], B [batch, L, d_state], C [batch, L, d_state]
            A = self.A  # [d_inner, d_state], negative
            y_ssm = selective_scan_fn(
                x_conv.transpose(1, 2),      # [batch, d_inner, L]
                delta_t.transpose(1, 2),     # [batch, d_inner, L]
                A,                            # [d_inner, d_state]
                B_t.transpose(1, 2),         # [batch, d_state, L]
                C_t.transpose(1, 2),         # [batch, d_state, L]
                D=self.D,                     # [d_inner]
                z=None,
                return_last_state=False,
            )
            y_ssm = y_ssm.transpose(1, 2)   # [batch, L, d_inner]
            scan_intermediates = {}

        # --- Gating ---
        z_gate = torch.nn.functional.silu(z)
        y = y_ssm * z_gate  # [batch, L, d_inner]

        # --- Output projection + residual ---
        y = self.out_proj(y) + residual  # [batch, L, d_model]

        # --- LM head ---
        logits = self.lm_head(y)  # [batch, L, vocab_size]

        # --- Store intermediates ---
        intermediates = {
            'A': self.A,
            'delta_t': delta_t,
            'B_t': B_t,
            'C_t': C_t,
            **scan_intermediates,
        }
        self._last_intermediates = intermediates

        return logits, intermediates

    def get_intermediates(self) -> dict[str, Tensor]:
        """Return intermediates from the last forward pass."""
        if self._last_intermediates is None:
            raise RuntimeError("No forward pass has been run yet")
        return self._last_intermediates
