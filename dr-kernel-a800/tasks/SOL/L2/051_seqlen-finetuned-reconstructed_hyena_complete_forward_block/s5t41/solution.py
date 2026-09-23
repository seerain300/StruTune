import math
import torch
import triton
import triton.language as tl


# ---------- Triton kernels ----------

@triton.jit
def ln_forward_kernel(x_ptr, weight_ptr, bias_ptr, y_ptr,
                       M, D, eps,
                       BLOCK_SIZE: tl.constexpr):
    """
    LayerNorm forward for rows of a 2D tensor [M, D].
    - x_ptr: input flattened to [M*D], float32
    - weight_ptr, bias_ptr: [D] float32
    - y_ptr: output flattened to [M*D], float32
    """
    row = tl.program_id(0)
    offs = row * D + tl.arange(0, BLOCK_SIZE)
    mask = offs < M * D
    # We compute mean/variance across the D columns for this row.
    # Triton program processes one row: row index is offs // D.
    valid = offs >= row * D
    cols = offs - row * D  # local column indices in [0, D)
    x_valid = tl.load(x_ptr + offs, mask=mask & valid, other=0.0)
    # mean
    mean = tl.sum(x_valid, axis=0) / D
    x_center = x_valid - mean
    # var
    var = tl.sum(x_center * x_center, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + eps)
    # scale and shift
    w = tl.load(weight_ptr + cols, mask=valid, other=1.0)
    b = tl.load(bias_ptr + cols, mask=valid, other=0.0)
    y = x_center * inv_std * w + b
    tl.store(y_ptr + offs, y, mask=mask & valid)


@triton.jit
def matmul_bias_kernel(A_ptr, B_ptr, Bias_ptr, C_ptr,
                       M, K, N, eps,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute C[M, N] = A[M, K] @ B[K, N] + Bias[N].
    A_ptr: [M, K], B_ptr: [K, N], Bias_ptr: [N], C_ptr: [M, N]
    Launch grid over (M, N).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        A = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0)
        B = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0)
        acc += tl.dot(A, B)

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc += bias[None, :]

    # Store result
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def conv1d_per_channel_kernel(u_ptr, w_ptr, y_ptr,
                               B, C, L_in, K, F,
                               BLOCK_L: tl.constexpr):
    """
    Per-channel 1D conv on u[B, C, L_in] with weight w[K, 1, F], groups=C.
    Outputs y[B, C, L_out] where L_out = L_in - F + 1.
    Assumes no padding; kernel handles range via L_out.
    """
    pid = tl.program_id(0)  # one program per (b, c)
    b = pid // C
    c = pid % C
    L_out = L_in - F + 1
    for l0 in range(0, L_out, BLOCK_L):
        offs_l = l0 + tl.arange(0, BLOCK_L)
        mask_l = offs_l < L_out
        acc = tl.zeros((BLOCK_L,), dtype=tl.float32)
        # w_ptr is [K, 1, F]; for groups=C, conv1d weight shape is [C, 1, F].
        # We use row c: w[c, 1, f].
        # Access linear index: c*(1*F) + f
        for f in range(0, F):
            u_idx = b * (C * L_in) + c * L_in + (offs_l + f) * 1
            u_val = tl.load(u_ptr + u_idx, mask=mask_l, other=0.0)
            w_idx = c * (1 * F) + f
            w_val = tl.load(w_ptr + w_idx)
            acc += u_val * w_val
        y_idx = b * (C * L_out) + c * L_out + offs_l
        tl.store(y_ptr + y_idx, acc, mask=mask_l)


@triton.jit
def exp_mod_kernel(h_ptr, delta_ptr, shift, M, D,
                    BLOCK_SIZE: tl.constexpr):
    """
    Compute h_mod = h * (exp(-t * |delta|) + shift) for h flattened [M*D].
    delta is [D], shift is scalar float32.
    t is per-position: t = col / (D - 1), where col is the local column index within the row.
    """
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (row < M) & (col < D)
    h = tl.load(h_ptr + row * D + col, mask=mask, other=0.0)
    delta = tl.load(delta_ptr + col, mask=mask, other=0.0)
    t = col.to(tl.float32) / (D - 1)
    exp_term = tl.exp(-t * tl.abs(delta))
    h_mod = h * (exp_term + shift)
    tl.store(h_ptr + row * D + col, h_mod, mask=mask)


# ---------- ModelNew forward using Triton ----------

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                hidden_states: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                in_proj_weight: torch.Tensor,  # [inner_width, d_model]
                in_proj_bias: torch.Tensor,    # [inner_width]
                short_conv_weight: torch.Tensor,  # [C, 1, F], here C=inner_width, F=3
                short_conv_bias: torch.Tensor,    # not used
                filter_linear1_weight: torch.Tensor,
                filter_linear1_bias: torch.Tensor,
                sin_freq: torch.Tensor,  # not used
                filter_linear2_weight: torch.Tensor,
                filter_linear2_bias: torch.Tensor,
                filter_linear3_weight: torch.Tensor,
                filter_linear3_bias: torch.Tensor,
                filter_linear_final_weight: torch.Tensor,
                filter_bias: torch.Tensor,
                exp_mod_deltas: torch.Tensor,  # [1, 1, d_model]
                out_proj_weight: torch.Tensor,  # [d_model, d_model]
                out_proj_bias: torch.Tensor,    # [d_model]
                mlp_fc1_weight: torch.Tensor,   # [d_inner, d_model]
                mlp_fc1_bias: torch.Tensor,     # [d_inner]
                mlp_fc2_weight: torch.Tensor,   # [d_model, d_inner]
                mlp_fc2_bias: torch.Tensor,     # [d_model]
                layer_norm_eps: float,
                exp_mod_shift: float):
        """
        Triton-only forward: all heavy computations are done by Triton kernels.
        """
        B, S, D = hidden_states.shape
        device = hidden_states.device

        # 1) LayerNorm 1 on hidden_states
        x = hidden_states
        M = B * S
        x_flat = x.reshape(M, D).contiguous()
        y1_flat = torch.empty_like(x_flat)
        ln_forward_kernel[(M,)](
            x_flat, norm1_weight, norm1_bias, y1_flat,
            M, D, layer_norm_eps,
            BLOCK_SIZE=256
        )
        residual = y1_flat.reshape(B, S, D)  # [B, S, D]

        # 2) Input projection: u = F.linear(residual, in_proj_weight, in_proj_bias)
        # Implement via Triton matmul bias
        inner_width = D * 3  # order=2
        A = residual.reshape(M, D).contiguous()              # [M, D]
        Bt = in_proj_weight.transpose(0, 1).contiguous()     # [D, inner_width]
        C_mat = torch.empty((M, inner_width), device=device, dtype=torch.float32)
        matmul_bias_kernel[(M, inner_width)](
            A, Bt, in_proj_bias, C_mat,
            M, D, inner_width, 0.0,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
        )
        u = C_mat.reshape(B, S, inner_width)  # [B, S, inner_width]

        # 3) Short conv: conv1d per-channel with groups=C, no padding
        C = inner_width
        F_filter = short_conv_weight.shape[-1]  # e.g., 3
        # u_bcS: [B, C, S]
        u_bcS = u.reshape(B, C, S).contiguous()
        y = torch.empty((B, C, S - F_filter + 1), device=device, dtype=torch.float32)
        conv1d_per_channel_kernel[(B * C,)](
            u_bcS, short_conv_weight, y,
            B, C, S, 1, F_filter,
            BLOCK_L=64
        )

        # 4) Split conv output into x (first d_model slices) and v (last d_model)
        d_model = D
        x0 = y[:, :d_model, :]            # [B, d_model, L_out]
        x1 = y[:, d_model:2 * d_model, :] # [B, d_model, L_out]
        v = y[:, 2 * d_model:, :]         # [B, d_model, L_out]

        # 5) Exponential modulation: v_mod = v * (exp(-t * |delta|) + shift)
        B_mod, _, L_out = y.shape
        v_flat = v.reshape(B_mod * L_out, d_model).contiguous()
        delta = exp_mod_deltas[0, 0, :].contiguous()  # [d_model]
        exp_mod_kernel[(B_mod * L_out, (d_model + 256 - 1) // 256)](
            v_flat, delta, exp_mod_shift,
            B_mod * L_out, d_model,
            BLOCK_SIZE=256
        )
        v_mod = v_flat.reshape(B_mod, L_out, d_model)

        # 6) Iterative gating: v = v * x1; v = v * x0 (order=2)
        v = v_mod * x1
        v = v * x0

        # 7) Output projection: F.linear(v, out_proj_weight, out_proj_bias) -> [B, L_out, D


def run(*args):
    return ModelNew()(*args)
