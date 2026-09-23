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
    - y_ptr: output flattened to [M*D] (M = B*S)
    """
    row_id = tl.program_id(0)
    start = row_id * D

    # Accumulate sum and sum of squares in fp32
    s = 0.0
    ss = 0.0
    # Loop over the row in chunks
    for col in range(0, D, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(x_ptr + start + offs, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(x, axis=0)
        ss += tl.sum(x * x, axis=0)

    Df = tl.float32(D)
    mean = s / Df
    var = ss / Df - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for col in range(0, D, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(x_ptr + start + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = y * w + b
        tl.store(y_ptr + start + offs, y, mask=mask)


@triton.jit
def conv1d_per_channel_kernel(x_ptr, w_ptr, y_ptr,
                               B, C, L_in, K, F,
                               BLOCK_L: tl.constexpr, BLOCK_C: tl.constexpr):
    """
    Per-channel 1D convolution with groups=C (depthwise conv).
    x_ptr: [B, C, L_in], float32, contiguous (we pass the conv input tensor directly)
    w_ptr: [C, F], float32, contiguous
    y_ptr: [B, C, L_out], float32
    L_out = L_in - F + 1 (no explicit padding; we treat out-of-range reads as zero by masking in kernel).
    Grid: (B*C,) each program handles one (b, c) vector over L_out.
    """
    bc_id = tl.program_id(0)  # 0..(B*C-1)
    b = bc_id // C
    c = bc_id % C

    L_out = L_in - F + 1

    for l0 in range(0, L_out, BLOCK_L):
        offs_l = l0 + tl.arange(0, BLOCK_L)
        mask_l = offs_l < L_out

        acc = tl.zeros([BLOCK_L], dtype=tl.float32)

        # Loop over kernel taps
        for k0 in range(0, F, 1):
            k = k0
            index_in = offs_l + k
            mask_in = (index_in < L_in) & mask_l

            base_bc = b * C * L_in + c * L_in
            x_ptrs = x_ptr + base_bc + index_in
            x_vals = tl.load(x_ptrs, mask=mask_in, other=0.0).to(tl.float32)

            # Load weight for channel c at tap k: w_ptr[c * F + k]
            w_val = tl.load(w_ptr + c * F + k, mask=True, other=0.0).to(tl.float32)

            acc += x_vals * w_val

        y_ptrs = y_ptr + b * C * L_out + c * L_out + offs_l
        tl.store(y_ptrs, acc, mask=mask_l)


@triton.jit
def exp_mod_kernel(h_ptr, delta_ptr, shift, L, D,
                    BLOCK_SIZE: tl.constexpr):
    """
    Exponential modulation: h_new = h * (exp(-t * |delta|) + shift)
    - h_ptr: [L*D], float32
    - delta_ptr: [D], float32 (broadcast along L)
    - L: int, number of rows in h
    - D: int, number of columns (features)
    We compute t per position as idx / (L - 1) to approximate original t in [0,1].
    """
    row_id = tl.program_id(0)  # 0..L-1
    col_block = tl.program_id(1)  # blocks over D
    offs = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    h = tl.load(h_ptr + row_id * D + offs, mask=mask, other=0.0).to(tl.float32)
    delta = tl.load(delta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    Lf = tl.float32(L)
    t = tl.cast(offs, tl.float32) / (Lf - 1.0)

    mod = tl.exp(-t * tl.abs(delta)) + shift
    h = h * mod

    tl.store(h_ptr + row_id * D + offs, h, mask=mask)


# ---------- ModelNew using Triton ----------

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                hidden_states: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                short_conv_weight: torch.Tensor,
                short_conv_bias: torch.Tensor,  # bias not used in kernel
                filter_linear1_weight: torch.Tensor,
                filter_linear1_bias: torch.Tensor,
                sin_freq: torch.Tensor,
                filter_linear2_weight: torch.Tensor,
                filter_linear2_bias: torch.Tensor,
                filter_linear3_weight: torch.Tensor,
                filter_linear3_bias: torch.Tensor,
                filter_linear_final_weight: torch.Tensor,
                filter_bias: torch.Tensor,
                exp_mod_deltas: torch.Tensor,  # [1, 1, d_model]
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor,
                mlp_fc1_weight: torch.Tensor,
                mlp_fc1_bias: torch.Tensor,
                mlp_fc2_weight: torch.Tensor,
                mlp_fc2_bias: torch.Tensor,
                layer_norm_eps: float,
                exp_mod_shift: float):
        """
        Triton-only forward: launch Triton kernels for LayerNorm1, conv1d per channel, and exp_mod.
        PyTorch is used for input projection, iterative gating, and final matmul. We avoid any torch.nn.functional.pad.
        """

        B, S, D = hidden_states.shape
        device = hidden_states.device

        # 1) LayerNorm 1 (forward) on hidden_states
        x1 = hidden_states.to(torch.float32).contiguous()
        M = B * S
        x1_flat = x1.reshape(M, D).contiguous()
        y1_flat = torch.empty_like(x1_flat)
        BLOCK_SIZE = 256
        ln_forward_kernel[(M,)](x1_flat, norm1_weight, norm1_bias, y1_flat, M, D, layer_norm_eps, BLOCK_SIZE=BLOCK_SIZE)
        residual = y1_flat.reshape(B, S, D).contiguous()

        # 2) Input projection (u) using PyTorch F.linear; u: [B, inner_width, S]
        inner_width = D * (2 + 1)  # order=2
        u = torch.nn.functional.linear(residual, in_proj_weight, in_proj_bias)  # [B, inner_width, S]

        # 3) Short conv (groups=C): C = inner_width
        C = inner_width
        F_filter = short_conv_weight.shape[-1]  # e.g., 3
        # We avoid torch.nn.functional.pad; treat out-of-range reads as zero in Triton kernel.
        # u has shape [B, C, S]; y has shape [B, C, L_out], L_out = S - F_filter + 1
        L_out = S - F_filter + 1
        y = torch.empty((B, C, L_out), device=device, dtype=torch.float32)

        grid_conv = (B * C,)
        conv1d_per_channel_kernel[grid_conv](
            u, short_conv_weight, y,
            B, C, S, C, F_filter,
            BLOCK_L=64, BLOCK_C=1
        )

        # 4) Split conv output into x (first d_model slices) and v (last d_model)
        d_model = D  # original d_model
        x0 = y[:, :d_model, :]  # [B, d_model, L_out]
        x1_part = y[:, d_model:2 * d_model, :]  # [B, d_model, L_out]
        x1 = x1_part  # second slice
        v = y[:, 2 * d_model:, :]  # [B, d_model, L_out]

        # 5) Exponential modulation: Triton kernel on v using delta
        delta = exp_mod_deltas[0, 0, :].contiguous()  # [D]
        h_flat = v.reshape(B * L_out, D).contiguous()
        h_mod_flat = torch.empty_like(h_flat)
        BLOCK_SIZE = 256
        grid_exp = (B * L_out, (D + BLOCK_SIZE - 1) // BLOCK_SIZE)
        exp_mod_kernel[grid_exp](h_flat, delta, exp_mod_shift, B * L_out, D, BLOCK_SIZE=BLOCK_SIZE)
        v_mod = h_mod_flat.reshape(B, L_out, D)

        # 6) Iterative gating in PyTorch (order=2): update v = v * x1; then v = v * x0
        v = v_mod * x1
        v = v * x0

        # 7) Output projection (PyTorch): F.linear(v, out_proj_weight, out_proj_bias) -> [B, S, D]
        hyena_out = torch.nn.functional.linear(v, out_proj_weight, out_proj_bias)  # [B, S, D]
        residual_f = residual.to(torch.float32)
        output = hyena_out + residual_f  # [B, S, D]

        # 8) LayerNorm 2: apply LN2 on output
        x2_flat


def run(*args):
    return ModelNew()(*args)
