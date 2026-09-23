import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layernorm_stats_kernel(x_ptr, sums_ptr, sumsq_ptr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    n = tl.program_id(0)
    sum_val = 0.0
    sumsq_val = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + n * D + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)
    tl.store(sums_ptr + n, sum_val)
    tl.store(sumsq_ptr + n, sumsq_val)


@triton.jit
def layernorm_apply_kernel(x_ptr, sums_ptr, sumsq_ptr, weight_ptr, bias_ptr, out_ptr,
                            N, D, eps, BLOCK_D: tl.constexpr):
    n = tl.program_id(0)
    sum_val = tl.load(sums_ptr + n)
    sumsq_val = tl.load(sumsq_ptr + n)
    mean = sum_val / D
    var = sumsq_val / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + n * D + offs, mask=mask, other=0.0)
        y = (x - mean) * inv_std
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        y = y * w + b
        tl.store(out_ptr + n * D + offs, y, mask=mask)


@triton.jit
def linear_in_proj_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                           N, D, INNER_WIDTH, BLOCK_D: tl.constexpr):
    # x: [N*D], w: [INNER_WIDTH*D], b: [INNER_WIDTH], y: [N*INNER_WIDTH*D]
    # Each program handles (n, iw, tile over D)
    n = tl.program_id(0)
    iw = tl.program_id(1)
    tile = tl.program_id(2)
    d0 = tile * BLOCK_D
    offs = d0 + tl.arange(0, BLOCK_D)
    mask = offs < D
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    # Accumulate over D: y[n, iw, d] = sum over d of x[n, d] * w[iw, d] + b[iw]
    # We cannot loop over D here because D is a runtime parameter; Triton expects static loops.
    # To work around, we keep iw fixed and for each d tile, load x[n, d] and w[iw, d] and accumulate.
    # y_ptr indexing: linearized as y[n * (INNER_WIDTH * D) + iw * D + d]
    # x_ptr indexing: x[n * D + d]
    # w_ptr indexing: w[iw * D + d]
    for t in range(BLOCK_D):
        d = d0 + t
        mask_t = d < D
        xval = tl.load(x_ptr + n * D + d, mask=mask_t, other=0.0)
        wval = tl.load(w_ptr + iw * D + d, mask=mask_t, other=0.0)
        acc[t] = xval * wval
    # add bias b[iw]
    bval = tl.load(b_ptr + iw)
    acc = acc + bval
    # store y[n, iw, d]
    for t in range(BLOCK_D):
        d = d0 + t
        mask_t = d < D
        tl.store(y_ptr + n * (INNER_WIDTH * D) + iw * D + d, acc[t], mask=mask_t)


@triton.jit
def conv1d_short_groups_kernel(u_ptr, w_ptr, bias_ptr, out_ptr,
                                N, D, L_in, OUT_L, K: tl.constexpr, pad_left: tl.constexpr,
                                BLOCK_D: tl.constexpr):
    # u: [N, D, L_in], w: [D, 1, K], bias: [D], out: [N, D, OUT_L]
    # Grid over (n, d). Each program handles one (n, d).
    n = tl.program_id(0)  # N
    d = tl.program_id(1)  # D
    for t_out in range(OUT_L):
        acc = 0.0
        for k in range(K):
            pos = t_out + pad_left + k  # in [0, L_in)
            val = tl.load(u_ptr + n * (D * L_in) + d * L_in + pos)
            wval = tl.load(w_ptr + d * K + k)  # w[d, k]
            acc += val * wval
        acc += tl.load(bias_ptr + d)
        tl.store(out_ptr + n * (D * OUT_L) + d * OUT_L + t_out, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                short_conv_weight: torch.Tensor, short_conv_bias: torch.Tensor,
                filter_linear1_weight: torch.Tensor, filter_linear1_bias: torch.Tensor,
                sin_freq: torch.Tensor,
                filter_linear2_weight: torch.Tensor, filter_linear2_bias: torch.Tensor,
                filter_linear3_weight: torch.Tensor, filter_linear3_bias: torch.Tensor,
                filter_linear_final_weight: torch.Tensor, filter_bias: torch.Tensor,
                exp_mod_deltas: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor,
                mlp_fc1_weight: torch.Tensor, mlp_fc1_bias: torch.Tensor,
                mlp_fc2_weight: torch.Tensor, mlp_fc2_bias: torch.Tensor,
                layer_norm_eps: float, exp_mod_shift: float):

        N, L, D = hidden_states.shape

        # First LayerNorm: compute stats and apply
        x_flat = hidden_states.reshape(N * D).contiguous()
        sums = torch.empty((N,), dtype=torch.float32, device=hidden_states.device)
        sumsq = torch.empty((N,), dtype=torch.float32, device=hidden_states.device)
        layernorm_stats_kernel[(N,)](x_flat, sums, sumsq, D, BLOCK_D=256)
        out_ln1 = torch.empty_like(x_flat, dtype=torch.float32, device=hidden_states.device)
        layernorm_apply_kernel[(N,)](x_flat, sums, sumsq, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32), out_ln1, N, D, layer_norm_eps, BLOCK_D=256)
        normed = out_ln1.view(N, D)  # [N, D]

        # Add residual: original does residual = hidden_states.float(); then adds it before linear.
        residual = hidden_states.to(torch.float32)
        normed = normed + residual  # [N, D]

        # Input projection u = F.linear(normed, in_proj_weight, in_proj_bias)
        INNER_WIDTH = in_proj_weight.shape[0]
        # Implement in Triton
        x_flat_in = normed.reshape(N * D).contiguous()  # [N*D]
        W_flat = in_proj_weight.to(torch.float32).reshape(INNER_WIDTH * D)  # [INNER_WIDTH*D]
        b_flat = in_proj_bias.to(torch.float32)  # [INNER_WIDTH]
        y_flat = torch.empty((N * INNER_WIDTH * D,), dtype=torch.float32, device=hidden_states.device)
        linear_in_proj_kernel[(N, INNER_WIDTH, triton.cdiv(D, 256))](
            x_flat_in, W_flat, b_flat, y_flat, N, D, INNER_WIDTH, BLOCK_D=256
        )
        u = y_flat.view(N, INNER_WIDTH, D)

        # Short 1D convolution with groups=D and K=3
        L_in = L + 4  # pad left and right by 2
        OUT_L = L
        pad_left = 2
        # Construct u_padded [N, D, L_in]
        u_padded = torch.zeros((N, D, L_in), dtype=torch.float32, device=hidden_states.device)
        # Place original u (center) into u_padded: u_padded[:, :, 2:L+2] = u
        # u shape [N, INNER_WIDTH, D] and conv is along D, but original code applies conv on hidden_states after linear.
        # To align with original, we conv u_padded: zeros + left/right pad of 2. However, original u is [N, D, L] post-linear.
        # In our setup, u is [N, INNER_WIDTH, D]. We need to build u as [N, D, L_in]. We can create u_padded with zeros and place u at positions corresponding to inner channels,
        # but original u for conv is built from hidden_states. Since original conv is defined on u, which is [N, D, L] after linear, we cannot access it here.
        # Therefore, we implement short conv on hidden_states padded in Triton: conv on u_padded as [N, D, L_in] filled with zeros and adding left/right pad.
        # But to match original code, we must conv on u. We'll implement conv by using u_padded as [N, D, L


def run(*args):
    return ModelNew()(*args)
