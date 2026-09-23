import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton LayerNorm: compute per-row sum and sumsq across last dimension D (row is one [L, D] row, flattened)
@triton.jit
def layernorm_forward_stats_kernel(x_ptr, sums_ptr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    row_id = tl.program_id(0)  # each program handles one row (flattened across L and N)
    sum_val = 0.0
    sumsq_val = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + row_id * D + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)
    tl.store(sums_ptr + row_id, sum_val)
    tl.store(sums_ptr + row_id + 1, sumsq_val)  # store sumsq at next location


# Triton LayerNorm: apply normalization using precomputed sums and sumsq, affine with weight/bias
@triton.jit
def layernorm_apply_kernel(x_ptr, sums_ptr, sumsq_ptr, weight_ptr, bias_ptr, out_ptr, M, D, eps, BLOCK_D: tl.constexpr):
    row_id = tl.program_id(0)  # each program handles one row (flattened across L and N)
    sum_val = tl.load(sums_ptr + row_id)
    sumsq_val = tl.load(sums_ptr + row_id + 1)
    mean = sum_val / D
    var = sumsq_val / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + row_id * D + offs, mask=mask, other=0.0)
        y = (x - mean) * inv_std
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        y = y * w + b
        tl.store(out_ptr + row_id * D + offs, y, mask=mask)


# Triton Short 1D conv with groups=D and K=3, padding=2 on both sides (conv1d with groups, stride=1)
@triton.jit
def conv1d_short_groups_kernel(u_ptr, w_ptr, bias_ptr, out_ptr,
                                N, D, L_in, OUT_L, BLOCK_OUT: tl.constexpr):
    n = tl.program_id(0)  # batch
    d = tl.program_id(1)  # group/channel
    base = d * 3  # weight indices for this group: [base+0], [base+1], [base+2]
    w0 = tl.load(w_ptr + base + 0)
    w1 = tl.load(w_ptr + base + 1)
    w2 = tl.load(w_ptr + base + 2)
    b = tl.load(bias_ptr + d)
    for out_l in range(0, OUT_L):
        in0 = out_l + 0  # left pad index
        in1 = out_l + 1
        in2 = out_l + 2
        mask0 = in0 >= 0 and in0 < L_in
        mask1 = in1 >= 0 and in1 < L_in
        mask2 = in2 >= 0 and in2 < L_in
        v0 = tl.load(u_ptr + n * (D * L_in) + d * L_in + in0, mask=mask0, other=0.0)
        v1 = tl.load(u_ptr + n * (D * L_in) + d * L_in + in1, mask=mask1, other=0.0)
        v2 = tl.load(u_ptr + n * (D * L_in) + d * L_in + in2, mask=mask2, other=0.0)
        y = v0 * w0 + v1 * w1 + v2 * w2 + b
        tl.store(out_ptr + n * D * OUT_L + d * OUT_L + out_l, y)


# Triton input projection: out[n, w, d] = sum_d hidden_states[n, d, l] * in_proj_weight[w, d] + in_proj_bias[w]
@triton.jit
def linear_in_proj_kernel(hs_ptr, weight_ptr, bias_ptr, out_ptr,
                           N, D, W, L, BLOCK_D: tl.constexpr):
    # Grid: (N, W, L)
    n = tl.program_id(0)
    w = tl.program_id(1)
    l = tl.program_id(2)
    acc = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        hs = tl.load(hs_ptr + n * (D * L) + offs * L + l, mask=mask, other=0.0)  # hidden_states[n, d, l]
        wv = tl.load(weight_ptr + w * D + offs, mask=mask, other=0.0)           # in_proj_weight[w, d]
        acc += tl.sum(hs * wv, axis=0)
    b = tl.load(bias_ptr + w)
    tl.store(out_ptr + n * (W * D) + w * D + l, acc + b)


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    # Read dynamic sizes from axes_and_scalars
    batch_size = int(axes_and_scalars["batch_size"])
    seq_len = int(axes_and_scalars["seq_len"])
    # Original defaults from provided code; we will use these constants for kernels.
    d_model = 256
    order = 2
    l_max = 32768
    short_filter_order = 3
    filter_order = 64
    emb_dim = 5
    inner_width = d_model * (order + 1)  # 768

    hidden_states = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32, device=device)
    norm1_weight = torch.ones(d_model, dtype=torch.float32, device=device)
    norm1_bias = torch.zeros(d_model, dtype=torch.float32, device=device)
    norm2_weight = torch.ones(d_model, dtype=torch.float32, device=device)
    norm2_bias = torch.zeros(d_model, dtype=torch.float32, device=device)
    in_proj_weight = torch.randn(inner_width, d_model, dtype=torch.float32, device=device) * 0.02
    in_proj_bias = torch.randn(inner_width, dtype=torch.float32, device=device) * 0.02
    short_conv_weight = torch.randn(inner_width, 1, short_filter_order, dtype=torch.float32, device=device) * 0.02
    short_conv_bias = torch.randn(inner_width, dtype=torch.float32, device=device) * 0.02
    filter_linear1_weight = torch.randn(filter_order, emb_dim, dtype=torch.float32, device=device) * 0.02
    filter_linear1_bias = torch.randn(filter_order, dtype=torch.float32, device=device) * 0.02
    sin_freq = torch.ones(1, filter_order, dtype=torch.float32, device=device)
    filter_linear2_weight = torch.randn(filter_order, filter_order, dtype=torch.float32, device=device) * 0.02
    filter_linear2_bias = torch.randn(filter_order, dtype=torch.float32, device=device) * 0.02
    filter_linear3_weight = torch.randn(filter_order, filter_order, dtype=torch.float32, device=device) * 0.02
    filter_linear3_bias = torch.randn(filter_order, dtype=torch.float32, device=device) * 0.02
    filter_linear_final_weight = torch.randn(d_model, filter_order, dtype=torch.float32, device=device) * 0.02
    filter_bias = torch.randn(d_model, dtype=torch.float32, device=device) * 0.02
    max_decay = math.log(0.01) / 0.3
    min_decay = math.log(0.01) / 1.5
    deltas = torch.linspace(min_decay, max_decay, d_model, device=device)[None, None, :]
    exp_mod_deltas = deltas.to(torch.float32)
    out_proj_weight = torch.randn(d_model, d_model, dtype=torch.float32, device=device) * 0.02
    out_proj_bias = torch.randn(d_model, dtype=torch.float32, device=device) * 0.02
    mlp_fc1_weight = torch.randn(d_inner, d_model, dtype=torch.float32, device=device) * 0.02
    mlp_fc1_bias = torch.randn(d_inner, dtype=torch.float32, device=device) * 0.02
    mlp_fc2_weight = torch.randn(d_model, d_inner, dtype=torch.float32, device=device) * 0.02
    mlp_fc2_bias = torch.randn(d_model, dtype=torch.float32, device=device) * 0.02

    return {
        "hidden_states": hidden_states,
        "norm1_weight": norm1_weight,
        "norm1_bias": norm1_bias,
        "norm2_weight": norm2_weight,
        "norm2_bias": norm2_bias,
        "in_proj_weight": in_proj_weight,
        "in_proj_bias": in_proj_bias,
        "short_conv_weight": short_conv_weight,
        "short_conv_bias": short_conv_bias,
        "filter_linear1_weight": filter_linear1_weight,
        "filter_linear1_bias": filter_linear1_bias,
        "sin_freq": sin_freq,
        "filter_linear2_weight": filter_linear2_weight,
        "filter_linear2_bias": filter_linear2_bias,
        "filter_linear3_weight": filter_linear3_weight,
        "filter_linear3_bias": filter_linear3_bias,
        "filter_linear_final_weight": filter_linear_final_weight,
        "filter_bias": filter_bias,
        "exp_mod_deltas": exp_mod_deltas,
        "out_proj_weight": out_proj_weight,
        "out_proj_bias": out_proj_bias,
        "mlp_fc1_weight": mlp_fc1_weight,
        "mlp_fc1_bias": mlp_fc1_bias,
        "mlp_fc2_weight": mlp_fc2_weight,
        "mlp_fc2_bias": mlp_fc2_bias,
        "layer_norm_eps": 1e-5,
        "exp_mod_shift": 0.05
    }


class ModelNew(nn.Module):
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
        # Read sizes
        N, L, D = hidden_states.shape
        W = in_proj_weight.shape[0]  # inner


def run(*args):
    return ModelNew()(*args)
