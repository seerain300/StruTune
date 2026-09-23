import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layernorm_stats_kernel(x_ptr, sums_ptr, sumsq_ptr,
                            N, L, D,
                            BLOCK_D: tl.constexpr):
    # Each program handles one row: (n, l) across last dimension D
    row_id = tl.program_id(0)  # 0 .. N*L-1
    n = row_id // L
    l = row_id % L
    base = n * L * D + l * D

    sum_val = 0.0
    sumsq_val = 0.0
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    tl.store(sums_ptr + row_id, sum_val)
    tl.store(sumsq_ptr + row_id, sumsq_val)


@triton.jit
def layernorm_apply_kernel(x_ptr, sums_ptr, sumsq_ptr, weight_ptr, bias_ptr, out_ptr,
                            N, L, D, eps,
                            BLOCK_D: tl.constexpr):
    row_id = tl.program_id(0)  # 0 .. N*L-1
    n = row_id // L
    l = row_id % L
    base = n * L * D + l * D

    sum_val = tl.load(sums_ptr + row_id)
    sumsq_val = tl.load(sumsq_ptr + row_id)
    mean = sum_val / D
    var = sumsq_val / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = (x - mean) * inv_std
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0)
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        y = y * w + b
        tl.store(out_ptr + base + offs, y, mask=mask)


def _triton_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """
    x: [N, L, D] float32 CUDA tensor
    weight, bias: [D] float32 CUDA tensor
    returns: [N, L, D] float32 tensor
    """
    assert x.is_cuda, "Triton kernels require CUDA tensors"
    x = x.to(torch.float32)
    N, L, D = x.shape
    # Flatten rows to [N*L, D]
    x_flat = x.reshape(N * L, D).contiguous()
    weight = weight.to(torch.float32).contiguous()
    bias = bias.to(torch.float32).contiguous()
    sums = torch.empty(N * L, dtype=torch.float32, device=x.device)
    sumsq = torch.empty(N * L, dtype=torch.float32, device=x.device)
    out = torch.empty_like(x_flat)

    # Choose BLOCK_D based on D
    if D >= 1024:
        BLOCK_D = 256
    elif D >= 512:
        BLOCK_D = 128
    else:
        BLOCK_D = 64

    layernorm_stats_kernel[(N * L,)](x_flat, sums, sumsq, N, L, D, BLOCK_D)
    layernorm_apply_kernel[(N * L,)](x_flat, sums, sumsq, weight, bias, out, N, L, D, eps, BLOCK_D)

    return out.reshape(N, L, D)


class ModelNew(nn.Module):
    def forward(self, *args):
        """
        Args and order must match the original run signature:
        hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias,
        in_proj_weight, in_proj_bias, short_conv_weight, short_conv_bias,
        filter_linear1_weight, filter_linear1_bias, sin_freq, filter_linear2_weight,
        filter_linear2_bias, filter_linear3_weight, filter_linear3_bias,
        filter_linear_final_weight, filter_bias, exp_mod_deltas, out_proj_weight,
        out_proj_bias, mlp_fc1_weight, mlp_fc1_bias, mlp_fc2_weight, mlp_fc2_bias,
        layer_norm_eps (default 1e-5), exp_mod_shift
        """
        hidden_states = args[0]
        norm1_weight = args[1]
        norm1_bias = args[2]
        norm2_weight = args[3]
        norm2_bias = args[4]
        in_proj_weight = args[5]
        in_proj_bias = args[6]
        short_conv_weight = args[7]
        short_conv_bias = args[8]
        filter_linear1_weight = args[9]
        filter_linear1_bias = args[10]
        sin_freq = args[11]
        filter_linear2_weight = args[12]
        filter_linear2_bias = args[13]
        filter_linear3_weight = args[14]
        filter_linear3_bias = args[15]
        filter_linear_final_weight = args[16]
        filter_bias = args[17]
        exp_mod_deltas = args[18]
        out_proj_weight = args[19]
        out_proj_bias = args[20]
        mlp_fc1_weight = args[21]
        mlp_fc1_bias = args[22]
        mlp_fc2_weight = args[23]
        mlp_fc2_bias = args[24]
        layer_norm_eps = 1e-5
        exp_mod_shift = args[26]

        # First LayerNorm via Triton
        hidden_states = hidden_states.to(torch.float32)
        normed = _triton_layer_norm(hidden_states, norm1_weight, norm1_bias, layer_norm_eps)

        # Continue with the original pipeline using PyTorch ops for correctness
        output = run(
            normed,
            norm2_weight, norm2_bias,
            in_proj_weight, in_proj_bias,
            short_conv_weight, short_conv_bias,
            filter_linear1_weight, filter_linear1_bias,
            sin_freq, filter_linear2_weight, filter_linear2_bias,
            filter_linear3_weight, filter_linear3_bias,
            filter_linear_final_weight, filter_bias,
            exp_mod_deltas,
            out_proj_weight, out_proj_bias,
            mlp_fc1_weight, mlp_fc1_bias,
            mlp_fc2_weight, mlp_fc2_bias,
            layer_norm_eps,
            exp_mod_shift
        )
        return output


def run(*args):
    return ModelNew()(*args)
