import math
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_3d_affine(
    X, Y, W, BIAS,
    EPS,
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    stride_x_b, stride_x_l, stride_x_d,
    stride_y_b, stride_y_l, stride_y_d,
    stride_w, stride_bias,
    num_warps=4, num_stages=2
):
    """
    LayerNorm over last dimension for 3D tensor [B, L, D], with affine weight and bias.
    Launch as grid = (B, L). Each program handles one (b, l) row across D, looping over D in tiles.
    """
    b = tl.program_id(0)
    l = tl.program_id(1)

    base_x = b * stride_x_b + l * stride_x_l
    base_y = b * stride_y_b + l * stride_y_l

    # Accumulate sum and sum of squares across D
    sum_x = 0.0
    sum_x2 = 0.0
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < D
        x_ptrs = X + base_x + d * stride_x_d
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x_vals, axis=0)
        sum_x2 += tl.sum(x_vals * x_vals, axis=0)

    D_f = tl.float32(D)
    mean = sum_x / D_f
    var = sum_x2 / D_f - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Second pass: normalize and apply affine
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < D
        x_ptrs = X + base_x + d * stride_x_d
        y_ptrs = Y + base_y + d * stride_y_d
        w_ptrs = W + d * stride_w
        b_ptrs = BIAS + d * stride_bias

        x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        w_vals = tl.load(w_ptrs, mask=mask, other=1.0).to(tl.float32)
        b_vals = tl.load(b_ptrs, mask=mask, other=0.0).to(tl.float32)

        y_vals = (x_vals - mean) * inv_std
        y_vals = y_vals * w_vals + b_vals
        tl.store(y_ptrs, y_vals, mask=mask)


@triton.jit
def linear_3d_constK(
    X,            # [B, L, D]
    W,            # [K, D]
    BIAS,         # [K]
    Y,            # [B, L, K]
    B: tl.constexpr,  # batch size (for type hints)
    L: tl.constexpr,  # seq_len
    D: tl.constexpr,  # hidden size
    K: tl.constexpr,  # output channels
    BLOCK_D: tl.constexpr,
    stride_x_b, stride_x_l, stride_x_d,
    stride_w_o, stride_w_d,
    stride_y_b, stride_y_l, stride_y_d,
    stride_bias_o,
    num_warps=4, num_stages=2
):
    """
    Compute y[b, l, o] = sum_{d=0..D-1} x[b, l, d] * w[o, d] + bias[o]
    Launch as grid = (B, L, K). Each program handles one output channel o for a given (b, l).
    """
    b = tl.program_id(0)
    l = tl.program_id(1)
    o = tl.program_id(2)

    acc = 0.0

    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < D

        # Load x[b, l, d]
        x_ptrs = X + b * stride_x_b + l * stride_x_l + d * stride_x_d
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Load w[o, d]
        w_ptrs = W + o * stride_w_o + d * stride_w_d
        w_vals = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Accumulate dot product
        acc += tl.sum(x_vals * w_vals, axis=0)

    # Add bias[o]
    bias_val = tl.load(BIAS + o * stride_bias_o).to(tl.float32)
    y_val = acc + bias_val

    # Store to Y[b, l, o]
    y_ptr = Y + b * stride_y_b + l * stride_y_l + o * stride_y_d
    tl.store(y_ptr, y_val)


class ModelNew(torch.nn.Module):
    def __init__(self, d_model: int = 256, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.layer_norm_eps = layer_norm_eps

    def forward(self, *args):
        # Unpack inputs exactly as in the original run:
        # 0: hidden_states [B, L, D]
        # 1: norm1_weight [D]
        # 2: norm1_bias [D]
        # 3: norm2_weight [D]
        # 4: norm2_bias [D]
        # 5: in_proj_weight [inner_width, D]
        # 6: in_proj_bias [inner_width]
        # 7: short_conv_weight unused in Triton path (we keep in PyTorch conv)
        # 8: short_conv_bias unused
        # 9: filter_linear1_weight
        # 10: filter_linear1_bias
        # 11: sin_freq
        # 12: filter_linear2_weight
        # 13: filter_linear2_bias
        # 14: filter_linear3_weight
        # 15: filter_linear3_bias
        # 16: filter_linear_final_weight
        # 17: filter_bias
        # 18: exp_mod_deltas
        # 19: out_proj_weight [D, D]
        # 20: out_proj_bias [D]
        # 21: mlp_fc1_weight [D, D]
        # 22: mlp_fc1_bias [D]
        # 23: mlp_fc2_weight [D, D]
        # 24: mlp_fc2_bias [D]
        # 25: layer_norm_eps (float)
        # 26: exp_mod_shift (float)

        hidden_states = args[0]               # [B, L, D]
        norm1_weight = args[1]                # [D]
        norm1_bias = args[2]                  # [D]
        norm2_weight = args[3]                # [D]
        norm2_bias = args[4]                  # [D]
        in_proj_weight = args[5]              # [inner_width, D]
        in_proj_bias = args[6]                # [inner_width]
        short_conv_weight = args[7]           # unused in this Triton path
        short_conv_bias = args[8]             # unused
        filter_linear1_weight = args[9]
        filter_linear1_bias = args[10]
        sin_freq = args[11]                   # not needed for our output
        filter_linear2_weight = args[12]
        filter_linear2_bias = args[13]
        filter_linear3_weight = args[14]
        filter_linear3_bias = args[15]
        filter_linear_final_weight = args[16]
        filter_bias = args[17]


def run(*args):
    return ModelNew()(*args)
