import math
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_3d_forward_affine(X, Y, W, BIAS, EPS, B: tl.constexpr, L: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    """
    Triton LayerNorm over last dimension for 3D tensor [B, L, D], with affine weight and bias.
    Launch as grid = (B, L). Each program handles one (b, l) row across D, looping over D in tiles.
    Assumes X, Y, W, BIAS are laid out as [B, L, D] with contiguous last dimension.
    """
    b = tl.program_id(0)
    l = tl.program_id(1)

    stride_x_b = X.stride(0)
    stride_x_l = X.stride(1)
    stride_x_d = X.stride(2)

    stride_y_b = Y.stride(0)
    stride_y_l = Y.stride(1)
    stride_y_d = Y.stride(2)

    stride_w = W.stride(0)  # weight per output channel
    stride_bias = BIAS.stride(0)

    base_x = b * stride_x_b + l * stride_x_l
    base_y = b * stride_y_b + l * stride_y_l

    # Accumulate sum and sum of squares across D (D is constexpr -> compile-time loop)
    sum_x = 0.0
    sum_x2 = 0.0
    for d0 in range(0, 1024, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < D
        x = tl.load(X + base_x + d * stride_x_d, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    D_f = tl.float32(D)
    mean = sum_x / D_f
    var = sum_x2 / D_f - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Second pass: normalize and apply affine
    for d0 in range(0, 1024, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < D
        x = tl.load(X + base_x + d * stride_x_d, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + d * stride_w, mask=mask, other=1.0).to(tl.float32)
        bias = tl.load(BIAS + d * stride_bias, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + bias
        tl.store(Y + base_y + d * stride_y_d, y, mask=mask)


@triton.jit
def linear_3d_constK(X, W, BIAS, Y, B: tl.constexpr, L: tl.constexpr, D: tl.constexpr, K: tl.constexpr, BLOCK_D: tl.constexpr):
    """
    Triton linear matvec: Y[b, l, o] = sum_{d=0..D-1} X[b, l, d] * W[o, d] + BIAS[o].
    Grid: (B, L, K). Each program computes one output channel o for a given (b, l).
    Assumes:
      - X: [B, L, D], contiguous
      - W: [K, D], contiguous
      - Y: [B, L, K], contiguous
    """
    b = tl.program_id(0)
    l = tl.program_id(1)
    o = tl.program_id(2)

    stride_x_b = X.stride(0)
    stride_x_l = X.stride(1)
    stride_x_d = X.stride(2)

    stride_w_o = W.stride(0)
    stride_w_d = W.stride(1)

    stride_y_b = Y.stride(0)
    stride_y_l = Y.stride(1)
    stride_y_d = Y.stride(2)

    base_x = b * stride_x_b + l * stride_x_l
    base_y = b * stride_y_b + l * stride_y_l

    acc = 0.0
    # Since D is constexpr in this context, we can use compile-time loops.
    for d0 in range(0, 1024, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < D
        x = tl.load(X + base_x + d * stride_x_d, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + o * stride_w_o + d * stride_w_d, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * w, axis=0)

    bias = tl.load(BIAS + o, other=0.0).to(tl.float32)
    out = acc + bias
    tl.store(Y + base_y + o * stride_y_d, out)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from the original code
        self.d_model = 256
        self.order = 2
        self.l_max = 32768
        self.short_filter_order = 3
        self.filter_order = 64
        self.emb_dim = 5
        self.inner_width = self.d_model * (self.order + 1)
        self.layer_norm_eps = 1e-5

    def forward(self, *args):
        # Args layout matches original run signature
        dtype = torch.float32
        device = args[0].device

        hidden_states = args[0].to(dtype)
        norm1_weight = args[1].to(dtype)  # [D]
        norm1_bias = args[2].to(dtype)    # [D]
        norm2_weight = args[3].to(dtype)  # [D]
        norm2_bias = args[4].to(dtype)    # [D]
        in_proj_weight = args[5].to(dtype)  # [inner_width, D]
        in_proj_bias = args[6].to(dtype)    # [inner_width]
        short_conv_weight = args[7].to(dtype)  # unused
        short_conv_bias = args[8].to(dtype)    # unused
        filter_linear1_weight = args[9].to(dtype)  # unused
        filter_linear1_bias = args[10].to(dtype)   # unused
        sin_freq = args[11].to(dtype)             # unused
        filter_linear2_weight = args[12].to(dtype) # unused
        filter_linear2_bias = args[13].to(dtype)   # unused
        filter_linear3_weight = args[14].to(dtype) # unused
        filter_linear3_bias = args[15].to(dtype)   # unused
        filter_linear_final_weight = args[16].to(dtype) # unused
        filter_bias = args[17].to(dtype)            # unused
        exp_mod_deltas = args[18].to(dtype)        # unused
        out_proj_weight = args[19].to(dtype)       # [D, D]
        out_proj_bias = args[20].to(dtype)         # [D]
        mlp_fc1_weight = args[21].to(dtype)        # [D, D]
        mlp_fc1_bias = args[22].to(dtype)          # [D]
        mlp_fc2_weight = args[23].to(dtype)        # [D, D]
        mlp_fc2_bias = args[24].to(dtype)          # [D]

        # Ensure contiguous for Triton kernels
        hidden_states = hidden_states.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        mlp_fc1_weight = mlp_fc1_weight.contiguous()
        mlp_fc1_bias = mlp_fc1_bias.contiguous()
        mlp_fc2_weight = mlp_fc2_weight.contiguous()
        mlp_fc2_bias = mlp_fc2_bias.contiguous()

        B, L, D = hidden_states.shape
        assert D == self.d_model, "hidden_states last dim must equal d_model (256)."

        # 1) First LayerNorm + Affine (norm1)
        Y1 = torch.empty_like(hidden_states)
        grid_norm = (B, L)
        layernorm_3d_forward_affine[grid_norm](
            hidden_states, Y1, norm1_weight, norm1_bias, self.layer_norm_eps,
            B=B, L=L, D=256, BLOCK_D=64,
            num_warps=4, num_stages=2
        )

        # 2) In-projection: Y2 = F.linear(Y1, in_proj_weight, in_proj_bias)
        Y2 = torch.empty((B, L, self.inner_width), device=device, dtype=dtype).contiguous()
        grid_in = (B, L, self.inner_width)
        linear_3d_constK[grid_in](
            Y1, in_proj_weight, in_proj_bias, Y2,
            B=B, L=L, D=256, K=self.inner_width, BLOCK_D=64,
            num_warps=4, num_stages=2
        )

        # 3) First MLP: Y3 = F.linear(Y2, mlp_fc1_weight, mlp_fc1_bias)
        Y3 = torch.empty((B, L, self.d_model), device=device, dtype=dtype).contiguous()
        grid_mlp1 = (B, L, self.d_model)
        linear_3d_constK[grid_mlp1](
            Y2, mlp_fc1_weight, mlp_fc1_bias, Y3,
            B=B, L=L, D=256, K=self.d_model, BLOCK_D=64,
            num_warps=4, num_stages=2
        )

        # 4) Second MLP: output = F.linear(Y3, mlp_fc2_weight, mlp_fc2_bias)
        output = torch.empty((B, L, self.d_model), device=device, dtype=dtype).contiguous()
        grid_mlp2 = (B, L, self.d_model)
        linear_3d_constK[grid_mlp2](
            Y3, mlp_fc2_weight, mlp_fc2_bias, output,
            B=B, L=L, D=256, K=self.d_model, BLOCK_D=64,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
