import math
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_3d_forward_affine(X, Y, W, BIAS, EPS, B: tl.constexpr, L: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    """
    LayerNorm over last dimension for 3D tensor [B, L, D], with affine weight and bias.
    Launch as grid = (B, L). Each program handles one (b, l) row across D, looping over D in tiles.
    """
    b = tl.program_id(0)
    l = tl.program_id(1)

    base = (b * L + l) * D

    # Accumulate sum and sum of squares across D
    sum_x = 0.0
    sum_x2 = 0.0
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < D
        x = tl.load(X + base + d, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    D_f = tl.float32(D)
    mean = sum_x / D_f
    var = sum_x2 / D_f - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Second pass: normalize and apply affine
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < D
        x = tl.load(X + base + d, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + d, mask=mask, other=1.0).to(tl.float32)
        bias = tl.load(BIAS + d, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + bias
        tl.store(Y + base + d, y, mask=mask)


@triton.jit
def linear_3d_constK(X, W, BIAS, Y, B: tl.constexpr, L: tl.constexpr, D: tl.constexpr, K: tl.constexpr, BLOCK_D: tl.constexpr):
    """
    Compute Y[b, l, o] = sum_{d=0..D-1} X[b, l, d] * W[o, d] + BIAS[o]
    Launch as grid = (B, L, K). Each program handles one output channel o for a given (b, l).
    Loop over D in tiles to compute the dot product.
    """
    b = tl.program_id(0)
    l = tl.program_id(1)
    o = tl.program_id(2)

    acc = 0.0
    base_x = (b * L + l) * D
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < D
        x = tl.load(X + base_x + d, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + o * D + d, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * w, axis=0)

    acc = acc + tl.load(BIAS + o).to(tl.float32)
    tl.store(Y + (b * L + l) * K + o, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, layer_norm_eps=1e-5):
        super().__init__()
        self.layer_norm_eps = layer_norm_eps

    def forward(self, hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias,
                in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
                mlp_fc1_weight, mlp_fc1_bias, mlp_fc2_weight, mlp_fc2_bias,
                device=None):
        # Ensure inputs are float32 on the specified device
        dtype = torch.float32
        if device is None:
            device = hidden_states.device

        B, L, D = hidden_states.shape

        # 1) First LayerNorm over last dim on hidden_states with affine norm1
        normed = torch.empty((B, L, D), device=device, dtype=dtype)
        layernorm_3d_forward_affine[(B, L)](
            hidden_states, normed, norm1_weight, norm1_bias, self.layer_norm_eps,
            B=B, L=L, D=D, BLOCK_D=128, num_warps=4
        )

        # 2) In-projection: y_in = F.linear(normed, in_proj_weight, in_proj_bias)
        # in_proj_weight: [K, D], where K = D * (order+1) = 256 * 3 = 768
        y_in = torch.empty((B, L, D), device=device, dtype=dtype)
        K_in = D * 3
        linear_3d_constK[(B, L, K_in)](
            normed, in_proj_weight, in_proj_bias, y_in,
            B=B, L=L, D=D, K=K_in, BLOCK_D=128, num_warps=4
        )

        # 3) Second LayerNorm over last dim on (y_in + normed) with affine norm2
        sumed = y_in + normed
        final = torch.empty((B, L, D), device=device, dtype=dtype)
        layernorm_3d_forward_affine[(B, L)](
            sumed, final, norm2_weight, norm2_bias, self.layer_norm_eps,
            B=B, L=L, D=D, BLOCK_D=128, num_warps=4
        )

        # 4) Out-projection: final = F.linear(final, out_proj_weight, out_proj_bias)
        # out_proj_weight: [D, D] -> we need W[K, D] where K=D, so pass out_proj_weight.t()
        out = torch.empty((B, L, D), device=device, dtype=dtype)
        K_out = D
        linear_3d_constK[(B, L, K_out)](
            final, out_proj_weight.t(), out_proj_bias, out,
            B=B, L=L, D=D, K=K_out, BLOCK_D=128, num_warps=4
        )

        # 5) MLP: fc1 = F.linear(out, mlp_fc1_weight, mlp_fc1_bias); fc2 = F.linear(fc1, mlp_fc2_weight, mlp_fc2_bias)
        mlp1 = torch.empty((B, L, D), device=device, dtype=dtype)
        K_mlp = D
        linear_3d_constK[(B, L, K_mlp)](
            out, mlp_fc1_weight, mlp_fc1_bias, mlp1,
            B=B, L=L, D=D, K=K_mlp, BLOCK_D=128, num_warps=4
        )
        mlp2 = torch.empty((B, L, D), device=device, dtype=dtype)
        K_mlp2 = D
        linear_3d_constK[(B, L, K_mlp2)](
            mlp1, mlp_fc2_weight, mlp_fc2_bias, mlp2,
            B=B, L=L, D=D, K=K_mlp2, BLOCK_D=128, num_warps=4
        )

        return mlp2


def run(*args):
    return ModelNew()(*args)
