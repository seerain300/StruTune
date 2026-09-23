import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel: GroupNorm + SiLU for NCHW
# x: (B, C, H, W) contiguous
# norm_w, norm_b: (C,) affine scale/bias
# out: (B, C, H, W)
@triton.jit
def groupnorm_silu_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, out_ptr,
    B, C, H, W, num_groups, eps,
    C_PER_GROUP: tl.constexpr,
):
    # One program per (n, group)
    pid = tl.program_id(0)  # range [0, B * num_groups)
    n = pid // num_groups
    g = pid % num_groups

    # Compute group statistics: sum and sum of squares
    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    start_ci = g * C_PER_GROUP
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)
                s += x_val
                s2 += x_val * x_val

    group_size = C_PER_GROUP * H * W
    mean = s / group_size
    var = s2 / group_size - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, affine, SiLU, store
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        w = tl.load(norm_w_ptr + ci)
        b = tl.load(norm_b_ptr + ci)
        for h in range(0, H):
            for w_idx in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w_idx
                x = tl.load(x_ptr + idx)
                y = (x - mean) * invstd * w + b
                # SiLU: y * sigmoid(y)
                z = y * tl.sigmoid(y)
                tl.store(out_ptr + idx, z)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        # Validate shapes
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C, H, W = x.shape
        assert C % self.num_groups == 0, "num_groups must divide C"
        assert conv1_weight.shape[1] == C and conv1_weight.shape[0] == C, "conv1_weight must be (C, C, 3, 3)"
        assert conv2_weight.shape[1] == C and conv2_weight.shape[0] == C, "conv2_weight must be (C, C, 3, 3)"

        # Ensure contiguous and float32 for compute
        x = x.contiguous()
        x_f32 = x.to(torch.float32)

        # First conv: use cuDNN
        out1 = F.conv2d(x_f32, conv1_weight, bias=None, stride=1, padding=1)

        # GroupNorm + SiLU via Triton for out1
        out1_gn_silu = torch.empty_like(out1, dtype=torch.float32, device=x.device)
        C_per_group = C // self.num_groups
        grid = (B * self.num_groups,)
        groupnorm_silu_kernel[grid](
            out1, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32), out1_gn_silu,
            B, C, H, W, self.num_groups, self.eps,
            C_PER_GROUP=C_per_group
        )

        # Second conv: use cuDNN
        out2 = F.conv2d(out1_gn_silu, conv2_weight, bias=None, stride=1, padding=1)

        # GroupNorm + SiLU via Triton for out2
        out2_gn_silu = torch.empty_like(out2, dtype=torch.float32, device=x.device)
        groupnorm_silu_kernel[grid](
            out2, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32), out2_gn_silu,
            B, C, H, W, self.num_groups, self.eps,
            C_PER_GROUP=C_per_group
        )

        # Residual add
        out = out2_gn_silu + x_f32

        # Cast back to original dtype
        return out.to(x.dtype)


def run(*args):
    return ModelNew()(*args)
