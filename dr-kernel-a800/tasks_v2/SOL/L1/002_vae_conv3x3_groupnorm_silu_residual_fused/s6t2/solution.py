import torch
import triton
import triton.language as tl

# Triton kernel: compute per-(N, group) sums and sum of squares for GroupNorm
# x: input tensor after conv, shape (N, C, H, W)
# sums_ptr: float32 tensor of shape (N, num_groups, 2), stores [sum, sumsq] per (n, group)
@triton.jit
def groupnorm_reduce_sums(
    x_ptr, sums_ptr,
    N, C, H, W, num_groups,
    X_sN, X_sC, X_sH, X_sW,
    BLOCK_HW: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)  # group id
    C_per_group = C // num_groups

    # Handle all channels in group pid_g for sample pid_n
    start_chan = pid_n * C_per_group + pid_g

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Loop over channels in this group
    for i in range(0, C_per_group):
        chan = start_chan + i
        num_hw = H * W
        s = tl.zeros((), dtype=tl.float32)
        ss = tl.zeros((), dtype=tl.float32)
        # Iterate over H*W in chunks
        for off in range(0, num_hw, BLOCK_HW):
            hw_off = off + tl.arange(0, BLOCK_HW)
            mask_hw = hw_off < num_hw
            h = hw_off // W
            w = hw_off % W
            x_off = pid_n * X_sN + chan * X_sC + h * X_sH + w * X_sW
            x_val = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)
            x_f32 = x_val.to(tl.float32)
            s += tl.sum(x_f32, axis=0)
            ss += tl.sum(x_f32 * x_f32, axis=0)
        sum_val += s
        sumsq_val += ss

    M = C_per_group * H * W
    base = pid_n * (num_groups * 2) + pid_g * 2  # address within sums_ptr for this (n, g)
    tl.store(sums_ptr + base + 0, sum_val)       # sum at index 0
    tl.store(sums_ptr + base + 1, sumsq_val)     # sumsq at index 1


# Triton kernel: apply GroupNorm (using precomputed per-group sums) + affine + SiLU
@triton.jit
def groupnorm_apply_affine_silu(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, H, W, num_groups, eps,
    X_sN, X_sC, X_sH, X_sW,
    Y_sN, Y_sC, Y_sH, Y_sW,
    GROUP_sN  # pointer to sums, layout (N, num_groups, 2) with two floats per group: [sum, sumsq]
):
    pid = tl.program_id(0)
    hw_per_n = H * W
    n = pid // (C * hw_per_n)
    rem = pid % (C * hw_per_n)
    c = rem // hw_per_n
    rem2 = rem % hw_per_n
    h = rem2 // W
    w = rem2 % W

    C_per_group = C // num_groups
    g = c // C_per_group

    base = n * (num_groups * 2) + g * 2  # address within GROUP_sN for this (n, g)
    sum_val = tl.load(GROUP_sN + base + 0)
    sumsq_val = tl.load(GROUP_sN + base + 1)

    M = C_per_group * hw_per_n
    mean = sum_val / M
    var = sumsq_val / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    x_off = n * X_sN + c * X_sC + h * X_sH + w * X_sW
    x_val = tl.load(x_ptr + x_off)
    x_f32 = x_val.to(tl.float32)

    y_norm = (x_f32 - mean) * rstd
    weight_c = tl.load(weight_ptr + c)
    bias_c = tl.load(bias_ptr + c)
    y_affine = y_norm * weight_c + bias_c

    # SiLU
    sig = 1.0 / (1.0 + tl.exp(-y_affine))
    y_silu = y_affine * sig

    y_off = n * Y_sN + c * Y_sC + h * Y_sH + w * Y_sW
    tl.store(y_ptr + y_off, y_silu)


# Triton kernel: elementwise add (out = x1 + x2)
@triton.jit
def add_inplace(x1_ptr, x2_ptr, out_ptr, N, C, H, W):
    pid = tl.program_id(0)
    hw_per_n = H * W
    n = pid // (C * hw_per_n)
    rem = pid % (C * hw_per_n)
    c = rem // hw_per_n
    rem2 = rem % hw_per_n
    h = rem2 // W
    w = rem2 % W

    off = n * C * H * W + c * H * W + h * W + w
    a = tl.load(x1_ptr + off)
    b = tl.load(x2_ptr + off)
    tl.store(out_ptr + off, a + b)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # Ensure CUDA and float32
        assert x.is_cuda and conv1_weight.is_cuda and conv2_weight.is_cuda, "All tensors must be on CUDA"
        assert x.dtype == torch.float32 and conv1_weight.dtype == torch.float32 and conv2_weight.dtype == torch.float32, "Use float32 tensors"
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        N, C, H, W = x.shape
        assert C % 32 == 0, "num_groups=32 requires C % 32 == 0"
        num_groups = 32
        C_per_group = C // num_groups

        # 1) First conv (PyTorch/cuDNN)
        out1 = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # 2) GroupNorm + affine + SiLU using Triton
        out1 = out1.contiguous()
        # Compute per-(N, group) sums and sumsq (layout (N, num_groups, 2))
        sums1 = torch.empty((N, num_groups, 2), dtype=torch.float32, device=x.device)
        grid_reduce = (N, num_groups)
        groupnorm_reduce_sums[grid_reduce](
            out1, sums1,
            N, C, H, W, num_groups,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            BLOCK_HW=1024,
        )
        # Apply GroupNorm + affine + SiLU
        out1_silu = torch.empty_like(out1)
        total_elems = N * C * H * W
        grid_apply = (total_elems,)
        groupnorm_apply_affine_silu[grid_apply](
            out1, out1_silu, norm1_weight, norm1_bias,
            N, C, H, W, num_groups, eps,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            out1_silu.stride(0), out1_silu.stride(1), out1_silu.stride(2), out1_silu.stride(3),
            sums1,
        )

        # 3) Second conv (PyTorch/cuDNN)
        out2 = torch.nn.functional.conv2d(out1_silu, conv2_weight, bias=None, stride=1, padding=1)
        out2 = out2.contiguous()

        # 4) GroupNorm + affine + SiLU using Triton
        sums2 = torch.empty((N, num_groups, 2), dtype=torch.float32, device=x.device)
        groupnorm_reduce_sums[grid_reduce](
            out2, sums2,
            N, C, H, W, num_groups,
            out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
            BLOCK_HW=1024,
        )
        out2_silu = torch.empty_like(out2)
        groupnorm_apply_affine_silu[grid_apply](
            out2, out2_silu, norm2_weight, norm2_bias,
            N, C, H, W, num_groups, eps,
            out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
            out2_silu.stride(0), out2_silu.stride(1), out2_silu.stride(2), out2_silu.stride(3),
            sums2,
        )

        # 5) Residual add (out2_silu + x), Triton elementwise
        out_final = torch.empty_like(out2_silu)
        add_inplace[grid_apply](
            out2_silu, x, out_final,
            N, C, H, W,
        )

        return out_final


def run(*args):
    return ModelNew()(*args)
