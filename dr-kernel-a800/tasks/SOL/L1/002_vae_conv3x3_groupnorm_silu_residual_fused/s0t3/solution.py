import torch
import triton
import triton.language as tl

# Constants
NUM_GROUPS = 32

# Convolution kernel: per-output element with static loops for kh/kw/ci
@triton.jit
def conv3x3_per_elem(
    x_ptr,          # *float32, input (B, C_in, H, W)
    w_ptr,          # *float32, weights (C_out, C_in, 3, 3)
    out_ptr,        # *float32, output (B, C_out, H, W)
    B, C_in, C_out, H, W,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_HW: tl.constexpr,  # not used directly, but kept for signature
):
    # program ids: (batch, out_channel, h, w)
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    h = pid_h
    w = pid_w

    # accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # input channels loop
    for ci in range(0, C_in):
        # 3x3 kernel: static loops for unrolling
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = h + kh - 1
                w_in = w + kw - 1
                # in-bounds mask
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                x_ptrs = x_ptr + pid_b * x_stride_b + ci * x_stride_c + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(x_ptrs, mask=in_bounds, other=0.0)
                # load corresponding weight for (co, ci, kh, kw)
                w_idx = pid_co * (C_in * 9) + ci * 9 + kh * 3 + kw
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val

    # store output
    out_ptrs = out_ptr + pid_b * out_stride_b + pid_co * out_stride_c + h * out_stride_h + w * out_stride_w
    tl.store(out_ptrs, acc)


# GroupNorm: one program per (n, group)
@triton.jit
def group_norm_kernel(
    in_ptr, out_ptr,
    norm_weight_ptr, norm_bias_ptr,
    B, C, H, W, num_groups, eps,
    in_stride_b, in_stride_c, in_stride_h, in_stride_w,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_GROUP_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_group = tl.program_id(1)

    group_size = C // num_groups
    n = pid_n
    group = pid_group
    start_c = group * group_size

    total_sum = tl.zeros((), dtype=tl.float32)
    total_sumsq = tl.zeros((), dtype=tl.float32)

    # First pass: compute sum and sumsq over group
    for ci in range(0, group_size):
        c = start_c + ci
        HW = H * W
        for hw in range(0, HW):
            h = hw // W
            w = hw % W
            in_ptrs = in_ptr + n * in_stride_b + c * in_stride_c + h * in_stride_h + w * in_stride_w
            x = tl.load(in_ptrs, mask=True, other=0.0)
            total_sum += x
            total_sumsq += x * x

    group_num = group_size * HW
    mean = total_sum / group_num
    var = total_sumsq / group_num - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for ci in range(0, group_size):
        c = start_c + ci
        gamma = tl.load(norm_weight_ptr + c)
        beta = tl.load(norm_bias_ptr + c)
        for hw in range(0, H * W):
            h = hw // W
            w = hw % W
            in_ptrs = in_ptr + n * in_stride_b + c * in_stride_c + h * in_stride_h + w * in_stride_w
            x = tl.load(in_ptrs)
            y = (x - mean) * inv_std
            y = y * gamma + beta
            out_ptrs = out_ptr + n * out_stride_b + c * out_stride_c + h * out_stride_h + w * out_stride_w
            tl.store(out_ptrs, y)


# SiLU elementwise
@triton.jit
def silu_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x * tl.sigmoid(x)
    tl.store(out_ptr + offsets, y, mask=mask)


# Residual add elementwise: out += x
@triton.jit
def add_residual_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.load(out_ptr + offsets, mask=mask, other=0.0)
    out = out + x
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        # Ensure tensors on CUDA
        assert x.is_cuda, "Input x must be on CUDA for Triton kernels."
        assert conv1_weight.is_cuda and norm1_weight.is_cuda and norm1_bias.is_cuda and conv2_weight.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "All weights and norms must be on CUDA."

        # Cast to float32 for numerical stability
        x_f = x.contiguous().to(torch.float32)
        B, C, H, W = x_f.shape
        # Check GroupNorm compatibility
        assert C % NUM_GROUPS == 0, "Channel count must be divisible by NUM_GROUPS=32 for GroupNorm."

        C_in = conv1_weight.shape[0]
        C_out1 = conv1_weight.shape[1]

        # First conv: out1
        out1 = torch.empty((B, C_out1, H, W), dtype=torch.float32, device=x.device)
        grid_conv1 = (B, C_out1, H, W)
        conv3x3_per_elem[grid_conv1](
            x_f, conv1_weight, out1,
            B, C_in, C_out1, H, W,
            x_f.stride(0), x_f.stride(1), x_f.stride(2), x_f.stride(3),
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            BLOCK_HW=1,  # not used; static loops handle 3x3
        )

        # GroupNorm1
        out1_gn = torch.empty_like(out1)
        grid_gn1 = (B, NUM_GROUPS)
        group_norm_kernel[grid_gn1](
            out1, out1_gn,
            norm1_weight, norm1_bias,
            B, C_out1, H, W, NUM_GROUPS, eps,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            out1_gn.stride(0), out1_gn.stride(1), out1_gn.stride(2), out1_gn.stride(3),
            BLOCK_GROUP_SIZE=NUM_GROUPS,
        )

        # SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # Second conv: out2
        C_out2 = conv2_weight.shape[1]
        out2 = torch.empty((B, C_out2, H, W), dtype=torch.float32, device=x.device)
        grid_conv2 = (B, C_out2, H, W)
        conv3x3_per_elem[grid_conv2](
            out1_silu, conv2_weight, out2,
            B, C_out1, C_out2, H, W,
            out1_silu.stride(0), out1_silu.stride(1), out1_silu.stride(2), out1_silu.stride(3),
            out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
            BLOCK_HW=1,
        )

        # GroupNorm2
        out2_gn = torch.empty_like(out2)
        grid_gn2 = (B, NUM_GROUPS)
        group_norm_kernel[grid_gn2](
            out2, out2_gn,
            norm2_weight, norm2_bias,
            B, C_out2, H, W, NUM_GROUPS, eps,
            out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
            out2_gn.stride(0), out2_gn.stride(1), out2_gn.stride(2), out2_gn.stride(3),
            BLOCK_GROUP_SIZE=NUM_GROUPS,
        )

        # SiLU2
        out2_silu = torch.empty_like(out2_gn)
        N2 = out2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](out2_gn, out2_silu, N2, BLOCK=1024)

        # Add residual x
        out = torch.empty_like(out2_silu)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[x_f, out2_silu, out](x_f, out2_silu, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
