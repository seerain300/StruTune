import torch
import triton
import triton.language as tl

# Triton kernel: 3x3 convolution, stride=1, padding=1
# Inputs:
#   X_ptr: *float32, shape (N, C_in, H_in, W_in)
#   W_ptr: *float32, shape (C_out, C_in, 3, 3)
#   Y_ptr: *float32, shape (N, C_out, H_out, W_out) -- initialized to zeros
# Sizes:
#   N, C_in, H_in, W_in, C_out, H_out, W_out
@triton.jit
def conv3x3_stride1_pad1(
    X_ptr, W_ptr, Y_ptr,
    N, C_in, H_in, W_in, C_out, H_out, W_out,
    X_sN, X_sC, X_sH, X_sW,
    W_sCo, W_sCi, W_sKh, W_sKw,
    Y_sN, Y_sC, Y_sH, Y_sW,
    BLOCK_HW: tl.constexpr, BLOCK_CO: tl.constexpr
):
    # Grid: (N, C_out, tiles over H_out*W_out)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Compute vector of output positions for this tile
    num_hw = H_out * W_out
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < num_hw
    h_out = offs_hw // W_out
    w_out = offs_hw % W_out

    # Accumulator for BLOCK_HW outputs for channel pid_co
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                # Input indices with padding=1
                h_in = h_out + kh - 1
                w_in = w_out + kw - 1
                # Compute base offset for X at (n, ci, h_in, w_in)
                x_off = pid_n * X_sN + ci * X_sC + h_in * X_sH + w_in * X_sW
                # Load input vector with mask
                x_val = tl.load(X_ptr + x_off, mask=mask_hw, other=0.0)  # float32
                # Load weight scalar W[pid_co, ci, kh, kw]
                w_off = pid_co * W_sCo + ci * W_sCi + kh * W_sKh + kw * W_sKw
                w_val = tl.load(W_ptr + w_off)  # scalar
                acc += x_val * w_val

    # Store results to Y at channel pid_co
    y_off = pid_n * Y_sN + pid_co * Y_sC + h_out * Y_sH + w_out * Y_sW
    tl.store(Y_ptr + y_off, acc, mask=mask_hw)


# Triton kernel: reduce per-(N, group) sums for GroupNorm
# Computes sum and sumsq for channels in 'group' for each sample 'n'
# x: input tensor after conv, shape (N, C, H, W), any dtype
# sums_ptr: float32, shape (N, num_groups), stores [sum, sumsq] per (n,g)
@triton.jit
def groupnorm_reduce_sums(
    x_ptr, sums_ptr,
    N, C, H, W, num_groups,
    X_sN, X_sC, X_sH, X_sW,
    BLOCK_HW: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)  # group index
    C_per_group = C // num_groups

    # This program handles one channel in group pid_g for sample pid_n
    ci = pid_n * C_per_group + pid_g  # one program per channel? Too many.
    # Correction: we need to iterate over all channels in the group; each program handles one (n,g),
    # and loops over its channels. To do that, we need a third dimension over channels.
    # Triton does not support 3D grid like (N, num_groups, C_per_group) easily; so we instead
    # make pid_g be the combined id over channels in group by using a 2D grid (N, C_per_group*num_groups).
    # But simpler: we pass grid=(N, num_groups), and inside we loop over channels in that group.
    # Implement loop over channels: we don't have separate pid for channel, so we need to remap.
    # Better approach: launch grid=(N, C_per_group) with channels as program_id(2), but Triton supports up to 3 dims.
    # So, we instead do grid=(N, num_groups), and in each program loop over all channels of that group.
    # We compute start and then iterate:
    start = pid_n * C_per_group + pid_g
    # Loop over channels in this group
    # We need dynamic loop over C_per_group channels. Triton supports while loops.
    num_channels_in_group = C_per_group
    i = 0
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)
    while i < num_channels_in_group:
        chan = start + i
        # Sum over spatial
        num_hw = H * W
        s = tl.zeros((), dtype=tl.float32)
        ss = tl.zeros((), dtype=tl.float32)
        # loop over HW in chunks
        for off in range(0, num_hw, BLOCK_HW):
            hw_off = off + tl.arange(0, BLOCK_HW)
            mask_hw = hw_off < num_hw
            h = hw_off // W
            w = hw_off % W
            x_off = pid_n * X_sN + chan * X_sC + h * X_sH + w * X_sW
            x_val = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)  # dtype derived from tensor
            x_val_f32 = x_val.to(tl.float32)
            s += tl.sum(x_val_f32, axis=0)
            ss += tl.sum(x_val_f32 * x_val_f32, axis=0)
        sum_val += s
        sumsq_val += ss
        i += 1

    # Store as two scalars in sums_ptr at (pid_n, pid_g)
    tl.store(sums_ptr + pid_n * num_groups + pid_g, sum_val)
    tl.store(sums_ptr + pid_n * num_groups + pid_g + 1, sumsq_val)


# Triton kernel: apply GroupNorm (using precomputed per-group sums) + affine + SiLU
@triton.jit
def groupnorm_apply_affine_silu(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, H, W, num_groups, eps,
    X_sN, X_sC, X_sH, X_sW,
    Y_sN, Y_sC, Y_sH, Y_sW,
    GROUP_sN  # pointer to sums, we access as GROUP_sN[pid_n * num_groups + pid_g]
):
    # Grid over all elements (N, C, H, W)
    pid = tl.program_id(0)
    # Map pid -> (n, c, h, w)
    hw_per_n = H * W
    n = pid // (C * hw_per_n)
    rem = pid % (C * hw_per_n)
    c = rem // hw_per_n
    rem2 = rem % hw_per_n
    h = rem2 // W
    w = rem2 % W

    # Compute group index for channel c
    C_per_group = C // num_groups
    g = c // C_per_group

    # Load per-group sums (sum and sumsq) from GROUP_sN
    base = n * num_groups + g
    sum_val = tl.load(GROUP_sN + base)  # float32
    sumsq_val = tl.load(GROUP_sN + base + 1)  # float32

    # Compute mean and rstd
    M = C_per_group * hw_per_n  # number of elements in this group
    mean = sum_val / M
    var = sumsq_val / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Load x
    x_off = n * X_sN + c * X_sC + h * X_sH + w * X_sW
    x_val = tl.load(x_ptr + x_off)
    x_val_f32 = x_val.to(tl.float32)

    # Normalize
    y_norm = (x_val_f32 - mean) * rstd

    # Apply affine: weight[c] and bias[c]
    weight_c = tl.load(weight_ptr + c)  # float32
    bias_c = tl.load(bias_ptr + c)      # float32
    y_affine = y_norm * weight_c + bias_c

    # SiLU
    # sigmoid(y) = 1 / (1 + exp(-y))
    sig = 1.0 / (1.0 + tl.exp(-y_affine))
    y_silu = y_affine * sig

    # Store (cast back to original dtype of x_ptr/y_ptr)
    y_off = n * Y_sN + c * Y_sC + h * Y_sH + w * Y_sW
    # We don't know dtype here, rely on store casting
    tl.store(y_ptr + y_off, y_silu)


# Triton kernel: SiLU elementwise (in-place)
@triton.jit
def silu_inplace(x_ptr, N, C, H, W):
    pid = tl.program_id(0)
    hw_per_n = H * W
    n = pid // (C * hw_per_n)
    rem = pid % (C * hw_per_n)
    c = rem // hw_per_n
    rem2 = rem % hw_per_n
    h = rem2 // W
    w = rem2 % W

    off = n * C * H * W + c * H * W + h * W + w
    x_val = tl.load(x_ptr + off)
    x_f32 = x_val.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x_f32))
    y = x_f32 * sig
    tl.store(x_ptr + off, y)


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

# Forward function using Triton-only kernels
def run_triton_only(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    # Ensure CUDA tensors
    assert x.is_cuda, "ModelNew requires CUDA tensors"
    # Shapes
    N, C, H, W = x.shape
    # num_groups fixed to 32, ensure divisible
    num_groups = 32
    assert C % num_groups == 0, "C must be divisible by num_groups (32)"
    C_per_group = C // num_groups

    # Allocate outputs for convs (we'll compute in float32 and cast back)
    # But Triton kernels here assume float32 tensors; we'll operate in float32 for simplicity.
    # Make contiguous
    x_f32 = x.contiguous().to(torch.float32)
    conv1_weight_f32 = conv1_weight.contiguous().to(torch.float32)
    conv2_weight_f32 = conv2_weight.contiguous().to(torch.float32)
    # Output for first conv
    out1 = torch.empty((N, C, H, W), dtype=torch.float32, device=x.device)
    # Output for second conv
    out2 = torch.empty((N, C, H, W), dtype=torch.float32, device=x.device)

    # Launch first conv
    H_out = H
    W_out = W
    # Grid over (N, C_out=C, tiles over H*W)
    tiles_hw = (H_out * W_out + 127) // 128  # tile size 128
    grid_conv1 = (N, C, tiles_hw)
    conv3x3_stride1_pad1[grid_conv1](
        x_f32, conv1_weight_f32, out1,
        N, C, H, W, C, H_out, W_out,
        x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
        conv1_weight_f32.stride(0), conv1_weight_f32.stride(1), conv1_weight_f32.stride(2), conv1_weight_f32.stride(3),
        out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        BLOCK_HW=128, BLOCK_CO=1
    )

    # GroupNorm + SiLU for first conv output
    # Compute per-(N, group) sums
    sums1 = torch.empty((N, num_groups), dtype=torch.float32, device=x.device)
    grid_reduce1 = (N, num_groups)
    groupnorm_reduce_sums[grid_reduce1](
        out1, sums1,
        N, C, H, W, num_groups,
        out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        BLOCK_HW=1024
    )

    # Apply GroupNorm + affine + SiLU (in-place on out1)
    grid_apply1 = (N * C * H * W,)
    groupnorm_apply_affine_silu[grid_apply1](
        out1, out1, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32),
        N, C, H, W, num_groups, eps,
        out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        sums1
    )

    # SiLU (in-place)
    silu_inplace[(N * C * H * W,)](out1, N, C, H, W)

    # Second conv
    conv3x3_stride1_pad1[(N, C, tiles_hw)](
        out1, conv2_weight_f32, out2,
        N, C, H, W, C, H_out, W_out,
        out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        conv2_weight_f32.stride(0), conv2_weight_f32.stride(1), conv2_weight_f32.stride(2), conv2_weight_f32.stride(3),
        out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
        BLOCK_HW=128, BLOCK_CO=1
    )

    # GroupNorm + SiLU for second conv output
    sums2 = torch.empty((N, num_groups), dtype=torch.float32, device=x.device)
    groupnorm_reduce_sums[(N, num_groups)](
        out2, sums2,
        N, C, H, W, num_groups,
        out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
        BLOCK_HW=1024
    )

    groupnorm_apply_affine_silu[(N * C * H * W,)](
        out2, out2, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32),
        N, C, H, W, num_groups, eps,
        out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
        out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
        sums2
    )

    # SiLU (in-place)
    silu_inplace[(N * C * H * W,)](out2, N, C, H, W)

    # Residual add: out2 = out2 + x
    out_final = torch.empty_like(out2)
    add_inplace[(N * C * H * W,)](
        out2, x_f32, out_final,
        N, C, H, W
    )

    return out_final

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
        # Ensure Triton execution
        if not x.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensors for Triton execution.")
        return run_triton_only(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
