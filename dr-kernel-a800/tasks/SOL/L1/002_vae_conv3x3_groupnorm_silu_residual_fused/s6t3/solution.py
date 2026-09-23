import torch
import triton
import triton.language as tl

# Conv3x3 stride=1, padding=1
# Input: x[N, C_in, H, W], weight[C_out, C_in, 3, 3]
# Output: y[N, C_out, H, W]
@triton.jit
def conv3x3_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C_in, H, W, C_out, OH, OW,
    X_sN, X_sC, X_sH, X_sW,
    W_sCO, W_sCI, W_sKH, W_sKW,
    Y_sN, Y_sC, Y_sH, Y_sW,
    BLOCK_HW: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)

    # Vector of output spatial positions to handle per program
    off_vec = tl.program_id(2) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_vec = off_vec < (OH * OW)
    h_out = off_vec // OW
    w_out = off_vec % OW

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Loop over input channels
    for cin in range(0, C_in):
        # Accumulate over 3x3 neighborhood with padding (implicit via masked loads)
        for kh in range(0, 3):
            ih = h_out + (1 - kh)  # padding=1: output h_out maps to ih in [-1, 1]
            # ih boundary check: mask vectorized
            mask_h = (ih >= 0) & (ih < H) & mask_vec
            for kw in range(0, 3):
                iw = w_out + (1 - kw)
                mask_w = (iw >= 0) & (iw < W) & mask_vec
                mask = mask_h & mask_w

                x_off = pid_n * X_sN + cin * X_sC + ih * X_sH + iw * X_sW
                # Load input values for this cin and 3x3 neighborhood; masked out-of-bounds -> 0
                x_val = tl.load(x_ptr + x_off, mask=mask, other=0.0)
                x_val = x_val.to(tl.float32)

                # Load corresponding weight for output channel pid_co
                w_off = pid_co * W_sCO + cin * W_sCI + kh * W_sKH + kw * W_sKW
                w_val = tl.load(w_ptr + w_off)  # scalar weight
                w_val = w_val.to(tl.float32)

                # FMA accumulate
                acc += x_val * w_val

    # Store results
    y_off = pid_n * Y_sN + pid_co * Y_sC + h_out * Y_sH + w_out * Y_sW
    tl.store(y_ptr + y_off, acc, mask=mask_vec)


# Triton kernel: reduce per (n, group) sum and sum of squares across channels in group and all spatial positions.
# x_ptr: input pointer (after first conv)
# sums_ptr: per-(n, group) storage of length 2 floats [sum, sumsq]
@triton.jit
def groupnorm_reduce_sums(
    x_ptr, sums_ptr,
    N, C, H, W, num_groups,
    X_sN, X_sC, X_sH, X_sW,
    BLOCK_HW: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    C_per_group = C // num_groups
    assert C % num_groups == 0, "C must be divisible by num_groups"
    start_chan = pid_n * C_per_group + pid_g

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Loop channels in this group
    for i in range(0, C_per_group):
        chan = start_chan + i
        num_hw = H * W
        s = tl.zeros((), dtype=tl.float32)
        ss = tl.zeros((), dtype=tl.float32)
        # Iterate H*W in chunks
        for off in range(0, num_hw, BLOCK_HW):
            hw_vec = off + tl.arange(0, BLOCK_HW)
            mask_hw = hw_vec < num_hw
            h = hw_vec // W
            w = hw_vec % W
            x_off = pid_n * X_sN + chan * X_sC + h * X_sH + w * X_sW
            x_val = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)
            x_f32 = x_val.to(tl.float32)
            s += tl.sum(x_f32, axis=0)
            ss += tl.sum(x_f32 * x_f32, axis=0)
        sum_val += s
        sumsq_val += ss

    M = C_per_group * H * W
    base = pid_n * (num_groups * 2) + pid_g * 2  # two floats per group
    tl.store(sums_ptr + base + 0, sum_val)       # sum
    tl.store(sums_ptr + base + 1, sumsq_val)     # sumsq


# Triton kernel: apply GroupNorm using precomputed sums, then affine + SiLU.
# Grid: (N*C,) one program per (n, c)
@triton.jit
def groupnorm_apply_affine_silu_nc(
    x_ptr, y_ptr, weight_ptr, bias_ptr, GROUP_sN,
    N, C, H, W, num_groups, eps,
    X_sN, X_sC, X_sH, X_sW,
    Y_sN, Y_sC, Y_sH, Y_sW
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    # Compute mean and rstd for this (n, c) using GROUP_sN which holds per-(n, group) stats
    C_per_group = C // num_groups
    g = c // C_per_group

    base = n * (num_groups * 2) + g * 2
    sum_val = tl.load(GROUP_sN + base + 0)
    sumsq_val = tl.load(GROUP_sN + base + 1)
    M = C_per_group * H * W
    mean = sum_val / M
    var = sumsq_val / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Load weight and bias for this channel
    weight_c = tl.load(weight_ptr + c)
    bias_c = tl.load(bias_ptr + c)

    # Apply GroupNorm affine
    # We need H*W iteration; do it vectorized with a single vector over H*W.
    num_hw = H * W
    hw_vec = tl.arange(0, num_hw)
    mask_hw = hw_vec < num_hw
    h = hw_vec // W
    w = hw_vec % W

    x_off = n * X_sN + c * X_sC + h * X_sH + w * X_sW
    x_val = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)
    x_f32 = x_val.to(tl.float32)

    y_norm = (x_f32 - mean) * rstd
    y_affine = y_norm * weight_c + bias_c

    # SiLU
    sig = 1.0 / (1.0 + tl.exp(-y_affine))
    y_silu = y_affine * sig

    # Store
    y_off = n * Y_sN + c * Y_sC + h * Y_sH + w * Y_sW
    tl.store(y_ptr + y_off, y_silu, mask=mask_hw)


# Triton kernel: elementwise add (out = a + b) over N*C*H*W
@triton.jit
def add_inplace(a_ptr, b_ptr, out_ptr, N, C, H, W):
    pid = tl.program_id(0)
    num_total = N * C * H * W
    idx = pid * 1024 + tl.arange(0, 1024)
    mask = idx < num_total
    # Decode idx into (n, c, h, w) for linear addressing
    n = idx // (C * H * W)
    rem = idx % (C * H * W)
    c = rem // (H * W)
    rem2 = rem % (H * W)
    h = rem2 // W
    w = rem2 % W

    off = n * C * H * W + c * H * W + h * W + w
    a = tl.load(a_ptr + off, mask=mask, other=0.0)
    b = tl.load(b_ptr + off, mask=mask, other=0.0)
    tl.store(out_ptr + off, a + b, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        # Ensure tensors are contiguous and float32
        assert x.dtype == torch.float32, "ModelNew expects float32 tensors"
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        N, C, H, W = x.shape
        assert C % self.num_groups == 0, "Input channels must be divisible by num_groups"

        # Output buffers for convs
        y1 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        y3 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        # Buffers for GroupNorm stats
        sums1 = torch.empty(N * self.num_groups * 2, device=x.device, dtype=torch.float32)
        sums2 = torch.empty(N * self.num_groups * 2, device=x.device, dtype=torch.float32)

        # Strides
        X_sN, X_sC, X_sH, X_sW = x.stride()
        W1_sCO, W1_sCI, W1_sKH, W1_sKW = conv1_weight.stride()
        W2_sCO, W2_sCI, W2_sKH, W2_sKW = conv2_weight.stride()
        Y1_sN, Y1_sC, Y1_sH, Y1_sW = y1.stride()

        # Launch conv1: y1 = conv3x3(x)
        BLOCK_HW = 256
        grid_conv1 = (N, C, triton.cdiv(H * W, BLOCK_HW))
        conv3x3_kernel[grid_conv1](
            x, conv1_weight, y1,
            N, C, H, W, C, H, W,
            X_sN, X_sC, X_sH, X_sW,
            W1_sCO, W1_sCI, W1_sKH, W1_sKW,
            Y1_sN, Y1_sC, Y1_sH, Y1_sW,
            BLOCK_HW=BLOCK_HW,
            num_warps=4
        )

        # GroupNorm and SiLU for y1: y2
        groupnorm_reduce_sums[(N, self.num_groups)](
            y1, sums1,
            N, C, H, W, self.num_groups,
            Y1_sN, Y1_sC, Y1_sH, Y1_sW,
            BLOCK_HW=BLOCK_HW,
            num_warps=1
        )
        y2 = torch.empty_like(y1)
        groupnorm_apply_affine_silu_nc[(N * C)](
            y1, y2, norm1_weight, norm1_bias, sums1,
            N, C, H, W, self.num_groups, eps,
            Y1_sN, Y1_sC, Y1_sH, Y1_sW,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=4
        )

        # Conv2: y3 = conv3x3(y2)
        y3.zero_()  # ensure clean output; not strictly necessary if we only use conv result
        grid_conv2 = (N, C, triton.cdiv(H * W, BLOCK_HW))
        conv3x3_kernel[grid_conv2](
            y2, conv2_weight, y3,
            N, C, H, W, C, H, W,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            W2_sCO, W2_sCI, W2_sKH, W2_sKW,
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            BLOCK_HW=BLOCK_HW,
            num_warps=4
        )

        # GroupNorm and SiLU for y3: y4
        groupnorm_reduce_sums[(N, self.num_groups)](
            y3, sums2,
            N, C, H, W, self.num_groups,
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            BLOCK_HW=BLOCK_HW,
            num_warps=1
        )
        y4 = torch.empty_like(y3)
        groupnorm_apply_affine_silu_nc[(N * C)](
            y3, y4, norm2_weight, norm2_bias, sums2,
            N, C, H, W, self.num_groups, eps,
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            y4.stride(0), y4.stride(1), y4.stride(2), y4.stride(3),
            num_warps=4
        )

        # Residual: out = y4 + x
        out = torch.empty_like(x)
        add_inplace[(triton.cdiv(N * C * H * W, 1024))](y4, x, out, N, C, H, W)

        return out


def run(*args):
    return ModelNew()(*args)
