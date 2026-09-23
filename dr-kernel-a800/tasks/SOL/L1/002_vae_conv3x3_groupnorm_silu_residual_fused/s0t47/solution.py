import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_nobias_single(
    x_ptr,       # *const float, input [B, C_in, H, W]
    w_ptr,       # *const float, weights [C_out, C_in, 3, 3] flattened
    out_ptr,     # *float, output [B, C_out, H, W]
    # sizes and strides (as constexpr for Triton JIT)
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)  # batch
    pid_co = tl.program_id(1) # output channel
    pid_h = tl.program_id(2)  # output height
    pid_w = tl.program_id(3)  # output width

    h_out = pid_h
    w_out = pid_w

    # accumulate in float32
    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and 3x3 kernel, with masks for padding
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = h_out - 1 + kh  # padding=1
                w_in = w_out - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                base_x = pid_n * x_stride_n + ci * x_stride_c
                ptr_x = x_ptr + base_x + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(ptr_x, mask=in_bounds, other=0.0).to(tl.float32)
                # weight index: w_ptr is flattened as [C_out, C_in, 9], contiguous
                w_idx = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx).to(tl.float32)
                acc += x_val * w_val

    # store result
    base_out = pid_n * out_stride_n + pid_co * out_stride_c
    ptr_out = out_ptr + base_out + h_out * out_stride_h + w_out * out_stride_w
    tl.store(ptr_out, acc)


@triton.jit
def groupnorm_two_pass(
    x_ptr,        # *const float, input [B, C, H, W]
    gamma_ptr,    # *const float, per-channel scale [C]
    beta_ptr,     # *const float, per-channel bias [C]
    out_ptr,      # *float, output [B, C, H, W]
    # sizes and strides (as constexpr for Triton JIT)
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    group_size: tl.constexpr, eps: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
    GROUP: tl.constexpr, NUM_GROUPS: tl.constexpr,
):
    # program ids: combine batch and group into one dimension
    pid = tl.program_id(0)
    n = pid // GROUP
    g = pid % GROUP

    # compute group channel range
    start_c = g * group_size
    # vectors for channel within group
    cvec = start_c + tl.arange(0, group_size)
    H_vec = tl.arange(0, H)
    W_vec = tl.arange(0, W)
    # build flat indices for x: ((n*C + c)*H + h)*W + w
    idx = ((n * C + cvec) * H + H_vec[:, None]) * W + W_vec[None, :]
    # bounds mask
    mask_hw = (H_vec[:, None] < H) & (W_vec[None, :] < W)
    mask = (cvec < C) & mask_hw

    # load x as 2D: [group_size, H*W]
    x_2d = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    # compute sum and sum of squares across H*W for each channel in group
    sum_vec = tl.zeros((group_size,), dtype=tl.float32)
    sq_vec = tl.zeros((group_size,), dtype=tl.float32)
    for i in tl.static_range(group_size):
        sum_vec[i] = tl.sum(x_2d[i, :])
        sq_vec[i] = tl.sum(x_2d[i, :] * x_2d[i, :])

    mean = sum_vec / (H * W)
    var = sq_vec / (H * W) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # second pass: normalize and apply affine, then store
    for i in tl.static_range(group_size):
        c = start_c + i
        # load gamma/beta
        gamma = tl.load(gamma_ptr + c).to(tl.float32)
        beta = tl.load(beta_ptr + c).to(tl.float32)
        mean_i = mean[i]
        inv_std_i = inv_std[i]
        # output pointer for this (n, c)
        out_idx = ((n * C + c) * H + H_vec) * W + W_vec
        x_norm = tl.load(x_ptr + out_idx, mask=(c < C) & (H_vec < H) & (W_vec < W), other=0.0).to(tl.float32)
        y = (x_norm - mean_i) * inv_std_i
        y = y * gamma + beta
        tl.store(out_ptr + out_idx, y, mask=(c < C) & (H_vec < H) & (W_vec < W))


@triton.jit
def silu_kernel(in_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # sigmoid
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def add_residual_kernel(x1_ptr, x2_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(x1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a + b
    tl.store(out_ptr + offs, y, mask=mask)


def run_triton(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    # x: [B, C, H, W], float32, contiguous
    assert x.dim() == 4, "x must be [B, C, H, W]"
    B, C, H, W = x.shape
    # conv weights: [C, C, 3, 3], float32
    C_in1 = conv1_weight.shape[1]
    C_out1 = conv1_weight.shape[0]
    C_in2 = conv2_weight.shape[1]
    C_out2 = conv2_weight.shape[0]
    # GroupNorm num_groups=32, C must be divisible by 32
    num_groups = 32
    assert C % num_groups == 0, "C must be divisible by num_groups=32"

    device = x.device
    dtype = torch.float32
    x_f = x.to(dtype).contiguous()
    conv1_w_f = conv1_weight.to(dtype).contiguous()
    conv2_w_f = conv2_weight.to(dtype).contiguous()
    norm1_weight_f = norm1_weight.to(dtype).contiguous()
    norm1_bias_f = norm1_bias.to(dtype).contiguous()
    norm2_weight_f = norm2_weight.to(dtype).contiguous()
    norm2_bias_f = norm2_bias.to(dtype).contiguous()

    # 1) Conv1: out1 = conv3x3(x, conv1_weight, bias=None, stride=1, padding=1)
    out1 = torch.empty((B, C_out1, H, W), device=device, dtype=torch.float32)
    grid_conv1 = (B, C_out1, H, W)
    conv3x3_stride1_pad1_nobias_single[grid_conv1](
        x_f, conv1_w_f, out1,
        B, C_in1, H, W, C_out1, H, W,
        x_f.stride(0), x_f.stride(1), x_f.stride(2), x_f.stride(3),
        out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        num_warps=4, num_stages=2,
    )

    # 2) GroupNorm1 (num_groups=32)
    out1_gn = torch.empty_like(out1)
    group_size = C // num_groups
    grid_gn1 = (B * num_groups,)
    groupnorm_two_pass[grid_gn1](
        out1, norm1_weight_f, norm1_bias_f, out1_gn,
        B, C, H, W, group_size, eps,
        out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        out1_gn.stride(0), out1_gn.stride(1), out1_gn.stride(2), out1_gn.stride(3),
        GROUP=group_size, NUM_GROUPS=num_groups,
        num_warps=4, num_stages=2,
    )

    # 3) SiLU1
    out1_silu = torch.empty_like(out1_gn)
    N1 = out1_gn.numel()
    grid_silu1 = (triton.cdiv(N1, 1024),)
    silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

    # 4) Conv2: out2_pre = conv3x3(out1_silu, conv2_weight, bias=None, stride=1, padding=1)
    out2_pre = torch.empty((B, C_out2, H, W), device=device, dtype=torch.float32)
    grid_conv2 = (B, C_out2, H, W)
    conv3x3_stride1_pad1_nobias_single[grid_conv2](
        out1_silu, conv2_w_f, out2_pre,
        B, C_in2, H, W, C_out2, H, W,
        out1_silu.stride(0), out1_silu.stride(1), out1_silu.stride(2), out1_silu.stride(3),
        out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        num_warps=4, num_stages=2,
    )

    # 5) GroupNorm2 (num_groups=32)
    out2_gn = torch.empty_like(out2_pre)
    grid_gn2 = (B * num_groups,)
    groupnorm_two_pass[grid_gn2](
        out2_pre, norm2_weight_f, norm2_bias_f, out2_gn,
        B, C_out2, H, W, group_size, eps,
        out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        out2_gn.stride(0), out2_gn.stride(1), out2_gn.stride(2), out2_gn.stride(3),
        GROUP=group_size, NUM_GROUPS=num_groups,
        num_warps=4, num_stages=2,
    )

    # 6) SiLU2
    out2_silu = torch.empty_like(out2_gn)
    N2 = out2_gn.numel()
    grid_silu2 = (triton.cdiv(N2, 1024),)
    silu_kernel[grid_silu2](out2_gn, out2_silu, N2, BLOCK=1024)

    # 7) Add residual x
    out = torch.empty_like(out2_silu)
    Nfinal = out2_silu.numel()
    grid_add = (triton.cdiv(Nfinal, 1024),)
    add_residual_kernel[grid_add](out2_silu, x_f, out, Nfinal, BLOCK=1024)

    return out


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5, num_groups: int = 32):
        super().__init__()
        self.eps = float(eps)
        self.num_groups = int(num_groups)

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        # All computation in Triton; no torch ops in forward
        return run_triton(
            x, conv1_weight, norm1_weight, norm1_bias,
            conv2_weight, norm2_weight, norm2_bias,
            self.eps
        )


def run(*args):
    return ModelNew()(*args)
