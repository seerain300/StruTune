import torch
import triton
import triton.language as tl


# Conv2d 3x3, stride=1, padding=1, bias=None
# Kernel 1: single output element per program. Simple, robust, but not fastest.
@triton.jit
def conv3x3_stride1_pad1_single_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C, H, W,
    C_in, C_out, K,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cout, w_stride_cin, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
):
    pid = tl.program_id(axis=0)
    total = N * C_out * H * W
    if pid >= total:
        return
    n = pid // (C_out * H * W)
    tmp = pid % (C_out * H * W)
    c_out = tmp // (H * W)
    ow = tmp % (H * W)
    # derive oh from tmp? We can't, since H*W was used above. So we instead use grid over (oh, ow) in forward.
    # To keep one program per element, we rely on grid sized exactly N*C_out*H*W and compute oh = (pid // (C_out*W)) % H,
    # but Triton limits to 1D grid. Therefore, this kernel is redefined below as conv2d_tile which uses 2D grid.
    # For now, we will not use this kernel in forward; conv2d_tile is the one used.
    # Placeholder return to satisfy Triton JIT; not executed in forward.
    return


# Conv2d 3x3, stride=1, padding=1, bias=None
# Kernel 2: tile of output channels per program. Faster, still robust.
@triton.jit
def conv3x3_stride1_pad1_tile_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C_out, H, W,
    C_in, BLOCK_OC: tl.constexpr,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cout, w_stride_cin, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
):
    # 2D grid: axis 0 over N*H*W, axis 1 over ceil_div(C_out, BLOCK_OC)
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    total_ow = W  # not directly used here; we infer ow from pid0
    n = pid0 // (H * W)
    tmp = pid0 % (H * W)
    oh = tmp // W
    ow = tmp % W

    oc_start = pid1 * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    # accumulator for output channels
    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # loop over input channels
    for ci in range(0, C_in):
        # loop over 3x3 taps
        for kh in range(0, 3):
            ih = oh + kh - 1
            valid_h = (ih >= 0) & (ih < H)
            for kw in range(0, 3):
                iw = ow + kw - 1
                valid_w = (iw >= 0) & (iw < W)
                valid = valid_h & valid_w
                if valid:
                    # load x[n, ci, ih, iw]
                    x_off = n * x_stride_n + ci * x_stride_c + ih * x_stride_h + iw * x_stride_w
                    x_val = tl.load(x_ptr + x_off)  # float32 assumed
                    # load weights for oc tile: w[c_out_tile, ci, kh, kw]
                    w_off = oc_offsets * w_stride_cout + ci * w_stride_cin + kh * w_stride_kh + kw * w_stride_kw
                    w_vec = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                    acc += x_val * w_vec

    # store results
    y_off_base = n * y_stride_n + oc_offsets * y_stride_c + oh * y_stride_h + ow * y_stride_w
    tl.store(y_ptr + y_off_base, acc, mask=oc_mask)


# GroupNorm first pass: compute sum and sumsq over channels in group and all spatial positions
@triton.jit
def group_norm_first_pass(
    x_ptr, sum_ptr, sumsq_ptr,
    N, C, H, W, G, eps,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    MAX_C: tl.constexpr,
):
    n = tl.program_id(axis=0)
    group = tl.program_id(axis=1)
    group_size = C // G
    start_c = group * group_size

    sum_val = 0.0
    sumsq_val = 0.0

    for ch in range(0, MAX_C):
        if ch >= C:
            break
        if (ch < start_c) or (ch >= start_c + group_size):
            continue
        for h in range(0, H):
            for w in range(0, W):
                x_off = n * x_stride_n + ch * x_stride_c + h * x_stride_h + w * x_stride_w
                x_val = tl.load(x_ptr + x_off)
                sum_val += x_val
                sumsq_val += x_val * x_val

    denom = float(H * W * group_size)
    mean = sum_val / denom
    var = sumsq_val / denom - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # write mean and inv_std to sum_ptr/sumsq_ptr if we need them? For now, we only compute; second pass will use them.
    # We will instead pass them via y_ptr for second pass. To keep simple, we recompute in second pass.


# GroupNorm second pass: normalize and apply affine
@triton.jit
def group_norm_second_pass(
    x_ptr, y_ptr, gamma_ptr, beta_ptr,
    N, C, H, W, G, eps,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    MAX_C: tl.constexpr,
):
    n = tl.program_id(axis=0)
    group = tl.program_id(axis=1)
    group_size = C // G
    start_c = group * group_size

    # We need mean and inv_std; recompute them here for simplicity:
    sum_val = 0.0
    sumsq_val = 0.0
    for ch in range(0, MAX_C):
        if ch >= C:
            break
        if (ch < start_c) or (ch >= start_c + group_size):
            continue
        for h in range(0, H):
            for w in range(0, W):
                x_off = n * x_stride_n + ch * x_stride_c + h * x_stride_h + w * x_stride_w
                x_val = tl.load(x_ptr + x_off)
                sum_val += x_val
                sumsq_val += x_val * x_val
    denom = float(H * W * group_size)
    mean = sum_val / denom
    var = sumsq_val / denom - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # normalize and apply affine per element
    for ch in range(0, MAX_C):
        if ch >= C:
            break
        if (ch < start_c) or (ch >= start_c + group_size):
            continue
        for h in range(0, H):
            for w in range(0, W):
                x_off = n * x_stride_n + ch * x_stride_c + h * x_stride_h + w * x_stride_w
                x_val = tl.load(x_ptr + x_off)
                norm = (x_val - mean) * inv_std
                gamma = tl.load(gamma_ptr + ch)
                beta = tl.load(beta_ptr + ch)
                y_val = norm * gamma + beta
                y_off = n * y_stride_n + ch * y_stride_c + h * y_stride_h + w * y_stride_w
                tl.store(y_ptr + y_off, y_val)


# SiLU elementwise
@triton.jit
def silu_kernel(x_ptr, y_ptr, N, C, H, W,
                x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                y_stride_n, y_stride_c, y_stride_h, y_stride_w):
    total = N * C * H * W
    pid = tl.program_id(axis=0)
    if pid >= total:
        return
    n = pid // (C * H * W)
    tmp = pid % (C * H * W)
    c = tmp // (H * W)
    hw = tmp % (H * W)
    h = hw // W
    w = hw % W
    x_off = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
    x_val = tl.load(x_ptr + x_off)
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    y_off = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
    tl.store(y_ptr + y_off, y_val)


# Residual add elementwise
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, C, H, W,
               x_stride_n, x_stride_c, x_stride_h, x_stride_w,
               y_stride_n, y_stride_c, y_stride_h, y_stride_w,
               out_stride_n, out_stride_c, out_stride_h, out_stride_w):
    total = N * C * H * W
    pid = tl.program_id(axis=0)
    if pid >= total:
        return
    n = pid // (C * H * W)
    tmp = pid % (C * H * W)
    c = tmp // (H * W)
    hw = tmp % (H * W)
    h = hw // W
    w = hw % W
    x_off = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
    y_off = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
    out_off = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w
    a = tl.load(x_ptr + x_off)
    b = tl.load(y_ptr + y_off)
    tl.store(out_ptr + out_off, a + b)


def _launch_conv_tile(x, weight, y, N, C, H, W, C_in, C_out, BLOCK_OC=8):
    # x: (N, C, H, W), weight: (C_out, C_in, 3, 3), y: (N, C_out, H, W)
    x_f32 = x.contiguous().float()
    w_f32 = weight.contiguous().float()
    y_f32 = y
    grid = (N * H * W, (C_out + BLOCK_OC - 1) // BLOCK_OC)
    conv3x3_stride1_pad1_tile_kernel[grid](
        x_f32, w_f32, y_f32,
        N, C_out, H, W,
        C_in,
        BLOCK_OC,
        x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
        w_f32.stride(0), w_f32.stride(1), w_f32.stride(2), w_f32.stride(3),
        y_f32.stride(0), y_f32.stride(1), y_f32.stride(2), y_f32.stride(3),
        num_warps=4, num_stages=2,
    )


def _launch_group_norm(x, y, gamma, beta, N, C, H, W, G, eps):
    # x,y: (N, C, H, W); gamma,beta: (C,)
    x_f32 = x.contiguous().float()
    y_f32 = y
    gamma_f32 = gamma.contiguous().float()
    beta_f32 = beta.contiguous().float()
    grid = (N, G)
    # First pass: compute nothing here; we recompute mean/var in second pass due to simplicity.
    group_norm_second_pass[grid](
        x_f32, y_f32, gamma_f32, beta_f32,
        N, C, H, W, G, eps,
        x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
        y_f32.stride(0), y_f32.stride(1), y_f32.stride(2), y_f32.stride(3),
        MAX_C=256,
        num_warps=4, num_stages=2,
    )


def _launch_silu(x, y, N, C, H, W):
    x_f32 = x.contiguous().float()
    y_f32 = y
    total = N * C * H * W
    grid = (total,)
    silu_kernel[grid](
        x_f32, y_f32, N, C, H, W,
        x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
        y_f32.stride(0), y_f32.stride(1), y_f32.stride(2), y_f32.stride(3),
        num_warps=4, num_stages=2,
    )


def _launch_add(x, y, out, N, C, H, W):
    x_f32 = x.contiguous().float()
    y_f32 = y.contiguous().float()
    out_f32 = out
    total = N * C * H * W
    grid = (total,)
    add_kernel[grid](
        x_f32, y_f32, out_f32, N, C, H, W,
        x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
        y_f32.stride(0), y_f32.stride(1), y_f32.stride(2), y_f32.stride(3),
        out_f32.stride(0), out_f32.stride(1), out_f32.stride(2), out_f32.stride(3),
        num_warps=4, num_stages=2,
    )


class ModelNew(torch.nn.Module):
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        # x: (N, C, H, W), weights on CUDA
        N, C, H, W = x.shape
        assert C % 32 == 0, "C must be divisible by num_groups=32 for GroupNorm"
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels"

        # 1) conv1
        out1 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        _launch_conv_tile(x, conv1_weight, out1, N, C, H, W, C, C, BLOCK_OC=8)

        # 2) GroupNorm1 (num_groups=32)
        out1_gn = torch.empty_like(out1)
        _launch_group_norm(out1, out1_gn, norm1_weight, norm1_bias, N, C, H, W, G=32, eps=eps)

        # 3) SiLU1


def run(*args):
    return ModelNew()(*args)
