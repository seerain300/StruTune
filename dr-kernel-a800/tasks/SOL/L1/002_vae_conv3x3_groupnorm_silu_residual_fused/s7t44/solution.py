import torch
import triton
import triton.language as tl


# Triton kernel: Conv2d 3x3 stride=1, padding=1, bias=None
# Computes one output element y[n, c_out, oh, ow] per program
@triton.jit
def conv3x3_stride1_pad1_single_kernel(
    x_ptr,              # *float32, shape (N, C_in, H, W)
    w_ptr,              # *float32, shape (C_in, C_out, 3, 3)
    y_ptr,              # *float32, shape (N, C_out, H, W)
    N, C_in, C_out, H, W,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
):
    # Grid: (N * C_out * H * W,)
    idx = tl.program_id(0)
    HW = H * W
    n = idx // (C_out * HW)
    rem = idx % (C_out * HW)
    c_out = rem // HW
    rem2 = rem % HW
    oh = rem2 // W
    ow = rem2 % W

    # Accumulator as scalar float32
    acc = 0.0

    # Flattened patch length and weight length both are Cin*9
    Cin = C_in
    patch_len = Cin * 9
    w_vec = tl.zeros((patch_len,), dtype=tl.float32)

    # Build flattened input patch (Cin*9) with masks for padding
    # pos = cin * 9 + kh*3 + kw
    # Input coordinates: ih = oh + kh - 1, iw = ow + kw - 1 (padding=1)
    for cin in range(Cin):
        for kh in range(3):
            for kw in range(3):
                ih = oh + kh - 1
                iw = ow + kw - 1
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x_offset = n * x_stride_n + cin * x_stride_c + ih * x_stride_h + iw * x_stride_w
                # Load with mask; if invalid, load 0.0
                x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)
                w_vec[cin * 9 + kh * 3 + kw] = x_val

    # Load corresponding flattened weight vector (Cin*9) for c_out
    # Weight tensor shape: (C_in, C_out, 3, 3)
    w_base = w_ptr + c_out * w_stride_cout
    for cin in range(Cin):
        for kh in range(3):
            for kw in range(3):
                w_off = cin * w_stride_cin + c_out * w_stride_cout + kh * w_stride_kh + kw * w_stride_kw
                w_val = tl.load(w_base + w_off)
                w_vec[cin * 9 + kh * 3 + kw] = w_val

    # Accumulate dot product
    # For each cin, kh, kw we have a scalar. We can sum directly.
    # Note: The patch and weight vectors are constructed similarly.
    # Dot product of two vectors of length Cin*9
    # Build another vector b from the weight values to perform dot
    # Here we directly accumulate since patch_vec is constructed from x and w separately.
    # But to compute dot, we need to extract scalar contributions. Given our construction,
    # acc = sum over cin,kh,kw of (patch scalar) * (weight scalar). Our w_vec already holds the weights.
    # However, we mistakenly set w_vec from weights above. For conv, the w_vec should be all zeros except computed
    # from weights. Let's correct by computing acc = sum over cin,kh,kw of x_vals * w_vals, where
    # x_vals correspond to the 3x3 patch and w_vals correspond to the same indices in the weight vector.
    # Since we have already built w_vec from weights, we can compute acc by:
    # acc = sum_i (w_vec[i] * w_vec[i]) is wrong. We must reconstruct the contribution.
    # Easier: Reconstruct directly during accumulation from weights loaded separately.
    # We'll recompute acc by iterating cin,kh,kw and multiplying x value by corresponding weight.
    acc = 0.0
    # Now recompute acc by iterating over cin,kh,kw and multiply x_vals (constructed above) by corresponding weight.
    # Note: Our previous w_vec set from weights is incorrect for contribution; instead, we directly load weights
    # in this loop and multiply with corresponding x_vals that we constructed by loading from x.
    # However, x_vals were masked; to make this simple and correct, we'll load x and weight per kh,kw and accumulate.
    # So we do this properly:
    # Reinitialize w_vec from weights and x_vec from x. Better: directly accumulate in loop below.
    # To simplify, we'll reconstruct contribution by loading x and weight per kh,kw.
    # But since we already have w_vec filled from weights, and x values stored in those positions, we can compute
    # acc = sum_i w_vec[i] * w_vec[i] is wrong. Instead, we must multiply each x value (valid or 0) by its weight.
    # Since w_vec has weights and x_patch values, we cannot distinguish which were x. Hence we recompute contribution
    # by loading x and weight per kh,kw. Since the kernel loads x with mask and sets others to 0, w_vec has weights
    # at those indices, multiplying w_vec by itself would not be correct. Therefore, we will not rely on w_vec here.
    # Instead, we will recompute contribution in the loop by loading weight for each kh,kw, and the corresponding x value
    # by reloading from x_ptr (which is fine since loop is simple).

    # The above approach is not ideal. To ensure correctness, we revert to a simpler approach: compute contribution
    # by reloading the weights and x values in the final accumulation loop. For clarity and correctness, we'll
    # compute acc = sum over cin,kh,kw of x_val * w_val, by reloading w per kh,kw and x per kh,kw. This avoids
    # relying on a constructed w_vec.

    # Reinitialize acc and compute proper contribution
    acc = 0.0
    for cin in range(Cin):
        for kh in range(3):
            for kw in range(3):
                ih = oh + kh - 1
                iw = ow + kw - 1
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x_offset = n * x_stride_n + cin * x_stride_c + ih * x_stride_h + iw * x_stride_w
                x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)
                # Load weight for this (cin, c_out, kh, kw)
                w_off = cin * w_stride_cin + c_out * w_stride_cout + kh * w_stride_kh + kw * w_stride_kw
                w_val = tl.load(w_ptr + w_off)
                # Multiply and accumulate
                acc += x_val * w_val

    # Store result
    y_offset = n * y_stride_n + c_out * y_stride_c + oh * y_stride_h + ow * y_stride_w
    tl.store(y_ptr + y_offset, acc)


# Triton kernel: GroupNorm first pass (num_groups=32, affine)
# Computes sum and sumsq per (n, group)
@triton.jit
def group_norm_first_pass(
    x_ptr, y_ptr, gamma_ptr, beta_ptr,
    N, C, H, W, G, eps,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    MAX_C: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per (n, group)
    n = pid // G
    g = pid % G
    # channels in this group
    start_c = g * (C // G)
    end_c = start_c + (C // G)
    # Accumulate sum and sumsq across all channels and spatial positions
    s = 0.0
    ss = 0.0
    for c in range(start_c, end_c):
        for h in range(0, H):
            for w in range(0, W):
                x_off = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
                x_val = tl.load(x_ptr + x_off)
                s += x_val
                ss += x_val * x_val
    # Number of elements in group
    group_size = (C // G) * H * W
    mean = s / group_size
    var = ss / group_size - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    # store mean and inv_std to y_ptr for this (n,g)
    # We'll store scalars into y_ptr at index (n * G + g)
    y_index = n * G + g
    tl.store(y_ptr + y_index, mean)
    tl.store(y_ptr + y_index + 1, inv_std)


# Triton kernel: GroupNorm second pass (num_groups=32, affine)
# Normalizes and applies per-channel affine
@triton.jit
def group_norm_second_pass(
    x_ptr, y_ptr, gamma_ptr, beta_ptr,
    N, C, H, W, G, eps,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    MAX_C: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per (n, group)
    n = pid // G
    g = pid % G
    start_c = g * (C // G)
    end_c = start_c + (C // G)
    # Load mean and inv_std from y_ptr at index (n * G + g)
    y_index = n * G + g
    mean = tl.load(y_ptr + y_index)
    inv_std = tl.load(y_ptr + y_index + 1)
    for c in range(start_c, end_c):
        for h in range(0, H):
            for w in range(0, W):
                x_off = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
                x_val = tl.load(x_ptr + x_off)
                norm = (x_val - mean) * inv_std
                gamma = tl.load(gamma_ptr + c)
                beta = tl.load(beta_ptr + c)
                y_off = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
                tl.store(y_ptr + y_off, norm * gamma + beta)


# Triton kernel: SiLU elementwise y = x * sigmoid(x)
@triton.jit
def silu_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
):
    total = N * C * H * W
    pid = tl.program_id(0)  # one program per element
    # Compute indices
    n = pid // (C * H * W)
    rem = pid % (C * H * W)
    c = rem // (H * W)
    rem2 = rem % (H * W)
    h = rem2 // W
    w = rem2 % W
    x_off = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
    y_off = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
    x_val = tl.load(x_ptr + x_off)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    tl.store(y_ptr + y_off, y_val)


# Triton kernel: Add residual y = x1 + x2 (elementwise)
@triton.jit
def add_kernel(
    x1_ptr, x2_ptr, y_ptr,
    N, C, H, W,
    x1_stride_n, x1_stride_c, x1_stride_h, x1_stride_w,
    x2_stride_n, x2_stride_c, x2_stride_h, x2_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
):
    total = N * C * H * W
    pid = tl.program_id(0)  # one program per element
    n = pid // (C * H * W)
    rem = pid % (C * H * W)
    c = rem // (H * W)
    rem2 = rem % (H * W)
    h = rem2 // W
    w = rem2 % W
    off1 = n * x1_stride_n + c * x1_stride_c + h * x1_stride_h + w * x1_stride_w
    off2 = n * x2_stride_n + c * x2_stride_c + h * x2_stride_h + w * x2_stride_w
    offy = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
    v1 = tl.load(x1_ptr + off1)
    v2 = tl.load(x2_ptr + off2)
    tl.store(y_ptr + offy, v1 + v2)


def _launch_conv_single(x, weight, out, N, C_in, C_out, H, W):
    # x, weight, out are float32 and contiguous
    x_f32 = x.contiguous().float()
    w_f32 = weight.contiguous().float()
    y_f32 = out
    grid = (N * C_out * H * W,)
    conv3x3_stride1_pad1_single_kernel[grid](
        x_f32, w_f32, y_f32,
        N, C_in, C_out, H, W,
        x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
        w_f32.stride(0), w_f32.stride(1), w_f32.stride(2), w_f32.stride(3),
        y_f32.stride(0), y_f32.stride(1), y_f32.stride(2), y_f32.stride(3),
        num_warps=4, num_stages=2,
    )


def _launch_groupnorm_first_pass(x, y, gamma, beta, N, C, H, W, G, eps):
    assert C % G == 0, "Channels must be divisible by num_groups"
    x_f32 = x.contiguous().float()
    y_mean = torch.empty((N * G,), device=x.device, dtype=torch.float32)
    # y is output; we also store mean and inv_std into y_mean (n*G + g)
    grid = (N * G,)
    group_norm_first_pass[grid](
        x_f32, y_f32, gamma, beta,
        N, C, H, W, G, eps,
        x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
        y_f32.stride(0), y_f32.stride(1), y_f32.stride(2), y_f32.stride(3),
        MAX_C=256,  # loop upper bound; works for small C
        num_warps=4, num_stages=2,
    )


def _launch_groupnorm_second_pass(x, y, gamma, beta, N, C, H, W, G, eps):
    grid = (N * G,)
    group_norm_second_pass[grid](
        x, y, gamma, beta,
        N, C, H, W, G, eps,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        MAX_C=256,
        num_warps=4, num_stages=2,
    )


def _launch_silu(x, y, N, C, H, W):
    x_f32 = x.contiguous().float()
    y_f32 = y
    total = N * C * H * W
    grid = (total,)
    silu_kernel[grid](
        x_f32, y_f32,
        N, C, H, W,
        x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
        y_f32.stride(0), y_f32.stride(1), y_f32.stride(2), y_f32.stride(3),
        num_warps=4, num_stages=2,
    )


def _launch_add(x1, x2, out, N, C, H, W):
    x1_f32 = x1.contiguous().float()
    x2_f32 = x2.contiguous().float()
    out_f32 = out
    total = N * C * H * W
    grid = (total,)
    add_kernel[grid](
        x1_f32, x2_f32, out_f32,
        N, C, H, W,
        x1_f32.stride(0), x1_f32.stride(1), x1_f32.stride(2), x1_f32.stride(3),
        x2_f32.stride(0), x2_f32.stride(1), x2_f32.stride(2), x2_f32.stride(3),
        out_f32.stride(0), out_f32.stride(1), out_f32.stride(2), out_f32.stride(3),
        num_warps=4, num_stages=2,
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        # x: (N, C, H, W), conv weights: (C_in, C_out, 3, 3), affine params: (C,)
        assert x.is_cuda, "Input must be on CUDA for Triton kernels"
        N, C, H, W = x.shape
        assert C % 32 == 0, "Channels must be divisible by num_groups=32 for GroupNorm"
        # 1) conv1
        out1 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        _launch_conv_single(x, conv1_weight, out1, N, C, C, H, W)  # C_in=C, C_out=C

        # 2) GroupNorm1 (num_groups=32)
        out1_gn = torch.empty_like(out1)
        _launch_groupnorm_first_pass(out1, out1_gn, norm1_weight, norm1_bias, N, C, H, W, G=32, eps=eps)

        # 3) SiLU1
        out1_silu = torch.empty_like(out1_gn)
        _launch_silu(out1_gn, out1_silu, N, C, H, W)

        # 4) conv2
        out2 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        _launch_conv_single(out1_silu, conv2_weight, out2, N, C, C, H, W)  # C_in=C_out=C

        # 5) GroupNorm2
        out2_gn = torch.empty_like(out2)
        _launch_groupnorm_first_pass(out2, out2_gn, norm2_weight, norm2_bias, N, C, H, W, G=32, eps=eps)

        # 6) SiLU2
        out2_silu = torch.empty_like(out2_gn)
        _launch_silu(out2_gn, out2_silu, N, C, H, W)

        # 7) Add residual x
        out = torch.empty_like(out2_silu)
        _launch_add(out2_silu, x, out, N, C, H, W)

        return out


def run(*args):
    return ModelNew()(*args)
