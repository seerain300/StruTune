import torch
import triton
import triton.language as tl


# Triton kernel: Conv2d 3x3 stride=1, padding=1, bias=None
# Computes one output element y[n, c_out, oh, ow] per program
@triton.jit
def conv3x3_stride1_pad1_single_kernel(
    x_ptr,            # *float32
    w_ptr,            # *float32
    y_ptr,            # *float32
    N, C, H, W,       # int32
    C_IN, C_OUT,      # int32 (not used in this kernel since weight is per c_out)
    H_OUT, W_OUT,     # int32
    # strides
    x_sN, x_sC, x_sH, x_sW,
    w_sCin, w_sCout, w_sKH, w_sKW,
    y_sN, y_sC, y_sH, y_sW,
):
    pid = tl.program_id(axis=0)
    total = N * C_OUT * H_OUT * W_OUT
    # guard: Triton grid is set to total, so pid < total always
    # compute indices
    n = pid // (C_OUT * H_OUT * W_OUT)
    rem = pid % (C_OUT * H_OUT * W_OUT)
    c_out = rem // (H_OUT * W_OUT)
    rem2 = rem % (H_OUT * W_OUT)
    oh = rem2 // W_OUT
    ow = rem2 % W_OUT

    acc = 0.0
    # iterate over 3x3 taps
    for kh in range(3):
        for kw in range(3):
            ih = oh + kh - 1  # padding=1
            iw = ow + kw - 1  # padding=1
            # mask for valid input coords
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            # flatten patch index (Cin, kh, kw) => pos in [0, Cin*9)
            # For each input channel, weight per c_out
            for c_in in range(0, C_IN):  # loop over input channels
                x_offset = n * x_sN + c_in * x_sC + ih * x_sH + iw * x_sW
                w_offset = c_in * w_sCin + c_out * w_sCout + kh * w_sKH + kw * w_sKW
                # load with mask; if out-of-bounds, value 0
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    y_offset = n * y_sN + c_out * y_sC + oh * y_sH + ow * y_sW
    tl.store(y_ptr + y_offset, acc)


# Triton kernel: GroupNorm first pass (compute sum and sumsq per (n, group))
# Assumes C divisible by num_groups (32 here). Group size = C // 32.
@triton.jit
def group_norm_first_pass(
    x_ptr,            # *float32
    sum_ptr,          # *float32, shape (N * G)
    sumsq_ptr,        # *float32, shape (N * G)
    N, C, H, W,       # int32
    G,                # int32 (num_groups)
    # strides
    x_sN, x_sC, x_sH, x_sW,
):
    pid = tl.program_id(axis=0)
    # one program per (n, group)
    num_groups_per_n = G
    n = pid // G
    group = pid % G
    group_size = C // G
    start_c = group * group_size

    s = 0.0
    ss = 0.0
    # loop over channels in group and all spatial positions
    for c in range(0, 256):  # MAX_C loop; C should be <= 256 for this task
        if c >= start_c and c < start_c + group_size:
            for h in range(0, H):
                for w in range(0, W):
                    x_offset = n * x_sN + c * x_sC + h * x_sH + w * x_sW
                    x_val = tl.load(x_ptr + x_offset)
                    s += x_val
                    ss += x_val * x_val
    # store per (n, group)
    out_index = n * G + group
    tl.store(sum_ptr + out_index, s)
    tl.store(sumsq_ptr + out_index, ss)


# Triton kernel: GroupNorm second pass (normalize and apply affine per channel)
@triton.jit
def group_norm_second_pass(
    x_ptr,            # *float32
    y_ptr,            # *float32
    gamma_ptr,        # *float32 (per-channel scale)
    beta_ptr,         # *float32 (per-channel bias)
    N, C, H, W,       # int32
    G,                # int32 (num_groups)
    eps,              # float32
    # strides
    x_sN, x_sC, x_sH, x_sW,
    y_sN, y_sC, y_sH, y_sW,
):
    pid = tl.program_id(axis=0)
    # one program per (n, group)
    n = pid // G
    group = pid % G
    group_size = C // G
    start_c = group * group_size

    # mean and var for this group: read from sum and sumsq arrays (precomputed)
    # We need to pass sum and sumsq somehow. Triton kernels don't have global read of sum_ptr,
    # so we compute them in first pass and pass to second pass via params? Instead, compute here by re-reading x.
    # To keep it simple and correct, recompute per program (less efficient but acceptable for correctness).
    s = 0.0
    ss = 0.0
    for c in range(0, 256):
        if c >= start_c and c < start_c + group_size:
            for h in range(0, H):
                for w in range(0, W):
                    x_offset = n * x_sN + c * x_sC + h * x_sH + w * x_sW
                    x_val = tl.load(x_ptr + x_offset)
                    s += x_val
                    ss += x_val * x_val
    num = group_size * H * W
    mean = s / num
    var = ss / num - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # normalize and affine, write to y
    for c in range(0, 256):
        if c >= start_c and c < start_c + group_size:
            gamma = tl.load(gamma_ptr + c)
            beta = tl.load(beta_ptr + c)
            for h in range(0, H):
                for w in range(0, W):
                    x_offset = n * x_sN + c * x_sC + h * x_sH + w * x_sW
                    y_offset = n * y_sN + c * y_sC + h * y_sH + w * y_sW
                    x_val = tl.load(x_ptr + x_offset)
                    norm = (x_val - mean) * inv_std
                    y_val = norm * gamma + beta
                    tl.store(y_ptr + y_offset, y_val)


# Triton kernel: SiLU elementwise
@triton.jit
def silu_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    x_sN, x_sC, x_sH, x_sW,
    y_sN, y_sC, y_sH, y_sW,
):
    total = N * C * H * W
    pid = tl.program_id(axis=0)
    # compute indices
    n = pid // (C * H * W)
    rem = pid % (C * H * W)
    c = rem // (H * W)
    rem2 = rem % (H * W)
    h = rem2 // W
    w = rem2 % W

    x_offset = n * x_sN + c * x_sC + h * x_sH + w * x_sW
    y_offset = n * y_sN + c * y_sC + h * y_sH + w * y_sW

    x_val = tl.load(x_ptr + x_offset)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    tl.store(y_ptr + y_offset, y_val)


# Triton kernel: elementwise add
@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    N, C, H, W,
    x_sN, x_sC, x_sH, x_sW,
    y_sN, y_sC, y_sH, y_sW,
    out_sN, out_sC, out_sH, out_sW,
):
    total = N * C * H * W
    pid = tl.program_id(axis=0)
    n = pid // (C * H * W)
    rem = pid % (C * H * W)
    c = rem // (H * W)
    rem2 = rem % (H * W)
    h = rem2 // W
    w = rem2 % W

    x_offset = n * x_sN + c * x_sC + h * x_sH + w * x_sW
    y_offset = n * y_sN + c * y_sC + h * y_sH + w * y_sW
    out_offset = n * out_sN + c * out_sC + h * out_sH + w * out_sW

    x_val = tl.load(x_ptr + x_offset)
    y_val = tl.load(y_ptr + y_offset)
    out_val = x_val + y_val
    tl.store(out_ptr + out_offset, out_val)


def _launch_conv(x, weight, out, N, C, H, W, C_IN, C_OUT):
    # x, weight must be on CUDA, float32
    x_f32 = x.contiguous().float()
    w_f32 = weight.contiguous().float()
    y_f32 = out  # float32 tensor
    total = N * C_OUT * H * W
    grid = (total,)
    conv3x3_stride1_pad1_single_kernel[grid](
        x_f32, w_f32, y_f32,
        N, C, H, W,
        C_IN, C_OUT, H, W,
        x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
        w_f32.stride(0), w_f32.stride(1), w_f32.stride(2), w_f32.stride(3),
        y_f32.stride(0), y_f32.stride(1), y_f32.stride(2), y_f32.stride(3),
        num_warps=4, num_stages=2,
    )


def _launch_group_norm(x, y, gamma, beta, N, C, H, W, G=32, eps=1e-5):
    # y must be float32, same shape as x
    x_f32 = x.contiguous().float()
    y_f32 = y
    # sum and sumsq buffers
    sum_buf = torch.empty(N * G, device=x.device, dtype=torch.float32)
    sumsq_buf = torch.empty(N * G, device=x.device, dtype=torch.float32)

    # first pass: compute sum and sumsq per (n, group)
    grid = (N * G,)
    group_norm_first_pass[grid](
        x_f32, sum_buf, sumsq_buf,
        N, C, H, W, G,
        x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
        num_warps=4, num_stages=2,
    )

    # second pass: normalize and apply affine
    grid2 = (N * G,)
    group_norm_second_pass[grid2](
        x_f32, y_f32, gamma.contiguous().float(), beta.contiguous().float(),
        N, C, H, W, G, eps,
        x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
        y_f32.stride(0), y_f32.stride(1), y_f32.stride(2), y_f32.stride(3),
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


def _launch_add(x, y, out, N, C, H, W):
    x_f32 = x.contiguous().float()
    y_f32 = y.contiguous().float()
    out_f32 = out
    total = N * C * H * W
    grid = (total,)
    add_kernel[grid](
        x_f32, y_f32, out_f32,
        N, C, H, W,
        x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
        y_f32.stride(0), y_f32.stride(1), y_f32.stride(2), y_f32.stride(3),
        out_f32.stride(0), out_f32.stride(1), out_f32.stride(2), out_f32.stride(3),
        num_warps=4, num_stages=2,
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        assert x.is_cuda, "Input must be on CUDA for Triton kernels"
        assert C % 32 == 0, "C must be divisible by num_groups=32 for GroupNorm"
        # Ensure weights are on the same device and dtype float32
        conv1_weight = conv1_weight.contiguous().to(torch.float32)
        conv2_weight = conv2_weight.contiguous().to(torch.float32)
        norm1_weight = norm1_weight.contiguous().to(torch.float32)
        norm1_bias = norm1_bias.contiguous().to(torch.float32)
        norm2_weight = norm2_weight.contiguous().to(torch.float32)
        norm2_bias = norm2_bias.contiguous().to(torch.float32)

        # 1) conv1
        out1 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        _launch_conv(x, conv1_weight, out1, N, C, H, W, C, C)

        # 2) GroupNorm1 (num_groups=32)
        out1_gn = torch.empty_like(out1)
        _launch_group_norm(out1, out1_gn, norm1_weight, norm1_bias, N, C, H, W, G=32, eps=eps)

        # 3) SiLU1
        out1_silu = torch.empty_like(out1_gn)
        _launch_silu(out1_gn, out1_silu, N, C, H, W)

        # 4) conv2
        out2 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        _launch_conv(out1_silu, conv2_weight, out2, N, C, H, W, C, C)

        # 5) GroupNorm2
        out2_gn = torch.empty_like(out2)
        _launch_group_norm(out2, out2_gn, norm2_weight, norm2_bias, N, C, H, W, G=32, eps=eps)

        # 6) SiLU2
        out2_silu = torch.empty_like(out2_gn)
        _launch_silu(out2_gn, out2_silu, N, C, H, W)

        # 7) Add residual x
        out = torch.empty_like(out2_silu)
        _launch_add(out2_silu, x, out, N, C, H, W)

        return out


def run(*args):
    return ModelNew()(*args)
