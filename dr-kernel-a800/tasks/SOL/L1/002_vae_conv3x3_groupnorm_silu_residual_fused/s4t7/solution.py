import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_nobias(x_ptr, w_ptr, y_ptr,
                         B, C_in, C_out, H, W, H_out, W_out,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    # Grid: (B, C_out, H_out, W_out)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    for ci in range(0, C_in):
        for dh in range(0, 3):
            for dw in range(0, 3):
                ih = pid_h + dh - 1
                iw = pid_w + dw - 1
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x_offset = ((pid_n * C_in + ci) * H + ih) * W + iw
                w_offset = ((pid_co * C_in + ci) * 9) + (dh * 3 + dw)
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    out_offset = ((pid_n * C_out + pid_co) * H_out + pid_h) * W_out + pid_w
    tl.store(y_ptr + out_offset, acc)


@triton.jit
def group_norm_triton(y_ptr, y_norm_ptr, weight_ptr, bias_ptr,
                      B, C, H, W, num_groups, eps,
                      num_warps: tl.constexpr, num_stages: tl.constexpr):
    # Grid: (B, num_groups)
    n = tl.program_id(0)
    group = tl.program_id(1)

    channels_per_group = C // num_groups
    group_start_c = group * channels_per_group

    # First pass: per-channel sum and sum of squares across group's elements
    sum_c = tl.zeros((channels_per_group,), dtype=tl.float32)
    sumsq_c = tl.zeros((channels_per_group,), dtype=tl.float32)

    for ch in range(channels_per_group):
        c = group_start_c + ch
        M = H * W
        base = (n * C + c) * M
        for h in range(H):
            for w in range(W):
                idx = base + h * W + w
                x = tl.load(y_ptr + idx)
                sum_c[ch] += x
                sumsq_c[ch] += x * x

    M_total = channels_per_group * M
    mean = sum_c / float(M)  # per-channel mean
    var = sumsq_c / float(M) - mean * mean  # per-channel variance
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for ch in range(channels_per_group):
        c = group_start_c + ch
        scale = tl.load(weight_ptr + c)
        bias = tl.load(bias_ptr + c)
        base = (n * C + c) * M
        for h in range(H):
            for w in range(W):
                idx = base + h * W + w
                x = tl.load(y_ptr + idx)
                y = (x - mean[ch]) * rstd[ch]
                y = y * scale + bias
                tl.store(y_norm_ptr + idx, y)


@triton.jit
def silu_triton(x_ptr, y_ptr, N, num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.load(x_ptr + pid)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + pid, y)


@triton.jit
def add_residual_triton(x_ptr, out_ptr, N, num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    a = tl.load(x_ptr + pid)
    b = tl.load(out_ptr + pid)
    tl.store(out_ptr + pid, b + a)


@triton.jit
def upsample_nearest_2d_nchw(x_ptr, y_ptr,
                              B, C, H, W,
                              H_out, W_out,
                              num_warps: tl.constexpr, num_stages: tl.constexpr):
    # Grid: (B*C*H_out*W_out,)
    total = B * C * H_out * W_out
    pid = tl.program_id(0)
    # Map linear index to (n, c, h_out, w_out)
    nchout = B * C * H_out
    chout = C * H_out
    n = pid // chout
    rem1 = pid % chout
    c = rem1 // (H_out * W_out)
    rem2 = rem1 % (H_out * W_out)
    h_out = rem2 // W_out
    w_out = rem2 % W_out

    # Nearest mapping: h_in = h_out * 2, w_in = w_out * 2 (assume 2x upsample)
    h_in = h_out * 2
    w_in = w_out * 2

    # Bounds check for original H/W
    if (h_in >= H) or (w_in >= W):
        # If out-of-bounds, set to 0 (for edges when H/W not multiples of 2)
        val = 0.0
    else:
        base = (n * C + c) * H * W
        idx = base + h_in * W + w_in
        val = tl.load(x_ptr + idx)

    # Compute output index
    out_base = (n * C + c) * H_out * W_out
    out_idx = out_base + h_out * W_out + w_out
    tl.store(y_ptr + out_idx, val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
        eps: float,
    ):
        """
        Triton-only fused residual block:
        conv3x3 -> GroupNorm(num_groups=32) -> SiLU -> conv3x3 -> GroupNorm(num_groups=32) -> SiLU -> Add(x upscaled)
        All computations are done in Triton kernels. No torch operations in forward.
        """
        # Ensure contiguous NCHW
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        B, C, H, W = x.shape

        # First conv: (C_in=C, C_out=C), stride=1, padding=1
        C_out1 = C
        H_out1 = H - 2
        W_out1 = W - 2
        y1 = torch.empty((B, C_out1, H_out1, W_out1), device=x.device, dtype=x.dtype)

        grid1 = (B, C_out1, H_out1, W_out1)
        conv3x3_nchw_nobias[grid1](
            x, conv1_weight, y1,
            B, C, C_out1, H, W, H_out1, W_out1,
            num_warps=4, num_stages=2
        )

        # GroupNorm 1
        y1_norm = torch.empty_like(y1)
        grid_gn1 = (B, 32)
        group_norm_triton[grid_gn1](
            y1, y1_norm, norm1_weight, norm1_bias,
            B, C, H_out1, W_out1, 32, eps,
            num_warps=4, num_stages=2
        )

        # SiLU 1
        y1_silu = torch.empty_like(y1_norm)
        N1 = y1_norm.numel()
        grid_silu1 = (N1,)
        silu_triton[grid_silu1](
            y1_norm, y1_silu, N1,
            num_warps=4, num_stages=2
        )

        # Second conv: (C_in=C_out1=C, C_out=C), stride=1, padding=1
        C_out2 = C
        H_out2 = H_out1 - 2  # since conv2 also 3x3 padding=1
        W_out2 = W_out1 - 2
        y2 = torch.empty((B, C_out2, H_out2, W_out2), device=x.device, dtype=x.dtype)

        grid2 = (B, C_out2, H_out2, W_out2)
        conv3x3_nchw_nobias[grid2](
            y1_silu, conv2_weight, y2,
            B, C, C_out2, H_out1, W_out1, H_out2, W_out2,
            num_warps=4, num_stages=2
        )

        # GroupNorm 2
        y2_norm = torch.empty_like(y2)
        grid_gn2 = (B, 32)
        group_norm_triton[grid_gn2](
            y2, y2_norm, norm2_weight, norm2_bias,
            B, C, H_out2, W_out2, 32, eps,
            num_warps=4, num_stages=2
        )

        # SiLU 2
        y2_silu = torch.empty_like(y2_norm)
        N2 = y2_norm.numel()
        grid_silu2 = (N2,)
        silu_triton[grid_silu2](
            y2_norm, y2_silu, N2,
            num_warps=4, num_stages=2
        )

        # Upsample x to match y2_silu spatial size: nearest neighbor by 2x (assumes even H/W divisible by 2).
        # If H or W is odd, some edges may be discarded (set to 0). For typical workloads like 128, this is fine.
        H_up = (H + 1) // 2  # floor division yielding nearest 2x target
        W_up = (W + 1) // 2
        # Ensure H_up, W_up >= 1
        H_up = max(H_up, 1)
        W_up = max(W_up, 1)

        x_up = torch.empty((B, C, H_up, W_up), device=x.device, dtype=x.dtype)
        grid_up = (B * C * H_up * W_up,)
        upsample_nearest_2d_nchw[grid_up](
            x, x_up,
            B, C, H, W,
            H_up, W_up,
            num_warps=4, num_stages=2
        )

        # Add residual x_up to y2_silu (elementwise)
        # Flatten for elementwise Triton kernel
        y2_silu_flat = y2_silu.view(-1)
        x_up_flat = x_up.view(-1)
        N_add = y2_silu_flat.numel()
        add_residual_triton[(N_add,)](
            x_up_flat, y2_silu_flat, N_add,
            num_warps=4, num_stages=2
        )

        return y2_silu


def run(*args):
    return ModelNew()(*args)
