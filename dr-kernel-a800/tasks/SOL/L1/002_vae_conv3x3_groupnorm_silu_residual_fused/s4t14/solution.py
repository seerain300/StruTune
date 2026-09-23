import torch
import triton
import triton.language as tl

# Triton kernel: 3x3 Conv (NCHW, stride=1, padding=1, no bias)
@triton.jit
def conv3x3_nchw_nobias(x_ptr, w_ptr, y_ptr,
                         B, C_in, C_out, H, W, H_out, W_out,
                         x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                         w_stride_co, w_stride_ci, w_stride_dh, w_stride_dw,
                         y_stride_n, y_stride_c, y_stride_h, y_stride_w,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    # program ids
    pid_n = tl.program_id(0)  # batch
    pid_co = tl.program_id(1)  # output channel
    pid_h = tl.program_id(2)  # output height index
    pid_w = tl.program_id(3)  # output width index

    # compute output pointers
    y_offset = pid_n * y_stride_n + pid_co * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w

    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for dh in range(0, 3):
            ih = pid_h + dh - 1  # input h index with padding
            for dw in range(0, 3):
                iw = pid_w + dw - 1  # input w index with padding
                # bounds check for padding (1-padded)
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x_offset = pid_n * x_stride_n + ci * x_stride_c + ih * x_stride_h + iw * x_stride_w
                # masked load; out-of-bounds contribute 0
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
                w_offset = pid_co * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    # store result
    tl.store(y_ptr + y_offset, acc)


# Triton kernel: GroupNorm over 32 groups (num_groups=32), per-channel scale and bias
# Assumes input tensor y has shape (B, C, H_out, W_out) and is contiguous NCHW.
@triton.jit
def group_norm_triton_fixed(y_ptr, weight_ptr, bias_ptr, eps,
                             B, C, H_out, W_out,
                             y_stride_n, y_stride_c, y_stride_h, y_stride_w,
                             num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_b = tl.program_id(0)  # batch index
    pid_g = tl.program_id(1)  # group index in [0..31]
    channels_per_group = C // 32
    group_channels = pid_g * channels_per_group + tl.arange(0, channels_per_group)
    mask_ch = group_channels < C

    # Compute per-channel sum and sumsq across all spatial positions for the batch
    sum_c = tl.zeros([channels_per_group], dtype=tl.float32)
    sumsq_c = tl.zeros([channels_per_group], dtype=tl.float32)

    for ch in range(0, channels_per_group):
        c = group_channels[ch]
        if mask_ch[ch]:
            # iterate over spatial positions
            for h in range(0, H_out):
                for w in range(0, W_out):
                    y_offset = pid_b * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
                    val = tl.load(y_ptr + y_offset)
                    sum_c[ch] += val
                    sumsq_c[ch] += val * val

    # compute mean and rstd per channel
    M = H_out * W_out
    mean = sum_c / M
    var = sumsq_c / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # apply normalization, per-channel weight and bias
    for ch in range(0, channels_per_group):
        c = group_channels[ch]
        if mask_ch[ch]:
            for h in range(0, H_out):
                for w in range(0, W_out):
                    y_offset = pid_b * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
                    val = tl.load(y_ptr + y_offset)
                    normed = (val - mean[ch]) * rstd[ch]
                    w = tl.load(weight_ptr + c)
                    b = tl.load(bias_ptr + c)
                    val = normed * w + b
                    tl.store(y_ptr + y_offset, val)


# Triton kernel: elementwise SiLU activation
@triton.jit
def silu_triton(y_ptr, N, num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    x = tl.load(y_ptr + idx, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + idx, y, mask=mask)


# Triton kernel: elementwise residual add y = x + y
@triton.jit
def add_residual_triton(x_ptr, y_ptr, N, num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    a = tl.load(x_ptr + idx, mask=mask, other=0.0)
    b = tl.load(y_ptr + idx, mask=mask, other=0.0)
    y = b + a
    tl.store(y_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                 conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        super().__init__()
        self.conv1_weight = conv1_weight
        self.norm1_weight = norm1_weight
        self.norm1_bias = norm1_bias
        self.conv2_weight = conv2_weight
        self.norm2_weight = norm2_weight
        self.norm2_bias = norm2_bias
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous and float32
        x = x.contiguous().to(torch.float32)

        B, C, H, W = x.shape

        # Allocate output for conv1
        H1 = H - 2
        W1 = W - 2
        y1 = torch.empty((B, C, H1, W1), device=x.device, dtype=x.dtype)

        # Launch conv1 Triton kernel
        grid_conv1 = (B, C, H1, W1)
        conv3x3_nchw_nobias[grid_conv1](
            x, self.conv1_weight, y1,
            B, C, C, H, W, H1, W1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.conv1_weight.stride(0), self.conv1_weight.stride(1), self.conv1_weight.stride(2), self.conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm1 (num_groups=32)
        y1_contig = y1  # already contiguous
        grid_gn1 = (B, 32)
        group_norm_triton_fixed[grid_gn1](
            y1_contig, self.norm1_weight, self.norm1_bias, self.eps,
            B, C, H1, W1,
            y1_contig.stride(0), y1_contig.stride(1), y1_contig.stride(2), y1_contig.stride(3),
            num_warps=4, num_stages=2
        )

        # SiLU1
        N1 = y1_contig.numel()
        y1_silu = torch.empty_like(y1_contig)
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_triton[grid_silu1](y1_contig, N1, num_warps=4, num_stages=2)
        # Since silu_triton expects y_ptr to be the output, copy result back from tmp:
        # We can recompute or use the above, but to avoid confusion, we directly apply SiLU via Triton.
        # The above kernel writes back into y1_silu implicitly; we need to ensure y_ptr is y1_silu.
        # Therefore, launch the kernel again with y1_silu as y_ptr:
        # But Triton kernel only supports pointer; so we run:
        silu_triton[grid_silu1](y1_silu, N1, num_warps=4, num_stages=2)

        # conv2
        H2 = H1 - 2  # = H - 4
        W2 = W1 - 2  # = W - 4
        y2 = torch.empty((B, C, H2, W2), device=x.device, dtype=x.dtype)
        grid_conv2 = (B, C, H2, W2)
        conv3x3_nchw_nobias[grid_conv2](
            y1_silu, self.conv2_weight, y2,
            B, C, C, H1, W1, H2, W2,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            self.conv2_weight.stride(0), self.conv2_weight.stride(1), self.conv2_weight.stride(2), self.conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm2 (num_groups=32)
        y2_contig = y2
        grid_gn2 = (B, 32)
        group_norm_triton_fixed[grid_gn2](
            y2_contig, self.norm2_weight, self.norm2_bias, self.eps,
            B, C, H2, W2,
            y2_contig.stride(0), y2_contig.stride(1), y2_contig.stride(2), y2_contig.stride(3),
            num_warps=4, num_stages=2
        )

        # SiLU2
        N2 = y2_contig.numel()
        y2_silu = torch.empty_like(y2_contig)
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_triton[grid_silu2](y2_contig, N2, num_warps=4, num_stages=2)
        silu_triton[grid_silu2](y2_silu, N2, num_warps=4, num_stages=2)

        # Residual add y = y2_silu + x
        N = x.numel()
        y_out = torch.empty_like(x)
        grid_add = (triton.cdiv(N, 1024),)
        add_residual_triton[grid_add](x, y2_silu, N, num_warps=4, num_stages=2)

        return y_out


def run(*args):
    return ModelNew()(*args)
