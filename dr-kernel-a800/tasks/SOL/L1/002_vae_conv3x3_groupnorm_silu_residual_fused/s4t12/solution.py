import torch
import triton
import triton.language as tl

# Triton kernel: 3x3 Convolution (NCHW, stride=1, padding=1, no bias)
@triton.jit
def conv3x3_nchw_nobias(x_ptr, w_ptr, y_ptr,
                         B, C_in, C_out, H, W, H_out, W_out,
                         x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                         w_stride_co, w_stride_ci, w_stride_dh, w_stride_dw,
                         y_stride_n, y_stride_c, y_stride_h, y_stride_w,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    # program ids
    pid_n = tl.program_id(0)  # batch
    pid_co = tl.program_id(1) # output channel
    pid_h = tl.program_id(2)  # output height
    pid_w = tl.program_id(3)  # output width

    # accumulate in float32
    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for dh in range(0, 3):
            h_in = pid_h + dh
            for dw in range(0, 3):
                w_in = pid_w + dw
                # masked load for padding
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                x_offset = pid_n * x_stride_n + ci * x_stride_c + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
                # load weight scalar
                w_offset = pid_co * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    # store to output
    y_offset = pid_n * y_stride_n + pid_co * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w
    tl.store(y_ptr + y_offset, acc)

# Triton kernel: GroupNorm (num_groups=32, per-channel scale/bias, per-channel variance)
@triton.jit
def group_norm_triton_fixed(y_ptr, w_ptr, b_ptr, out_ptr,
                             B, C, H_out, W_out, groups,
                             y_stride_n, y_stride_c, y_stride_h, y_stride_w,
                             out_stride_n, out_stride_c, out_stride_h, out_stride_w,
                             num_warps: tl.constexpr, num_stages: tl.constexpr):
    # program ids: per batch and per group
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    # channels per group and starting channel index
    ch_per_group = C // groups
    ch_start = pid_g * ch_per_group

    # accumulate sum and sumsq per channel
    total_sum = tl.zeros((), dtype=tl.float32)
    total_sumsq = tl.zeros((), dtype=tl.float32)

    # loop channels in this group
    for ch in range(0, ch_per_group):
        c = ch_start + ch
        N_elems = H_out * W_out
        sum_c = tl.zeros((), dtype=tl.float32)
        sumsq_c = tl.zeros((), dtype=tl.float32)
        # loop over all elements in channel c
        for i in range(0, N_elems):
            h = i // W_out
            w = i % W_out
            y_offset = pid_n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
            y_val = tl.load(y_ptr + y_offset)
            sum_c += y_val
            sumsq_c += y_val * y_val
        total_sum += sum_c
        total_sumsq += sumsq_c

    # compute mean and variance (per-channel across the group's elements)
    mean = total_sum / (ch_per_group * N_elems)
    var = total_sumsq / (ch_per_group * N_elems) - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)  # eps

    # apply normalization and affine per channel
    for ch in range(0, ch_per_group):
        c = ch_start + ch
        gamma = tl.load(w_ptr + c)
        beta = tl.load(b_ptr + c)
        for i in range(0, N_elems):
            h = i // W_out
            w = i % W_out
            y_offset = pid_n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
            y_val = tl.load(y_ptr + y_offset)
            norm = (y_val - mean) * rstd
            out_val = norm * gamma + beta
            out_offset = pid_n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w
            tl.store(out_ptr + out_offset, out_val)

# Triton kernel: SiLU activation (elementwise: y = x * sigmoid(x))
@triton.jit
def silu_triton(x_ptr, y_ptr, N,
                num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + idx, y, mask=mask)

# Triton kernel: elementwise residual add y = x + y
@triton.jit
def add_residual_triton(x_ptr, y_ptr, N,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    y = tl.load(y_ptr + idx, mask=mask, other=0.0)
    y = y + x
    tl.store(y_ptr + idx, y, mask=mask)

# Triton kernel: upsample y2_silu (B, C, H-4, W-4) into y_out (B, C, H, W) by placing at (h+2, w+2)
@triton.jit
def upsample_shift_add(y2_ptr, x_ptr, y_out_ptr,
                        B, C, H_src, W_src, H_dst, W_dst,
                        y2_stride_n, y2_stride_c, y2_stride_h, y2_stride_w,
                        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                        y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
                        num_warps: tl.constexpr, num_stages: tl.constexpr):
    # This kernel assumes H_dst = H_src + 2 and W_dst = W_src + 2 by construction
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h2 = tl.program_id(2)  # source h index in [0..H_src-4]
    pid_w2 = tl.program_id(3)  # source w index in [0..W_src-4]
    # Guard: ensure pid_h2, pid_w2 are valid within y2 domain
    if (pid_h2 >= 0) & (pid_w2 >= 0) & (pid_h2 < H_src) & (pid_w2 < W_src):
        # read y2_silu at (pid_n, pid_c, pid_h2, pid_w2)
        y2_offset = pid_n * y2_stride_n + pid_c * y2_stride_c + pid_h2 * y2_stride_h + pid_w2 * y2_stride_w
        val = tl.load(y2_ptr + y2_offset)
        # write into y_out at (pid_n, pid_c, pid_h2+2, pid_w2+2)
        h_out = pid_h2 + 2
        w_out = pid_w2 + 2
        if (h_out >= 0) & (h_out < H_dst) & (w_out >= 0) & (w_out < W_dst):
            y_out_offset = pid_n * y_out_stride_n + pid_c * y_out_stride_c + h_out * y_out_stride_h + w_out * y_out_stride_w
            # add residual from x[n, c, h_out, w_out]
            x_offset = pid_n * x_stride_n + pid_c * x_stride_c + h_out * x_stride_h + w_out * x_stride_w
            x_val = tl.load(x_ptr + x_offset)
            tl.store(y_out_ptr + y_out_offset, val + x_val)

# ModelNew: Triton-only forward (no torch ops)
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
        # Ensure contiguity and dtype
        B, C, H, W = x.shape
        assert x.is_contiguous(), "Input must be contiguous NCHW"
        assert x.dtype == torch.float32, "Input must be float32"

        # First conv: output shape (B, C, H-2, W-2)
        H_out1 = H - 2
        W_out1 = W - 2
        y1 = torch.empty((B, C, H_out1, W_out1), dtype=torch.float32, device=x.device)

        grid_conv = (B, C, H_out1, W_out1)
        conv3x3_nchw_nobias[grid_conv](
            x, self.conv1_weight, y1,
            B, C, C, H, W, H_out1, W_out1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.conv1_weight.stride(0), self.conv1_weight.stride(1), self.conv1_weight.stride(2), self.conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm1: (B, C, H_out1, W_out1), num_groups=32
        y1_norm = torch.empty_like(y1)
        grid_gn1 = (B, 32)
        group_norm_triton_fixed[grid_gn1](
            y1, self.norm1_weight, self.norm1_bias, y1_norm,
            B, C, H_out1, W_out1, 32,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            num_warps=4, num_stages=2
        )

        # SiLU1
        y1_silu = torch.empty_like(y1_norm)
        N1 = B * C * H_out1 * W_out1
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_triton[grid_silu1](
            y1_norm, y1_silu, N1,
            num_warps=4, num_stages=2
        )

        # Second conv: output shape (B, C, H-4, W-4)
        H_out2 = H_out1 - 2  # = H - 4
        W_out2 = W_out1 - 2  # = W - 4
        y2 = torch.empty((B, C, H_out2, W_out2), dtype=torch.float32, device=x.device)

        grid_conv2 = (B, C, H_out2, W_out2)
        conv3x3_nchw_nobias[grid_conv2](
            y1_silu, self.conv2_weight, y2,
            B, C, C, H_out1, W_out1, H_out2, W_out2,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            self.conv2_weight.stride(0), self.conv2_weight.stride(1), self.conv2_weight.stride(2), self.conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm2
        y2_norm = torch.empty_like(y2)
        grid_gn2 = (B, 32)
        group_norm_triton_fixed[grid_gn2](
            y2, self.norm2_weight, self.norm2_bias, y2_norm,
            B, C, H_out2, W_out2, 32,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            num_warps=4, num_stages=2
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_norm)
        N2 = B * C * H_out2 * W_out2
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_triton[grid_silu2](
            y2_norm, y2_silu, N2,
            num_warps=4, num_stages=2
        )

        # Residual addition y = y2_silu + x (upsample x to y2_silu spatial shape by shifting: place at (h+2, w+2))
        # Note: y2_silu has shape (B, C, H-4, W-4). We need to add x shifted by +2 in height and width.
        # Construct output tensor y_out with same shape as y2_silu, then use upsample_shift_add to fill.
        y_out = torch.empty_like(y2_silu)

        # We need to launch upsample_shift_add which writes into y_out and adds x at corresponding locations.
        # y2_silu and x are expected to be contiguous NCHW. Compute strides:
        grid_up = (B, C, H_out2, W_out2)
        upsample_shift_add[grid_up](
            y2_silu, x, y_out,
            B, C, H_out2, W_out2, H, W,
            y2_silu.stride(0), y2_silu.stride(1), y2_silu.stride(2), y2_silu.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            y_out.stride(0), y_out.stride(1), y_out.stride(2), y_out.stride(3),
            num_warps=4, num_stages=2
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
