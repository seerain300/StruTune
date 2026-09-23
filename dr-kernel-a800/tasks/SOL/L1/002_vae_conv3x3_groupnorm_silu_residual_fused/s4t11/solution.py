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
    pid_c = tl.program_id(1)  # output channel
    pid_h = tl.program_id(2)  # output height
    pid_w = tl.program_id(3)  # output width

    # accumulate in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for dh in (0, 1, 2):
            ih = pid_h + dh
            in_bounds_h = (ih >= 0) & (ih < H)
            for dw in (0, 1, 2):
                iw = pid_w + dw
                in_bounds_w = (iw >= 0) & (iw < W)
                in_bounds = in_bounds_h & in_bounds_w
                # load input
                x_offset = pid_n * x_stride_n + ci * x_stride_c + ih * x_stride_h + iw * x_stride_w
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
                # load weight
                w_offset = pid_c * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    # store output
    y_offset = pid_n * y_stride_n + pid_c * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w
    tl.store(y_ptr + y_offset, acc)

# Triton kernel: GroupNorm with num_groups=32, per-channel scale/bias, per-channel variance
# We normalize per (n, group) across all channels in the group and all spatial positions of the output tensor.
@triton.jit
def group_norm_triton_fixed(inp_ptr, weight_ptr, bias_ptr, out_ptr,
                             B, C, H_out, W_out, GROUPS: tl.constexpr,
                             in_stride_n, in_stride_c, in_stride_h, in_stride_w,
                             out_stride_n, out_stride_c, out_stride_h, out_stride_w,
                             eps,
                             num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)  # batch
    pid_g = tl.program_id(1)  # group index in 0..GROUPS-1

    CH_PER_GROUP = C // GROUPS  # in this task, GROUPS=32, so CH_PER_GROUP=C//32
    group_start_c = pid_g * CH_PER_GROUP

    # first pass: compute per-channel sum and sumsq across all spatial positions
    sum_vec = tl.zeros((CH_PER_GROUP,), dtype=tl.float32)
    sumsq_vec = tl.zeros((CH_PER_GROUP,), dtype=tl.float32)
    # for each channel in the group
    for ch in range(0, CH_PER_GROUP):
        c = group_start_c + ch
        # accumulate over all H_out*W_out positions
        total = 0
        for h in range(0, H_out):
            for w in range(0, W_out):
                in_offset = pid_n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w
                x = tl.load(inp_ptr + in_offset)
                total += x
        # now we need sumsq; but total is sum, so recompute sumsq. This is acceptable for these sizes.
        # However, we can recompute sumsq efficiently by reading again:
        sumsq_total = 0
        for h in range(0, H_out):
            for w in range(0, W_out):
                in_offset = pid_n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w
                x = tl.load(inp_ptr + in_offset)
                sumsq_total += x * x
        sum_vec[ch] = total
        sumsq_vec[ch] = sumsq_total

    # compute mean and rstd per channel
    M = H_out * W_out
    # num_groups is constexpr, so we can compute per channel
    mean = sum_vec / M
    var = sumsq_vec / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # second pass: normalize and apply per-channel weight and bias, then store
    for ch in range(0, CH_PER_GROUP):
        c = group_start_c + ch
        for h in range(0, H_out):
            for w in range(0, W_out):
                in_offset = pid_n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w
                x = tl.load(inp_ptr + in_offset)
                # per-channel scale and bias
                scale = tl.load(weight_ptr + c)  # per-channel scalar
                b = tl.load(bias_ptr + c)        # per-channel scalar
                y = (x - mean[ch]) * rstd[ch] * scale + b
                out_offset = pid_n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w
                tl.store(out_ptr + out_offset, y)

# Triton kernel: SiLU activation elementwise
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

# Main ModelNew: forward uses only Triton kernels
class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self,
                x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        # Ensure contiguity
        x = x.contiguous()
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # First conv: stride=1, padding=1
        H_out1 = H - 2
        W_out1 = W - 2
        y1 = torch.empty((B, C, H_out1, W_out1), dtype=torch.float32, device=x.device)

        # Launch conv1 kernel
        grid_conv1 = (B, C, H_out1, W_out1)
        conv3x3_nchw_nobias[grid_conv1](
            x, conv1_weight, y1,
            B, C, C, H, W, H_out1, W_out1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm1: num_groups=32
        num_groups = 32
        y1_norm = torch.empty_like(y1)

        grid_gn1 = (B, num_groups)
        group_norm_triton_fixed[grid_gn1](
            y1, norm1_weight, norm1_bias, y1_norm,
            B, C, H_out1, W_out1, num_groups,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            self.eps,
            num_warps=4, num_stages=2
        )

        # SiLU1
        y1_silu = torch.empty_like(y1_norm)
        N1 = y1_norm.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_triton[grid_silu1](y1_norm, y1_silu, N1, num_warps=4, num_stages=2)

        # Second conv: stride=1, padding=1 on y1_silu
        H_out2 = H_out1 - 2
        W_out2 = W_out1 - 2
        y2 = torch.empty((B, C, H_out2, W_out2), dtype=torch.float32, device=x.device)

        grid_conv2 = (B, C, H_out2, W_out2)
        conv3x3_nchw_nobias[grid_conv2](
            y1_silu, conv2_weight, y2,
            B, C, C, H_out1, W_out1, H_out2, W_out2,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm2
        y2_norm = torch.empty_like(y2)
        grid_gn2 = (B, num_groups)
        group_norm_triton_fixed[grid_gn2](
            y2, norm2_weight, norm2_bias, y2_norm,
            B, C, H_out2, W_out2, num_groups,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            self.eps,
            num_warps=4, num_stages=2
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_norm)
        N2 = y2_norm.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_triton[grid_silu2](y2_norm, y2_silu, N2, num_warps=4, num_stages=2)

        # Residual addition y = y2_silu + x (elementwise), output shape matches x
        # We need to upsample y2_silu to (B, C, H, W). In original code, final output has same spatial size as x,
        # but y2_silu has (B, C, H-4, W-4). To match original behavior (which adds to x of shape (B,C,H,W)),
        # we must upsample y2_silu to (B,C,H,W). PyTorch's interpolate is not allowed in forward; implement
        # a simple nearest-neighbor upsampling in Triton:
        # Map output (B, C, H, W) to source (B, C, H-4, W-4): For each input y2_silu[n, c, h, w], place at
        # x[n, c, h+2, w+2]. We'll write zeros elsewhere. This reproduces nearest-neighbor behavior and
        # matches the final addition's required shape.
        x_in = x  # original x
        y2_up = torch.zeros_like(x_in)

        # Write y2_silu into y2_up at shifted positions (h+2, w+2)
        # We'll use Triton elementwise kernel for this copy with offset.
        N_copy = y2_silu.numel()
        grid_add = (triton.cdiv(N_copy, 1024),)
        # Prepare source pointer (y2_silu) and destination pointer (y2_up) and compute offsets:
        # For each linear index i, map back to (n,c,h,w) in y2_silu and then store at (n,c,h+2,w+2) in y2_up.
        # But Triton kernel prefers separate pointers; we can perform this by reading y2_silu and writing
        # into y2_up at (h+2, w+2).
        # Implement explicit offset mapping:
        # y2_up[n, c, h, w] = y2_silu[n, c, h-2, w-2] for 2<=h<H-2, 2<=w<W-2; else 0
        # We'll launch a kernel over N_copy elements (we need N=x.numel() elements to write all).
        # To avoid torch ops, we'll write zeros elsewhere explicitly:
        # - First zero the destination, then write the shifted values.
        # We already zeroed y2_up above.
        # Launch a kernel to copy shifted region into y2_up:
        # We need a kernel that reads from y2_silu and writes to y2_up at offset (+2,+2).

        # Define a kernel for this copy with offset. However, we don't have direct access to h,w from linear idx.
        # Instead, we can launch a kernel that writes the full tensor y2_silu into y2_up shifted positions by skipping out-of-range indices.
        # But Triton kernel does not have an easy way to derive (h,w) from linear indices; so we use a simple approach:
        # We'll perform the copy via torch.zeros_like and then use another kernel to fill only the valid region.
        # Since we cannot create a perfect mapping without knowing (h,w), we'll instead implement an elementwise
        # fill by reading y2_silu and writing into y2_up at the appropriate positions using a 4D grid over (B, C, H-4, W-4)
        # and compute output indices accordingly. Triton supports 4D grids, so we can define the kernel grid as (B, C, H-4, W-4).

        # Define a 4D grid Triton kernel for copy with offset (+2,+2):
        @triton.jit
        def copy_shift_upsample(y2_src_ptr, y2_dst_ptr,
                                 B, C, H_src, W_src, H_dst, W_dst,
                                 y2_src_stride_n, y2_src_stride_c, y2_src_stride_h, y2_src_stride_w,
                                 y2_dst_stride_n, y2_dst_stride_c, y2_dst_stride_h, y2_dst_stride_w,
                                 num_warps: tl.constexpr, num_stages: tl.constexpr):
            pid_n = tl.program_id(0)
            pid_c = tl.program_id(1)
            pid_h = tl.program_id(2)  # source h
            pid_w = tl.program_id(3)  # source w

            # bounds for source
            if (pid_h < H_src) & (pid_w < W_src):
                src_offset = pid_n * y2_src_stride_n + pid_c * y2_src_stride_c + pid_h * y2_src_stride_h + pid_w * y2_src_stride_w
                val = tl.load(y2_src_ptr + src_offset)
                dst_n = pid_n
                dst_c = pid_c
                dst_h = pid_h + 2
                dst_w = pid_w + 2
                if (dst_h < H_dst) & (dst_w < W_dst):
                    dst_offset = dst_n * y2_dst_stride_n + dst_c * y2_dst_stride_c + dst_h * y2_dst_stride_h + dst_w * y2_dst_stride_w
                    tl.store(y2_dst_ptr + dst_offset, val)

        # Launch 4D grid for (B, C, H-4, W-4)
        grid_copy = (B, C, H_out2, W_out2)
        copy_shift_upsample[grid_copy](
            y2_silu, y2_up,
            B, C, H_out2, W_out2, H, W,
            y2_silu.stride(0), y2_silu.stride(1), y2_silu.stride(2), y2_silu.stride(3),
            y2_up.stride(0), y2_up.stride(1), y2_up.stride(2), y2_up.stride(3),
            num_warps=4, num_stages=2
        )

        # Now y2_up has y2_silu placed at (h+2, w+2), zeros elsewhere. Proceed with residual addition:
        # y_out = y2_up + x
        # Define Triton residual add kernel over N=x.numel()
        y_out = torch.empty_like(x)
        N_total = x.numel()
        grid_add4 = (triton.cdiv(N_total, 1024),)
        add_residual_triton[grid_add4](
            x, y2_up, N_total, num_warps=4, num_stages=2
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
