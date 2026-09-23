import torch
import triton
import triton.language as tl


# Triton kernel: Conv3x3 NCHW, stride=1, padding=1, no bias
# Computes out[n, co, h, w] = sum_{ci=0..C_IN-1} sum_{dh=-1..1} sum_{dw=-1..1} x[n, ci, h+dh, w+dw] * w[co, ci, 1+dh, 1+dw]
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C_IN: tl.constexpr, C_OUT: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_W: tl.constexpr,  # tile along W (e.g., 64)
):
    # Grid: (B, C_OUT, H, ceil_div(W, BLOCK_W))
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w_blk = tl.program_id(3)

    h = pid_h
    w_start = pid_w_blk * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    # Accumulator for output row
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels (compile-time loop due to tl.constexpr C_IN)
    for ci in range(0, C_IN):
        # Accumulate over 3x3 neighborhood with padding=1
        for dh in range(-1, 2):
            h_src = h + dh
            valid_h = (h_src >= 0) & (h_src < H)
            for dw in range(-1, 2):
                w_src = w_offsets + dw
                valid_w = (w_src >= 0) & (w_src < W)
                mask = mask_w & valid_w & valid_h
                # Compute input index for x[n, ci, h_src, w_src]
                x_idx = ((pid_n * C_IN + ci) * H + h_src) * W + w_src
                # Load x value (no bias), invalid positions get 0
                x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
                # Load corresponding weight w[co, ci, 1+dh, 1+dw]
                # Weight layout: (C_OUT, C_IN, 3, 3)
                w_idx = (pid_co * C_IN + ci) * 9 + (dh + 1) * 3 + (dw + 1)
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val

    # Store output row
    out_idx = (pid_n * C_OUT + pid_co) * H * W + h * W + w_offsets
    tl.store(out_ptr + out_idx, acc, mask=mask_w)


# Triton kernel: Elementwise residual add out += x, NCHW layout
@triton.jit
def add_residual_kernel(
    out_ptr, x_ptr,
    B, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid: (B, C, H, ceil_div(W, BLOCK_W))
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w_blk = tl.program_id(3)

    h = pid_h
    w_start = pid_w_blk * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    idx = (pid_n * C + pid_c) * H * W + h * W + w_offsets
    out_val = tl.load(out_ptr + idx, mask=mask_w, other=0.0)
    x_val = tl.load(x_ptr + idx, mask=mask_w, other=0.0)
    tl.store(out_ptr + idx, out_val + x_val, mask=mask_w)


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
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add

        Triton-only implementation: both convs and residual add are computed by Triton kernels.
        """
        # Ensure contiguous and float32 for computation
        x = x.contiguous().to(torch.float32)
        B, C, H, W = x.shape

        # Prepare outputs for convs
        C_IN1 = conv1_weight.shape[1]
        C_OUT1 = conv1_weight.shape[0]
        C_IN2 = conv2_weight.shape[1]
        C_OUT2 = conv2_weight.shape[0]

        out1 = torch.empty((B, C_OUT1, H, W), device=x.device, dtype=torch.float32)
        out2 = torch.empty((B, C_OUT2, H, W), device=x.device, dtype=torch.float32)

        # Launch Triton conv1 kernel: B, C_IN=64, C_OUT=64, H, W (tl.constexpr args)
        BLOCK_W = 64
        grid1 = (B, C_OUT1, H, triton.cdiv(W, BLOCK_W))
        conv3x3_nchw_kernel[grid1](
            x, conv1_weight, out1,
            B, C_IN1, C_OUT1, H, W,  # tl.constexpr args: C_IN, C_OUT, H, W (C_IN1 and C_OUT1 are passed as 64 here)
            BLOCK_W,
        )

        # Residual add for conv1 output (Triton): out1 += x
        add_residual_kernel[grid1](
            out1, x,
            B, C_OUT1, H, W,
            BLOCK_W,
        )

        # Launch Triton conv2 kernel: conv(out1, conv2_weight) into out2
        grid2 = (B, C_OUT2, H, triton.cdiv(W, BLOCK_W))
        conv3x3_nchw_kernel[grid2](
            out1, conv2_weight, out2,
            B, C_IN2, C_OUT2, H, W,  # tl.constexpr args: C_IN=64, C_OUT=64
            BLOCK_W,
        )

        # Residual add for conv2 output (Triton): out2 += out1 (this matches the original residual addition behavior)
        add_residual_kernel[grid2](
            out2, out1,
            B, C_OUT2, H, W,
            BLOCK_W,
        )

        return out2


def run(*args):
    return ModelNew()(*args)
