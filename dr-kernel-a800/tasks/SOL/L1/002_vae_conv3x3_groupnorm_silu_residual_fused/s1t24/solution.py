import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,        # *float32, input [N, C_in, H, W]
    w_ptr,        # *float32, weight [C_in, C_out, 3, 3]
    y_ptr,        # *float32, output [N, C_out, H, W]
    N, C_in, H, W,
    C_out,
    BLOCK_IN: tl.constexpr,
):
    # One program computes one output element: y[n, c_out, h_out, w_out]
    pid = tl.program_id(axis=0)

    # Compute n, c_out, h_out, w_out from pid
    n = pid // (C_out * H * W)
    tmp = pid % (C_out * H * W)
    c_out = tmp // (H * W)
    tmp = tmp % (H * W)
    h_out = tmp // W
    w_out = tmp % W

    # Accumulator
    acc = 0.0

    # Loop over input channels in chunks
    for c_start in range(0, C_in, BLOCK_IN):
        c_idx = c_start + tl.arange(0, BLOCK_IN)
        mask_c = c_idx < C_in

        # Accumulate over 3x3 window with padding
        for kh in range(3):
            for kw in range(3):
                ih = h_out + kh - 1  # pad=1: conv output dims equal input dims
                iw = w_out + kw - 1
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

                # Build input pointer for all c in the chunk and all input channels
                # Address: x[n, c_idx, ih, iw]
                x_offsets = n * (C_in * H * W) + c_idx * (H * W) + ih * W + iw
                x_mask = mask_c & in_bounds
                x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

                # Load weights for these c_idx -> c_out
                # Address: w[c_idx, c_out, kh, kw]
                w_offsets = c_idx * (3 * 3) + kh * 3 + kw
                w_vals = tl.load(w_ptr + w_offsets, mask=mask_c, other=0.0)

                # Outer product and accumulate
                # acc += sum_{c} x_vals[c] * w_vals[c]
                acc += tl.sum(x_vals * w_vals, axis=0)

    # Store output: y[n, c_out, h_out, w_out]
    y_offset = n * (C_out * H * W) + c_out * (H * W) + h_out * W + w_out
    tl.store(y_ptr + y_offset, acc)


@triton.jit
def groupnorm_affine_fp32(
    x_ptr,        # *float32, input [N, C, H, W]
    scale_ptr,    # *float32, per-channel scale [C]
    bias_ptr,     # *float32, per-channel bias [C]
    y_ptr,        # *float32, output [N, C, H, W]
    N, C, H, W,
    group_id,     # int
    group_size,   # int, C / num_groups (here num_groups=32)
    num_groups: tl.constexpr,  # number of groups, used for bounds checks (not used in math but kept for clarity)
    eps,          # float
):
    # One program handles one (n, group)
    n = tl.program_id(axis=0)  # we launch one program per (n, group)
    # Compute group bounds
    c_start = group_id * group_size
    # First pass: compute mean and variance over channels in the group and all spatial positions
    sum_val = 0.0
    sum_sq = 0.0
    ci = 0
    while ci < group_size:
        c = c_start + ci
        # loop over all H*W
        spatial_idx = 0
        hw = H * W
        while spatial_idx < hw:
            h = spatial_idx // W
            w = spatial_idx % W
            x_offset = n * (C * H * W) + c * (H * W) + h * W + w
            x_val = tl.load(x_ptr + x_offset)
            sum_val += x_val
            sum_sq += x_val * x_val
            spatial_idx += 1
        ci += 1

    mean = sum_val / (group_size * H * W)
    var = sum_sq / (group_size * H * W) - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, then store
    ci = 0
    while ci < group_size:
        c = c_start + ci
        # per-channel scale and bias
        scale = tl.load(scale_ptr + c)
        b = tl.load(bias_ptr + c)
        spatial_idx = 0
        hw = H * W
        while spatial_idx < hw:
            h = spatial_idx // W
            w = spatial_idx % W
            x_offset = n * (C * H * W) + c * (H * W) + h * W + w
            x_val = tl.load(x_ptr + x_offset)
            y_val = (x_val - mean) * rstd
            y_val = y_val * scale + b
            y_offset = n * (C * H * W) + c * (H * W) + h * W + w
            tl.store(y_ptr + y_offset, y_val)
            spatial_idx += 1
        ci += 1


@triton.jit
def silu_kernel_fp32(x_ptr, y_ptr, total_elems: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def add_residual_fp32(a_ptr, b_ptr, y_ptr, total_elems: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    y = a + b
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32):
        super().__init__()
        self.num_groups = num_groups

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
        conv1 (3x3) -> GroupNorm -> SiLU -> conv2 (3x3) -> GroupNorm -> SiLU -> add original x
        Shapes preserved: N, C, H, W
        """
        # Ensure float32 and contiguous
        device = x.device
        dtype = x.dtype
        assert device.type == "cuda", "This implementation requires CUDA and Triton kernels."
        # Make contiguous and cast to float32 for kernels
        x_in = x.contiguous().to(torch.float32)
        N, C_in, H, W = x_in.shape

        # Assert channels divisible by num_groups (32 here)
        assert C_in % self.num_groups == 0, "Channels must be divisible by num_groups (32)."

        # 1) conv1
        y1 = torch.empty((N, C_in, H, W), device=device, dtype=torch.float32)
        C_out1 = C_in  # same as input channels
        grid_conv1 = (N * C_out1 * H * W,)
        conv3x3_nchw_fp32[grid_conv1](
            x_in, conv1_weight.contiguous().to(torch.float32), y1,
            N, C_in, H, W, C_out1,
            BLOCK_IN=16,
        )

        # 2) GroupNorm1
        y1_gn = torch.empty_like(y1, device=device, dtype=torch.float32)
        group_size1 = C_in // self.num_groups
        for group_id in range(self.num_groups):
            grid_gn1 = (1,)  # one program per (n, group), n is implicit in loop and pointer math uses batch dim
            groupnorm_affine_fp32[grid_gn1](
                y1, norm1_weight.contiguous().to(torch.float32), norm1_bias.contiguous().to(torch.float32),
                y1_gn,
                N, C_in, H, W,
                group_id, group_size1, self.num_groups, eps,
            )

        # 3) SiLU1
        y1_silu = torch.empty_like(y1_gn, device=device, dtype=torch.float32)
        total_silu1 = y1_gn.numel()
        grid_silu1 = (triton.cdiv(total_silu1, 1024),)
        silu_kernel_fp32[grid_silu1](y1_gn, y1_silu, total_silu1, BLOCK=1024)

        # 4) conv2
        y2 = torch.empty((N, C_in, H, W), device=device, dtype=torch.float32)
        C_out2 = C_in
        grid_conv2 = (N * C_out2 * H * W,)
        conv3x3_nchw_fp32[grid_conv2](
            y1_silu, conv2_weight.contiguous().to(torch.float32), y2,
            N, C_in, H, W, C_out2,
            BLOCK_IN=16,
        )

        # 5) GroupNorm2
        y2_gn = torch.empty_like(y2, device=device, dtype=torch.float32)
        group_size2 = C_in // self.num_groups
        for group_id in range(self.num_groups):
            grid_gn2 = (1,)
            groupnorm_affine_fp32[grid_gn2](
                y2, norm2_weight.contiguous().to(torch.float32), norm2_bias.contiguous().to(torch.float32),
                y2_gn,
                N, C_in, H, W,
                group_id, group_size2, self.num_groups, eps,
            )

        # 6) SiLU2
        y2_silu = torch.empty_like(y2_gn, device=device, dtype=torch.float32)
        total_silu2 = y2_gn.numel()
        grid_silu2 = (triton.cdiv(total_silu2, 1024),)
        silu_kernel_fp32[grid_silu2](y2_gn, y2_silu, total_silu2, BLOCK=1024)

        # 7) Residual add (original x)
        x_fp32 = x_in  # original input as fp32 contiguous
        total_add = y2_silu.numel()
        final_out = torch.empty_like(y2_silu, device=device, dtype=torch.float32)
        grid_add = (triton.cdiv(total_add, 1024),)
        add_residual_fp32[grid_add](y2_silu, x_fp32, final_out, total_add, BLOCK=1024)

        return final_out


def run(*args):
    return ModelNew()(*args)
