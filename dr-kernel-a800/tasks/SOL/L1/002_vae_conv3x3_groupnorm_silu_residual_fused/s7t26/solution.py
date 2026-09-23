import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_per_oc_kernel(
    x_ptr,           # *float32 input tensor [N, C_in, H, W]
    w_ptr,           # *float32 weights tensor [C_out, C_in, 3, 3]
    y_ptr,           # *float32 output tensor [N, C_out, H, W]
    N: tl.int32, C_in: tl.int32, H: tl.int32, W: tl.int32, C_out: tl.int32,
):
    # Grid: (N, C_out)
    n = tl.program_id(0)
    oc = tl.program_id(1)

    # Accumulator for this single output channel
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                # For stride=1, padding=1, output dims are H and W.
                for oh in range(H):
                    ih = oh + kh - 1  # handle padding
                    for ow in range(W):
                        iw = ow + kw - 1  # handle padding
                        valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

                        # Compute input linear index: (((n * C_in + cin) * H + ih) * W + iw)
                        in_index = (((n * C_in + cin) * H + ih) * W + iw)
                        x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0)

                        # Load corresponding weight for this (oc, cin, kh, kw)
                        # weight layout: [C_out, C_in, 3, 3]
                        w_index = ((oc * C_in + cin) * 9) + (kh * 3 + kw)
                        w_val = tl.load(w_ptr + w_index)

                        acc += x_val * w_val

    # Store result to y[n, oc, :, :]
    for oh in range(H):
        for ow in range(W):
            y_index = (((n * C_out + oc) * H + oh) * W + ow)
            tl.store(y_ptr + y_index, acc)


@triton.jit
def group_norm_32groups_kernel(
    x_ptr,           # *float32 input tensor [N, C, H, W]
    scale_ptr,       # *float32 GroupNorm scale tensor [C]
    bias_ptr,        # *float32 GroupNorm bias tensor [C]
    y_ptr,           # *float32 output tensor [N, C, H, W]
    N: tl.int32, C: tl.int32, H: tl.int32, W: tl.int32,
    eps: tl.float32,
):
    # Grid: (N, 32) since num_groups=32
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // 32
    group_start = g * channels_per_group
    C_group = channels_per_group

    # First pass: compute sum and sum of squares over this group
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for c in range(C_group):
        cin = group_start + c
        for h in range(H):
            for w in range(W):
                x_index = (((n * C + cin) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index)
                sum_val += x_val
                sum_sq += x_val * x_val

    HxW = H * W
    mean = sum_val / (C_group * HxW)
    var = sum_sq / (C_group * HxW) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for c in range(C_group):
        cin = group_start + c
        scale_c = tl.load(scale_ptr + cin)
        bias_c = tl.load(bias_ptr + cin)
        for h in range(H):
            for w in range(W):
                x_index = (((n * C + cin) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index)
                norm = (x_val - mean) * inv_std
                y_val = norm * scale_c + bias_c
                y_index = x_index  # same memory layout
                tl.store(y_ptr + y_index, y_val)


@triton.jit
def silu_kernel(
    x_ptr, y_ptr,
    N: tl.int32, C: tl.int32, H: tl.int32, W: tl.int32,
):
    # Grid: (N, C, H, W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    x_index = (((n * C + c) * H + h) * W + w)
    x_val = tl.load(x_ptr + x_index)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    tl.store(y_ptr + x_index, y_val)


@triton.jit
def add_residual_kernel(
    y_ptr, x_ptr, z_ptr,
    N: tl.int32, C: tl.int32, H: tl.int32, W: tl.int32,
):
    # Grid: (N, C, H, W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    y_index = (((n * C + c) * H + h) * W + w)
    z_index = y_index
    y_val = tl.load(y_ptr + y_index)
    x_val = tl.load(x_ptr + z_index)  # x is the original input
    out_val = y_val + x_val
    tl.store(z_ptr + z_index, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        # Ensure CUDA and contiguous; compute in float32
        assert x.is_cuda and conv1_weight.is_cuda and norm1_weight.is_cuda and norm1_bias.is_cuda and \
               conv2_weight.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "All tensors must be on CUDA"
        N, C, H, W = x.shape
        # Enforce GroupNorm constraint: C must be divisible by 32
        if C % 32 != 0:
            raise ValueError(f"GroupNorm requires C divisible by num_groups (32). Got C={C}.")

        # Cast to float32 for Triton kernels
        x32 = x.contiguous().to(torch.float32)
        conv1_w32 = conv1_weight.contiguous().to(torch.float32)  # [C, C, 3, 3]
        conv2_w32 = conv2_weight.contiguous().to(torch.float32)  # [C, C, 3, 3]
        norm1_s32 = norm1_weight.contiguous().to(torch.float32)  # [C]
        norm1_b32 = norm1_bias.contiguous().to(torch.float32)    # [C]
        norm2_s32 = norm2_weight.contiguous().to(torch.float32)  # [C]
        norm2_b32 = norm2_bias.contiguous().to(torch.float32)    # [C]

        # 1) Conv1: Triton
        y1 = torch.empty((N, C, H, W), dtype=torch.float32, device=x.device)
        grid_conv = (N, C)
        conv3x3_stride1_pad1_per_oc_kernel[grid_conv](
            x32, conv1_w32, y1, N, C, H, W, C, num_warps=4, num_stages=2
        )

        # 2) GroupNorm1: Triton
        y1_norm = torch.empty_like(y1)
        grid_gn = (N, 32)
        group_norm_32groups_kernel[grid_gn](
            y1, norm1_s32, norm1_b32, y1_norm, N, C, H, W, eps, num_warps=4, num_stages=2
        )

        # 3) SiLU1: Triton
        y1_silu = torch.empty_like(y1_norm)
        grid_silu = (N, C, H, W)
        silu_kernel[grid_silu](y1_norm, y1_silu, N, C, H, W, num_warps=4, num_stages=2)

        # 4) Conv2: Triton
        y2 = torch.empty((N, C, H, W), dtype=torch.float32, device=x.device)
        grid_conv2 = (N, C)
        conv3x3_stride1_pad1_per_oc_kernel[grid_conv2](
            y1_silu, conv2_w32, y2, N, C, H, W, C, num_warps=4, num_stages=2
        )

        # 5) GroupNorm2: Triton
        y2_norm = torch.empty_like(y2)
        grid_gn2 = (N, 32)
        group_norm_32groups_kernel[grid_gn2](
            y2, norm2_s32, norm2_b32, y2_norm, N, C, H, W, eps, num_warps=4, num_stages=2
        )

        # 6) SiLU2: Triton
        y2_silu = torch.empty_like(y2_norm)
        grid_silu2 = (N, C, H, W)
        silu_kernel[grid_silu2](y2_norm, y2_silu, N, C, H, W, num_warps=4, num_stages=2)

        # 7) Add residual (original x): Triton
        y_out = torch.empty_like(y2_silu)
        grid_add = (N, C, H, W)
        add_residual_kernel[grid_add](y2_silu, x32, y_out, N, C, H, W, num_warps=4, num_stages=2)

        # Cast back to original dtype if needed
        if x.dtype != torch.float32:
            y_out = y_out.to(x.dtype)
        return y_out


def run(*args):
    return ModelNew()(*args)
