import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_per_oc_kernel(
    x_ptr,           # *float32, input [N, C_in, H, W]
    w_ptr,           # *float32, weights [C_out, C_in, 3, 3]
    y_ptr,           # *float32, output [N, C_out, H, W]
    N, C_in, H, W, C_out,
):
    # Grid: (N, C_out)
    n = tl.program_id(0)
    oc = tl.program_id(1)

    # Accumulator for this output channel
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 taps, handle padding via masks
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                for oh in range(H):
                    ih = oh + kh - 1  # padding=1
                    for ow in range(W):
                        iw = ow + kw - 1
                        valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

                        # Input linear index: (((n * C_in + cin) * H + ih) * W + iw)
                        in_index = (((n * C_in + cin) * H + ih) * W + iw)
                        x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0)

                        # Weight scalar for (oc, cin, kh, kw)
                        # weight layout: w[oc, cin, kh, kw] -> linear index = oc*C_in*9 + cin*9 + kh*3 + kw
                        w_index = (oc * C_in * 9) + (cin * 9) + (kh * 3) + kw
                        w_val = tl.load(w_ptr + w_index)

                        acc += x_val * w_val

    # Store result: y[n, oc, :, :] -> linear index (((n * C_out + oc) * H + oh) * W + ow)
    # We need to write acc into each (oh, ow). Since we loop over oh, ow, we compute here using oh as the outer index.
    # However, we cannot store inside the nested loop; Triton requires explicit writes per element.
    # To simplify, we store after loops using oh, ow again by iterating, but Triton expects vectorized stores.
    # Instead, we use a second small loop to write back. For each oh, ow, recompute in_index but store acc.
    # We do this by iterating over spatial positions after accumulation. We can write using linear indexing.

    # Create base offset for output tensor starting from n, oc
    base_n_oc = (n * C_out + oc) * H * W

    # Write acc back to all spatial positions
    # We need to write y[n, oc, oh, ow] = acc for all oh, ow. We iterate and store.
    # Note: acc is scalar; Triton can broadcast it when storing, but Triton does not allow implicit broadcasting.
    # So we need a vectorized approach: store acc into each y_ptr position.
    # Triton allows scalar store, but we must ensure we store for each spatial position.
    for oh in range(H):
        for ow in range(W):
            out_index = base_n_oc + (oh * W + ow)
            tl.store(y_ptr + out_index, acc)


@triton.jit
def group_norm_32groups_kernel(
    x_ptr,          # *float32, input [N, C, H, W]
    gamma_ptr,      # *float32, scale [C]
    beta_ptr,       # *float32, bias [C]
    y_ptr,          # *float32, output [N, C, H, W]
    N, C, H, W,
    eps: tl.constexpr,
):
    # GroupNorm requires C % 32 == 0
    channels_per_group = C // 32
    num_groups = 32

    # Grid: (N, num_groups)
    n = tl.program_id(0)
    g = tl.program_id(1)

    # Compute statistics for this (n, group)
    group_start = g * channels_per_group
    sum_ = tl.zeros((), dtype=tl.float32)
    sumsq_ = tl.zeros((), dtype=tl.float32)

    # First pass: accumulate sum and sum of squares across group channels and all H*W
    for ch in range(channels_per_group):
        c = group_start + ch
        for oh in range(H):
            for ow in range(W):
                in_index = (((n * C + c) * H + oh) * W + ow)
                x_val = tl.load(x_ptr + in_index)
                sum_ += x_val
                sumsq_ += x_val * x_val

    size = channels_per_group * H * W
    mean = sum_ / size
    var = sumsq_ / size - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for ch in range(channels_per_group):
        c = group_start + ch
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        for oh in range(H):
            for ow in range(W):
                in_index = (((n * C + c) * H + oh) * W + ow)
                x_val = tl.load(x_ptr + in_index)
                y_val = ((x_val - mean) * inv_std) * gamma + beta
                out_index = in_index  # same linear index mapping for output tensor
                tl.store(y_ptr + out_index, y_val)


@triton.jit
def silu_kernel(x_ptr, y_ptr, N, C, H, W):
    # Elementwise: y = x * sigmoid(x)
    for n in range(N):
        for c in range(C):
            for oh in range(H):
                for ow in range(W):
                    in_index = (((n * C + c) * H + oh) * W + ow)
                    x_val = tl.load(x_ptr + in_index)
                    # sigmoid
                    sig = 1.0 / (1.0 + tl.exp(-x_val))
                    y_val = x_val * sig
                    out_index = in_index
                    tl.store(y_ptr + out_index, y_val)


@triton.jit
def add_residual_kernel(y_ptr, x_ptr, out_ptr, N, C, H, W):
    # Elementwise: out = y + x
    for n in range(N):
        for c in range(C):
            for oh in range(H):
                for ow in range(W):
                    in_index = (((n * C + c) * H + oh) * W + ow)
                    y_val = tl.load(y_ptr + in_index)
                    x_val = tl.load(x_ptr + in_index)
                    out_val = y_val + x_val
                    tl.store(out_ptr + in_index, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        # Ensure CUDA and float32 for stable computation
        device = x.device
        assert device.type == 'cuda', "Input tensor must be on CUDA for Triton kernels."
        N, C, H, W = x.shape
        # Check GroupNorm divisibility (PyTorch requirement)
        assert C % 32 == 0, "GroupNorm requires C % 32 == 0 for num_groups=32."

        x32 = x.to(torch.float32)

        # First Conv3x3 (bias=None, stride=1, padding=1)
        # Prepare output y1
        y1 = torch.empty((N, C, H, W), dtype=torch.float32, device=device)

        # Ensure weights are float32 and contiguous
        w1 = conv1_weight.to(torch.float32).contiguous()
        w2 = conv2_weight.to(torch.float32).contiguous()

        # Launch conv kernel: grid=(N, C)
        grid_conv = (N, C)
        conv3x3_stride1_pad1_per_oc_kernel[grid_conv](
            x32, w1, y1, N, C, H, W, C,
            num_warps=4, num_stages=2
        )

        # First GroupNorm (num_groups=32)
        y_gn1 = torch.empty_like(y1)
        grid_gn1 = (N, 32)
        group_norm_32groups_kernel[grid_gn1](
            y1, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32), y_gn1,
            N, C, H, W, eps=eps,
            num_warps=4, num_stages=2
        )

        # SiLU
        y_silu1 = torch.empty_like(y_gn1)
        grid_silu1 = (N, C, H, W)
        silu_kernel[grid_silu1](y_gn1, y_silu1, N, C, H, W,
                                num_warps=4, num_stages=2)

        # Save residual x
        # We will add it later in Triton.

        # Second Conv3x3
        y2_pre = torch.empty((N, C, H, W), dtype=torch.float32, device=device)
        conv3x3_stride1_pad1_per_oc_kernel[grid_conv](
            y_silu1, w2, y2_pre, N, C, H, W, C,
            num_warps=4, num_stages=2
        )

        # Second GroupNorm
        y2_gn = torch.empty_like(y2_pre)
        grid_gn2 = (N, 32)
        group_norm_32groups_kernel[grid_gn2](
            y2_pre, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32), y2_gn,
            N, C, H, W, eps=eps,
            num_warps=4, num_stages=2
        )

        # SiLU
        y2_silu = torch.empty_like(y2_gn)
        grid_silu2 = (N, C, H, W)
        silu_kernel[grid_silu2](y2_gn, y2_silu, N, C, H, W,
                                num_warps=4, num_stages=2)

        # Residual add (original x in float32)
        y_out = torch.empty_like(y2_silu)
        grid_add = (N, C, H, W)
        add_residual_kernel[grid_add](y2_silu, x32, y_out, N, C, H, W,
                                      num_warps=4, num_stages=2)

        # Cast back to original dtype if needed
        if x.dtype != torch.float32:
            y_out = y_out.to(x.dtype)
        return y_out


def run(*args):
    return ModelNew()(*args)
