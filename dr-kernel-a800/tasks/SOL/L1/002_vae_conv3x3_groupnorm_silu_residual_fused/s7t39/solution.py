import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_tile_kernel(
    x_ptr,        # *float32, input (B, C_in, H, W), contiguous
    w_ptr,        # *float32, weight (C_out, C_in, 3, 3), contiguous
    y_ptr,        # *float32, output (B, C_out, H, W), contiguous
    N, C_in, C_out, H, W,
    BLOCK_OC: tl.constexpr,  # tile size for output channels (e.g., 8)
):
    # Grid: (N, ceil_div(C_out, BLOCK_OC), H, W)
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)  # [BLOCK_OC] int32
    oc_mask = oc_offsets < C_out

    # Accumulator for this (n, tile of oc, oh, ow)
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Loop over input channels and 3x3 taps with static bounds
    for cin in range(0, C_in):  # C_in is runtime, Triton allows such loops
        for kh in range(3):
            for kw in range(3):
                ih = oh + kh - 1  # output -> input with padding
                iw = ow + kw - 1
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                # input linear index: (((n * C_in + cin) * H + ih) * W + iw)
                in_index = (((n * C_in + cin) * H + ih) * W + iw)
                # load x element (masked for padding)
                x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0)

                # load weight vector for this cin and (kh, kw) across BLOCK_OC oc's
                for j in range(BLOCK_OC):
                    if oc_mask[j]:
                        # weight linear index: (((c_out * C_in + cin) * 9) + (kh * 3 + kw))
                        w_index = (((oc_offsets[j] * C_in + cin) * 9) + (kh * 3 + kw))
                        w_val = tl.load(w_ptr + w_index)
                        acc[j] += x_val * w_val

    # Store results y[n, oc, oh, ow] for all oc in the tile
    # y linear index: (((n * C_out + c_out) * H + oh) * W + ow)
    for j in range(BLOCK_OC):
        if oc_mask[j]:
            out_index = (((n * C_out + oc_offsets[j]) * H + oh) * W + ow)
            tl.store(y_ptr + out_index, acc[j])


@triton.jit
def group_norm_kernel(
    x_ptr,        # *float32, input (N, C, H, W), contiguous
    gn_weight_ptr,  # *float32, per-channel scale (C,)
    gn_bias_ptr,    # *float32, per-channel bias (C,)
    y_ptr,        # *float32, output (N, C, H, W), contiguous
    N, C, H, W,
    num_groups: tl.constexpr,  # must be 32
    eps: tl.constexpr,
):
    # Each program handles one (n, group)
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    # Accumulate sum and sumsq over this group and all spatial
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # First pass: compute mean and variance
    for c in range(0, channels_per_group):  # small, compile-time-like loop in Triton
        for oh in range(0, H):
            for ow in range(0, W):
                x_index = (((n * C + (group_start + c)) * H + oh) * W + ow)
                x_val = tl.load(x_ptr + x_index)
                sum_val += x_val
                sum_sq += x_val * x_val

    # Compute mean and variance
    M = channels_per_group * H * W
    mean = sum_val / M
    var = sum_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, then store
    for c in range(0, channels_per_group):
        for oh in range(0, H):
            for ow in range(0, W):
                x_index = (((n * C + (group_start + c)) * H + oh) * W + ow)
                x_val = tl.load(x_ptr + x_index)
                norm = (x_val - mean) * inv_std
                gamma = tl.load(gn_weight_ptr + (group_start + c))
                beta = tl.load(gn_bias_ptr + (group_start + c))
                y_val = norm * gamma + beta
                y_index = x_index  # same indexing; y has same shape
                tl.store(y_ptr + y_index, y_val)


@triton.jit
def silu_kernel(
    x_ptr, y_ptr, N, C, H, W,
):
    # elementwise y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    for n in range(0, N):
        for c in range(0, C):
            for h in range(0, H):
                for w in range(0, W):
                    idx = (((n * C + c) * H + h) * W + w)
                    x_val = tl.load(x_ptr + idx)
                    # compute in float32
                    x_f = x_val.to(tl.float32)
                    sig = 1.0 / (1.0 + tl.exp(-x_f))
                    y_val = x_f * sig
                    tl.store(y_ptr + idx, y_val)


@triton.jit
def add_residual_kernel(x_ptr, y_ptr, out_ptr, N, C, H, W):
    # elementwise out = y + x
    for n in range(0, N):
        for c in range(0, C):
            for h in range(0, H):
                for w in range(0, W):
                    idx = (((n * C + c) * H + h) * W + w)
                    a = tl.load(x_ptr + idx)
                    b = tl.load(y_ptr + idx)
                    out_val = a + b
                    tl.store(out_ptr + idx, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

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
        Triton-only implementation of:
          out = SiLU(GroupNorm(SiLU(GroupNorm(Conv3x3(Conv3x3(x)))))) + x
        with num_groups=32 and eps provided.
        """
        assert x.is_cuda and x.dtype == torch.float32, "x must be CUDA float32"
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Weights must be CUDA"
        assert conv1_weight.dtype == torch.float32 and conv2_weight.dtype == torch.float32, "Weights must be float32"
        B, C, H, W = x.shape
        assert conv1_weight.shape[1] == C, "conv1_weight in_channels must match x channels"
        assert conv2_weight.shape[1] == C, "conv2_weight in_channels must match x channels"
        # Enforce GroupNorm requirement
        if C % 32 != 0:
            raise ValueError(f"Channels {C} must be divisible by num_groups=32 for GroupNorm.")
        # Ensure contiguity
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        norm1_weight = norm1_bias = norm2_weight = norm2_bias = norm1_weight  # oh no, just names

        # 1) Conv1 (Triton)
        y1 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        BLOCK_OC = 8  # tile size for output channels
        grid_conv1 = (B, triton.cdiv(C, BLOCK_OC), H, W)
        conv3x3_stride1_pad1_tile_kernel[grid_conv1](
            x, conv1_weight, y1, B, C, C, H, W,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        # 2) GroupNorm1 (Triton)
        y1gn = torch.empty_like(y1)
        grid_gn1 = (B, 32)
        group_norm_kernel[grid_gn1](
            y1, norm1_weight, norm1_bias, y1gn, B, C, H, W,
            num_groups=32,
            eps=eps,
            num_warps=4,
            num_stages=2,
        )

        # 3) SiLU1 (Triton)
        y1_silu = torch.empty_like(y1gn)
        grid_silu1 = (B, C, H, W)
        silu_kernel[grid_silu1](
            y1gn, y1_silu, B, C, H, W,
            num_warps=4,
            num_stages=2,
        )

        # 4) Conv2 (Triton)
        y2 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        grid_conv2 = (B, triton.cdiv(C, BLOCK_OC), H, W)
        conv3x3_stride1_pad1_tile_kernel[grid_conv2](
            y1_silu, conv2_weight, y2, B, C, C, H, W,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        # 5) GroupNorm2 (Triton)
        y2gn = torch.empty_like(y2)
        group_norm_kernel[grid_gn1](
            y2, norm2_weight, norm2_bias, y2gn, B, C, H, W,
            num_groups=32,
            eps=eps,
            num_warps=4,
            num_stages=2,
        )

        # 6) SiLU2 (Triton)
        y2_silu = torch.empty_like(y2gn)
        silu_kernel[grid_silu1](
            y2gn, y2_silu, B, C, H, W,
            num_warps=4,
            num_stages=2,
        )

        # 7) Add residual (Triton)
        y_out = torch.empty_like(y2_silu)
        add_residual_kernel[grid_silu1](
            x, y2_silu, y_out, B, C, H, W,
            num_warps=4,
            num_stages=2,
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
