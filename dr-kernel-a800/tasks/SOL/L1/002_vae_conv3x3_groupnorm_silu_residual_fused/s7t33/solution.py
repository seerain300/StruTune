import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_kernel(
    x_ptr,          # *float32 input tensor (B, C_in, H, W)
    w_ptr,          # *float32 weights tensor (C_out, C_in, 3, 3)
    y_ptr,          # *float32 output tensor (B, C_out, H, W)
    N: tl.constexpr,  # batch size
    C_in: tl.constexpr,  # input channels
    C_out: tl.constexpr,  # output channels
    H: tl.constexpr,  # input height
    W: tl.constexpr,  # input width
    BLOCK_OC: tl.constexpr,  # tile size for output channels per program
):
    # Grid: (N, ceil_div(C_out, BLOCK_OC))
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    # Accumulator for the tile of output channels
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel taps
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                # Output spatial dimensions are H and W (stride=1, padding=1)
                for h in range(H):
                    ih = h + kh - 1  # input row index for this output row h
                    valid_h = (ih >= 0) & (ih < H)
                    for wj in range(W):
                        iw = wj + kw - 1  # input col index for this output col wj
                        valid = valid_h & (iw >= 0) & (iw < W)
                        # Input linear index: (((n * C_in + cin) * H + ih) * W + iw)
                        in_index = (((n * C_in + cin) * H + ih) * W + iw)
                        x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0).to(tl.float32)
                        # Weight linear index for oc tile: (((oc * C_in + cin) * 9) + (kh * 3 + kw))
                        for j in range(BLOCK_OC):
                            if oc_mask[oc_offsets[j]]:
                                w_index = (((oc_offsets[j] * C_in + cin) * 9) + (kh * 3 + kw))
                                w_val = tl.load(w_ptr + w_index).to(tl.float32)
                                acc[j] += x_val * w_val

    # Store results for all spatial positions (h, w)
    for j in range(BLOCK_OC):
        if oc_mask[oc_offsets[j]]:
            for h in range(H):
                for wj in range(W):
                    out_index = (((n * C_out + oc_offsets[j]) * H + h) * W + wj)
                    tl.store(y_ptr + out_index, acc[j])


@triton.jit
def group_norm_affine_kernel(
    x_ptr,           # *float32 input tensor (B, C, H, W)
    gamma_ptr,       # *float32 scale tensor (C,)
    beta_ptr,        # *float32 bias tensor (C,)
    y_ptr,           # *float32 output tensor (B, C, H, W)
    N: tl.constexpr,  # batch size
    C: tl.constexpr,  # channels
    H: tl.constexpr,  # height
    W: tl.constexpr,  # width
    num_groups: tl.constexpr,  # number of groups (32)
    eps: tl.constexpr,          # epsilon for numerical stability
):
    # One program per (n, group)
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    group_start = g * channels_per_group
    group_end = group_start + channels_per_group

    # Compute sum and sum of squares across the group over all spatial positions
    sum_val = 0.0
    sum_sq = 0.0
    for c in range(group_start, group_end):
        for h in range(H):
            for w in range(W):
                index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + index)
                sum_val += x_val
                sum_sq += x_val * x_val

    # Compute mean and variance
    M = channels_per_group * H * W
    mean = sum_val / M
    var = sum_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine: y = ((x - mean) * inv_std) * gamma + beta
    for c in range(group_start, group_end):
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        for h in range(H):
            for w in range(W):
                index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + index)
                norm = ((x_val - mean) * inv_std) * gamma + beta
                tl.store(y_ptr + index, norm)


@triton.jit
def silu_kernel(
    x_ptr,            # *float32 input tensor (B, C, H, W)
    y_ptr,            # *float32 output tensor (B, C, H, W)
    N: tl.constexpr,  # batch size
    C: tl.constexpr,  # channels
    H: tl.constexpr,  # height
    W: tl.constexpr,  # width
):
    # One program per (n, c, h, w)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    index = (((n * C + c) * H + h) * W + w)
    x_val = tl.load(x_ptr + index)
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    tl.store(y_ptr + index, y_val)


@triton.jit
def add_residual_kernel(
    y_ptr,            # *float32 input tensor (B, C, H, W)
    x_ptr,            # *float32 residual tensor (B, C, H, W)
    out_ptr,          # *float32 output tensor (B, C, H, W)
    N: tl.constexpr,  # batch size
    C: tl.constexpr,  # channels
    H: tl.constexpr,  # height
    W: tl.constexpr,  # width
):
    # One program per (n, c, h, w)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    in_index = (((n * C + c) * H + h) * W + w)
    y_val = tl.load(y_ptr + in_index)
    x_val = tl.load(x_ptr + in_index)
    sum_val = y_val + x_val
    tl.store(out_ptr + in_index, sum_val)


@triton.jit
def conv3x3_stride1_pad1_kernel_two(
    x_ptr,          # *float32 input tensor (B, C_in, H, W)
    w_ptr,          # *float32 weights tensor (C_out, C_in, 3, 3)
    y_ptr,          # *float32 output tensor (B, C_out, H, W)
    N: tl.constexpr,  # batch size
    C_in: tl.constexpr,  # input channels
    C_out: tl.constexpr,  # output channels
    H: tl.constexpr,  # input height
    W: tl.constexpr,  # input width
    BLOCK_OC: tl.constexpr,  # tile size for output channels per program
):
    # Grid: (N, ceil_div(C_out, BLOCK_OC))
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                for h in range(H):
                    ih = h + kh - 1
                    valid_h = (ih >= 0) & (ih < H)
                    for wj in range(W):
                        iw = wj + kw - 1
                        valid = valid_h & (iw >= 0) & (iw < W)
                        in_index = (((n * C_in + cin) * H + ih) * W + iw)
                        x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0).to(tl.float32)
                        for j in range(BLOCK_OC):
                            if oc_mask[oc_offsets[j]]:
                                w_index = (((oc_offsets[j] * C_in + cin) * 9) + (kh * 3 + kw))
                                w_val = tl.load(w_ptr + w_index).to(tl.float32)
                                acc[j] += x_val * w_val

    for j in range(BLOCK_OC):
        if oc_mask[oc_offsets[j]]:
            for h in range(H):
                for wj in range(W):
                    out_index = (((n * C_out + oc_offsets[j]) * H + h) * W + wj)
                    tl.store(y_ptr + out_index, acc[j])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        Triton-only implementation for convs, GroupNorms, SiLU, and residual add.
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype  # keep dtype as float32 for numerical stability

        # Ensure contiguity and float32 for kernels
        x_c = x.contiguous()
        if x_c.dtype != torch.float32:
            x_c = x_c.float()
        conv1_weight_c = conv1_weight.contiguous().float()
        conv2_weight_c = conv2_weight.contiguous().float()
        norm1_weight_c = norm1_weight.contiguous().float()
        norm1_bias_c = norm1_bias.contiguous().float()
        norm2_weight_c = norm2_weight.contiguous().float()
        norm2_bias_c = norm2_bias.contiguous().float()

        # 1) Conv1: Triton kernel
        C_in1 = C
        C_out1 = conv1_weight_c.shape[0]
        y1 = torch.empty((B, C_out1, H, W), device=device, dtype=torch.float32)
        grid_conv1 = (B, triton.cdiv(C_out1, 32))
        conv3x3_stride1_pad1_kernel[grid_conv1](
            x_c, conv1_weight_c, y1,
            N=B, C_in=C_in1, C_out=C_out1, H=H, W=W,
            BLOCK_OC=32,
            num_warps=4,
            num_stages=2,
        )

        # 2) GroupNorm1 (32 groups) on y1: Triton kernel
        if C_out1 % 32 != 0:
            raise ValueError(f"GroupNorm requires C divisible by num_groups (32). Got C_out1={C_out1}.")
        y_norm1 = torch.empty_like(y1, dtype=torch.float32)
        grid_gn1 = (B, 32)
        group_norm_affine_kernel[grid_gn1](
            y1, norm1_weight_c, norm1_bias_c, y_norm1,
            N=B, C=C_out1, H=H, W=W,
            num_groups=32, eps=eps,
            num_warps=4,
            num_stages=2,
        )

        # 3) SiLU1: Triton kernel
        y_silu1 = torch.empty_like(y_norm1, dtype=torch.float32)
        grid_silu1 = (B, C_out1, H, W)
        silu_kernel[grid_silu1](
            y_norm1, y_silu1,
            N=B, C=C_out1, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # 4) Conv2: Triton kernel on y_silu1
        C_in2 = C_out1
        C_out2 = conv2_weight_c.shape[0]
        y2 = torch.empty((B, C_out2, H, W), device=device, dtype=torch.float32)
        grid_conv2 = (B, triton.cdiv(C_out2, 32))
        conv3x3_stride1_pad1_kernel[grid_conv2](
            y_silu1, conv2_weight_c, y2,
            N=B, C_in=C_in2, C_out=C_out2, H=H, W=W,
            BLOCK_OC=32,
            num_warps=4,
            num_stages=2,
        )

        # 5) GroupNorm2 (32 groups) on y2: Triton kernel
        if C_out2 % 32 != 0:
            raise ValueError(f"GroupNorm requires C divisible by num_groups (32). Got C_out2={C_out2}.")
        y_norm2 = torch.empty_like(y2, dtype=torch.float32)
        grid_gn2 = (B, 32)
        group_norm_affine_kernel[grid_gn2](
            y2, norm2_weight_c, norm2_bias_c, y_norm2,
            N=B, C=C_out2, H=H, W=W,
            num_groups=32, eps=eps,
            num_warps=4,
            num_stages=2,
        )

        # 6) SiLU2: Triton kernel
        y_silu2 = torch.empty_like(y_norm2, dtype=torch.float32)
        grid_silu2 = (B, C_out2, H, W)
        silu_kernel[grid_silu2](
            y_norm2, y_silu2,
            N=B, C=C_out2, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # 7) Residual add (Triton): y_out = y_silu2 + x
        y_out = torch.empty_like(y_silu2, dtype=torch.float32)
        grid_add = (B, C, H, W)
        add_residual_kernel[grid_add](
            y_silu2, x_c, y_out,
            N=B, C=C, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        return y_out


# Example usage:
# model = ModelNew().cuda()
# x = torch.randn(16, 64, 64, 64, device='cuda', dtype=torch.float32)
# conv1_w = torch.randn(64, 64, 3, 3, device='cuda', dtype=torch.float32)
# norm1_w = torch.randn(64, device='cuda', dtype=torch.float32)
# norm1_b = torch.randn(64, device='cuda', dtype=torch.float32)
# conv2_w = torch.randn(64, 64, 3, 3, device='cuda', dtype=torch.float32)
# norm2_w = torch.randn(64, device='cuda', dtype=torch.float32)
# norm2_b = torch.randn(64, device='cuda', dtype=torch.float32)
# eps = 1e-5
# y = model(x, conv1_w, norm1_w, norm1_b, conv2_w, norm2_w, norm2_b, eps)


def run(*args):
    return ModelNew()(*args)
