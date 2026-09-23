import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_kernel(
    x_ptr,          # *float32 input tensor (B, C_in, H, W)
    w_ptr,          # *float32 weights tensor (C_out, C_in, 3, 3)
    y_ptr,          # *float32 output tensor (B, C_out, H, W)
    N,              # int
    C_in,           # int
    H,              # int
    W,              # int
    C_out,          # int
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
                # Output spatial dims equal input due to stride=1, padding=1
                for h in range(H):
                    ih = h + kh - 1
                    valid_h = (ih >= 0) & (ih < H)
                    for wj in range(W):
                        iw = wj + kw - 1
                        valid = valid_h & (iw >= 0) & (iw < W)
                        # Input linear index: (((n * C_in + cin) * H + ih) * W + iw)
                        in_index = (((n * C_in + cin) * H + ih) * W + iw)
                        x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0).to(tl.float32)
                        # Weight linear index: (((oc * C_in + cin) * 9) + (kh * 3 + kw))
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
    x_ptr,            # *float32 input tensor (B, C, H, W)
    gamma_ptr,        # *float32 scale (C,)
    beta_ptr,         # *float32 bias (C,)
    y_ptr,            # *float32 output tensor (B, C, H, W)
    N: tl.constexpr,  # batch size
    C: tl.constexpr,  # channels
    H: tl.constexpr,  # height
    W: tl.constexpr,  # width
    num_groups: tl.constexpr,  # number of groups (e.g., 32)
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
    # Loop over channels in the group
    for c in range(group_start, group_end):
        # Loop over all H*W positions
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
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    tl.store(y_ptr + index, y_val)


@triton.jit
def add_kernel(
    a_ptr, b_ptr, out_ptr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr
):
    # One program per (n, c, h, w)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    index = (((n * C + c) * H + h) * W + w)
    a_val = tl.load(a_ptr + index)
    b_val = tl.load(b_ptr + index)
    tl.store(out_ptr + index, a_val + b_val)


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
        """
        Fused residual block entirely in Triton:
        Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        Returns (B, C, H, W)
        """
        assert x.dim() == 4, "Input x must be (B, C, H, W)"
        B, C, H, W = x.shape
        # Ensure dtype float32 for numerical stability
        x = x.contiguous().to(torch.float32)
        conv1_weight = conv1_weight.contiguous().to(torch.float32)
        conv2_weight = conv2_weight.contiguous().to(torch.float32)

        # GroupNorm requires C % 32 == 0
        num_groups = 32
        if C % num_groups != 0:
            raise ValueError(f"GroupNorm requires C % num_groups == 0, got C={C}, num_groups={num_groups}")

        # 1) Conv1 (Triton)
        y1 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        grid_conv1 = (B, triton.cdiv(C, 64))  # tile of 64 output channels per program
        conv3x3_stride1_pad1_kernel[grid_conv1](
            x, conv1_weight, y1, B, C, H, W, C,
            BLOCK_OC=64,
            num_warps=4, num_stages=2,
        )

        # 2) GroupNorm1 (Triton)
        y2 = torch.empty_like(y1)
        grid_gn1 = (B, num_groups)
        group_norm_affine_kernel[grid_gn1](
            y1, norm1_weight, norm1_bias, y2,
            B, C, H, W,
            num_groups=32, eps=self.eps,
            num_warps=4, num_stages=2,
        )

        # 3) SiLU1 (Triton)
        y3 = torch.empty_like(y2)
        grid_silu1 = (B, C, H, W)
        silu_kernel[grid_silu1](
            y2, y3,
            B, C, H, W,
            num_warps=1, num_stages=1,
        )

        # 4) Conv2 (Triton)
        y4 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        grid_conv2 = (B, triton.cdiv(C, 64))
        conv3x3_stride1_pad1_kernel[grid_conv2](
            y3, conv2_weight, y4, B, C, H, W, C,
            BLOCK_OC=64,
            num_warps=4, num_stages=2,
        )

        # 5) GroupNorm2 (Triton)
        y5 = torch.empty_like(y4)
        group_norm_affine_kernel[grid_gn1](
            y4, norm2_weight, norm2_bias, y5,
            B, C, H, W,
            num_groups=32, eps=self.eps,
            num_warps=4, num_stages=2,
        )

        # 6) SiLU2 (Triton)
        y6 = torch.empty_like(y5)
        silu_kernel[grid_silu1](
            y5, y6,
            B, C, H, W,
            num_warps=1, num_stages=1,
        )

        # 7) Add residual (Triton)
        y_out = torch.empty_like(y6)
        add_kernel[grid_silu1](
            y6, x, y_out,
            B, C, H, W,
            num_warps=1, num_stages=1,
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
