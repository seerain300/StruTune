import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_kernel(
    x_ptr,            # *float32 input tensor (B, C_in, H, W)
    w_ptr,            # *float32 weights tensor (C_out, C_in, 3, 3)
    y_ptr,            # *float32 output tensor (B, C_out, H, W)
    N: tl.constexpr,  # batch size
    C_in: tl.constexpr,  # input channels
    H: tl.constexpr,  # input height
    W: tl.constexpr,  # input width
    C_out: tl.constexpr,  # output channels
    BLOCK_OC: tl.constexpr,  # tile size for output channels per program
):
    # Grid: (N, ceil_div(C_out, BLOCK_OC))
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    # Accumulator for the tile of output channels (float32)
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel taps
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                # Output spatial dimensions match input due to padding=1, stride=1
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
    x_ptr,           # *float32 input tensor (B, C, H, W)
    gamma_ptr,       # *float32 scale tensor (C,)
    beta_ptr,        # *float32 bias tensor (C,)
    y_ptr,           # *float32 output tensor (B, C, H, W)
    N: tl.constexpr, # batch size
    C: tl.constexpr, # channels
    H: tl.constexpr, # height
    W: tl.constexpr, # width
    num_groups: tl.constexpr,  # number of groups (32)
    eps: tl.constexpr,         # epsilon
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
                x_val = tl.load(x_ptr + index).to(tl.float32)
                sum_val += x_val
                sum_sq += x_val * x_val

    # Compute mean and variance
    M = channels_per_group * H * W
    mean = sum_val / M
    var = sum_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine: y = ((x - mean) * inv_std) * gamma + beta
    for c in range(group_start, group_end):
        gamma = tl.load(gamma_ptr + c).to(tl.float32)
        beta = tl.load(beta_ptr + c).to(tl.float32)
        for h in range(H):
            for w in range(W):
                index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + index).to(tl.float32)
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
def add_residual_kernel(
    y_ptr,            # *float32 input tensor (B, C, H, W), current output
    x_ptr,            # *float32 input tensor (B, C, H, W), original residual x
    out_ptr,          # *float32 output tensor (B, C, H, W)
    N: tl.constexpr,  # batch size
    C: tl.constexpr,  # channels
    H: tl.constexpr,  # height
    W: tl.constexpr,  # width
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    index = (((n * C + c) * H + h) * W + w)
    y_val = tl.load(y_ptr + index)
    x_val = tl.load(x_ptr + index)
    out_val = y_val + x_val
    tl.store(out_ptr + index, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        # Ensure tensors are CUDA and contiguous
        assert x.is_cuda and conv1_weight.is_cuda and norm1_weight.is_cuda and norm1_bias.is_cuda and \
               conv2_weight.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "All tensors must be CUDA."
        # Enforce dtype float32 for computation (Triton kernels assume float32)
        x = x.contiguous().to(torch.float32)
        conv1_weight = conv1_weight.contiguous().to(torch.float32)
        conv2_weight = conv2_weight.contiguous().to(torch.float32)
        norm1_weight = norm1_weight.contiguous().to(torch.float32)
        norm1_bias = norm1_bias.contiguous().to(torch.float32)
        norm2_weight = norm2_weight.contiguous().to(torch.float32)
        norm2_bias = norm2_bias.contiguous().to(torch.float32)

        B, C, H, W = x.shape
        C_in1 = conv1_weight.shape[1]  # (C_out, C_in, 3, 3)
        C_out1 = conv1_weight.shape[0]

        # First conv: (B, C_in1, H, W) -> (B, C_out1, H, W)
        y1 = torch.empty((B, C_out1, H, W), dtype=torch.float32, device=x.device)
        BLOCK_OC = 32
        grid_conv1 = (B, triton.cdiv(C_out1, BLOCK_OC))
        conv3x3_stride1_pad1_kernel[grid_conv1](
            x, conv1_weight, y1,
            N=B, C_in=C_in1, H=H, W=W, C_out=C_out1,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm1 (num_groups=32), affine
        if C_out1 % 32 != 0:
            raise ValueError(f"GroupNorm requires C % num_groups == 0, got C_out1={C_out1}, num_groups=32.")
        y2 = torch.empty_like(y1, dtype=torch.float32)
        grid_gn1 = (B, 32)
        group_norm_affine_kernel[grid_gn1](
            y1, norm1_weight, norm1_bias, y2,
            N=B, C=C_out1, H=H, W=W,
            num_groups=32,
            eps=float(eps),
            num_warps=4,
            num_stages=2,
        )

        # SiLU1
        y3 = torch.empty_like(y2, dtype=torch.float32)
        grid_silu1 = (B, C_out1, H, W)
        silu_kernel[grid_silu1](
            y2, y3,
            N=B, C=C_out1, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # Second conv: (B, C_out1, H, W) -> (B, C_out2, H, W)
        C_out2 = conv2_weight.shape[0]
        y4 = torch.empty((B, C_out2, H, W), dtype=torch.float32, device=x.device)
        grid_conv2 = (B, triton.cdiv(C_out2, BLOCK_OC))
        conv3x3_stride1_pad1_kernel[grid_conv2](
            y3, conv2_weight, y4,
            N=B, C_in=C_out1, H=H, W=W, C_out=C_out2,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm2 (num_groups=32), affine
        if C_out2 % 32 != 0:
            raise ValueError(f"GroupNorm requires C % num_groups == 0, got C_out2={C_out2}, num_groups=32.")
        y5 = torch.empty_like(y4, dtype=torch.float32)
        grid_gn2 = (B, 32)
        group_norm_affine_kernel[grid_gn2](
            y4, norm2_weight, norm2_bias, y5,
            N=B, C=C_out2, H=H, W=W,
            num_groups=32,
            eps=float(eps),
            num_warps=4,
            num_stages=2,
        )

        # SiLU2
        y6 = torch.empty_like(y5, dtype=torch.float32)
        grid_silu2 = (B, C_out2, H, W)
        silu_kernel[grid_silu2](
            y5, y6,
            N=B, C=C_out2, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # Add residual: original input x (shape (B, C, H, W)) should be added. Note: original x has C channels, final y6 has C_out2 channels.
        # The original PyTorch code adds the residual input x to the final output after both paths, where x has the same channel count as final output.
        # Ensure channel counts match.
        if C != C_out2:
            raise ValueError(f"Residual add requires matching channel counts; got x.C={C}, conv2 output.C={C_out2}.")
        y_out = torch.empty_like(x, dtype=torch.float32)
        grid_add = (B, C, H, W)
        add_residual_kernel[grid_add](
            y6, x, y_out,
            N=B, C=C, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
