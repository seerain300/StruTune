import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv, stride=1, pad=1, bias=None
# x: [B, C_in, H, W] input
# w: [C_out, C_in, 3, 3] weights
# y: [B, C_out, H, W] output
@triton.jit
def conv3x3_stride1_pad1_kernel(
    x_ptr,          # *fp32
    w_ptr,          # *fp32
    y_ptr,          # *fp32
    N: tl.int32,
    C_in: tl.int32,
    H: tl.int32,
    W: tl.int32,
    C_out: tl.int32,
    BLOCK_OC: tl.constexpr,  # tile of output channels per program (e.g., 8)
):
    # Grid: (N, ceil_div(C_out, BLOCK_OC))
    pid_n = tl.program_id(0)  # batch index
    pid_oc = tl.program_id(1) # oc tile id

    oc_start = pid_oc * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)

    # Accumulator for this (n, oc_tile)
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for cin in range(0, C_in):
        for kh in range(3):
            for kw in range(3):
                # For each output spatial position (h, w)
                for oh in range(0, H):
                    ih = oh + kh - 1  # pad=1
                    for ow in range(0, W):
                        iw = ow + kw - 1
                        valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                        x_index = (((pid_n * C_in + cin) * H + ih) * W + iw)
                        x_val = tl.load(x_ptr + x_index, mask=valid, other=0.0)
                        # load weight vector for this (cin, kh, kw) across oc tile
                        # weight layout: [C_out, C_in, 3, 3] -> w[oc, cin, kh, kw]
                        for j in range(0, BLOCK_OC):
                            w_index = (oc_offsets[j] * C_in + cin) * 9 + (kh * 3 + kw)
                            w_val = tl.load(w_ptr + w_index)
                            acc[j] += x_val * w_val

    # Store results to y[n, oc, :, :] for all oc in tile
    # y_ptr is contiguous: offset = n*C_out*H*W + oc*C_in*H*W + h*W + w
    for j in range(0, BLOCK_OC):
        oc = oc_offsets[j]
        # write acc[j] to all positions y[n, oc, :, :]
        for oh in range(0, H):
            for ow in range(0, W):
                y_index = (pid_n * C_out + oc) * H * W + oh * W + ow
                tl.store(y_ptr + y_index, acc[j])


# Triton kernel: GroupNorm with num_groups=32, per-channel affine
# y_in: [B, C, H, W] input to normalize
# scale: [C] GroupNorm weight
# bias: [C] GroupNorm bias
# y_out: [B, C, H, W] output
@triton.jit
def group_norm_affine_kernel(
    y_in_ptr,       # *fp32
    scale_ptr,      # *fp32, length C
    bias_ptr,       # *fp32, length C
    y_out_ptr,      # *fp32
    N: tl.int32,
    C: tl.int32,
    H: tl.int32,
    W: tl.int32,
    num_groups: tl.int32,    # must be 32
    eps: tl.float32,
):
    # Grid: (N, num_groups)
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    # Accumulate sum and sum of squares over the group across all spatial H*W
    sum_val = 0.0
    sum_sq = 0.0
    for ch in range(0, channels_per_group):
        c = group_start + ch
        for h in range(0, H):
            for w in range(0, W):
                y_index = (n * C + c) * H * W + h * W + w
                val = tl.load(y_in_ptr + y_index)
                sum_val += val
                sum_sq += val * val

    size = channels_per_group * H * W
    mean = sum_val / size
    var = sum_sq / size - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine, then write to output
    for ch in range(0, channels_per_group):
        c = group_start + ch
        scale = tl.load(scale_ptr + c)
        bias = tl.load(bias_ptr + c)
        for h in range(0, H):
            for w in range(0, W):
                y_index = (n * C + c) * H * W + h * W + w
                val = tl.load(y_in_ptr + y_index)
                norm_val = (val - mean) * inv_std
                out = norm_val * scale + bias
                tl.store(y_out_ptr + y_index, out)


# Triton kernel: SiLU activation (elementwise)
@triton.jit
def silu_kernel(
    x_ptr,          # *fp32
    y_ptr,          # *fp32
    N: tl.int32,
    C: tl.int32,
    H: tl.int32,
    W: tl.int32,
):
    # Grid: (N, C, H, W) one program per element
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    x_index = (n * C + c) * H * W + h * W + w
    x_val = tl.load(x_ptr + x_index)
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    tl.store(y_ptr + x_index, y_val)


# Triton kernel: elementwise add (y += x), y is output, x is input
@triton.jit
def add_residual_kernel(
    y_ptr,          # *fp32, input tensor to which we add residual
    x_ptr,          # *fp32, residual tensor
    N: tl.int32,
    C: tl.int32,
    H: tl.int32,
    W: tl.int32,
):
    # Grid: (N, C, H, W) one program per element
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    y_index = (n * C + c) * H * W + h * W + w
    y_val = tl.load(y_ptr + y_index)
    x_val = tl.load(x_ptr + y_index)  # residual is x
    out = y_val + x_val
    tl.store(y_ptr + y_index, out)


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
        Triton-only fused residual block:
        Conv3x3 -> GroupNorm(32) -> SiLU -> Conv3x3 -> GroupNorm(32) -> SiLU -> Add residual
        """
        # Ensure CUDA and contiguous
        assert x.is_cuda and conv1_weight.is_cuda and conv2_weight.is_cuda, "All tensors must be on CUDA"
        x = x.contiguous()
        # Cast to float32 for numerical stability and Triton compatibility
        x = x.float()
        conv1_weight = conv1_weight.contiguous().float()
        conv2_weight = conv2_weight.contiguous().float()

        N, C, H, W = x.shape
        C1_in = C
        C1_out = conv1_weight.shape[0]  # C_out for first conv
        C2_in = C1_out
        C2_out = conv2_weight.shape[0]  # C_out for second conv

        # 1) Conv1: Triton kernel
        y1 = torch.empty((N, C1_out, H, W), device=x.device, dtype=torch.float32)
        BLOCK_OC = 8
        grid_conv1 = (N, triton.cdiv(C1_out, BLOCK_OC))
        conv3x3_stride1_pad1_kernel[grid_conv1](
            x, conv1_weight, y1,
            N, C1_in, H, W, C1_out,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        # 2) GroupNorm1 (num_groups=32) with affine
        if C1_out % 32 != 0:
            raise ValueError(f"GroupNorm requires channels divisible by num_groups=32, got C={C1_out}")
        y1_norm = torch.empty_like(y1)
        grid_gn1 = (N, 32)
        group_norm_affine_kernel[grid_gn1](
            y1, norm1_weight.float(), norm1_bias.float(), y1_norm,
            N, C1_out, H, W, 32, eps,
            num_warps=4,
            num_stages=2,
        )

        # 3) SiLU1
        y1_silu = torch.empty_like(y1_norm)
        grid_silu1 = (N, C1_out, H, W)
        silu_kernel[grid_silu1](
            y1_norm, y1_silu,
            N, C1_out, H, W,
            num_warps=4,
            num_stages=2,
        )

        # 4) Conv2: Triton kernel
        y2 = torch.empty((N, C2_out, H, W), device=x.device, dtype=torch.float32)
        grid_conv2 = (N, triton.cdiv(C2_out, BLOCK_OC))
        conv3x3_stride1_pad1_kernel[grid_conv2](
            y1_silu, conv2_weight, y2,
            N, C2_in, H, W, C2_out,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        # 5) GroupNorm2 (num_groups=32) with affine
        if C2_out % 32 != 0:
            raise ValueError(f"GroupNorm requires channels divisible by num_groups=32, got C={C2_out}")
        y2_norm = torch.empty_like(y2)
        grid_gn2 = (N, 32)
        group_norm_affine_kernel[grid_gn2](
            y2, norm2_weight.float(), norm2_bias.float(), y2_norm,
            N, C2_out, H, W, 32, eps,
            num_warps=4,
            num_stages=2,
        )

        # 6) SiLU2
        y2_silu = torch.empty_like(y2_norm)
        grid_silu2 = (N, C2_out, H, W)
        silu_kernel[grid_silu2](
            y2_norm, y2_silu,
            N, C2_out, H, W,
            num_warps=4,
            num_stages=2,
        )

        # 7) Add residual x
        y_out = torch.empty_like(y2_silu)
        grid_add = (N, C2_out, H, W)
        add_residual_kernel[grid_add](
            y2_silu, x.float(),  # residual is original input x
            N, C2_out, H, W,
            num_warps=4,
            num_stages=2,
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
