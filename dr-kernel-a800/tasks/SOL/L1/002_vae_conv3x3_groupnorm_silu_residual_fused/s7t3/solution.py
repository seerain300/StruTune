import torch
import triton
import triton.language as tl

# Conv2D 3x3, stride=1, padding=1, bias=None
@triton.jit
def conv3x3_stride1_pad1_biasNone(
    x_ptr,              # *f16 or *f32 input: [N, C_in, H, W]
    w_ptr,              # *f16 or *f32 weight: [C_out, C_in, 3, 3]
    y_ptr,              # *f16 or *f32 output: [N, C_out, H, W]
    N, C_in, H, W, C_out,
    BLOCK_OC: tl.constexpr,
):
    # Grid: (N, ceil_div(C_out, BLOCK_OC))
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    # Accumulator for each output channel in the tile
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel taps
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                # Loop over all output spatial positions
                OHW = H * W
                for p in range(OHW):
                    oh = p // W
                    ow = p % W
                    ih = oh + kh - 1
                    iw = ow + kw - 1
                    valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                    # Input linear index: ((n * C_in + cin) * H + ih) * W + iw
                    in_index = ((n * C_in + cin) * H + ih) * W + iw
                    x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0)
                    # Weight linear index: ((oc * C_in + cin) * 9) + (kh * 3 + kw)
                    for j in range(BLOCK_OC):
                        if oc_mask[oc_offsets[j]]:
                            w_index = ((oc_offsets[j] * C_in + cin) * 9) + (kh * 3 + kw)
                            w_val = tl.load(w_ptr + w_index)
                            acc[j] += x_val * w_val

    # Store results: y[n, oc, h, w] for all positions
    # We store using flattened index by writing per (oh, ow)
    for p in range(OHW):
        oh = p // W
        ow = p % W
        base = n * C_out * OHW + oh * W + ow  # per (n, oh, ow) row base
        for j in range(BLOCK_OC):
            if oc_mask[oc_offsets[j]]:
                y_index = base + oc_offsets[j]
                tl.store(y_ptr + y_index, acc[j])


# GroupNorm with affine, num_groups = 32, per sample n
@triton.jit
def group_norm_affine(
    y_in_ptr,           # *f16 or *f32 input after conv: [N, C, H, W]
    weight_ptr,         # *f16 or *f32 per-channel scale: [C]
    bias_ptr,           # *f16 or *f32 per-channel bias: [C]
    y_out_ptr,          # *f16 or *f32 output normalized + affine: [N, C, H, W]
    N, C, H, W,
    num_groups: tl.constexpr,  # 32
    eps: tl.constexpr,
):
    # Grid: (N, num_groups)
    n = tl.program_id(0)
    g = tl.program_id(1)

    # Elements per group
    channels_in_group = C // num_groups
    group_size = channels_in_group * H * W

    # First pass: compute sum and sumsq per group
    total_sum = tl.zeros((), dtype=tl.float32)
    total_sumsq = tl.zeros((), dtype=tl.float32)

    for c_local in range(channels_in_group):
        c = g * channels_in_group + c_local
        base = (n * C + c) * (H * W)
        for p in range(H * W):
            lin = base + p
            x = tl.load(y_in_ptr + lin)
            total_sum += x
            total_sumsq += x * x

    mean = total_sum / group_size
    var = total_sumsq / group_size - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for c_local in range(channels_in_group):
        c = g * channels_in_group + c_local
        base = (n * C + c) * (H * W)
        scale = tl.load(weight_ptr + c)
        beta = tl.load(bias_ptr + c)
        for p in range(H * W):
            lin = base + p
            x = tl.load(y_in_ptr + lin)
            y_norm = (x - mean) * inv_std
            y = y_norm * scale + beta
            tl.store(y_out_ptr + lin, y)


# SiLU activation: y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
@triton.jit
def silu(
    x_ptr,              # *f16 or *f32 input
    y_ptr,              # *f16 or *f32 output
    N, C, H, W,
):
    # Grid: (N, C, H, W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    base = (n * C + c) * (H * W) + h * W + w
    x = tl.load(x_ptr + base)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + base, y)


# Elementwise residual add: y = y + x
@triton.jit
def add_residual(
    y_ptr,              # *f16 or *f32 output to which residual will be added
    x_ptr,              # *f16 or *f32 residual input
    N, C, H, W,
):
    # Grid: (N, C, H, W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    base = (n * C + c) * (H * W) + h * W + w
    y = tl.load(y_ptr + base)
    x = tl.load(x_ptr + base)
    y = y + x
    tl.store(y_ptr + base, y)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        All computation is done by Triton kernels. No torch ops are used in forward.
        """
        # Ensure CUDA tensors
        assert x.is_cuda and conv1_weight.is_cuda and norm1_weight.is_cuda and norm1_bias.is_cuda \
               and conv2_weight.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "All tensors must be on CUDA."
        # Enforce GroupNorm constraint: C % 32 == 0
        num_groups = 32
        if x.shape[1] % num_groups != 0:
            raise ValueError(f"GroupNorm requires channels divisible by num_groups. Got C={x.shape[1]}, num_groups={num_groups}.")

        N, C, H, W = x.shape
        C_in1 = C  # input channels of conv1 equal output channels (conv1_weight: [C, C, 3, 3])
        C_out1 = C

        # 1) Conv1 (Triton)
        y1 = torch.empty((N, C_out1, H, W), device=x.device, dtype=x.dtype)
        BLOCK_OC1 = 16
        grid1 = (N, (C_out1 + BLOCK_OC1 - 1) // BLOCK_OC1)
        conv3x3_stride1_pad1_biasNone[grid1](
            x, conv1_weight, y1, N, C_in1, H, W, C_out1,
            BLOCK_OC=BLOCK_OC1,
            num_warps=4,
            num_stages=2,
        )

        # 2) GroupNorm1 (Triton)
        y2 = torch.empty_like(y1)
        grid_gn1 = (N, num_groups)
        group_norm_affine[grid_gn1](
            y1, norm1_weight, norm1_bias, y2, N, C, H, W,
            num_groups=32, eps=eps,
            num_warps=4,
            num_stages=2,
        )

        # 3) SiLU1 (Triton)
        y3 = torch.empty_like(y2)
        grid_silu1 = (N, C, H, W)
        silu[grid_silu1](y2, y3, N, C, H, W,
                         num_warps=4,
                         num_stages=2,
                         )

        # 4) Conv2 (Triton)
        C_in2 = C_out1  # from conv1 output
        C_out2 = C  # output channels equal input channels (conv2_weight: [C, C, 3, 3])
        y4 = torch.empty((N, C_out2, H, W), device=x.device, dtype=x.dtype)
        conv3x3_stride1_pad1_biasNone[grid1](
            y3, conv2_weight, y4, N, C_in2, H, W, C_out2,
            BLOCK_OC=BLOCK_OC1,
            num_warps=4,
            num_stages=2,
        )

        # 5) GroupNorm2 (Triton)
        y5 = torch.empty_like(y4)
        group_norm_affine[grid_gn1](
            y4, norm2_weight, norm2_bias, y5, N, C, H, W,
            num_groups=32, eps=eps,
            num_warps=4,
            num_stages=2,
        )

        # 6) SiLU2 (Triton)
        y6 = torch.empty_like(y5)
        silu[grid_silu1](y5, y6, N, C, H, W,
                         num_warps=4,
                         num_stages=2,
                         )

        # 7) Add residual (Triton)
        y_out = torch.empty_like(y6)
        add_residual[grid_silu1](y6, x, N, C, H, W,
                                 num_warps=4,
                                 num_stages=2,
                                 )

        return y_out


def run(*args):
    return ModelNew()(*args)
