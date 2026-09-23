import torch
import triton
import triton.language as tl


# Triton conv2d 3x3, stride=1, padding=1, bias=None
# Input: x_in (N, C_in, H, W), weight (C_out, C_in, 3, 3)
# Output: y_out (N, C_out, H, W)
@triton.jit
def conv2d_3x3_stride1_pad1_kernel(
    x_ptr,        # *float or *half input tensor
    w_ptr,        # *float or *half weight tensor
    y_ptr,        # *float or *half output tensor
    N,            # int
    C_in,         # int
    H,            # int
    W,            # int
    C_out,        # int
    BLOCK_OC: tl.constexpr,  # tile size for output channels per program
):
    # Grid: (N, ceil_div(C_out, BLOCK_OC))
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    # Accumulator for this (n, oc_tile)
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Loop over input channels
    for cin in range(C_in):
        # Loop over 3x3 kernel
        for kh in range(3):
            for kw in range(3):
                # Output coordinates (oh, ow) correspond to input coordinates ih=oh+kh-1, iw=ow+kw-1
                # Since stride=1, pad=1, output dims are H and W
                for oh in range(H):
                    ih = oh + kh - 1
                    for ow in range(W):
                        iw = ow + kw - 1
                        valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                        # Input linear index: (((n * C_in + cin) * H + ih) * W + iw)
                        in_index = (((n * C_in + cin) * H + ih) * W + iw)
                        x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0)
                        # Weight linear index: (((oc * C_in + cin) * 9) + (kh * 3 + kw))
                        for j in range(BLOCK_OC):
                            if oc_mask[oc_offsets[j]]:
                                w_index = (((oc_offsets[j] * C_in + cin) * 9) + (kh * 3 + kw))
                                w_val = tl.load(w_ptr + w_index)
                                acc[j] += x_val * w_val
    # Store results: write acc[oc] into y[n, oc, :, :] across all positions
    for j in range(BLOCK_OC):
        if oc_mask[oc_offsets[j]]:
            for oh in range(H):
                for ow in range(W):
                    y_index = (((n * C_out + oc_offsets[j]) * H + oh) * W + ow)
                    tl.store(y_ptr + y_index, acc[j])


# Triton GroupNorm: per-sample, per-group reduction + normalization + affine
# Input: y_in (N, C, H, W) contiguous; num_groups must divide C
# Weight and bias are per-channel (C,), applied after normalization.
@triton.jit
def group_norm_triton(
    y_in_ptr,     # *float or *half input tensor after conv
    y_out_ptr,    # *float or *half output tensor
    weight_ptr,   # *float per-channel scale (C,)
    bias_ptr,     # *float per-channel bias (C,)
    N,            # int
    C,            # int (channels)
    H,            # int
    W,            # int
    num_groups,   # int
    eps,          # float
    BLOCK_HW: tl.constexpr,  # chunk size for spatial loops
):
    n = tl.program_id(0)   # sample index
    g = tl.program_id(1)   # group index

    channels_in_group = C // num_groups
    group_start_channel = g * channels_in_group
    group_size_hw = H * W
    elements_per_group = channels_in_group * group_size_hw

    # First pass: compute sum and sum of squares for this (n, g)
    total_sum = 0.0
    total_sumsq = 0.0

    # Iterate over channels in the group
    for c_local in range(channels_in_group):
        c = group_start_channel + c_local
        base = (n * C + c) * group_size_hw
        # Loop over spatial H*W in chunks
        off = 0
        while off < group_size_hw:
            idx = off + tl.arange(0, BLOCK_HW)
            mask = idx < group_size_hw
            lin = base + idx
            x = tl.load(y_in_ptr + lin, mask=mask, other=0.0)
            total_sum += tl.sum(x, axis=0)
            total_sumsq += tl.sum(x * x, axis=0)
            off += BLOCK_HW

    num_elems = elements_per_group
    mean = total_sum / num_elems
    var = total_sumsq / num_elems - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, write to output
    for c_local in range(channels_in_group):
        c = group_start_channel + c_local
        base = (n * C + c) * group_size_hw
        scale = tl.load(weight_ptr + c)
        beta = tl.load(bias_ptr + c)
        off = 0
        while off < group_size_hw:
            idx = off + tl.arange(0, BLOCK_HW)
            mask = idx < group_size_hw
            lin = base + idx
            x = tl.load(y_in_ptr + lin, mask=mask, other=0.0)
            y_norm = (x - mean) * inv_std
            y = y_norm * scale + beta
            tl.store(y_out_ptr + lin, y, mask=mask)
            off += BLOCK_HW


# Triton elementwise SiLU kernel: y = x * sigmoid(x)
@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, conv1_weight, conv2_weight, norm1_weight, norm1_bias, norm2_weight, norm2_bias, eps):
        super().__init__()
        # No buffers; we will receive tensors as inputs each forward
        self.eps = eps
        self.num_groups = 32  # hardcoded as in original

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        # Ensure CUDA tensors and contiguous layout
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        N, C, H, W = x.shape

        # Stage 1: Conv1 via Triton
        out1 = torch.empty((N, C, H, W), device=x.device, dtype=x.dtype)
        conv2d_3x3_stride1_pad1_kernel[(N, triton.cdiv(C, 32))](
            x, conv1_weight, out1,
            N, C, H, W, C,
            BLOCK_OC=32
        )

        # GroupNorm 1 (Triton), num_groups=32 requires C % 32 == 0
        if C % self.num_groups != 0:
            raise ValueError(f"GroupNorm requires channels divisible by num_groups. Got C={C}, num_groups={self.num_groups}.")
        y1 = torch.empty_like(out1)
        group_norm_triton[(N, self.num_groups)](
            out1, y1, norm1_weight, norm1_bias,
            N, C, H, W, self.num_groups, self.eps,
            BLOCK_HW=1024
        )

        # SiLU 1 (Triton)
        y1_silu = torch.empty_like(y1)
        silu_kernel[(triton.cdiv(y1.numel(), 1024),)](y1, y1_silu, y1.numel(), BLOCK=1024)

        # Save residual
        residual = x

        # Stage 2: Conv2 via Triton
        out2 = torch.empty((N, C, H, W), device=x.device, dtype=x.dtype)
        conv2d_3x3_stride1_pad1_kernel[(N, triton.cdiv(C, 32))](
            y1_silu, conv2_weight, out2,
            N, C, H, W, C,
            BLOCK_OC=32
        )

        # GroupNorm 2 (Triton)
        if C % self.num_groups != 0:
            raise ValueError(f"GroupNorm requires channels divisible by num_groups. Got C={C}, num_groups={self.num_groups}.")
        y2 = torch.empty_like(out2)
        group_norm_triton[(N, self.num_groups)](
            out2, y2, norm2_weight, norm2_bias,
            N, C, H, W, self.num_groups, self.eps,
            BLOCK_HW=1024
        )

        # SiLU 2 (Triton)
        y2_silu = torch.empty_like(y2)
        silu_kernel[(triton.cdiv(y2.numel(), 1024),)](y2, y2_silu, y2.numel(), BLOCK=1024)

        # Residual add
        y2_final = y2_silu + residual

        return y2_final


def run(*args):
    return ModelNew()(*args)
