import torch
import torch.nn.functional as F
import triton
import triton.language as tl


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


def triton_silu(y: torch.Tensor) -> torch.Tensor:
    # Ensure contiguous for linear indexing
    y = y.contiguous()
    n_elements = y.numel()
    y_out = torch.empty_like(y)
    BLOCK = 1024
    grid = (triton.cdiv(n_elements, BLOCK),)
    silu_kernel[grid](y, y_out, n_elements, BLOCK=BLOCK)
    return y_out


# Triton kernel for GroupNorm: per-sample, per-group reduction + normalization + affine
# Assumes y_in is NCHW contiguous; num_groups divides C.
@triton.jit
def group_norm_triton(
    y_in_ptr,     # *float or *half input after conv
    y_out_ptr,    # *float or *half output
    weight_ptr,   # *float per-channel scale (C,)
    bias_ptr,     # *float per-channel bias (C,)
    N,            # int
    C,            # int (channels)
    H,            # int
    W,            # int
    num_groups,   # int
    eps,          # float
    BLOCK: tl.constexpr,  # chunk size for loops
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
        off = 0
        while off < group_size_hw:
            idx = off + tl.arange(0, BLOCK)
            mask = idx < group_size_hw
            lin = base + idx
            x = tl.load(y_in_ptr + lin, mask=mask, other=0.0)
            total_sum += tl.sum(x, axis=0)
            total_sumsq += tl.sum(x * x, axis=0)
            off += BLOCK

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
            idx = off + tl.arange(0, BLOCK)
            mask = idx < group_size_hw
            lin = base + idx
            x = tl.load(y_in_ptr + lin, mask=mask, other=0.0)
            y_norm = (x - mean) * inv_std
            y = y_norm * scale + beta
            tl.store(y_out_ptr + lin, y, mask=mask)
            off += BLOCK


class ModelNew(torch.nn.Module):
    def __init__(self, conv1_weight, conv2_weight, norm1_weight, norm1_bias, norm2_weight, norm2_bias, eps):
        super().__init__()
        # Register weights as buffers (no gradients required)
        self.register_buffer('conv1_weight', conv1_weight)
        self.register_buffer('conv2_weight', conv2_weight)
        self.register_buffer('norm1_weight', norm1_weight)
        self.register_buffer('norm1_bias', norm1_bias)
        self.register_buffer('norm2_weight', norm2_weight)
        self.register_buffer('norm2_bias', norm2_bias)
        self.eps = eps
        self.num_groups = 32  # hardcoded as in original

    def forward(self, x: torch.Tensor):
        # Ensure CUDA tensors and contiguous layout
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."

        # Stage 1: Conv1
        # x: (N, C, H, W)
        out1 = F.conv2d(x, self.conv1_weight, bias=None, stride=1, padding=1)

        # GroupNorm 1
        C_out1 = out1.shape[1]
        if C_out1 % self.num_groups != 0:
            raise ValueError(f"GroupNorm requires channels divisible by num_groups. Got C_out1={C_out1}, num_groups={self.num_groups}.")
        y1 = torch.empty_like(out1)
        group_norm_triton[(out1.shape[0], self.num_groups)](
            out1, y1, self.norm1_weight, self.norm1_bias,
            out1.shape[0], C_out1, out1.shape[2], out1.shape[3], self.num_groups, self.eps,
            BLOCK=1024
        )

        # SiLU 1 (Triton)
        y1 = triton_silu(y1)

        # Save residual
        residual = x

        # Stage 2: Conv2
        out2 = F.conv2d(y1, self.conv2_weight, bias=None, stride=1, padding=1)

        # GroupNorm 2
        C_out2 = out2.shape[1]
        if C_out2 % self.num_groups != 0:
            raise ValueError(f"GroupNorm requires channels divisible by num_groups. Got C_out2={C_out2}, num_groups={self.num_groups}.")
        y2 = torch.empty_like(out2)
        group_norm_triton[(out2.shape[0], self.num_groups)](
            out2, y2, self.norm2_weight, self.norm2_bias,
            out2.shape[0], C_out2, out2.shape[2], out2.shape[3], self.num_groups, self.eps,
            BLOCK=1024
        )

        # SiLU 2 (Triton)
        y2 = triton_silu(y2)

        # Residual add
        y2 = y2 + residual

        return y2


def run(*args):
    return ModelNew()(*args)
