import torch
import triton
import triton.language as tl


# Simple conv3x3 (stride=1, padding=1, no bias) per output element.
# One Triton program computes one output element (n, co, h_out, w_out).
@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,        # *const float32, input [B, C_in, H, W]
    w_ptr,        # *const float32, weights [C_out, C_in, 3, 3]
    out_ptr,      # *float32, output [B, C_out, H_out, W_out]
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    h_out = pid_h
    w_out = pid_w

    acc = tl.zeros((), dtype=tl.float32)

    # sum over input channels and 3x3 kernel
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = h_out - 1 + kh  # padding=1
                w_in = w_out - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                # x index: n * stride_n + ci * stride_c + h_in * stride_h + w_in * stride_w
                base_x = pid_n * x_stride_n + ci * x_stride_c
                ptr_x = x_ptr + base_x + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(ptr_x, mask=in_bounds, other=0.0).to(tl.float32)
                # weight index: (co * (C_in * 9)) + (ci * 9) + (kh * 3 + kw)
                w_idx = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx).to(tl.float32)
                acc += x_val * w_val

    # store result
    base_out = pid_n * out_stride_n + pid_co * out_stride_c
    ptr_out = out_ptr + base_out + h_out * out_stride_h + w_out * out_stride_w
    tl.store(ptr_out, acc)


# GroupNorm two-pass kernel: one program per (batch, group).
# Pass 1: compute sum and sum of squares over elements in the group.
@triton.jit
def group_norm_reduce(
    inp_ptr,           # *const float32, input tensor
    sum_ptr,           # *float32, [B * num_groups]
    sqsum_ptr,         # *float32, [B * num_groups]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
    inp_stride_n: tl.constexpr, inp_stride_c: tl.constexpr, inp_stride_h: tl.constexpr, inp_stride_w: tl.constexpr,
):
    pid = tl.program_id(0)  # ranges over B * num_groups
    n = pid // num_groups
    g = pid % num_groups
    group_size = C // num_groups
    start_c = g * group_size

    total_elems = group_size * H * W
    sum_val = tl.zeros((), dtype=tl.float32)
    sqsum_val = tl.zeros((), dtype=tl.float32)

    # iterate over all channels in the group, all H, all W
    for ci in tl.static_range(group_size):
        c_idx = start_c + ci
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                base = n * inp_stride_n + c_idx * inp_stride_c
                ptr = inp_ptr + base + h * inp_stride_h + w * inp_stride_w
                val = tl.load(ptr).to(tl.float32)
                sum_val += val
                sqsum_val += val * val

    # write results
    out_idx = n * num_groups + g
    tl.store(sum_ptr + out_idx, sum_val)
    tl.store(sqsum_ptr + out_idx, sqsum_val)


# GroupNorm two-pass kernel: pass 2, normalize and apply affine (gamma, beta).
@triton.jit
def group_norm_apply(
    inp_ptr,           # *const float32, input tensor
    gamma_ptr,         # *const float32, per-channel gamma [C]
    beta_ptr,          # *const float32, per-channel beta  [C]
    out_ptr,           # *float32, output tensor
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
    eps: tl.constexpr,
    inp_stride_n: tl.constexpr, inp_stride_c: tl.constexpr, inp_stride_h: tl.constexpr, inp_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    pid = tl.program_id(0)  # ranges over B * num_groups
    n = pid // num_groups
    g = pid % num_groups
    group_size = C // num_groups
    start_c = g * group_size

    # read sum and sqsum from first pass
    # We'll recompute; alternatively, these are provided in PyTorch, but we do them here for simplicity.
    total_elems = group_size * H * W
    # Compute sum and sumsq in one pass over the group; reuse logic as above
    sum_val = tl.zeros((), dtype=tl.float32)
    sqsum_val = tl.zeros((), dtype=tl.float32)
    for ci in tl.static_range(group_size):
        c_idx = start_c + ci
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                base = n * inp_stride_n + c_idx * inp_stride_c
                ptr = inp_ptr + base + h * inp_stride_h + w * inp_stride_w
                val = tl.load(ptr).to(tl.float32)
                sum_val += val
                sqsum_val += val * val

    mean = sum_val / total_elems
    var = sqsum_val / total_elems - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply gamma/beta, write to out
    for ci in tl.static_range(group_size):
        c_idx = start_c + ci
        gamma = tl.load(gamma_ptr + c_idx).to(tl.float32)
        beta = tl.load(beta_ptr + c_idx).to(tl.float32)
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                base_in = n * inp_stride_n + c_idx * inp_stride_c
                ptr_in = inp_ptr + base_in + h * inp_stride_h + w * inp_stride_w
                val = tl.load(ptr_in).to(tl.float32)
                norm = (val - mean) * inv_std
                out_val = norm * gamma + beta
                base_out = n * out_stride_n + c_idx * out_stride_c
                ptr_out = out_ptr + base_out + h * out_stride_h + w * out_stride_w
                tl.store(ptr_out, out_val)


# Elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def silu_kernel(
    in_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # sigmoid(x) = 1 / (1 + exp(-x))
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(out_ptr + offsets, y, mask=mask)


# Elementwise add residual: out = in + residual
@triton.jit
def add_residual_kernel(
    in_ptr, residual_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    c = a + b
    tl.store(out_ptr + offsets, c, mask=mask)


# Constants for the module
class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        # Ensure contiguous and float32 for robustness
        device = x.device
        dtype = torch.float32
        x = x.contiguous().to(dtype)
        B, C, H, W = x.shape
        assert C % self.num_groups == 0, "C must be divisible by num_groups for GroupNorm."

        # First conv: conv3x3, stride=1, padding=1, no bias
        y1 = torch.empty((B, C, H, W), device=device, dtype=dtype)
        grid_conv1 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv1](
            x, conv1_weight.to(dtype).contiguous(), y1,
            B, C, H, W, C, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        )

        # GroupNorm1: two-pass per (batch, group)
        sum1 = torch.empty((B * self.num_groups,), device=device, dtype=dtype)
        sqsum1 = torch.empty((B * self.num_groups,), device=device, dtype=dtype)
        grid_reduce1 = (B * self.num_groups,)
        group_norm_reduce[grid_reduce1](
            y1, sum1, sqsum1,
            B, C, H, W,
            self.num_groups,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        )
        y1_gn = torch.empty_like(y1)
        grid_apply1 = (B * self.num_groups,)
        group_norm_apply[grid_apply1](
            y1, norm1_weight.to(dtype).contiguous(), norm1_bias.to(dtype).contiguous(), y1_gn,
            B, C, H, W,
            self.num_groups,
            self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_gn.stride(0), y1_gn.stride(1), y1_gn.stride(2), y1_gn.stride(3),
        )

        # SiLU1
        y1_silu = torch.empty_like(y1_gn)
        N1 = y1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](y1_gn, y1_silu, N1, BLOCK=1024)

        # Second conv: conv3x3, stride=1, padding=1, no bias
        y2_pre = torch.empty((B, C, H, W), device=device, dtype=dtype)
        grid_conv2 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv2](
            y1_silu, conv2_weight.to(dtype).contiguous(), y2_pre,
            B, C, H, W, C, H, W,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
        )

        # GroupNorm2: two-pass per (batch, group)
        sum2 = torch.empty((B * self.num_groups,), device=device, dtype=dtype)
        sqsum2 = torch.empty((B * self.num_groups,), device=device, dtype=dtype)
        grid_reduce2 = (B * self.num_groups,)
        group_norm_reduce[grid_reduce2](
            y2_pre, sum2, sqsum2,
            B, C, H, W,
            self.num_groups,
            y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
        )
        y2_gn = torch.empty_like(y2_pre)
        grid_apply2 = (B * self.num_groups,)
        group_norm_apply[grid_apply2](
            y2_pre, norm2_weight.to(dtype).contiguous(), norm2_bias.to(dtype).contiguous(), y2_gn,
            B, C, H, W,
            self.num_groups,
            self.eps,
            y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
            y2_gn.stride(0), y2_gn.stride(1), y2_gn.stride(2), y2_gn.stride(3),
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_gn)
        N2 = y2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](y2_gn, y2_silu, N2, BLOCK=1024)

        # Add residual x
        out = torch.empty_like(y2_silu)
        Nfinal = y2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[grid_add](y2_silu, x, out, Nfinal, BLOCK=1024)

        return out


# The following helper functions are not used by evaluation, but shown for completeness.
def run_triton_only(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    model = ModelNew(num_groups=32, eps=eps).to(x.device)
    return model(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias)


# Example usage:
# model = ModelNew(num_groups=32, eps=1e-5)
# out = model(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias)


def run(*args):
    return ModelNew()(*args)
