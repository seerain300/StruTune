import torch
import triton
import triton.language as tl


# Conv3x3 stride=1, padding=1, bias=None, compute one output element per program.
# x: [B, C_in, H, W] float32 contiguous
# w: [C_out, C_in, 3, 3] float32 contiguous (row-major)
# out: [B, C_out, H_out, W_out] float32 contiguous
@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,        # *const float32
    w_ptr,        # *const float32
    out_ptr,      # *float32
    B, C_in, H, W, C_out, H_out, W_out,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = pid_h - 1 + kh
                w_in = pid_w - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                # Flattened indexing for x: ((n * C_in + ci) * H + h_in) * W + w_in
                x_index = (pid_n * C_in + ci) * H * W + h_in * W + w_in
                x_val = tl.load(x_ptr + x_index, mask=in_bounds, other=0.0)
                # weights [C_out, C_in, 3, 3] contiguous; flatten to [C_out, C_in*9]
                w_index = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_index)
                acc += x_val * w_val

    out_index = (pid_n * C_out + pid_co) * H_out * W_out + pid_h * W_out + pid_w
    tl.store(out_ptr + out_index, acc)


# GroupNorm per (n, group): normalize across channels in the group and all spatial locations.
# Assumes out has shape [B, C, H, W], num_groups divides C.
# We normalize over channels [group*channels_per_group : (group+1)*channels_per_group] and all H*W spatial.
@triton.jit
def group_norm_two_pass(out_ptr, gamma_ptr, beta_ptr, out_norm_ptr,
                         B, C, H, W, num_groups, eps,
                         out_stride_n, out_stride_c, out_stride_h, out_stride_w):
    # Grid is (B, num_groups)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    channels_per_group = C // num_groups
    group_start = pid_g * channels_per_group

    # First pass: compute sum and sum of squares across group channels and spatial
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for ch in tl.static_range(channels_per_group):
        c = group_start + ch
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                ptr = out_ptr + pid_n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w
                x = tl.load(ptr)
                sum_val += x
                sum_sq += x * x

    N = channels_per_group * H * W
    mean = sum_val / N
    var = sum_sq / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine gamma/beta, store to out_norm
    for ch in tl.static_range(channels_per_group):
        c = group_start + ch
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                ptr = out_ptr + pid_n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w
                x = tl.load(ptr)
                y = ((x - mean) * inv_std) * gamma + beta
                out_norm_ptr_index = pid_n * (C * H * W) + c * (H * W) + h * W + w
                tl.store(out_norm_ptr + out_norm_ptr_index, y)


# Elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def silu_kernel(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(out_ptr + offs, y, mask=mask)


# Elementwise add residual: out = y + x
@triton.jit
def add_residual_kernel(y_ptr, x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = y + x
    tl.store(out_ptr + offs, out, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor) -> torch.Tensor:
        # Ensure float32 and contiguous
        x = x.contiguous().to(torch.float32)
        device = x.device
        B, C, H, W = x.shape

        # First conv: y1 = conv3x3(x)
        y1 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        conv3x3_nobias_one_elem[(B, C, H, W)](
            x, conv1_weight, y1,
            B, C, H, W, C, H, W
        )

        # First GroupNorm: y1_gn = GroupNorm(num_groups=32) on y1
        y1_gn = torch.empty_like(y1)
        grid_gn1 = (B, self.num_groups)
        # We need channels_per_group = C // num_groups. For correctness, assume C % self.num_groups == 0.
        channels_per_group = C // self.num_groups
        group_norm_two_pass[grid_gn1](
            y1, norm1_weight, norm1_bias, y1_gn,
            B, C, H, W, self.num_groups, self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        )

        # First SiLU
        y1_silu = torch.empty_like(y1_gn)
        N1 = y1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](y1_gn, y1_silu, N1, BLOCK=1024)

        # Second conv: y2_pre = conv3x3(y1_silu)
        y2_pre = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        conv3x3_nobias_one_elem[(B, C, H, W)](
            y1_silu, conv2_weight, y2_pre,
            B, C, H, W, C, H, W
        )

        # Second GroupNorm
        y2_gn = torch.empty_like(y2_pre)
        channels_per_group2 = C // self.num_groups
        group_norm_two_pass[(B, self.num_groups)](
            y2_pre, norm2_weight, norm2_bias, y2_gn,
            B, C, H, W, self.num_groups, self.eps,
            y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
        )

        # Second SiLU
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


def run(*args):
    return ModelNew()(*args)
