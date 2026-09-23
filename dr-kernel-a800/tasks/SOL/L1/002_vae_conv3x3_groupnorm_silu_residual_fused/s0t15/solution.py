import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,        # *const float, input [B, C_in, H, W]
    w_ptr,        # *const float, weights [C_out, C_in, 3, 3] flattened as [C_out, C_in*9]
    out_ptr,      # *float, output [B, C_out, H, W]
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    # program ids: batch, output channel, output spatial location
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    h_out = pid_h
    w_out = pid_w

    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and 3x3 kernel with padding=1
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = h_out - 1 + kh
                w_in = w_out - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                base_x = pid_n * x_stride_n + ci * x_stride_c
                ptr_x = x_ptr + base_x + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(ptr_x, mask=in_bounds, other=0.0).to(tl.float32)
                # weight index: (co * (C_in * 9)) + (ci * 9) + (kh * 3 + kw)
                w_idx = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx).to(tl.float32)
                acc += x_val * w_val

    # store result at (n, co, h_out, w_out)
    ptr_out = out_ptr + pid_n * out_stride_n + pid_co * out_stride_c + h_out * out_stride_h + w_out * out_stride_w
    tl.store(ptr_out, acc)


@triton.jit
def group_norm_two_pass(
    out_ptr,         # *const float, input tensor after conv, shape [B, C, H, W]
    gamma_ptr,       # *const float, per-channel gamma (weight) [C]
    beta_ptr,        # *const float, per-channel beta (bias) [C]
    out_norm_ptr,    # *float, output normalized + affine [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, eps: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    # Pass 1: compute sum and sum of squares per (n, group)
    group_size = C // num_groups
    for n in tl.static_range(B):
        for group in tl.static_range(num_groups):
            sum_val = tl.zeros((), dtype=tl.float32)
            sum_sq = tl.zeros((), dtype=tl.float32)
            # iterate over channels in this group and all spatial positions
            for c_off in tl.static_range(C):
                if (c_off % num_groups) == group:
                    for h in tl.static_range(H):
                        for w in tl.static_range(W):
                            ptr = out_ptr + n * out_stride_n + c_off * out_stride_c + h * out_stride_h + w * out_stride_w
                            x_val = tl.load(ptr).to(tl.float32)
                            sum_val += x_val
                            sum_sq += x_val * x_val
            M = group_size * H * W
            mean = sum_val / M
            var = sum_sq / M - mean * mean
            rstd = 1.0 / tl.sqrt(var + eps)

            # Pass 2: normalize and apply affine, then store
            for c_off in tl.static_range(C):
                if (c_off % num_groups) == group:
                    gamma = tl.load(gamma_ptr + c_off).to(tl.float32)
                    beta = tl.load(beta_ptr + c_off).to(tl.float32)
                    for h in tl.static_range(H):
                        for w in tl.static_range(W):
                            ptr = out_ptr + n * out_stride_n + c_off * out_stride_c + h * out_stride_h + w * out_stride_w
                            x_val = tl.load(ptr).to(tl.float32)
                            y = (x_val - mean) * rstd
                            y = y * gamma + beta
                            out_ptr2 = out_norm_ptr + n * out_stride_n + c_off * out_stride_c + h * out_stride_h + w * out_stride_w
                            tl.store(out_ptr2, y)


@triton.jit
def silu_kernel(in_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # SiLU: x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def add_residual_kernel(in_ptr, res_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(in_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(res_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a + b
    tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        # Ensure dtype float32 and contiguous
        device = x.device
        x_f = x.contiguous().to(torch.float32)
        B, C, H, W = x_f.shape
        H_out = H
        W_out = W

        # First conv
        out1 = torch.empty((B, C, H_out, W_out), device=device, dtype=torch.float32)
        grid_conv1 = (B, C, H_out, W_out)
        conv3x3_nobias_one_elem[grid_conv1](
            x_f, conv1_weight.contiguous().to(torch.float32), out1,
            B, C, H, W, C,
            x_f.stride(0), x_f.stride(1), x_f.stride(2), x_f.stride(3),
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        )

        # GroupNorm1 (num_groups=32)
        out1_norm = torch.empty_like(out1)
        grid_gn1 = (B, self.num_groups)
        group_norm_two_pass[grid_gn1](
            out1, norm1_weight.contiguous().to(torch.float32), norm1_bias.contiguous().to(torch.float32), out1_norm,
            B, C, H_out, W_out, self.num_groups, self.eps,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        )

        # SiLU1
        out1_silu = torch.empty_like(out1_norm)
        N1 = out1_norm.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_norm, out1_silu, N1, BLOCK=1024)

        # Second conv
        out2_pre = torch.empty((B, C, H_out, W_out), device=device, dtype=torch.float32)
        grid_conv2 = (B, C, H_out, W_out)
        conv3x3_nobias_one_elem[grid_conv2](
            out1_silu, conv2_weight.contiguous().to(torch.float32), out2_pre,
            B, C, H_out, W_out, C,
            out1_silu.stride(0), out1_silu.stride(1), out1_silu.stride(2), out1_silu.stride(3),
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )

        # GroupNorm2 (num_groups=32)
        out2_norm = torch.empty_like(out2_pre)
        grid_gn2 = (B, self.num_groups)
        group_norm_two_pass[grid_gn2](
            out2_pre, norm2_weight.contiguous().to(torch.float32), norm2_bias.contiguous().to(torch.float32), out2_norm,
            B, C, H_out, W_out, self.num_groups, self.eps,
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )

        # SiLU2
        out2_silu = torch.empty_like(out2_norm)
        N2 = out2_norm.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](out2_norm, out2_silu, N2, BLOCK=1024)

        # Add residual x
        out = torch.empty_like(out2_silu)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[grid_add](out2_silu, x_f, out, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
