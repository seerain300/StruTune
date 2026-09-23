import torch
import triton
import triton.language as tl


# Triton conv3x3 stride=1, padding=1, no bias: compute one output element per program
@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,        # *const float, input [B, C_in, H, W]
    w_ptr,        # *const float, weights [C_out, C_in, 9] (flattened (3,3))
    out_ptr,      # *float, output [B, C_out, H, W]
    B, C_in, H, W, C_out, H_out, W_out,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
):
    # program ids
    pid_n = tl.program_id(0)  # batch
    pid_co = tl.program_id(1) # output channel
    pid_h = tl.program_id(2)  # output height index
    pid_w = tl.program_id(3)  # output width index

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = pid_h - 1 + kh  # padding=1
                w_in = pid_w - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                # base offset for x at (n, ci, h_in, w_in)
                x_offset = pid_n * x_stride_n + ci * x_stride_c + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0).to(tl.float32)
                # weight index: (co * (C_in * 9)) + (ci * 9) + (kh * 3 + kw)
                w_idx = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx).to(tl.float32)
                acc += x_val * w_val

    # store to output
    out_offset = pid_n * out_stride_n + pid_co * out_stride_c + pid_h * out_stride_h + pid_w * out_stride_w
    tl.store(out_ptr + out_offset, acc)


# Triton GroupNorm over num_groups=32 (assumes C % 32 == 0)
@triton.jit
def group_norm_two_pass(
    inp_ptr,       # *const float, input to normalize [B, C, H, W]
    gamma_ptr,     # *const float, per-channel scale [C]
    beta_ptr,      # *const float, per-channel bias [C]
    out_ptr,       # *float, output [B, C, H, W]
    B, C, H, W,
    eps,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
):
    # Each program handles one (n, group)
    pid_n = tl.program_id(0)
    group = tl.program_id(1)  # 0..31
    group_size = C // 32
    group_start = group * group_size

    sum_vals = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # First pass: accumulate sum and sum of squares over the group for this sample
    for ci in tl.static_range(group_start, group_start + group_size):
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                in_offset = pid_n * (C * H * W) + ci * (H * W) + h * W + w
                x = tl.load(inp_ptr + in_offset).to(tl.float32)
                sum_vals += x
                sum_sq += x * x

    m = H * W
    M = group_size * m
    mean = sum_vals / M
    var = sum_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, store to output
    for ci in tl.static_range(group_start, group_start + group_size):
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                in_offset = pid_n * (C * H * W) + ci * (H * W) + h * W + w
                x = tl.load(inp_ptr + in_offset).to(tl.float32)
                gamma = tl.load(gamma_ptr + ci).to(tl.float32)
                beta = tl.load(beta_ptr + ci).to(tl.float32)
                y = (x - mean) * inv_std
                y = y * gamma + beta
                out_offset = pid_n * out_stride_n + ci * out_stride_c + h * out_stride_h + w * out_stride_w
                tl.store(out_ptr + out_offset, y)


# Triton SiLU activation: elementwise y = x * sigmoid(x)
@triton.jit
def silu_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Triton add residual: elementwise out = out + x
@triton.jit
def add_residual_kernel(out_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    out = tl.load(out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = out + x
    tl.store(out_ptr + offs, out, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
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
        # Ensure tensors are on CUDA and float32; make contiguous
        device = x.device
        B, C, H, W = x.shape
        assert C % self.num_groups == 0, "C must be divisible by num_groups for GroupNorm"
        assert conv1_weight.shape == (C, C, 3, 3), "conv1_weight must have shape (C, C, 3, 3)"
        assert conv2_weight.shape == (C, C, 3, 3), "conv2_weight must have shape (C, C, 3, 3)"
        assert norm1_weight.shape == (C,), "norm1_weight must have shape (C,)"
        assert norm1_bias.shape == (C,), "norm1_bias must have shape (C,)"
        assert norm2_weight.shape == (C,), "norm2_weight must have shape (C,)"
        assert norm2_bias.shape == (C,), "norm2_bias must have shape (C,)"

        # Make tensors contiguous and float32
        x_f = x.contiguous().to(torch.float32)
        # Flatten conv weights to [C_out, C_in, 9] for easier indexing in Triton
        conv1_weight_f = conv1_weight.contiguous().to(torch.float32).reshape(C, C, 9)
        conv2_weight_f = conv2_weight.contiguous().to(torch.float32).reshape(C, C, 9)
        norm1_weight_f = norm1_weight.contiguous().to(torch.float32)
        norm1_bias_f = norm1_bias.contiguous().to(torch.float32)
        norm2_weight_f = norm2_weight.contiguous().to(torch.float32)
        norm2_bias_f = norm2_bias.contiguous().to(torch.float32)

        H_out = H
        W_out = W

        # 1) First convolution: out1 = conv3x3(x)
        out1 = torch.empty((B, C, H_out, W_out), device=device, dtype=torch.float32)

        grid_conv1 = (B, C, H_out, W_out)
        conv3x3_nobias_one_elem[grid_conv1](
            x_f, conv1_weight_f, out1,
            B, C, H, W, C, H_out, W_out,
            x_f.stride(0), x_f.stride(1), x_f.stride(2), x_f.stride(3),
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        )

        # 2) GroupNorm1 (num_groups=32)
        out1_gn = torch.empty_like(out1)

        grid_gn1 = (B, self.num_groups)
        group_norm_two_pass[grid_gn1](
            out1, norm1_weight_f, norm1_bias_f, out1_gn,
            B, C, H_out, W_out, self.eps,
            out1_gn.stride(0), out1_gn.stride(1), out1_gn.stride(2), out1_gn.stride(3),
        )

        # 3) SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # 4) Second convolution: out2_pre = conv3x3(out1_silu)
        out2_pre = torch.empty((B, C, H_out, W_out), device=device, dtype=torch.float32)

        grid_conv2 = (B, C, H_out, W_out)
        conv3x3_nobias_one_elem[grid_conv2](
            out1_silu, conv2_weight_f, out2_pre,
            B, C, H_out, W_out, C, H_out, W_out,
            out1_silu.stride(0), out1_silu.stride(1), out1_silu.stride(2), out1_silu.stride(3),
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )

        # 5) GroupNorm2 (num_groups=32)
        out2_gn = torch.empty_like(out2_pre)

        grid_gn2 = (B, self.num_groups)
        group_norm_two_pass[grid_gn2](
            out2_pre, norm2_weight_f, norm2_bias_f, out2_gn,
            B, C, H_out, W_out, self.eps,
            out2_gn.stride(0), out2_gn.stride(1), out2_gn.stride(2), out2_gn.stride(3),
        )

        # 6) SiLU2
        out2_silu = torch.empty_like(out2_gn)
        N2 = out2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](out2_gn, out2_silu, N2, BLOCK=1024)

        # 7) Add residual x
        out = torch.empty_like(out2_silu)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[grid_add](out2_silu, x_f, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
