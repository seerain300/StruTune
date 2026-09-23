import torch
import triton
import triton.language as tl


# Triton kernels: Conv3x3 stride=1, padding=1, no bias, input NCHW, output NCHW
# We implement im2col + matmul per (n, co) and BLOCK_H x BLOCK_W output tile.
@triton.jit
def conv3x3_im2col_matmul(
    x_ptr,          # *const float, input [B, C_in, H, W]
    w_ptr,          # *const float, weights [C_out, C_in, 3, 3], flattened to [C_out, K] where K=C_in*9
    out_ptr,        # *float, output [B, C_out, H_out, W_out]
    B: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_y = tl.program_id(2)
    pid_x = tl.program_id(3)

    # output tile coordinates
    Y = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
    X = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_y = Y < H_out
    mask_x = X < W_out
    mask_out = mask_y[:, None] & mask_x[None, :]

    # flatten to one vector for im2col and matmul
    HW_out = H_out * W_out
    idx = Y[:, None] * W_out + X[None, :]  # shape [BLOCK_H, BLOCK_W]

    # im2col: build input matrix A of shape [BLOCK_H*BLOCK_W, K]
    K = C_in * 9  # kh in {0,1,2}, kw in {0,1,2}
    A = tl.zeros((BLOCK_H * BLOCK_W, K), dtype=tl.float32)

    # iterate over input channels and 3x3 kernel
    # We'll compute base pointers and offsets for each (ci, kh, kw)
    # Note: Triton requires static loops; C_in and C_out are constexpr here.
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                # compute input coordinates with padding
                h_in = Y - 1 + kh  # shape [BLOCK_H]
                w_in = X - 1 + kw  # shape [BLOCK_W]
                in_bounds = (h_in[:, None] >= 0) & (h_in[:, None] < H) & (w_in[None, :] >= 0) & (w_in[None, :] < W) & mask_out
                # base offset for x
                base_x = pid_n * x_stride_n + ci * x_stride_c
                # pointer for A rows: A[row, col] = x[pid_n, ci, h_in[row], w_in[col]]
                ptrs_x = x_ptr + base_x + h_in[:, None] * x_stride_h + w_in[None, :] * x_stride_w
                # load with mask
                vals = tl.load(ptrs_x, mask=in_bounds, other=0.0)  # shape [BLOCK_H, BLOCK_W], dtype inferred as float32
                # reshape vals to [BLOCK_H*BLOCK_W] by flattening indices
                # mapping: col = ci*(3*3) + kh*3 + kw
                col = ci * 9 + kh * 3 + kw
                A[:, col] = vals.flatten()  # broadcast to all rows

    # weight matrix W of shape [C_out, K], contiguous flattened
    W_mat = tl.load(w_ptr + pid_co * K + tl.arange(0, K), mask=tl.full((K,), True, dtype=tl.int1), other=0.0)  # shape [K]
    # output matrix multiply: C = A @ W_mat^T -> shape [BLOCK_H*BLOCK_W, C_out]
    # Note: Triton provides tl.dot for matrix multiplication; we reshape W_mat to [K, 1] implicitly via broadcasting.
    # We need to expand W_mat to [1, K] to match A's second dim.
    C = tl.dot(A, tl.trans(W_mat[None, :]))  # shape [BLOCK_H*BLOCK_W, 1], but since W_mat is [K], this yields [BLOCK_H*BLOCK_W, C_out]

    # above line is conceptual; Triton expects proper 2D loads. So instead, we compute per-co using tl.dot with proper shapes.
    # To ensure correctness, we compute per-co directly:
    # Prepare C as [BLOCK_H*BLOCK_W, C_out] via explicit loop over C_out:
    out_vec = tl.zeros((BLOCK_H * BLOCK_W,), dtype=tl.float32)
    for co in tl.static_range(C_out):
        w_vec = tl.load(w_ptr + co * K + tl.arange(0, K), mask=tl.full((K,), True, dtype=tl.int1), other=0.0)  # [K]
        # C = A @ w_vec -> [BLOCK_H*BLOCK_W]
        out_vec += tl.dot(A, tl.trans(w_vec[None, :]))  # [BLOCK_H*BLOCK_W]

    # store output vector to out[n, co, Y, X]
    base_out = pid_n * out_stride_n + co * out_stride_c
    out_ptrs = out_ptr + base_out + (Y[:, None] * out_stride_h + X[None, :] * out_stride_w) * mask_out
    tl.store(out_ptrs, out_vec[None, :].to(tl.float32), mask=mask_out)


# GroupNorm Triton kernel: two-pass per (n, group)
# 1) compute sum and sum of squares over group elements
# 2) normalize and apply affine gamma/beta, then write out
@triton.jit
def group_norm_kernel(
    inp_ptr,        # *const float, input tensor [B, C, H, W]
    scale_ptr,      # *const float, gamma [C]
    bias_ptr,       # *const float, beta [C]
    out_ptr,        # *float, output tensor [B, C, H, W]
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    num_groups: tl.constexpr,
    eps: tl.float32,
):
    # program ids
    pid = tl.program_id(0)  # one per (n, group)
    n = pid // num_groups
    g = pid % num_groups

    group_size = C // num_groups
    c_start = g * group_size

    # 1) compute sum and sum of squares over group
    total = 0.0
    total_sq = 0.0
    # iterate over channels in group and all spatial positions
    for ci in tl.static_range(group_size):
        c = c_start + ci
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                ptr = inp_ptr + n * (C * H * W) + c * (H * W) + h * W + w
                x = tl.load(ptr).to(tl.float32)
                total += x
                total_sq += x * x
    group_area = group_size * H * W
    mean = total / group_area
    var = total_sq / group_area - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # 2) normalize and apply affine, write out
    for ci in tl.static_range(group_size):
        c = c_start + ci
        scale = tl.load(scale_ptr + c).to(tl.float32)
        beta = tl.load(bias_ptr + c).to(tl.float32)
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                ptr_in = inp_ptr + n * (C * H * W) + c * (H * W) + h * W + w
                x = tl.load(ptr_in).to(tl.float32)
                y = (x - mean) * inv_std * scale + beta
                ptr_out = out_ptr + n * (C * H * W) + c * (H * W) + h * W + w
                tl.store(ptr_out, y)


# Elementwise SiLU kernel
@triton.jit
def silu_kernel(in_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    # SiLU: x * sigmoid(x)
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(out_ptr + offs, y, mask=mask)


# Elementwise residual add kernel: out = out + x
@triton.jit
def add_residual_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(out_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(out_ptr + offs, c, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; all work is done in Triton kernels in forward.

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        """
        Triton-only forward: conv3x3 -> GroupNorm -> SiLU -> conv3x3 -> GroupNorm -> SiLU -> Add
        x: (B, C, H, W), conv weights: (C, C, 3, 3), norm params: (C,), no bias in conv
        num_groups=32, eps provided
        """
        assert x.is_cuda and conv1_weight.is_cuda and conv2_weight.is_cuda and norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "All tensors must be on CUDA device for Triton kernels."

        B, C, H, W = x.shape
        assert C % 32 == 0, "GroupNorm num_groups=32 requires C divisible by 32."
        num_groups = 32
        group_size = C // num_groups

        # Prepare dtype: use float32 for stability; ensure inputs/weights are float32
        # We will operate in float32 in kernels; if inputs are not float32, cast to float32.
        x_f = x.to(torch.float32)
        # Weights: conv weights are (C, C, 3, 3). Flatten to (C, C, 9) then to contiguous [C*C, 9]
        # For conv1:
        w1 = conv1_weight.to(torch.float32).contiguous()  # (C, C, 3, 3)
        C_in1 = C  # input channels = C
        C_out1 = C  # output channels = C
        K1 = C_in1 * 3 * 3
        # For conv2:
        w2 = conv2_weight.to(torch.float32).contiguous()  # (C, C, 3, 3)
        C_in2 = C  # input channels = C
        C_out2 = C  # output channels = C
        K2 = C_in2 * 3 * 3

        # Output buffers for convs
        H_out1 = H  # stride=1, padding=1 -> same H_out as input for conv1
        W_out1 = W
        out1 = torch.empty((B, C, H_out1, W_out1), device=x.device, dtype=torch.float32)

        # Launch conv1 kernel
        BLOCK_H = 8
        BLOCK_W = 8
        grid_conv1 = (B, C, triton.cdiv(H_out1, BLOCK_H), triton.cdiv(W_out1, BLOCK_W))
        conv3x3_im2col_matmul[grid_conv1](
            x_f, w1.reshape(-1), out1,
            B, C_in1, H, W, C_out1, H_out1, W_out1,
            x_f.stride(0), x_f.stride(1), x_f.stride(2), x_f.stride(3),
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        )

        # GroupNorm1
        out1_gn = torch.empty_like(out1)
        grid_gn1 = (B * num_groups,)
        group_norm_kernel[grid_gn1](
            out1, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32), out1_gn,
            B, C, H_out1, W_out1, num_groups, eps,
        )

        # SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # Conv2
        H_out2 = H_out1  # same spatial size for conv2
        W_out2 = W_out1
        out2 = torch.empty((B, C, H_out2, W_out2), device=x.device, dtype=torch.float32)
        grid_conv2 = (B, C, triton.cdiv(H_out2, BLOCK_H), triton.cdiv(W_out2, BLOCK_W))
        conv3x3_im2col_matmul[grid_conv2](
            out1_silu, w2.reshape(-1), out2,
            B, C_in2, H_out1, W_out1, C_out2, H_out2, W_out2,
            out1_silu.stride(0), out1_silu.stride(1), out1_silu.stride(2), out1_silu.stride(3),
            out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        )

        # GroupNorm2
        out2_gn = torch.empty_like(out2)
        grid_gn2 = (B * num_groups,)
        group_norm_kernel[grid_gn2](
            out2, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32), out2_gn,
            B, C, H_out2, W_out2, num_groups, eps,
        )

        # SiLU2
        out2_silu = torch.empty_like(out2_gn)
        N2 = out2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](out2_gn, out2_silu, N2, BLOCK=1024)

        # Add residual x
        out = torch.empty_like(out2_silu)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[x_f, out2_silu, out](x_f, out2_silu, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
