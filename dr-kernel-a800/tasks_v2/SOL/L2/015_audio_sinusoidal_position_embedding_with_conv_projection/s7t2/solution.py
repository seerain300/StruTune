import math
import torch
import triton
import triton.language as tl


# Triton kernel: conv2d with 3x3, stride=2, padding=1, bias, followed by GELU (exact).
# Inputs:
#   X: [B, C_in, H, W], bfloat16
#   W: [C_out, C_in, 3, 3], bfloat16
#   Bias: [C_out], bfloat16
# Output:
#   Y: [B, C_out, H_out, W_out], bfloat16
@triton.jit
def conv2d_stride2_pad1_bias_gelu_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, C_in, H, W,
    C_out, H_out, W_out,
    stride_x_b, stride_x_ci, stride_x_h, stride_x_w,
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
    stride_y_b, stride_y_co, stride_y_h, stride_y_w,
    scale: tl.float32,
    BLOCK_H: tl.constexpr,  # tile size for H_out
    BLOCK_W: tl.constexpr,  # tile size for W_out
):
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h_block = tl.program_id(2)
    pid_w_block = tl.program_id(3)

    oh_offsets = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    ow_offsets = pid_w_block * BLOCK_W + tl.arange(0, BLOCK_W)

    # Create 2D mesh for output spatial positions
    oh = oh_offsets[:, None]  # shape [BLOCK_H, 1]
    ow = ow_offsets[None, :]  # shape [1, BLOCK_W]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Loop over input channels and kernel elements
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                # Compute input indices with padding=1, stride=2
                ih = 2 * oh + (1 - kh)  # shape [BLOCK_H, 1]
                iw = 2 * ow + (1 - kw)  # shape [1, BLOCK_W]

                # Bounds check
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & (oh < H_out) & (ow < W_out)

                # Load x[b, ci, ih, iw] -> shape [BLOCK_H, BLOCK_W]
                x_ptrs = X_ptr + pid_b * stride_x_b + ci * stride_x_ci + ih * stride_x_h + iw * stride_x_w
                x_val = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)

                # Load w[co, ci, kh, kw] scalar
                w_ptr = W_ptr + pid_co * stride_w_co + ci * stride_w_ci + kh * stride_w_kh + kw * stride_w_kw
                w_val = tl.load(w_ptr).to(tl.float32)

                # Accumulate
                acc += x_val * w_val

    # Add bias
    bias_val = tl.load(BIAS_ptr + pid_co).to(tl.float32)
    acc += bias_val

    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Scale
    acc = acc * scale

    # Store Y[b, co, oh, ow]
    y_ptrs = Y_ptr + pid_b * stride_y_b + pid_co * stride_y_co + oh * stride_y_h + ow * stride_y_w
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=(oh < H_out) & (ow < W_out))


# Triton GELU exact elementwise kernel
@triton.jit
def gelu_exact_elementwise_kernel(X_ptr, Y_ptr, N_total, scale):
    pid = tl.program_id(0)
    offsets = pid * tl.num_programs(0) + tl.arange(0, tl.num_programs(0))
    mask = offsets < N_total
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inv_sqrt2 = 0.7071067811865476
    y = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    y = y * scale
    tl.store(Y_ptr + offsets, y.to(tl.bfloat16), mask=mask)


# Triton matmul kernel: compute Y_rowwise = X_rowwise @ W^T, no bias
# X_rowwise: [B*S, K], W: [N, K], Y_rowwise: [B*S, N], we will view to [B, S, N] in host.
@triton.jit
def linear_matmul_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, S, K, N,
    stride_x_row, stride_x_k,
    stride_w_n, stride_w_k,
    stride_y_row, stride_y_n,
    BLOCK_N: tl.constexpr,  # tile size along output channels
    BLOCK_K: tl.constexpr,  # tile size along reduction
):
    pid_row = tl.program_id(0)  # row index in [0, B*S)
    pid_n_block = tl.program_id(1)

    s_idx = pid_row % S
    b_idx = pid_row // S

    n_offsets = pid_n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load X_row[b*s, k_offsets]
        x_ptrs = X_ptr + pid_row * stride_x_row + k_offsets * stride_x_k
        x_vec = tl.load(x_ptrs, mask=k_offsets < K, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Load W[n_offsets, k_offsets] as [BLOCK_N, BLOCK_K]
        w_ptrs = W_ptr + n_offsets[:, None] * stride_w_n + k_offsets[None, :] * stride_w_k
        w_tile = tl.load(w_ptrs, mask=(n_offsets[:, None] < N) & (k_offsets[None, :] < K), other=0.0).to(tl.float32)

        # Accumulate: per row over K-tile
        acc += tl.sum(w_tile * x_vec[None, :], axis=1)

    # Store Y[b*s, n_offsets]
    y_ptrs = Y_ptr + pid_row * stride_y_row + n_offsets * stride_y_n
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=n_offsets < N)


# Triton scale elementwise kernel
@triton.jit
def scale_elementwise_kernel(X_ptr, Y_ptr, N_total, scale):
    pid = tl.program_id(0)
    offsets = pid * tl.num_programs(0) + tl.arange(0, tl.num_programs(0))
    mask = offsets < N_total
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x * scale
    tl.store(Y_ptr + offsets, y.to(tl.bfloat16), mask=mask)


# Triton add positional embedding kernel: Y[B, S, N] += POS[S, N], broadcast over batch
# Launch grid over (B, S, N) and compute per element
@triton.jit
def add_pos_emb_kernel(
    Y_ptr, POS_ptr,
    B, S, N,
    stride_y_b, stride_y_s, stride_y_n,
    stride_pos_s, stride_pos_n,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_n = tl.program_id(2)

    n = pid_n
    # Load y[b, s, n]
    y_ptr = Y_ptr + pid_b * stride_y_b + pid_s * stride_y_s + n * stride_y_n
    y_val = tl.load(y_ptr).to(tl.float32)

    # Load pos[s, n]
    pos_ptr = POS_ptr + pid_s * stride_pos_s + n * stride_pos_n
    pos_val = tl.load(pos_ptr).to(tl.float32)

    # Add and store
    y_val = y_val + pos_val
    tl.store(y_ptr, y_val.to(tl.bfloat16))


def triton_conv2d_bias_gelu(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, scale: float):
    """
    x: [B, C_in, H, W], bfloat16, CUDA
    w: [C_out, C_in, 3, 3], bfloat16, CUDA
    bias: [C_out], bfloat16, CUDA
    Returns y: [B, C_out, H_out, W_out], bfloat16
    """
    B, C_in, H, W = x.shape
    C_out = w.shape[0]
    H_out = (H + 2 * 1 - 3) // 2 + 1
    W_out = (W + 2 * 1 - 3) // 2 + 1

    y = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=torch.bfloat16)

    # Strides
    stride_x_b, stride_x_ci, stride_x_h, stride_x_w = x.stride()
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw = w.stride()
    stride_y_b, stride_y_co, stride_y_h, stride_y_w = y.stride()

    # Grid: (B, C_out, ceil(H_out/8), ceil(W_out/32))
    BLOCK_H = 8
    BLOCK_W = 32
    grid = (B, C_out, triton.cdiv(H_out, BLOCK_H), triton.cdiv(W_out, BLOCK_W))
    conv2d_stride2_pad1_bias_gelu_kernel[grid](
        x, w, bias, y,
        B, C_in, H, W,
        C_out, H_out, W_out,
        stride_x_b, stride_x_ci, stride_x_h, stride_x_w,
        stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
        stride_y_b, stride_y_co, stride_y_h, stride_y_w,
        scale,  # embed_scale, used after GELU
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=4, num_stages=2
    )
    return y


def triton_gelu_exact(x: torch.Tensor, scale: float):
    """
    Apply exact GELU to x, scale by 'scale'. Returns tensor of same dtype as x.
    """
    x_c = x.contiguous()
    y = torch.empty_like(x_c, dtype=torch.bfloat16, device=x.device)
    N_total = x_c.numel()
    grid = (triton.cdiv(N_total, 1024),)
    gelu_exact_elementwise_kernel[grid](x_c, y, N_total, scale, num_warps=4, num_stages=2)
    return y


def triton_linear_proj(x_rowwise: torch.Tensor, w: torch.Tensor, embed_scale: float) -> torch.Tensor:
    """
    x_rowwise: [B, S, K], bfloat16, CUDA
    w: [N, K], bfloat16, CUDA
    Returns y: [B, S, N], bfloat16
    """
    B, S, K = x_rowwise.shape
    N, Kw = w.shape
    assert Kw == K, "Weight K must match x K dimension"
    x = x_rowwise.contiguous()  # [B*S, K]
    w_c = w.contiguous()        # [N, K]
    y = torch.empty((B * S, N), device=x.device, dtype=torch.bfloat16)  # [B*S, N]

    BLOCK_N = 128
    BLOCK_K = 64
    grid = (B * S, triton.cdiv(N, BLOCK_N))
    linear_matmul_kernel[grid](
        x, w_c, y,
        B, S, K, N,
        x.stride(0), x.stride(1),
        w_c.stride(0), w_c.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )
    return y.view(B, S, N)


def triton_scale_elementwise(y: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Scale y elementwise by scale. Returns tensor of same dtype as y.
    """
    y_c = y.contiguous()
    z = torch.empty_like(y_c, dtype=torch.bfloat16, device=y.device)
    N_total = y_c.numel()
    grid = (triton.cdiv(N_total, 1024),)
    scale_elementwise_kernel[grid](y_c, z, N_total, scale, num_warps=4, num_stages=2)
    return z


def triton_add_pos_emb(y: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
    """
    y: [B, S, N], bfloat16, CUDA
    pos_emb: [S, N], bfloat16, CUDA
    Adds pos_emb to y (broadcast over batch).
    """
    B, S, N = y.shape
    y_c = y.contiguous()
    pos_emb_c = pos_emb.contiguous()
    # Launch grid over (B, S, N)
    grid = (B, S, N)
    add_pos_emb_kernel[grid](
        y_c, pos_emb_c,
        B, S, N,
        y_c.stride(0), y_c.stride(1), y_c.stride(2),
        pos_emb_c.stride(0), pos_emb_c.stride(1),
        num_warps=2, num_stages=2
    )
    return y_c


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Unpack inputs
        input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale = args

        # Ensure inputs are on CUDA and bfloat16 (as per original)
        device = input_features.device
        # Stage 1: Conv2d (1 -> 384 channels) + GELU, using Triton
        x = triton_conv2d_bias_gelu(input_features, conv2d1_weight, conv2d1_bias, embed_scale)
        # Stage 2: Conv2d (384 -> 384 channels) + GELU, using Triton
        x = triton_conv2d_bias_gelu(x, conv2d2_weight, conv2d2_bias, embed_scale)
        # Stage 3: Conv2d (384 -> 384 channels) + GELU, using Triton
        x = triton_conv2d_bias_gelu(x, conv2d3_weight, conv2d3_bias, embed_scale)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Linear projection to d_model (no bias) via Triton
        B, S, K = x.shape
        N = conv_out_weight.shape[0]  # 1024
        x_rowwise = x.view(B * S, K).contiguous()  # [B*S, K]

        y_rowwise = triton_linear_proj(x_rowwise, conv_out_weight, embed_scale)  # [B*S, N]
        y = y_rowwise.view(B, S, N)

        # Scale embeddings
        y_scaled = triton_scale_elementwise(y, embed_scale)

        # Add positional embeddings (broadcast over batch) via Triton
        pos_emb = positional_embedding[:S, :].to(torch.bfloat16)  # [S, N]
        y_final = triton_add_pos_emb(y_scaled, pos_emb)

        return y_final


def run(*args):
    return ModelNew()(*args)
