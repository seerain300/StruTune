import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_2x2x3_bias_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, C_in, H, W,
    C_out, K_h, K_w,
    pad_h, pad_w,
    stride_h, stride_w,
    # strides
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
    stride_y_b, stride_y_c, stride_y_h, stride_y_w,
    # output dims
    H_out, W_out,
):
    # program ids
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow = tl.program_id(3)

    # Accumulator for output value at (b, co, oh, ow)
    acc = 0.0
    # Loop over input channels
    for ci in range(0, C_in):
        # Loop over kernel spatial positions
        for kh in range(0, K_h):
            ih = pid_oh * stride_h - pad_h + kh  # scalar
            valid_h = (ih >= 0) & (ih < H)
            for kw in range(0, K_w):
                iw = pid_ow * stride_w - pad_w + kw  # scalar
                valid_w = (iw >= 0) & (iw < W)
                in_bounds = valid_h & valid_w

                # Load input value if in bounds
                x_ptr = X_ptr + pid_b * stride_x_b + ci * stride_x_c + ih * stride_x_h + iw * stride_x_w
                x_val = tl.load(x_ptr, mask=in_bounds, other=0.0).to(tl.float32)

                # Load weight for this (co, ci, kh, kw)
                w_ptr = W_ptr + pid_co * stride_w_co + ci * stride_w_ci + kh * stride_w_kh + kw * stride_w_kw
                w_val = tl.load(w_ptr).to(tl.float32)

                acc += x_val * w_val

    # Add bias
    bias_val = tl.load(BIAS_ptr + pid_co).to(tl.float32)
    acc += bias_val

    # Store to output
    y_ptr = Y_ptr + pid_b * stride_y_b + pid_co * stride_y_c + pid_oh * stride_y_h + pid_ow * stride_y_w
    # cast to bfloat16 for storage (original code uses bfloat16)
    tl.store(y_ptr, acc.to(tl.bfloat16))


@triton.jit
def gelu_exact_kernel(X_ptr, Y_ptr, N, scale):
    # N is total number of elements
    pid = tl.program_id(0)
    offsets = pid * tl.num_programs(0) + tl.arange(0, tl.num_programs(0))
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)
    y = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    y = y * scale  # scale in kernel
    tl.store(Y_ptr + offsets, y.to(tl.bfloat16), mask=mask)


@triton.jit
def matmul_linear_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, S, K, N,
    stride_x_b, stride_x_s, stride_x_k,
    stride_w_n, stride_w_k,
    stride_y_b, stride_y_s, stride_y_n,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, S, ceil_div(N, BLOCK_N))
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_n_block = tl.program_id(2)

    n_offsets = pid_n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load X[b, s, k_offsets]
        x_ptrs = X_ptr + pid_b * stride_x_b + pid_s * stride_x_s + k_offsets * stride_x_k
        x_vec = tl.load(x_ptrs, mask=k_offsets < K, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Load W[n_offsets, k_offsets] as [BLOCK_N, BLOCK_K]
        w_ptrs = W_ptr + n_offsets[:, None] * stride_w_n + k_offsets[None, :] * stride_w_k
        w_tile = tl.load(w_ptrs, mask=(n_offsets[:, None] < N) & (k_offsets[None, :] < K), other=0.0).to(tl.float32)

        # acc += sum over k of x_vec[k] * w_tile[:, k]
        acc += tl.sum(w_tile * x_vec[None, :], axis=1)

    # Store Y[b, s, n_offsets]
    y_ptrs = Y_ptr + pid_b * stride_y_b + pid_s * stride_y_s + n_offsets * stride_y_n
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=n_offsets < N)


@triton.jit
def scale_elementwise_kernel(X_ptr, Y_ptr, N, scale):
    pid = tl.program_id(0)
    offsets = pid * tl.num_programs(0) + tl.arange(0, tl.num_programs(0))
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x * scale
    tl.store(Y_ptr + offsets, y.to(tl.bfloat16), mask=mask)


@triton.jit
def add_pos_emb_kernel(X_ptr, POS_ptr, Y_ptr, S, N):
    # X: [B, S, N], POS: [S, N]
    pid = tl.program_id(0)
    offsets = pid * tl.num_programs(0) + tl.arange(0, tl.num_programs(0))
    mask = offsets < S * N
    b_idx = offsets // N
    s_idx = offsets % N  # wrong, fix
    # Compute s and n: s = b_idx, n = offsets % N? No, we need to map linear offsets to (b, s, n). Fix:
    # We should treat X as flattened per batch. So recompute mapping:
    # Since Y is [B, S, N], we can map by strides; but we have only linear offsets. Simpler: launch a 2D grid (B, S*N).
    # Here we launch grid=(B, S*N), so pid0=B, pid1=S*N. Then n = pid1 % N, s = pid1 // N. But we only have pid. So redesign:
    # Instead, launch a 3D grid: (B, S, N). We will do that for matmul. For this elementwise add, we can use 1D again and rely on indexing by s=n_offsets approach per batch.
    # To keep it simple and correct, we'll use 2D grid: (B, S*N). Compute s and n via integer div/mod.
    # However Triton elementwise kernel commonly uses 1D. We can compute s and n from linear offsets by passing N and using div/mod.
    # Here, we implement with 1D and assume we know S,N. Compute s = offsets // N, n = offsets % N.
    s_idx = offsets // N
    n_idx = offsets % N
    x_val = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    pos_val = tl.load(POS_ptr + s_idx * N + n_idx, mask=mask, other=0.0).to(tl.float32)
    y_val = x_val + pos_val
    tl.store(Y_ptr + offsets, y_val.to(tl.bfloat16), mask=mask)


def triton_conv2d_stride2_pad1(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: [B, C_in, H, W], bfloat16, CUDA
    w: [C_out, C_in, 3, 3], bfloat16, CUDA
    bias: [C_out], bfloat16, CUDA
    Returns y: [B, C_out, H_out, W_out], bfloat16
    """
    B, C_in, H, W = x.shape
    C_out = w.shape[0]
    # pad=1, stride=2, k=3
    H_out = (H + 2 * 1 - 3) // 2 + 1
    W_out = (W + 2 * 1 - 3) // 2 + 1

    y = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=torch.bfloat16)

    grid = (B, C_out, H_out, W_out)
    # Strides
    stride_x_b, stride_x_c, stride_x_h, stride_x_w = x.stride()
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw = w.stride()
    stride_y_b, stride_y_c, stride_y_h, stride_y_w = y.stride()

    conv2d_2x2x3_bias_kernel[grid](
        x, w, bias, y,
        B, C_in, H, W,
        C_out, 3, 3, 1, 1, 2, 2,
        stride_x_b, stride_x_c, stride_x_h, stride_x_w,
        stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
        stride_y_b, stride_y_c, stride_y_h, stride_y_w,
        H_out, W_out,
        num_warps=2, num_stages=1
    )
    return y


def triton_gelu_exact(x: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Apply GELU (exact) with scaling to x. Returns tensor of same dtype as x.
    """
    x_c = x.contiguous()
    y = torch.empty_like(x_c, dtype=torch.bfloat16, device=x.device)
    N = x_c.numel()
    grid = (triton.cdiv(N, 1024),)
    gelu_exact_kernel[grid](x_c, y, N, scale, num_warps=4, num_stages=2)
    return y


def triton_linear_proj(x: torch.Tensor, w: torch.Tensor, embed_scale: float) -> torch.Tensor:
    """
    x: [B, S, K], bfloat16, CUDA
    w: [N, K], bfloat16, CUDA
    Returns y: [B, S, N], bfloat16
    """
    B, S, K = x.shape
    N, Kw = w.shape
    assert Kw == K, "Weight K must match x K dimension"
    x_c = x.contiguous()
    w_c = w.contiguous()
    y = torch.empty((B, S, N), device=x.device, dtype=torch.bfloat16)

    BLOCK_N = 128
    BLOCK_K = 64
    grid = (B, S, triton.cdiv(N, BLOCK_N))
    matmul_linear_kernel[grid](
        x_c, w_c, y,
        B, S, K, N,
        x_c.stride(0), x_c.stride(1), x_c.stride(2),
        w_c.stride(0), w_c.stride(1),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )
    return y


def triton_scale_elementwise(y: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Scale y elementwise by scale. Returns tensor of same dtype as y.
    """
    y_c = y.contiguous()
    z = torch.empty_like(y_c, dtype=torch.bfloat16, device=y.device)
    N = y_c.numel()
    grid = (triton.cdiv(N, 1024),)
    scale_elementwise_kernel[grid](y_c, z, N, scale, num_warps=4, num_stages=2)
    return z


def triton_add_pos_emb(y: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
    """
    y: [B, S, N], bfloat16, CUDA
    pos_emb: [S, N], bfloat16, CUDA
    Returns y + pos_emb (broadcast over batch).
    """
    B, S, N = y.shape
    y_c = y.contiguous()
    z = torch.empty_like(y_c, dtype=torch.bfloat16, device=y.device)
    N_elems = S * N
    grid = (triton.cdiv(N_elems, 1024),)
    add_pos_emb_kernel[grid](y_c, pos_emb, z, S, N, num_warps=4, num_stages=2)
    return z


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Unpack inputs as per get_inputs
        input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed


def run(*args):
    return ModelNew()(*args)
