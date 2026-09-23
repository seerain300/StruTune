import math
import triton
import triton.language as tl


# Triton kernel: GELU (tanh approximation), elementwise over N elements
@triton.jit
def gelu_tanh_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    # Load and compute in float32 for stability
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(Y + offsets, gelu, mask=mask)


# Triton kernel: Linear projection (no bias) Y = X @ W^T
# X: (B, T, K), W: (M, K), Y: (B, T, M)
# Tiled over M (output channels), reduce over K in chunks.
@triton.jit
def linear_no_bias_kernel(
    X, W, Y,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,   # tile size over M
    BLOCK_K: tl.constexpr    # tile size over reduction
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    t = pid_t
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load X[b, t, k_offsets] -> vector (BLOCK_K,)
        x_ptrs = X + b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Load W[:, k_offsets] for M tile -> matrix (BLOCK_M, BLOCK_K)
        w_ptrs = W + m_offsets[:, None] * stride_wm + k_offsets[None, :] * stride_wk
        w_mat = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate: acc += sum_k w_mat[m, k] * x_vec[k]
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

    # Store results Y[b, t, m_offsets]
    y_ptrs = Y + b * stride_yb + t * stride_yt + m_offsets * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Triton kernel: Elementwise add scaled positional embedding
# Y: (B, T, M), POS: (M, D), scale: float
# We treat Y as (B*T*M) and POS[:, :T, :] flattened, broadcasting over batch.
@triton.jit
def add_scaled_pos_emb_kernel(
    Y, POS, scale, B, T, M, D,
    stride_yb, stride_yt, stride_ym,
    stride_posm, stride_posd,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    N = B * T * M
    mask = idx < N

    # Compute (b, t, m) indices from idx
    m = M
    t = T
    b = B
    # Use integer math
    tm = tl.minimum(idx, (t * m - 1))
    b_idx = tm // (t * m)
    rem = tm % (t * m)
    t_idx = rem // m
    m_idx = rem % m

    y_ptrs = Y + b_idx * stride_yb + t_idx * stride_yt + m_idx * stride_ym
    y_val = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)

    pos_offsets = t_idx * D + m_idx
    pos_ptrs = POS + pos_offsets * stride_posd  # pos has shape (M, D) and we only use first T rows
    pos_val = tl.load(pos_ptrs, mask=mask, other=0.0).to(tl.float32)

    y_val += scale * pos_val
    tl.store(y_ptrs, y_val, mask=mask)


def _gelu_triton(x: torch.Tensor) -> torch.Tensor:
    # GELU via Triton, elementwise
    y = torch.empty_like(x)
    N = x.numel()
    # Launch 1D grid
    BLOCK = 1024
    gelu_tanh_kernel[(triton.cdiv(N, BLOCK),)](x, y, N, BLOCK=BLOCK)
    return y


def _linear_no_bias_triton(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    # X: (B, T, K), W: (M, K), output Y: (B, T, M)
    assert X.is_cuda and W.is_cuda
    B, T, K = X.shape
    M = W.shape[0]
    Y = torch.empty((B, T, M), device=X.device, dtype=X.dtype)

    # Strides
    stride_xb, stride_xt, stride_xk = X.stride()
    stride_wm, stride_wk = W.stride()
    stride_yb, stride_yt, stride_ym = Y.stride()

    # Tile sizes: M=1024, K=3840
    BLOCK_M = 64
    BLOCK_K = 256
    grid = (B, T, triton.cdiv(M, BLOCK_M))
    linear_no_bias_kernel[grid](
        X, W, Y,
        B, T, K, M,
        stride_xb, stride_xt, stride_xk,
        stride_wm, stride_wk,
        stride_yb, stride_yt, stride_ym,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K
    )
    return Y


def _add_scaled_pos_emb_triton(Y: torch.Tensor, POS: torch.Tensor, scale: float) -> None:
    # Y: (B, T, M), POS: (M, D), add scale * POS[:T, :] to Y
    B, T, M = Y.shape
    D = POS.shape[1]
    assert POS.shape[0] == M, "POS first dim must equal M"
    # Ensure POS is contiguous along D
    POS = POS.contiguous()

    # We flatten Y for 1D launch: N = B*T*M
    y_flat = Y.view(-1)
    N = y_flat.numel()
    # Strides for Y (contiguous)
    stride_yb = Y.stride(0)
    stride_yt = Y.stride(1)
    stride_ym = Y.stride(2)

    # For POS, we use strides on (M, D) layout
    stride_posm = POS.stride(0)
    stride_posd = POS.stride(1)

    BLOCK = 1024
    add_scaled_pos_emb_kernel[(triton.cdiv(N, BLOCK),)](
        y_flat, POS, scale, B, T, M, D,
        stride_yb, stride_yt, stride_ym,
        stride_posm, stride_posd,
        BLOCK=BLOCK
    )


@torch.no_grad()
def run_triton_only(
    input_features: torch.Tensor,
    conv2d1_weight: torch.Tensor,
    conv2d1_bias: torch.Tensor,
    conv2d2_weight: torch.Tensor,
    conv2d2_bias: torch.Tensor,
    conv2d3_weight: torch.Tensor,
    conv2d3_bias: torch.Tensor,
    conv_out_weight: torch.Tensor,
    positional_embedding: torch.Tensor,
    embed_scale: float,
):
    # Stage 1: Conv2d (1 -> 384 channels) + GELU
    x1 = torch.nn.functional.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
    x1 = _gelu_triton(x1)

    # Stage 2: Conv2d (384 -> 384 channels) + GELU
    x2 = torch.nn.functional.conv2d(x1, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
    x2 = _gelu_triton(x2)

    # Stage 3: Conv2d (384 -> 384 channels) + GELU
    x3 = torch.nn.functional.conv2d(x2, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
    x3 = _gelu_triton(x3)

    # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
    b, c, f, t = x3.size()
    x3 = x3.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

    # Linear projection to d_model (no bias) using Triton
    x4 = _linear_no_bias_triton(x3, conv_out_weight)

    # Scale embeddings
    x4 = x4.to(torch.float32)  # compute in fp32, output in fp32
    scale = float(embed_scale)
    # Add scaled positional embeddings via Triton
    _add_scaled_pos_emb_triton(x4, positional_embedding, scale)

    return x4


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect the same args as the original Model
        # args: input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale
        assert len(args) == 10, "ModelNew expects 10 arguments as in the original run function."
        return run_triton_only(*args)


def run(*args):
    return ModelNew()(*args)
