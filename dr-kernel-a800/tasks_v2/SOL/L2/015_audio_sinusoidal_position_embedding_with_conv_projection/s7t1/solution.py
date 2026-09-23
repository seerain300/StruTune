import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _linear_matmul_kernel(
    X_ptr,          # *bf16, [B*S, K], contiguous
    W_ptr,          # *bf16, [N, K], contiguous (conv_out_weight)
    Y_ptr,          # *bf16, [B*S, N], contiguous
    B,              # int
    S,              # int
    K,              # int
    N,              # int
    stride_x_row,   # int = K (stride along rows of X)
    stride_x_k,     # int = 1 (stride along K)
    stride_w_n,     # int = K (stride along N of W)
    stride_w_k,     # int = 1 (stride along K of W)
    stride_y_row,   # int = N (stride along rows of Y)
    stride_y_n,     # int = 1 (stride along N of Y)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # program ids: rows over B*S, columns over N tiles
    pid_row = tl.program_id(0)  # row index in [0, B*S)
    pid_n_block = tl.program_id(1)

    # compute (b, s) from pid_row
    s_idx = pid_row % S
    b_idx = pid_row // S

    # output channel offsets for this block
    n_offsets = pid_n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # loop over K dimension in tiles
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # load x_vec: X[pid_row, k_offsets]
        x_ptrs = X_ptr + pid_row * stride_x_row + k_offsets * stride_x_k
        x_vec = tl.load(x_ptrs, mask=k_offsets < K, other=0.0).to(tl.float32)  # [BLOCK_K]

        # load W_tile: W[n_offsets, k_offsets] -> shape [BLOCK_N, BLOCK_K]
        w_ptrs = W_ptr + n_offsets[:, None] * stride_w_n + k_offsets[None, :] * stride_w_k
        w_tile = tl.load(w_ptrs, mask=(n_offsets[:, None] < N) & (k_offsets[None, :] < K), other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_K]

        # accumulate acc += sum_k w_tile[:, k] * x_vec[k]
        acc += tl.sum(w_tile * x_vec[None, :], axis=1)

    # store Y[pid_row, n_offsets]
    y_ptrs = Y_ptr + pid_row * stride_y_row + n_offsets * stride_y_n
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=n_offsets < N)


@triton.jit
def _scale_elementwise_kernel(X_ptr, Y_ptr, N_total, scale):
    # N_total = B * S * N
    pid = tl.program_id(0)
    offsets = pid * tl.num_programs(0) + tl.arange(0, tl.num_programs(0))
    mask = offsets < N_total
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x * scale
    tl.store(Y_ptr + offsets, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _add_pos_emb_kernel(Y_ptr, POS_ptr, B, S, N, stride_y_b, stride_y_s, stride_y_n, stride_pos_s, stride_pos_n):
    # Launch grid: (B, S, ceil_div(N, BLOCK_N)) or 2D if desired. Here we use 2D with inner loop over N.
    # Triton kernels require 1D grid. We therefore launch grid=(B*S, ceil_div(N, BLOCK_N)) and compute b,s,n inside.
    pid_bs = tl.program_id(0)
    pid_n_block = tl.program_id(1)

    b_idx = pid_bs // S
    s_idx = pid_bs % S

    n_offsets = pid_n_block * 128 + tl.arange(0, 128)  # 128 is BLOCK_N; can be constexpr
    mask = n_offsets < N

    # Load y_row: Y[b, s, n_offsets]
    y_row_ptrs = Y_ptr + b_idx * stride_y_b + s_idx * stride_y_s + n_offsets * stride_y_n
    y_row = tl.load(y_row_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load pos_emb[s, n_offsets]
    pos_row_ptrs = POS_ptr + s_idx * stride_pos_s + n_offsets * stride_pos_n
    pos_row = tl.load(pos_row_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Add and store back
    y_row = y_row + pos_row
    tl.store(y_row_ptrs, y_row.to(tl.bfloat16), mask=mask)


def triton_linear_proj(x_rowwise: torch.Tensor, w: torch.Tensor, embed_scale: float) -> torch.Tensor:
    """
    x_rowwise: [B, S, K] viewed as [B*S, K], bfloat16, CUDA, contiguous
    w: [N, K], bfloat16, CUDA, contiguous
    Returns y: [B, S, N], bfloat16
    """
    B, S, K = x_rowwise.shape
    N, Kw = w.shape
    assert Kw == K, "Weight K must match x K dimension"
    x = x_rowwise.contiguous()  # [B*S, K]
    w_c = w.contiguous()        # [N, K]
    y = torch.empty((B, S, N), device=x.device, dtype=torch.bfloat16)  # [B, S, N]

    BLOCK_N = 128
    BLOCK_K = 64
    grid = (B * S, triton.cdiv(N, BLOCK_N))
    _linear_matmul_kernel[grid](
        x, w_c, y.view(-1, N),
        B, S, K, N,
        x.stride(0), x.stride(1),
        w_c.stride(0), w_c.stride(1),
        y.stride(0), y.stride(2),
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
    N_total = y_c.numel()
    grid = (triton.cdiv(N_total, 1024),)
    _scale_elementwise_kernel[grid](y_c, z, N_total, scale, num_warps=4, num_stages=2)
    return z


def triton_add_pos_emb(y: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
    """
    y: [B, S, N], bfloat16, CUDA
    pos_emb: [S, N], bfloat16, CUDA
    Adds pos_emb to y (broadcast across batch).
    """
    B, S, N = y.shape
    y_c = y.contiguous()
    # Ensure pos_emb is on same device and dtype
    pos_emb_c = pos_emb.to(torch.bfloat16).contiguous()
    # We need to write back to y_c; Triton kernel will add in-place per batch/sequence
    grid = (B * S, triton.cdiv(N, 128))
    _add_pos_emb_kernel[grid](
        y_c, pos_emb_c,
        B, S, N,
        y_c.stride(0), y_c.stride(1), y_c.stride(2),
        pos_emb_c.stride(0), pos_emb_c.stride(1),
        num_warps=2, num_stages=2
    )
    return y_c


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Unpack inputs as provided by get_inputs
        input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale = args

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Linear projection to d_model (no bias)
        B, S, K = x.shape
        N = conv_out_weight.shape[0]  # 1024
        x_rowwise = x.view(B * S, K).contiguous()  # [B*S, K]

        # Triton matmul: output y_rowwise [B*S, N], then view to [B, S, N]
        y = triton_linear_proj(x_rowwise, conv_out_weight, embed_scale)  # [B, S, N]

        # Scale embeddings
        y_scaled = triton_scale_elementwise(y, embed_scale)

        # Add positional embeddings (broadcast over batch)
        pos_emb = positional_embedding[:S, :].to(torch.bfloat16)  # [S, N]
        y_final = triton_add_pos_emb(y_scaled, pos_emb)

        return y_final


def run(*args):
    return ModelNew()(*args)
