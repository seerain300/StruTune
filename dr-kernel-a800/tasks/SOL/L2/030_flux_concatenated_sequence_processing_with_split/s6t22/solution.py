import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    A_ptr,        # *f32, [B, M, H] = encoder_hidden_states
    B_ptr,        # *f32, [B, N, H] = hidden_states (image)
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,    # batch size
    M: tl.constexpr,    # text_seq_len
    N: tl.constexpr,    # img_seq_len
    H: tl.constexpr,    # hidden_dim
    stride_ab,    # stride along batch for A
    stride_am,    # stride along seq (M) for A
    stride_ah,    # stride along hidden for A
    stride_bb,    # stride along batch for B
    stride_bn,    # stride along seq (N) for B
    stride_bh,    # stride along hidden for B
    stride_ob,    # stride along batch for Out
    stride_oc,    # stride along seq (C) for Out
    stride_oh,    # stride along hidden for Out
    BLOCK_M: tl.constexpr,  # tile size along M (text)
    BLOCK_N: tl.constexpr,  # tile size along N (image)
):
    # Grid: (B, tiles along C)
    b = tl.program_id(0)
    tile = tl.program_id(1)

    # We process the concatenation by filling Out[b, c, h] where c in [0, M) from A, and c in [M, M+N) from B (shift by M).
    # We iterate tiles over total C; for each tile, we copy up to BLOCK_M from A and up to BLOCK_N from B.
    total = M + N
    tiles = tl.cdiv(total, BLOCK_M)

    for t in range(0, tiles):
        start = t * BLOCK_M
        # First, copy from A for up to BLOCK_M lanes that are within M
        a_count = M - start
        if a_count > 0:
            a_count = tl.minimum(a_count, BLOCK_M)
            m_offsets = start + tl.arange(0, BLOCK_M)  # sequence indices from A
            # hidden dimension offsets
            h_offsets = tl.arange(0, H)
            # pointers
            a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h_offsets[None, :] * stride_ah
            out_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_oc + h_offsets[None, :] * stride_oh
            mask = (m_offsets[:, None] < M) & (h_offsets[None, :] < H)
            vals = tl.load(a_ptrs, mask=mask, other=0.0)
            tl.store(out_ptrs, vals, mask=mask)

        # Then, copy from B for up to BLOCK_N lanes that map to N
        n_offsets = tl.arange(0, BLOCK_N)  # sequence indices from B
        b_ptrs = B_ptr + b * stride_bb + n_offsets[:, None] * stride_bn + h_offsets[None, :] * stride_bh
        # out indices start at M + start, and cover BLOCK_N
        out_ptrs2 = Out_ptr + b * stride_ob + (M + start + n_offsets[:, None]) * stride_oc + h_offsets[None, :] * stride_oh
        mask2 = (start + n_offsets[:, None] < M + N) & (h_offsets[None, :] < H)
        vals2 = tl.load(b_ptrs, mask=mask2, other=0.0)
        tl.store(out_ptrs2, vals2, mask=mask2)


@triton.jit
def _batched_gemm_kernel(
    X_ptr,  # *f32, [B, C, K] where X is concatenated output
    W_ptr,  # *f32, [K, K] process_weight
    P_ptr,  # *f32, [B, C, K] result
    B: tl.constexpr,      # batch size
    C: tl.constexpr,      # sequence length (M + N)
    K: tl.constexpr,      # hidden dim
    stride_xb,  # stride along batch for X
    stride_xc,  # stride along seq for X
    stride_xk,  # stride along hidden for X
    stride_w0,  # stride along dim 0 for W (rows)
    stride_w1,  # stride along dim 1 for W (cols)
    stride_pb,  # stride along batch for P
    stride_pc,  # stride along seq for P
    stride_pk,  # stride along hidden for P
    BLOCK_M: tl.constexpr,  # tile along C (sequences)
    BLOCK_N: tl.constexpr,  # tile along K (hidden)
    BLOCK_K: tl.constexpr,  # tile along reduction dim
):
    # Grid: (B, tiles along C, tiles along K)
    b = tl.program_id(0)
    cm = tl.program_id(1)
    cn = tl.program_id(2)

    m_offsets = cm * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = cn * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_m = m_offsets < C
    mask_n = n_offsets < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # Load X tile [BLOCK_M, BLOCK_K]: X[b, m, k]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W^T tile [BLOCK_K, BLOCK_N]: W[k, n]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(x_tile, w_tile)  # [BLOCK_M, BLOCK_N]

    # Store result P[b, m, n]
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=store_mask)


def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-optimized concatenation + GEMM + split. Heavy computation in Triton.
    Returns:
      processed_encoder: [batch, text_seq_len, hidden_dim]
      processed_hidden:  [batch, img_seq_len, hidden_dim]
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA."
    B = hidden_states.shape[0]
    M = encoder_hidden_states.shape[1]
    N = hidden_states.shape[1]
    H = hidden_states.shape[2]

    # Ensure contiguous and float32
    A = encoder_hidden_states.contiguous().to(torch.float32)   # [B, M, H]
    Bseq = hidden_states.contiguous().to(torch.float32)        # [B, N, H]
    W = process_weight.contiguous().to(torch.float32)          # [H, H]

    # Concatenate into X [B, C, H], C = M + N
    C = M + N
    X = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

    # Triton concatenation
    BLOCK_M = 128
    BLOCK_N = 128
    grid_concat = (B, triton.cdiv(C, BLOCK_M))
    _concat_seq_kernel[grid_concat](
        A, Bseq, X,
        B, M, N, H,
        A.stride(0), A.stride(1), A.stride(2),
        Bseq.stride(0), Bseq.stride(1), Bseq.stride(2),
        X.stride(0), X.stride(1), X.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )

    # GEMM: P = X @ W^T -> [B, C, H]
    P = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

    stride_xb, stride_xc, stride_xk = X.stride(0), X.stride(1), X.stride(2)
    stride_w0, stride_w1 = W.stride(0), W.stride(1)
    stride_pb, stride_pc, stride_pk = P.stride(0), P.stride(1), P.stride(2)

    BLOCK_Mg = 64
    BLOCK_Ng = 64
    BLOCK_Kg = 64
    grid_gemm = (B, triton.cdiv(C, BLOCK_Mg), triton.cdiv(H, BLOCK_Ng))
    _batched_gemm_kernel[grid_gemm](
        X, W, P,
        B, C, H,
        stride_xb, stride_xc, stride_xk,
        stride_w0, stride_w1,
        stride_pb, stride_pc, stride_pk,
        BLOCK_M=BLOCK_Mg, BLOCK_N=BLOCK_Ng, BLOCK_K=BLOCK_Kg,
        num_warps=4, num_stages=2,
    )

    # Split into (encoded, hidden) streams, same shape and order as original:
    # processed_encoder: [B, M, H], processed_hidden: [B, N, H]
    processed_encoder = P[:, :M, :]
    processed_hidden = P[:, M:, :]

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
