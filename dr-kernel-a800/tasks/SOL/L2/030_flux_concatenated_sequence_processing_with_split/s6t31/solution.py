import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    A_ptr,        # *f32, [B, M, H] (encoder_hidden_states)
    B_ptr,        # *f32, [B, N, H] (hidden_states)
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,    # batch size (constexpr for specialization)
    M: tl.constexpr,    # text_seq_len (constexpr)
    N: tl.constexpr,    # img_seq_len (constexpr)
    H: tl.constexpr,    # hidden_dim (constexpr)
    stride_ab,    # int: stride along batch for A
    stride_am,    # int: stride along seq for A
    stride_ah,    # int: stride along hidden for A
    stride_bb,    # int: stride along batch for B
    stride_bn,    # int: stride along seq for B
    stride_bh,    # int: stride along hidden for B
    stride_ob,    # int: stride along batch for Out
    stride_oc,    # int: stride along seq for Out
    stride_oh,    # int: stride along hidden for Out
    BLOCK_M: tl.constexpr,  # tile along C for programs
):
    # Grid: (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    c_block = tl.program_id(1)

    C = M + N
    m_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # sequence indices [BLOCK_M]
    mask_c = m_offsets < C

    # Determine which positions come from A vs B
    from_A = m_offsets < M
    m_in_B = m_offsets - M  # valid where from_A is False

    # Hidden dimension offsets [0..H)
    h = tl.arange(0, H)

    # Compute pointers for A and B sources
    # A[b, m, h]
    a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h[None, :] * stride_ah
    a_mask = mask_c[:, None] & (h[None, :] < H)
    a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, H]

    # B[b, m-N, h] for m >= M
    b_ptrs = B_ptr + b * stride_bb + m_in_B[:, None] * stride_bn + h[None, :] * stride_bh
    b_mask = (~from_A)[:, None] & mask_c[:, None] & (h[None, :] < H)
    b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_M, H]

    # Select based on from_A
    vals = tl.where(from_A[:, None], a_vals, b_vals)  # [BLOCK_M, H]

    # Store to Out[b, m_offsets, h]
    out_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_oc + h[None, :] * stride_oh
    store_mask = mask_c[:, None] & (h[None, :] < H)
    tl.store(out_ptrs, vals, mask=store_mask)


@triton.jit
def _batched_matmul_kernel(
    X_ptr,  # *f32, [B, C, K] (C=M+N, K=H)
    W_ptr,  # *f32, [K, K] (process_weight)
    P_ptr,  # *f32, [B, C, K]
    B: tl.constexpr,          # batch size
    C: tl.constexpr,          # sequence length
    K: tl.constexpr,          # hidden dim (H)
    # Strides for X: treated as (B, C, K)
    stride_xb,  # int
    stride_xc,  # int
    stride_xk,  # int
    # Strides for W: [K, K]
    stride_w0,  # int: row stride
    stride_w1,  # int: col stride
    # Strides for P: [B, C, K]
    stride_pb,  # int
    stride_pc,  # int
    stride_pk,  # int
    BLOCK_M: tl.constexpr,  # tile over C (sequences)
    BLOCK_N: tl.constexpr,  # tile over K (hidden dim)
    BLOCK_K: tl.constexpr,  # reduction tile over K
):
    # 3D grid: (B, ceil_div(C, BLOCK_M), ceil_div(K, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_m = m_offsets < C
    mask_n = n_offsets < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # Load X[b, m, k] tile -> [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W[k, n] tile -> [BLOCK_K, BLOCK_N] (note: we need W[k, n], rows vary k, cols vary n)
        w_ptrs = W_ptr + k_offsets[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc += x_tile @ w_tile
        acc += tl.dot(x_tile, w_tile)

    # Store results to P[b, m, n]
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    tl.store(p_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, N, H] - image latent sequence
        encoder_hidden_states: [B, M, H] - text conditioning sequence
        process_weight: [H, H] - linear projection weight
        Returns: (processed_encoder, processed_hidden)
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32."
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]  # text_seq_len
        N = hidden_states.shape[1]          # img_seq_len
        H = hidden_states.shape[2]          # hidden_dim
        assert process_weight.shape == (H, H)

        C = M + N

        # 1) Concatenate sequences into X [B, C, H] using Triton
        X = torch.empty((B, C, H), device=hidden_states.device, dtype=hidden_states.dtype)

        BLOCK_M_concat = 128
        grid_concat = (B, triton.cdiv(C, BLOCK_M_concat))
        _concat_seq_kernel[grid_concat](
            encoder_hidden_states, hidden_states, X,
            B=B, M=M, N=N, H=H,
            stride_ab=encoder_hidden_states.stride(0), stride_am=encoder_hidden_states.stride(1), stride_ah=encoder_hidden_states.stride(2),
            stride_bb=hidden_states.stride(0), stride_bn=hidden_states.stride(1), stride_bh=hidden_states.stride(2),
            stride_ob=X.stride(0), stride_oc=X.stride(1), stride_oh=X.stride(2),
            BLOCK_M=BLOCK_M_concat,
            num_warps=4, num_stages=2
        )

        # 2) Compute P = X @ W^T using Triton GEMM
        P = torch.empty((B, C, H), device=hidden_states.device, dtype=hidden_states.dtype)

        BLOCK_M_gemm = 64
        BLOCK_N_gemm = 64
        BLOCK_K_gemm = 64
        grid_gemm = (B, triton.cdiv(C, BLOCK_M_gemm), triton.cdiv(H, BLOCK_N_gemm))
        _batched_matmul_kernel[grid_gemm](
            X, process_weight, P,
            B=B, C=C, K=H,
            stride_xb=X.stride(0), stride_xc=X.stride(1), stride_xk=X.stride(2),
            stride_w0=process_weight.stride(0), stride_w1=process_weight.stride(1),
            stride_pb=P.stride(0), stride_pc=P.stride(1), stride_pk=P.stride(2),
            BLOCK_M=BLOCK_M_gemm, BLOCK_N=BLOCK_N_gemm, BLOCK_K=BLOCK_K_gemm,
            num_warps=4, num_stages=2
        )

        # 3) Split P into two outputs
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
