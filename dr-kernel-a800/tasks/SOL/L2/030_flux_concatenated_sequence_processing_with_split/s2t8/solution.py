import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    encoder_ptr,    # *const T, shape [B, T, H]
    hidden_ptr,     # *const T, shape [B, I, H]
    out_ptr,        # *T, shape [B, L, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    hidden_stride_b, hidden_stride_i, hidden_stride_h,
    out_stride_b, out_stride_l, out_stride_h,
    BLOCK_M: tl.constexpr,  # tile over sequence positions (m)
    BLOCK_K: tl.constexpr,  # tile over hidden_dim (h)
):
    # program ids
    b = tl.program_id(0)  # batch
    m_block = tl.program_id(1)  # tile index along sequence

    # offsets
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = tl.arange(0, BLOCK_K)                      # [BLOCK_K]

    # bounds masks
    m_mask = m_offsets < (T + I)
    k_mask = k_offsets < H

    # loop over sequence positions in this tile
    for m in range(BLOCK_M):
        m_idx = m_offsets[m]
        # decide source: encoder or hidden
        from_encoder = m_idx < T
        # compute pointers
        # For encoder: ptr_e = encoder[b, m_idx, k]
        # For hidden: ptr_h = hidden[b, m_idx - T, k]
        # Note: m_idx is scalar; build a 1D pointer for k
        e_ptrs = encoder_ptr + b * encoder_stride_b + m_idx * encoder_stride_t + k_offsets * encoder_stride_h
        h_ptrs = hidden_ptr + b * hidden_stride_b + (m_idx - T) * hidden_stride_i + k_offsets * hidden_stride_h
        out_ptrs = out_ptr + b * out_stride_b + m_idx * out_stride_l + k_offsets * out_stride_h

        # load
        e_vals = tl.load(e_ptrs, mask=k_mask & from_encoder, other=0)
        h_vals = tl.load(h_ptrs, mask=k_mask & (~from_encoder), other=0)
        # combine
        a_vals = e_vals + h_vals  # either e_vals or h_vals depending on mask
        # store
        tl.store(out_ptrs, a_vals, mask=k_mask & m_mask[m])


@triton.jit
def _batched_matmul_kernel(
    A_ptr,  # *const T, [B, L, K], contiguous along K (last dim)
    B_ptr,  # *const T, [K, N], contiguous along K (last dim)
    C_ptr,  # *T, [B, L, N]
    B: tl.constexpr, L: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    A_stride_b, A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_b, C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr,  # tile over sequence positions (m)
    BLOCK_N: tl.constexpr,  # tile over output columns (n)
    BLOCK_K: tl.constexpr,  # tile over reduction (k)
):
    # program ids
    b = tl.program_id(0)       # batch
    m_block = tl.program_id(1) # tile along sequence
    n_block = tl.program_id(2) # tile along output columns

    # tile offsets
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # masks
    m_mask = m_offsets < L
    n_mask = n_offsets < N

    # accumulator [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # keep accumulation in fp32 for stability

    # loop over reduction dimension K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offsets < K

        # load A_tile: shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + b * A_stride_b + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        A_mask = m_mask[:, None] & k_mask[None, :]
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0)

        # load B_tile: shape [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k_offsets[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n
        B_mask = k_mask[:, None] & n_mask[None, :]
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0)

        # accumulate: acc += A_tile @ B_tile
        # A_tile: [BM, BK], B_tile: [BK, BN] -> [BM, BN]
        # cast to fp32 for accumulation
        acc += tl.dot(A_tile.to(tl.float32), B_tile.to(tl.float32))

    # store acc into C
    C_ptrs = C_ptr + b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    C_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version that:
          - Performs sequence concatenation in a Triton kernel (no torch.cat on host).
          - Performs the linear projection (matmul) in a Triton batched GEMM kernel.
          - Splits the result back into encoder and hidden streams.
        """
        # Validate device and ensure contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        B = hidden_states.size(0)
        T = encoder_hidden_states.size(1)
        I = hidden_states.size(1)
        H = hidden_states.size(2)
        K = H
        N = H
        L = T + I

        # Make inputs contiguous
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        W = process_weight.contiguous()  # [H, H]
        # We need B for matmul as [K, N] = [H, H]
        W_T = W.t().contiguous()  # [H, H]

        # Allocate output for concatenated sequence [B, L, H]
        A_cat = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch concat kernel
        # Choose tiles: BLOCK_M over sequence, BLOCK_K over hidden_dim
        BLOCK_M = 128
        BLOCK_K = 128

        grid_concat = (B, triton.cdiv(L, BLOCK_M))
        _concat_seq_kernel[grid_concat](
            encoder, hidden, A_cat,
            B=B, T=T, I=I, H=H,
            encoder_stride_b=encoder.stride(0), encoder_stride_t=encoder.stride(1), encoder_stride_h=encoder.stride(2),
            hidden_stride_b=hidden.stride(0), hidden_stride_i=hidden.stride(1), hidden_stride_h=hidden.stride(2),
            out_stride_b=A_cat.stride(0), out_stride_l=A_cat.stride(1), out_stride_h=A_cat.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Allocate output for processed [B, L, H]
        processed = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch batched matmul kernel: C = A_cat @ W_T
        BLOCK_M_gemm = 128
        BLOCK_N_gemm = 128
        BLOCK_K_gemm = 64

        grid_matmul = (B, triton.cdiv(L, BLOCK_M_gemm), triton.cdiv(H, BLOCK_N_gemm))
        _batched_matmul_kernel[grid_matmul](
            A_cat, W_T, processed,
            B=B, L=L, K=K, N=N,
            A_stride_b=A_cat.stride(0), A_stride_m=A_cat.stride(1), A_stride_k=A_cat.stride(2),
            B_stride_k=W_T.stride(0), B_stride_n=W_T.stride(1),
            C_stride_b=processed.stride(0), C_stride_m=processed.stride(1), C_stride_n=processed.stride(2),
            BLOCK_M=BLOCK_M_gemm, BLOCK_N=BLOCK_N_gemm, BLOCK_K=BLOCK_K_gemm,
            num_warps=4, num_stages=3,
        )

        # Split back into separate streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
