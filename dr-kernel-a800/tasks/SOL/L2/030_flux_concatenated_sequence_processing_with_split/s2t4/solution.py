import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    A1_ptr,  # [B, T, H]
    A2_ptr,  # [B, I, H]
    OUT_ptr, # [B, L, H], L = T + I
    B_size, T, I, H, L,
    A1_stride_b, A1_stride_m, A1_stride_k,
    A2_stride_b, A2_stride_m, A2_stride_k,
    OUT_stride_b, OUT_stride_m, OUT_stride_k,
):
    # Grid: (B, tiles along sequence L, 1)
    pid_b = tl.program_id(0)
    pid_m_tile = tl.program_id(1)

    # Choose a block size along L
    BLOCK_M = 128
    m_offsets = pid_m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < L

    # For each position in the tile, decide source and load
    for i in range(0, BLOCK_M):
        m = m_offsets[i]
        valid = m < L
        is_from_A1 = (m < T) & valid
        is_from_A2 = (m >= T) & valid
        src = tl.where(is_from_A1, m, m - T)

        # Load the row from the chosen source (k = 0..H-1)
        k = tl.arange(0, H)
        a1_addr = A1_ptr + pid_b * A1_stride_b + m * A1_stride_m + k * A1_stride_k
        a2_addr = A2_ptr + pid_b * A2_stride_b + src * A2_stride_m + k * A2_stride_k
        a1_vals = tl.load(a1_addr, mask=valid & is_from_A1, other=0.0)
        a2_vals = tl.load(a2_addr, mask=valid & is_from_A2, other=0.0)
        sel = tl.where(is_from_A1, a1_vals, a2_vals)

        # Store into OUT[b, m, :]
        out_addr = OUT_ptr + pid_b * OUT_stride_b + m * OUT_stride_m + k * OUT_stride_k
        tl.store(out_addr, sel, mask=valid)


@triton.jit
def _batched_gemm_kernel(
    A_ptr,  # [B, L, K] where K=H
    B_ptr,  # [K, N] where N=H, here B is process_weight.T
    C_ptr,  # [B, L, N]
    B_size, L, K, N,
    A_stride_b, A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_b, C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, tiles along L, tiles along N)
    pid_b = tl.program_id(0)
    pid_m_tile = tl.program_id(1)
    pid_n_tile = tl.program_id(2)

    m_offsets = pid_m_tile * BLOCK_M + tl.arange(0, BLOCK_M)  # along sequence
    n_offsets = pid_n_tile * BLOCK_N + tl.arange(0, BLOCK_N)  # along output columns

    # Accumulator for [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_tile_ptr = A_ptr + pid_b * A_stride_b + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        A_mask = (m_offsets[:, None] < L) & (k_offsets[None, :] < K)
        A_tile = tl.load(A_tile_ptr, mask=A_mask, other=0.0).to(tl.float32)

        # Load B tile: [BLOCK_K, BLOCK_N]
        B_tile_ptr = B_ptr + k_offsets[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        B_tile = tl.load(B_tile_ptr, mask=B_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Store results to C[b, m, n] for the tile
    C_tile_ptr = C_ptr + pid_b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    C_mask = (m_offsets[:, None] < L) & (n_offsets[None, :] < N)
    tl.store(C_tile_ptr, acc, mask=C_mask)


@triton.jit
def _split_streams_kernel(
    IN_ptr,  # [B, L, H]
    OUT1_ptr,  # [B, T, H]
    OUT2_ptr,  # [B, I, H]
    B_size, T, I, H, L,
    IN_stride_b, IN_stride_m, IN_stride_k,
    OUT1_stride_b, OUT1_stride_m, OUT1_stride_k,
    OUT2_stride_b, OUT2_stride_m, OUT2_stride_k,
):
    pid_b = tl.program_id(0)

    # Simple streaming copy: first T rows to OUT1, then next I rows to OUT2
    for ti in range(0, T):
        k = tl.arange(0, H)
        in_addr = IN_ptr + pid_b * IN_stride_b + ti * IN_stride_m + k * IN_stride_k
        out1_addr = OUT1_ptr + pid_b * OUT1_stride_b + ti * OUT1_stride_k + k * OUT1_stride_k
        val = tl.load(in_addr)
        tl.store(out1_addr, val)

    for ij in range(0, I):
        k = tl.arange(0, H)
        in_addr = IN_ptr + pid_b * IN_stride_b + (ij + T) * IN_stride_m + k * IN_stride_k
        out2_addr = OUT2_ptr + pid_b * OUT2_stride_b + ij * OUT2_stride_m + k * OUT2_stride_k
        val = tl.load(in_addr)
        tl.store(out2_addr, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
          - Concatenates sequences along the sequence dimension using a Triton kernel (no torch.cat).
          - Performs the linear projection using a Triton batched GEMM (no torch.matmul).
          - Splits results back into encoder and hidden streams using a Triton kernel (no torch slicing).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        # Ensure contiguous and float32 for stable math
        encoder = encoder_hidden_states.contiguous().float()
        hidden = hidden_states.contiguous().float()
        Wt = process_weight.t().contiguous().float()  # [H, H]

        # 1) Concatenate sequences into A_cat [B, L, H] using Triton
        A_cat = torch.empty((B, L, H), device=hidden.device, dtype=torch.float32)
        BLOCK_M_concat = 128
        grid_concat = (B, triton.cdiv(L, BLOCK_M_concat))
        _concat_sequences_kernel[grid_concat](
            encoder, hidden, A_cat,
            B, T, I, H, L,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            A_cat.stride(0), A_cat.stride(1), A_cat.stride(2),
            num_warps=4, num_stages=3,
        )

        # 2) GEMM: processed = A_cat @ Wt, output [B, L, H]
        processed = torch.empty((B, L, H), device=hidden.device, dtype=torch.float32)

        # Tile sizes chosen conservatively for robustness
        BLOCK_M_gemm = 64
        BLOCK_N_gemm = 64
        BLOCK_K_gemm = 64

        grid_gemm = (B, triton.cdiv(L, BLOCK_M_gemm), triton.cdiv(H, BLOCK_N_gemm))
        _batched_gemm_kernel[grid_gemm](
            A_cat, Wt, processed,
            B, L, H, H,  # A_cat: [B, L, H], Wt: [H, H]
            A_cat.stride(0), A_cat.stride(1), A_cat.stride(2),
            Wt.stride(0), Wt.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=BLOCK_M_gemm, BLOCK_N=BLOCK_N_gemm, BLOCK_K=BLOCK_K_gemm,
            num_warps=4, num_stages=3,
        )

        # 3) Split back into encoder and hidden streams using Triton
        processed_encoder = torch.empty((B, T, H), device=hidden.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=hidden.device, dtype=torch.float32)

        grid_split = (B,)
        _split_streams_kernel[grid_split](
            processed, processed_encoder, processed_hidden,
            B, T, I, H, L,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
