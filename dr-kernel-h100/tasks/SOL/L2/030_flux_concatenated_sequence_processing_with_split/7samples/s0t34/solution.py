import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_dim1_kernel(
    out_ptr,        # *fp32, output A: [B, M, K], M = T + I
    x1_ptr,         # *fp32, encoder_hidden_states: [B, T, K]
    x2_ptr,         # *fp32, hidden_states: [B, I, K]
    B: tl.constexpr, # batch size
    T: tl.constexpr, # text_seq_len
    I: tl.constexpr, # img_seq_len
    K: tl.constexpr, # hidden_dim
    out_s0, out_s1, out_s2,
    x1_s0, x1_s1, x1_s2,
    x2_s0, x2_s1, x2_s2,
):
    # Grid: (B, tiles over M = T + I, tiles over K)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    k_block = tl.program_id(2)

    BLOCK_M = 128
    BLOCK_K = 64

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # sequence indices in A
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # hidden indices

    M_total = T + I
    mask_m = m_offsets < M_total
    mask_k = k_offsets < K

    # Determine if this m belongs to encoder (x1) or image (x2)
    is_encoder = m_offsets < T

    # For each k in this tile, write to out[b, m, k] depending on segment
    for k_idx in range(0, BLOCK_K):
        k = k_offsets[k_idx]
        if not mask_k[k_idx]:
            break
        # Compute addresses for each source
        # Out: out[b, m, k] -> b*out_s0 + m*out_s1 + k*out_s2
        # X1: x1[b, m, k] -> b*x1_s0 + m*x1_s1 + k*x1_s2 (only if is_encoder)
        # X2: x2[b, m-T, k] -> b*x2_s0 + (m - T)*x2_s1 + k*x2_s2 (only if not is_encoder)
        out_row = out_ptr + b * out_s0 + m_offsets * out_s1 + k * out_s2
        x1_row = x1_ptr + b * x1_s0 + m_offsets * x1_s1 + k * x1_s2
        x2_row = x2_ptr + b * x2_s0 + (m_offsets - T) * x2_s1 + k * x2_s2

        # Masks
        mask = mask_m & mask_k[k_idx]
        # Select source based on is_encoder
        src_row = tl.where(is_encoder[:, None], x1_row, x2_row)

        # Load from source (masked), store to out
        val = tl.load(src_row, mask=mask[:, None], other=0.0)
        tl.store(out_row, val, mask=mask[:, None])


@triton.jit
def batched_matmul_kernel(
    C_ptr,  # *fp32, output [B, M, N] where N=K
    A_ptr,  # *fp32, input A: [B, M, K]
    W_ptr,  # *fp32, weight: [K, N] (note: N is typically K, but can be different)
    B: tl.constexpr,  # batch size
    M: tl.constexpr,  # sequence length after concat (T + I)
    K: tl.constexpr,  # hidden dim
    N: tl.constexpr,  # output hidden dim (can be K)
    A_s0, A_s1, A_s2,
    W_s0, W_s1,
    C_s0, C_s1, C_s2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A tiles: A[b, m, k] -> [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + pid_b * A_s0 + m_offsets[:, None] * A_s1 + k_offsets[None, :] * A_s2
        a = tl.load(a_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)

        # Load W tiles: W[k, n] -> [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * W_s0 + n_offsets[None, :] * W_s1
        w = tl.load(w_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)

        # Fused matmul update
        acc += tl.dot(a, w)

    # Store results
    c_ptrs = C_ptr + pid_b * C_s0 + m_offsets[:, None] * C_s1 + n_offsets[None, :] * C_s2
    tl.store(c_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.

        - Concatenates encoder_hidden_states and hidden_states along the sequence dimension (dim=1) into A [B, M, K], M=T+I.
        - Computes C = A @ process_weight.T (no bias).
        - Splits C back into encoder and image streams along the sequence dimension.

        Returns (processed_encoder_hidden_states, processed_hidden_states).
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = encoder_hidden_states.shape[2]
        assert hidden_states.shape[2] == K, "hidden_states and encoder_hidden_states must have the same hidden_dim"
        assert process_weight.shape[0] == K and process_weight.shape[1] == K, "process_weight must be [hidden_dim, hidden_dim]"

        # Ensure inputs are contiguous and float32 for Triton kernels
        x1 = encoder_hidden_states.contiguous().to(torch.float32)
        x2 = hidden_states.contiguous().to(torch.float32)
        W = process_weight.contiguous().to(torch.float32)  # [K, K]

        # 1) Concatenate along sequence dimension into A [B, M, K]
        M = T + I
        A = torch.empty((B, M, K), dtype=torch.float32, device=x1.device)

        # Grid for concat kernel: (B, tiles over M, tiles over K)
        BLOCK_M = 128
        BLOCK_K = 64
        grid_concat = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
        concat_seq_dim1_kernel[grid_concat](
            A, x1, x2,
            B, T, I, K,
            A.stride(0), A.stride(1), A.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            x2.stride(0), x2.stride(1), x2.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Compute C = A @ W, W is [K, K]
        C = torch.empty((B, M, K), dtype=torch.float32, device=A.device)

        BLOCK_M_G = 64
        BLOCK_N_G = 64
        BLOCK_K_G = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M_G), triton.cdiv(K, BLOCK_N_G))
        batched_matmul_kernel[grid_gemm](
            C, A, W,
            B, M, K, K,  # N=K, but we can generalize if needed
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2,
        )

        # 3) Split back into separate streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtypes to match the original API
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
