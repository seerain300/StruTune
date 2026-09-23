import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_kernel(
    out_ptr,         # *fp32, output [B, M, K], M = T + I
    x1_ptr,          # *fp32, encoder_hidden_states: [B, T, K]
    x2_ptr,          # *fp32, hidden_states: [B, I, K]
    B: tl.constexpr, # batch size
    T: tl.constexpr, # text_seq_len
    I: tl.constexpr, # img_seq_len
    K: tl.constexpr, # hidden_dim
    OUT_s0, OUT_s1, OUT_s2,
    X1_s0, X1_s1, X1_s2,
    X2_s0, X2_s1, X2_s2,
):
    # Grid: (B, tiles over M = T+I, tiles over K)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    k_block = tl.program_id(2)

    BLOCK_M = 128
    BLOCK_K = 64

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # indices in concatenated sequence
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # hidden_dim indices

    mask_m = m_offsets < (T + I)
    mask_k = k_offsets < K

    # Determine if this m corresponds to encoder (x1) or image (x2)
    is_encoder = m_offsets < T

    # Loop over K tile and write concatenated rows
    for k_idx in range(0, BLOCK_K):
        k = k_offsets[k_idx]
        if not mask_k[k_idx]:
            break
        # Compute addresses for each source
        # x1[b, m, k] -> addr1 = b*X1_s0 + m*X1_s1 + k*X1_s2
        # x2[b, (m-T), k] -> addr2 = b*X2_s0 + (m - T)*X2_s1 + k*X2_s2
        row_encoder = tl.where(is_encoder, m_offsets, 0)  # ignored when not encoder
        row_image = m_offsets - T  # valid when m >= T

        # Masks for loads: load from x1 when is_encoder True, from x2 otherwise
        mask_x1 = mask_m & is_encoder
        mask_x2 = mask_m & (~is_encoder)

        # Load values (float32 expected)
        v1 = tl.load(x1_ptr + b * X1_s0 + row_encoder * X1_s1 + k * X1_s2, mask=mask_x1, other=0.0)
        v2 = tl.load(x2_ptr + b * X2_s0 + row_image * X2_s1 + k * X2_s2, mask=mask_x2, other=0.0)

        # Sum (concatenation)
        val = v1 + v2  # v2 is 0 where mask_x2 is False

        # Store to output: out[b, m, k]
        out_addr = b * OUT_s0 + m_offsets * OUT_s1 + k * OUT_s2
        tl.store(out_ptr + out_addr, val, mask=mask_m)


@triton.jit
def batched_matmul_kernel(
    C_ptr,           # *fp32, output [B, M, K]
    A_ptr,           # *fp32, input [B, M, K]
    W_ptr,           # *fp32, weight [K, K]
    B: tl.constexpr, # batch size
    M: tl.constexpr, # total sequence length = T + I
    N: tl.constexpr, # K (output hidden_dim)
    K2: tl.constexpr,# K (input hidden_dim)
    A_s0, A_s1, A_s2,
    W_s0, W_s1, W_s2,
    C_s0, C_s1, C_s2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, tiles over M, tiles over N)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K2, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers to A and W tiles
        A_ptrs = A_ptr + b * A_s0 + offs_m[:, None] * A_s1 + offs_k[None, :] * A_s2
        W_ptrs = W_ptr + offs_k[:, None] * W_s0 + offs_n[None, :] * W_s1

        # Masks for boundaries
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K2)
        w_mask = (offs_k[:, None] < K2) & (offs_n[None, :] < N)

        # Load tiles
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        w = tl.load(W_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(a, w)

    # Store results
    C_ptrs = C_ptr + b * C_s0 + offs_m[:, None] * C_s1 + offs_n[None, :] * C_s2
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dim (Triton kernel).
        2) Apply linear projection: processed = concatenated @ process_weight.T (Triton kernel).
        3) Split back into separate encoder and image streams.
        """
        # Ensure inputs are on CUDA and contiguous
        device = encoder_hidden_states.device
        x1 = encoder_hidden_states.contiguous()
        x2 = hidden_states.contiguous()
        W = process_weight.contiguous()  # [K, K]
        B = x1.shape[0]
        T = x1.shape[1]
        I = x2.shape[1]
        K = x1.shape[2]
        M = T + I

        # 1) Concatenate along sequence dimension using Triton
        A = torch.empty((B, M, K), device=device, dtype=torch.float32)
        grid_concat = (B, triton.cdiv(M, 128), triton.cdiv(K, 64))
        concat_seq_kernel[grid_concat](
            A, x1, x2,
            B, T, I, K,
            A.stride(0), A.stride(1), A.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            x2.stride(0), x2.stride(1), x2.stride(2),
            num_warps=4, num_stages=2,
        )

        # 2) Batched GEMM: C = A @ W, output [B, M, K]
        C = torch.empty((B, M, K), device=device, dtype=torch.float32)
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        batched_matmul_kernel[grid_gemm](
            C, A, W,
            B, M, K, K,
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1), W.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split back into two streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtype if needed
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden