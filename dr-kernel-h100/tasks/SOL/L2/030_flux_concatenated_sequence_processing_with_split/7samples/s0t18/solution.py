import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_dim1_kernel(
    OUT_ptr,          # *fp32, output A: [B, M, K], M = T + I
    X1_ptr,           # *fp32, encoder_hidden_states: [B, T, K]
    X2_ptr,           # *fp32, hidden_states: [B, I, K]
    B: tl.constexpr,  # batch size
    T: tl.constexpr,  # text_seq_len
    I: tl.constexpr,  # img_seq_len
    K: tl.constexpr,  # hidden_dim
    OUT_s0, OUT_s1, OUT_s2,
    X1_s0, X1_s1, X1_s2,
    X2_s0, X2_s1, X2_s2,
    BLOCK_M: tl.constexpr,  # tile along sequence (M)
    BLOCK_K: tl.constexpr,  # tile along hidden dim (K)
):
    # Grid: (B, tiles over M = T+I, tiles over K)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    k_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # sequence positions in concatenated
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # hidden_dim indices

    mask_m = m_offsets < (T + I)
    mask_k = k_offsets < K

    # Determine if this m corresponds to encoder (x1) or image (x2)
    is_encoder = m_offsets < T

    # Loop over K tile and write concatenated rows
    for ki in range(0, BLOCK_K):
        k = k_offsets[ki]
        if not mask_k[ki]:
            break
        # Compute addresses for each source
        # x1[b, m, k] -> addr1 = b*X1_s0 + m*X1_s1 + k*X1_s2
        # x2[b, (m-T), k] -> addr2 = b*X2_s0 + (m - T)*X2_s1 + k*X2_s2
        addr1 = b * X1_s0 + m_offsets * X1_s1 + k * X1_s2
        addr2 = b * X2_s0 + (m_offsets - T) * X2_s1 + k * X2_s2

        # Load with masks (encoder or image)
        vals1 = tl.load(X1_ptr + addr1, mask=mask_m & is_encoder & (m_offsets < T), other=0.0)
        vals2 = tl.load(X2_ptr + addr2, mask=mask_m & (~is_encoder), other=0.0)

        # Select based on is_encoder; where is_encoder True -> vals1 else vals2
        # is_encoder is a vector (mask on m_offsets), broadcast over k
        val = tl.where(is_encoder, vals1, vals2)

        # Store to OUT[b, m, k]
        out_addr = b * OUT_s0 + m_offsets * OUT_s1 + k * OUT_s2
        tl.store(OUT_ptr + out_addr, val, mask=mask_m)


@triton.jit
def batched_matmul_kernel(
    C_ptr, A_ptr, W_ptr,
    B: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
    A_s0, A_s1, A_s2,       # strides for A: [B, M, K]
    W_s0, W_s1,             # strides for W: [K, K]
    C_s0, C_s1, C_s2,       # strides for C: [B, M, K]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 3D launch grid: (batch, tiles along M, tiles along N)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of A (sequence positions)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # output columns (hidden_dim)

    mask_m = m_offsets < M
    mask_n = n_offsets < K

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A[b, m, k] tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + b * A_s0 + m_offsets[:, None] * A_s1 + k_offsets[None, :] * A_s2
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load W[k, n] tile: shape [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * W_s0 + n_offsets[None, :] * W_s1
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # acc += a @ w  => [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, w)

    # Store the result tile
    c_ptrs = C_ptr + b * C_s0 + m_offsets[:, None] * C_s1 + n_offsets[None, :] * C_s2
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dim into A [B, M, K].
        2) Compute C = A @ process_weight using Triton batched matmul.
        3) Split C back into processed_encoder and processed_hidden.
        """
        # Inputs: encoder_hidden_states [B, T, K], hidden_states [B, I, K], process_weight [K, K]
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]

        # Ensure dtypes are float32 for kernels; keep a reference for output dtype
        in_dtype = hidden_states.dtype
        out_dtype = in_dtype  # keep same as input

        # Allocate A as float32
        A = torch.empty((B, T + I, K), dtype=torch.float32, device=hidden_states.device)

        # Launch concat kernel
        BLOCK_M = 128
        BLOCK_K = 64
        grid_concat = (B, triton.cdiv(T + I, BLOCK_M), triton.cdiv(K, BLOCK_K))
        concat_seq_dim1_kernel[grid_concat](
            A,
            encoder_hidden_states.to(torch.float32),
            hidden_states.to(torch.float32),
            B, T, I, K,
            A.stride(0), A.stride(1), A.stride(2),
            encoder_hidden_states.to(torch.float32).stride(0), encoder_hidden_states.to(torch.float32).stride(1), encoder_hidden_states.to(torch.float32).stride(2),
            hidden_states.to(torch.float32).stride(0), hidden_states.to(torch.float32).stride(1), hidden_states.to(torch.float32).stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Allocate C as float32
        C = torch.empty((B, T + I, K), dtype=torch.float32, device=hidden_states.device)

        # Ensure process_weight is float32 and contiguous
        W = process_weight.to(torch.float32).contiguous()

        # Launch batched matmul kernel
        BLOCK_M_G = 64
        BLOCK_N_G = 64
        BLOCK_K_G = 64
        grid_gemm = (B, triton.cdiv(T + I, BLOCK_M_G), triton.cdiv(K, BLOCK_N_G))
        batched_matmul_kernel[grid_gemm](
            C, A, W,
            B, T + I, K,
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2
        )

        # Split back into two streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast outputs back to original dtype
        processed_encoder = processed_encoder.to(out_dtype)
        processed_hidden = processed_hidden.to(out_dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
