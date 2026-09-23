import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_dim1_kernel(
    out_ptr,         # *fp32, A: [B, M, K], M = T + I
    x1_ptr,          # *fp32, encoder_hidden_states: [B, T, K]
    x2_ptr,          # *fp32, hidden_states: [B, I, K]
    B: tl.constexpr,
    T: tl.constexpr,  # text_seq_len
    I: tl.constexpr,  # img_seq_len
    K: tl.constexpr,  # hidden_dim
    OUT_s0, OUT_s1, OUT_s2,
    X1_s0, X1_s1, X1_s2,
    X2_s0, X2_s1, X2_s2,
):
    # Program IDs: batch, tiles over M, tiles over K
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    k_block = tl.program_id(2)

    BLOCK_M = 128
    BLOCK_K = 64

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)       # indices along M (sequence)
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)       # indices along K (hidden_dim)

    mask_m = m_offsets < (T + I)
    mask_k = k_offsets < K

    # Determine if this m corresponds to encoder (x1) or image (x2)
    is_encoder = m_offsets < T  # vector mask

    # Compute output pointers for the current tile
    out_base = out_ptr + b * OUT_s0
    out_ptrs = out_base + m_offsets[:, None] * OUT_s1 + k_offsets[None, :] * OUT_s2

    # Compute source pointers
    # For encoder: x1[b, m, k]
    x1_base = x1_ptr + b * X1_s0
    x1_ptrs = x1_base + m_offsets[:, None] * X1_s1 + k_offsets[None, :] * X1_s2

    # For image: x2[b, (m - T), k]
    x2_base = x2_ptr + b * X2_s0
    x2_m = m_offsets - T
    x2_ptrs = x2_base + x2_m[:, None] * X2_s1 + k_offsets[None, :] * X2_s2

    # Select source based on is_encoder
    # Create a 2D tile of zeros
    val_tile = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    # For encoder positions, load from x1; for image positions, load from x2.
    # We need to apply mask for valid m and k.
    mask_tile = mask_m[:, None] & mask_k[None, :]

    # Load from x1 where is_encoder is True
    load_x1 = tl.load(x1_ptrs, mask=(mask_tile & is_encoder[:, None]), other=0.0)
    # Load from x2 where is_encoder is False
    load_x2 = tl.load(x2_ptrs, mask=(mask_tile & (~is_encoder)[:, None]), other=0.0)

    # Select based on is_encoder
    # Cast is_encoder to float for selection
    is_encoder_f = is_encoder[:, None].to(tl.float32)
    val_tile = load_x1 * is_encoder_f + load_x2 * (1.0 - is_encoder_f)

    # Store to output
    tl.store(out_ptrs, val_tile, mask=mask_tile)


@triton.jit
def batched_matmul_kernel(
    C_ptr, A_ptr, W_ptr,
    B: tl.constexpr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,  # N is hidden_dim here (same as K)
    A_s0, A_s1, A_s2,
    W_s0, W_s1,
    C_s0, C_s1, C_s2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (B, tiles over M, tiles over N)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # A[b, m, k] tile
        A_base = A_ptr + b * A_s0
        A_ptrs = A_base + m_offsets[:, None] * A_s1 + k_offsets[None, :] * A_s2
        A_tile = tl.load(A_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)

        # W[k, n] tile (W is [K, N])
        W_ptrs = W_ptr + k_offsets[:, None] * W_s0 + n_offsets[None, :] * W_s1
        W_tile = tl.load(W_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, W_tile)

    # Store result to C[b, m, n]
    C_base = C_ptr + b * C_s0
    C_ptrs = C_base + m_offsets[:, None] * C_s1 + n_offsets[None, :] * C_s2
    tl.store(C_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, I, K]
        encoder_hidden_states: torch.Tensor,  # [B, T, K]
        process_weight: torch.Tensor,         # [K, K]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension in Triton
        - Apply linear projection (matmul) in Triton
        - Split back into separate streams
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]
        M = T + I

        # 1) Allocate and concatenate along sequence dim using Triton
        A = torch.empty((B, M, K), device=hidden_states.device, dtype=torch.float32)

        grid_concat = (B, triton.cdiv(M, 128), triton.cdiv(K, 64))
        concat_seq_dim1_kernel[grid_concat](
            A, encoder_hidden_states, hidden_states,
            B, T, I, K,
            A.stride(0), A.stride(1), A.stride(2),
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            BLOCK_M=128, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # 2) Compute C = A @ process_weight in Triton
        # Ensure process_weight is fp32
        W = process_weight.to(torch.float32)

        C = torch.empty((B, M, K), device=hidden_states.device, dtype=torch.float32)

        BLOCK_M_G = 128
        BLOCK_N_G = 128
        BLOCK_K_G = 64
        grid_matmul = (B, triton.cdiv(M, BLOCK_M_G), triton.cdiv(K, BLOCK_N_G))
        batched_matmul_kernel[grid_matmul](
            C, A, W,
            B, M, K, K,
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2,
        )

        # 3) Split back into two streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtype
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
