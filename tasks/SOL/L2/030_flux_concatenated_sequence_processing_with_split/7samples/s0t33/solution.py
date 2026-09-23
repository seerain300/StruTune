import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_dim1_kernel(
    out_ptr,         # *fp32, output A: [B, M, K], M = T + I
    x1_ptr,          # *fp32, encoder_hidden_states: [B, T, K]
    x2_ptr,          # *fp32, hidden_states: [B, I, K]
    B: tl.int32,
    T: tl.int32,     # text_seq_len
    I: tl.int32,     # img_seq_len
    K: tl.int32,     # hidden_dim
    OUT_s0, OUT_s1, OUT_s2,  # strides for out
    X1_s0, X1_s1, X1_s2,     # strides for x1
    X2_s0, X2_s1, X2_s2,     # strides for x2
    BLOCK_M: tl.constexpr,   # tile over sequence M
    BLOCK_K: tl.constexpr,   # tile over hidden dim K
):
    # program ids
    b = tl.program_id(0)      # batch
    m_block = tl.program_id(1)
    k_block = tl.program_id(2)

    M = T + I

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    mask_m = m_offsets < M
    mask_k = k_offsets < K

    # Determine which source to read from: x1 (encoder) if m < T, else x2 (image)
    use_x1 = m_offsets[:, None] < T  # [BLOCK_M, 1] boolean
    # Compute pointers for out
    out_ptrs = out_ptr + b * OUT_s0 + m_offsets[:, None] * OUT_s1 + k_offsets[None, :] * OUT_s2
    # Load from x1 or x2 depending on segment
    x1_ptrs = x1_ptr + b * X1_s0 + m_offsets[:, None] * X1_s1 + k_offsets[None, :] * X1_s2
    x2_ptrs = x2_ptr + b * X2_s0 + (m_offsets[:, None] - T) * X2_s1 + k_offsets[None, :] * X2_s2
    # Mask for x1: only when m < T and k in bounds; for x2: only when m >= T and k in bounds
    mask_x1 = mask_m[:, None] & mask_k[None, :] & (m_offsets[:, None] < T)
    mask_x2 = mask_m[:, None] & mask_k[None, :] & (m_offsets[:, None] >= T)

    # Load values from appropriate source; other=0 ensures out-of-mask contributes zero
    val1 = tl.load(x1_ptrs, mask=mask_x1, other=0.0)
    val2 = tl.load(x2_ptrs, mask=mask_x2, other=0.0)
    # Select based on use_x1; where use_x1 True -> val1 else val2
    val = tl.where(use_x1, val1, val2)

    # Store to out
    tl.store(out_ptr + b * OUT_s0 + m_offsets[:, None] * OUT_s1 + k_offsets[None, :] * OUT_s2, val, mask=mask_m[:, None] & mask_k[None, :])


@triton.jit
def batched_matmul_kernel(
    C_ptr,       # *fp32, output [B, M, N]
    A_ptr,       # *fp32, input A: [B, M, K]
    W_ptr,       # *fp32, weight: [K, N] (note: we pass W as process_weight.T)
    B: tl.int32, M: tl.int32, K: tl.int32, N: tl.int32,
    A_s0, A_s1, A_s2,  # strides for A
    W_s0, W_s1,        # strides for W ([K, N])
    C_s0, C_s1, C_s2,  # strides for C
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid over (B, tiles along M, tiles along N)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + b * A_s0 + m_offsets[:, None] * A_s1 + k_offsets[None, :] * A_s2
        mask_A = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=mask_A, other=0.0)

        # W tile: [BLOCK_K, BLOCK_N]
        W_ptrs = W_ptr + k_offsets[:, None] * W_s0 + n_offsets[None, :] * W_s1
        mask_W = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        W_tile = tl.load(W_ptrs, mask=mask_W, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, W_tile)

    # Store result
    C_ptrs = C_ptr + b * C_s0 + m_offsets[:, None] * C_s1 + n_offsets[None, :] * C_s2
    mask_C = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=mask_C)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, I, K]
        encoder_hidden_states: torch.Tensor,  # [B, T, K]
        process_weight: torch.Tensor,         # [K, K]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the given PyTorch function.
        - Concatenates encoder_hidden_states and hidden_states along sequence dim (dim=1).
        - Applies linear projection via batched matmul with process_weight.T.
        - Splits the result back into encoder and hidden streams.
        """
        # Shapes
        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = encoder_hidden_states.shape[2]
        M = T + I
        N = K  # since process_weight is [K, K], output per token is K

        # Ensure inputs are on CUDA
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "Inputs must be CUDA tensors for Triton kernels."

        # 1) Allocate A (concatenation output) and run Triton concat kernel
        A = torch.empty((B, M, K), dtype=torch.float32, device=hidden_states.device)

        # Launch concat kernel with explicit strides
        BLOCK_M = 128
        BLOCK_K = 64
        grid_concat = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
        concat_seq_dim1_kernel[grid_concat](
            A,
            encoder_hidden_states,
            hidden_states,
            B, T, I, K,
            A.stride(0), A.stride(1), A.stride(2),
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Prepare W as process_weight.T and launch GEMM Triton kernel
        W = process_weight.t().contiguous().to(torch.float32)  # [K, K]

        C = torch.empty((B, M, N), dtype=torch.float32, device=hidden_states.device)

        BLOCK_M_G = 64
        BLOCK_N_G = 64
        BLOCK_K_G = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M_G), triton.cdiv(N, BLOCK_N_G))
        batched_matmul_kernel[grid_gemm](
            C, A, W,
            B, M, K, N,
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2,
        )

        # 3) Split outputs back into encoder and hidden streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtypes to match the original API
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
