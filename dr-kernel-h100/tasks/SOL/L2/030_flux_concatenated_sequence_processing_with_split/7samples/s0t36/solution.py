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
    OUT_s0, OUT_s1, OUT_s2,    # strides for A
    X1_s0, X1_s1, X1_s2,       # strides for x1
    X2_s0, X2_s1, X2_s2,       # strides for x2
    BLOCK_M: tl.constexpr,     # tile over sequence
    BLOCK_K: tl.constexpr,     # tile over hidden
):
    # Grid: (B, tiles over M = T+I, tiles over K)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    k_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    M = T + I
    mask_m = m_offsets < M
    mask_k = k_offsets < K

    # Determine if this m corresponds to encoder (x1) or image (x2)
    is_encoder = m_offsets < T

    # For each k in the tile, write out values to A
    for kk in range(0, BLOCK_K):
        k = k_offsets[kk]
        if not mask_k[kk]:
            break
        # Compute base offsets for out, x1, x2 for each m
        out_base = b * OUT_s0 + m_offsets * OUT_s1 + k * OUT_s2  # [BLOCK_M]
        x1_base = b * X1_s0 + m_offsets * X1_s1 + k * X1_s2      # [BLOCK_M]
        x2_base = b * X2_s0 + (m_offsets - T) * X2_s1 + k * X2_s2  # [BLOCK_M]

        # Create pointers for this k
        out_ptrs = out_ptr + out_base
        # Load from x1 for encoder positions, x2 for image positions, masked by is_encoder
        x1_ptrs = x1_ptr + x1_base
        x2_ptrs = x2_ptr + x2_base

        # Masked load: for positions not in encoder, use x2; otherwise x1
        # Use tl.where to select source based on is_encoder
        # But since m_offsets may include non-encoder positions, we need to compute masks per m
        # For safety, compute combined source pointer: if m < T -> x1; else -> x2
        # We'll build the mask for each m:
        mask_m_encoder = m_offsets < T
        # Load from x1 where mask_m & mask_m_encoder, else load from x2
        val = tl.load(x1_ptrs, mask=mask_m & mask_m_encoder, other=0.0)
        val = tl.where(mask_m_encoder, val, tl.load(x2_ptrs, mask=mask_m & (~mask_m_encoder), other=0.0))

        # Store to output
        tl.store(out_ptrs, val, mask=mask_m)


@triton.jit
def batched_matmul_kernel(
    C_ptr,  # *fp32, output [B, M, K], but we'll compute into [B, M, N] where N=K
    A_ptr,  # *fp32, input A: [B, M, K]
    W_ptr,  # *fp32, weight: [K, N], here N=K, but we generalize
    B: tl.constexpr,  # batch size
    M: tl.constexpr,  # sequence length (T + I)
    K: tl.constexpr,  # hidden_dim (input feature)
    N: tl.constexpr,  # output feature (here K)
    stride_ab, stride_am, stride_ak,   # strides for A: (B, M, K)
    stride_wk, stride_wn,              # strides for W: (K, N)
    stride_cb, stride_cm, stride_cn,   # strides for C: (B, M, N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, tiles along M, tiles along N)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    k_offsets = tl.arange(0, BLOCK_K)  # [BLOCK_K]

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # Accumulator [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for kk in range(0, K, BLOCK_K):
        k = kk + k_offsets  # [BLOCK_K]
        mask_k = k < K

        # Load A[b, m, k] -> shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + k[None, :] * stride_ak
        A_tile = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load W[k, n] -> shape [BLOCK_K, BLOCK_N]
        W_ptrs = W_ptr + k[:, None] * stride_wk + n_offsets[None, :] * stride_wn
        W_tile = tl.load(W_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate: acc += A_tile @ W_tile
        acc += tl.dot(A_tile, W_tile)

    # Store result to C[b, m, n] = acc
    C_ptrs = C_ptr + b * stride_cb + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure inputs are on CUDA and contiguous
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA device"
        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = encoder_hidden_states.shape[2]
        assert hidden_states.shape[2] == K, "hidden_states and encoder_hidden_states must have the same hidden_dim"
        assert process_weight.shape[0] == K and process_weight.shape[1] == K, "process_weight must be [hidden_dim, hidden_dim]"

        # 1) Concatenate along sequence dimension using Triton
        M = T + I
        A = torch.empty((B, M, K), dtype=torch.float32, device=encoder_hidden_states.device)

        # Strides (assume float32 pointers)
        OUT_s0, OUT_s1, OUT_s2 = A.stride(0), A.stride(1), A.stride(2)
        X1_s0, X1_s1, X1_s2 = encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2)
        X2_s0, X2_s1, X2_s2 = hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2)

        # Tile sizes: moderate to avoid OOB
        BLOCK_M = 64
        BLOCK_K = 64
        grid_concat = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
        concat_seq_dim1_kernel[grid_concat](
            A, encoder_hidden_states, hidden_states,
            B, T, I, K,
            OUT_s0, OUT_s1, OUT_s2,
            X1_s0, X1_s1, X1_s2,
            X2_s0, X2_s1, X2_s2,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Prepare weight: W = process_weight.T
        W = process_weight.t().contiguous()
        # Ensure dtype float32 for kernel
        W = W.to(torch.float32)

        # 3) Compute C = A @ W using Triton GEMM
        N = K  # output feature size equals hidden_dim
        C = torch.empty((B, M, N), dtype=torch.float32, device=encoder_hidden_states.device)

        # Strides for A, W, C
        stride_ab, stride_am, stride_ak = A.stride(0), A.stride(1), A.stride(2)
        stride_wk, stride_wn = W.stride(0), W.stride(1)
        stride_cb, stride_cm, stride_cn = C.stride(0), C.stride(1), C.stride(2)

        BLOCK_M_G = 64
        BLOCK_N_G = 64
        BLOCK_K_G = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M_G), triton.cdiv(N, BLOCK_N_G))
        batched_matmul_kernel[grid_gemm](
            C, A, W,
            B, M, K, N,
            stride_ab, stride_am, stride_ak,
            stride_wk, stride_wn,
            stride_cb, stride_cm, stride_cn,
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2,
        )

        # 4) Split back into encoder and hidden streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtypes
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
