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
    M: tl.constexpr, # M = T + I
    K: tl.constexpr, # hidden_dim
    stride_x1b, stride_x1t, stride_x1k,
    stride_x2b, stride_x2i, stride_x2k,
    stride_outb, stride_outm, stride_outk,
    BLOCK_M: tl.constexpr, # tile size along M (sequence)
    BLOCK_K: tl.constexpr, # tile size along K (hidden)
):
    # program ids: b over batch, m_tile over sequence, k_tile over hidden
    b = tl.program_id(0)
    m_tile = tl.program_id(1)
    k_tile = tl.program_id(2)

    # compute tile offsets
    m_offsets = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    # masks to avoid out-of-bounds
    m_mask = m_offsets < M
    k_mask = k_offsets < K

    # Determine whether m belongs to encoder (0..T-1) or image (T..T+I-1)
    is_encoder = m_offsets < T  # boolean vector

    # Prepare pointers for loads
    # For encoder: x1[b, m_offsets, k_offsets]
    x1_ptrs = x1_ptr + b * stride_x1b + m_offsets[:, None] * stride_x1t + k_offsets[None, :] * stride_x1k
    x1_mask = m_mask[:, None] & k_mask[None, :]
    # For image: x2[b, m_offsets - T, k_offsets]
    x2_ptrs = x2_ptr + b * stride_x2b + (m_offsets[:, None] - T) * stride_x2i + k_offsets[None, :] * stride_x2k
    x2_mask = m_mask[:, None] & k_mask[None, :]

    # Load values with masking; other=0.0 for out-of-bounds
    a1 = tl.load(x1_ptrs, mask=x1_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
    a2 = tl.load(x2_ptrs, mask=x2_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

    # Select based on is_encoder
    # If m is encoder position, use a1; else use a2. Broadcasting over k.
    a = tl.where(is_encoder[:, None], a1, a2)

    # Store to output A[b, m_offsets, k_offsets]
    out_ptrs = out_ptr + b * stride_outb + m_offsets[:, None] * stride_outm + k_offsets[None, :] * stride_outk
    store_mask = m_mask[:, None] & k_mask[None, :]
    tl.store(out_ptrs, a, mask=store_mask)


@triton.jit
def batched_gemm_kernel(
    out_ptr,         # *fp32, output C: [B, M, K]
    a_ptr,           # *fp32, input A: [B, M, K]
    w_ptr,           # *fp32, weight W: [K, K]
    B: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
    stride_ab, stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid over (B, tiles along M, tiles along N=K)
    b = tl.program_id(0)
    m_tile = tl.program_id(1)
    n_tile = tl.program_id(2)  # N = K

    m_offsets = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    k_offsets = tl.arange(0, BLOCK_K)                     # [BLOCK_K]

    m_mask = m_offsets < M
    n_mask = n_offsets < K
    k_mask = k_offsets < K

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k_start in range(0, K, BLOCK_K):
        # A tile: a[b, m, k]
        a_ptrs = a_ptr + b * stride_ab + m_offsets[:, None] * stride_am + (k_start + k_offsets[None, :]) * stride_ak
        a_mask = m_mask[:, None] & k_mask[None, :]
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # W tile: w[k, n]
        w_ptrs = w_ptr + (k_start + k_offsets[:, None]) * stride_wk + n_offsets[None, :] * stride_wn
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc += a_tile @ w_tile
        # a_tile: [BM, BK], w_tile: [BK, BN] -> [BM, BN]
        acc += tl.dot(a_tile, w_tile)

    # Store results to C[b, m, n]
    c_ptrs = out_ptr + b * stride_cb + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenates encoder_hidden_states and hidden_states along the sequence dimension in a Triton kernel.
        - Applies linear projection using a Triton GEMM kernel.
        - Splits outputs back into two streams.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton."

        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]
        M = T + I

        # Ensure contiguous and compute in fp32
        # Concatenate along sequence dim using Triton
        x1 = encoder_hidden_states.contiguous().to(torch.float32)
        x2 = hidden_states.contiguous().to(torch.float32)

        # Allocate output A [B, M, K] in fp32
        A = torch.empty((B, M, K), device=x1.device, dtype=torch.float32)

        # Launch concat kernel
        BLOCK_M = 64
        BLOCK_K = 64
        grid_concat = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
        concat_seq_dim1_kernel[grid_concat](
            A, x1, x2,
            B=B, T=T, I=I, M=M, K=K,
            stride_x1b=x1.stride(0), stride_x1t=x1.stride(1), stride_x1k=x1.stride(2),
            stride_x2b=x2.stride(0), stride_x2i=x2.stride(1), stride_x2k=x2.stride(2),
            stride_outb=A.stride(0), stride_outm=A.stride(1), stride_outk=A.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Prepare weight W = process_weight.T in fp32, shape [K, K]
        # process_weight is [K, K] (from original code), transpose to [K, K] if needed
        W = process_weight.contiguous().to(torch.float32).transpose(0, 1)  # [K, K]

        # Allocate output C [B, M, K] in fp32
        C = torch.empty((B, M, K), device=A.device, dtype=torch.float32)

        # Launch GEMM kernel: C = A @ W
        BLOCK_M_G = 64
        BLOCK_N_G = 64
        BLOCK_K_G = 32
        grid_gemm = (B, triton.cdiv(M, BLOCK_M_G), triton.cdiv(K, BLOCK_N_G))
        batched_gemm_kernel[grid_gemm](
            C, A, W,
            B=B, M=M, K=K,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ak=A.stride(2),
            stride_wk=W.stride(0), stride_wn=W.stride(1),
            stride_cb=C.stride(0), stride_cm=C.stride(1), stride_cn=C.stride(2),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2,
        )

        # Split outputs back into encoder and hidden streams
        processed_encoder = C[:, :T, :]                          # [B, T, K]
        processed_hidden = C[:, T:, :]                          # [B, I, K]

        # Cast back to original input dtype for consistency
        # Original inputs typically use the same dtype; match encoder_hidden_states dtype
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden