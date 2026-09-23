import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_dim1_kernel(
    out_ptr,          # *fp32, output A: [B, M, K], M = T + I
    in1_ptr,          # *fp32, encoder_hidden_states: [B, T, K]
    in2_ptr,          # *fp32, hidden_states: [B, I, K]
    B: tl.constexpr,  # batch size
    T: tl.constexpr,  # text_seq_len
    I: tl.constexpr,  # img_seq_len
    K: tl.constexpr,  # hidden_dim
    out_s0, out_s1, out_s2,   # strides for out (A)
    in1_s0, in1_s1, in1_s2,   # strides for in1
    in2_s0, in2_s1, in2_s2,   # strides for in2
):
    # Grid: (B, tiles over M, tiles over K)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    k_block = tl.program_id(2)

    BLOCK_M = 128
    BLOCK_K = 64

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # indices in concatenated sequence
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # hidden_dim indices

    M_tot = T + I
    mask_m = m_offsets < M_tot
    mask_k = k_offsets < K

    # Determine source tensor for each m (encoder for m < T, image for m >= T)
    is_encoder = m_offsets < T  # boolean mask per m

    # For each k in the tile, load from the appropriate source and store to out
    for kk in range(BLOCK_K):
        k = k_offsets[kk]
        if not mask_k[kk]:
            continue
        # Compute addresses
        # out[b, m, k] -> out_addr = b*out_s0 + m*out_s1 + k*out_s2
        out_addr = b * out_s0 + m_offsets * out_s1 + k * out_s2  # vectorized over m_offsets

        # src addresses:
        # if encoder: in1[b, m, k] -> in1_addr = b*in1_s0 + m*in1_s1 + k*in1_s2
        # else:       in2[b, (m - T), k] -> in2_addr = b*in2_s0 + (m - T)*in2_s1 + k*in2_s2
        in1_addr = b * in1_s0 + m_offsets * in1_s1 + k * in1_s2
        in2_addr = b * in2_s0 + (m_offsets - T) * in2_s1 + k * in2_s2

        # Load from the correct source per m
        # Since is_encoder is per m, we use tl.where to pick the src pointer
        src_vals = tl.where(is_encoder, in1_ptr + in1_addr, in2_ptr + in2_addr)
        # Mask: only load/store when mask_m and mask_k
        load_mask = mask_m & mask_k[kk]
        vals = tl.load(src_vals, mask=load_mask, other=0.0)
        tl.store(out_ptr + out_addr, vals, mask=mask_m & mask_k[kk])


@triton.jit
def batched_matmul_kernel(
    C_ptr,    # *fp32, output [B, M, K]
    A_ptr,    # *fp32, input [B, M, K]
    W_ptr,    # *fp32, weight [K, K] (since process_weight is [hidden_dim, hidden_dim] with K)
    B: tl.constexpr,  # batch size
    M: tl.constexpr,  # total sequence length (T + I)
    K: tl.constexpr,  # hidden_dim (row/col of W)
    out_s0, out_s1, out_s2,  # strides for C
    A_s0, A_s1, A_s2,        # strides for A
    W_s0, W_s1,              # strides for W (2D)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, tiles over M, tiles over N), N == K here
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tiles: A[b, m, k]
        A_ptrs = A_ptr + b * A_s0 + m_offsets[:, None] * A_s1 + k_offsets[None, :] * A_s2
        mask_A = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_vals = tl.load(A_ptrs, mask=mask_A, other=0.0)

        # Load W tiles: W[k, n]
        W_ptrs = W_ptr + k_offsets[:, None] * W_s0 + n_offsets[None, :] * W_s1
        mask_W = (k_offsets[:, None] < K) & (n_offsets[None, :] < K)
        W_vals = tl.load(W_ptrs, mask=mask_W, other=0.0)

        # Accumulate
        acc += tl.dot(A_vals, W_vals)

    # Store results to C[b, m, n]
    C_ptrs = C_ptr + b * out_s0 + m_offsets[:, None] * out_s1 + n_offsets[None, :] * out_s2
    mask_C = (m_offsets[:, None] < M) & (n_offsets[None, :] < K)
    tl.store(C_ptrs, acc, mask=mask_C)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,            # [B, I, K]
        encoder_hidden_states: torch.Tensor,   # [B, T, K]
        process_weight: torch.Tensor,          # [K, K]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of:
          concatenated = cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, K]
          processed = concatenated @ process_weight.T                        # [B, T+I, K]
          processed_encoder = processed[:, :T, :]
          processed_hidden = processed[:, T:, :]
        """
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]

        # Ensure inputs are float32 for Triton
        x1 = encoder_hidden_states.contiguous().to(torch.float32)  # [B, T, K]
        x2 = hidden_states.contiguous().to(torch.float32)          # [B, I, K]
        W = process_weight.contiguous().to(torch.float32)          # [K, K]

        # Allocate A: concatenated along sequence dim -> [B, M, K], M = T + I
        M = T + I
        A = torch.empty((B, M, K), device=x1.device, dtype=torch.float32)

        # Launch concat kernel
        grid_concat = (B, triton.cdiv(M, 128), triton.cdiv(K, 64))
        concat_seq_dim1_kernel[grid_concat](
            A, x1, x2,
            B, T, I, K,
            A.stride(0), A.stride(1), A.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            x2.stride(0), x2.stride(1), x2.stride(2),
            BLOCK_M=128, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # Allocate C: [B, M, K]
        C = torch.empty((B, M, K), device=x1.device, dtype=torch.float32)

        # Launch GEMM kernel: C = A @ W
        BLOCK_M_G = 128
        BLOCK_N_G = 128
        BLOCK_K_G = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M_G), triton.cdiv(K, BLOCK_N_G))
        batched_matmul_kernel[grid_gemm](
            C, A, W,
            B, M, K,
            C.stride(0), C.stride(1), C.stride(2),
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2,
        )

        # Split outputs
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtypes
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden