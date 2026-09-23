import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    C_ptr,          # *fp32, output [B, M, N], N=K
    A_ptr,          # *fp32, input [B, M, K] = concatenated sequences
    W_ptr,          # *fp32, weight [K, N] = process_weight.T
    B: tl.int32,    # batch size
    M: tl.int32,    # total sequence length T + I
    K: tl.int32,    # hidden dim
    N: tl.int32,    # hidden dim (same as K)
    stride_ab: tl.int32, stride_am: tl.int32, stride_ak: tl.int32,  # strides for A
    stride_wk: tl.int32, stride_wn: tl.int32,                      # strides for W
    stride_cb: tl.int32, stride_cm: tl.int32, stride_cn: tl.int32,  # strides for C
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch: (B, tiles over M, tiles over N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A tile: [BM, BK]
        a_ptrs = A_ptr + pid_b * stride_ab + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = mask_m[:, None] & mask_k[None, :]
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BM, BK]

        # Load W tile: [BK, BN]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BK, BN]

        # Accumulate
        acc += tl.dot(a_tile, w_tile)

    # Store results to C: [B, M, N]
    c_ptrs = C_ptr + pid_b * stride_cb + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run:
        - Concatenate sequences along sequence dimension
        - Apply linear projection via Triton GEMM
        - Split back into separate encoder and image streams
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]  # text_seq_len
        I = hidden_states.shape[1]          # img_seq_len
        K = hidden_states.shape[2]          # hidden_dim

        # 1) Concatenate along sequence dimension: [B, T + I, K]
        A = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Ensure contiguous and dtype for kernel
        A = A.contiguous().to(torch.float32)
        process_weight = process_weight.contiguous().to(torch.float32)
        W = process_weight.t()  # [K, K]

        # Allocate output
        M = T + I
        C = torch.empty((B, M, K), device=A.device, dtype=torch.float32)

        # Launch Triton GEMM
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        batched_matmul_kernel[grid](
            C, A, W,
            B, M, K, K,   # N = K
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Split outputs back into encoder and hidden streams
        processed_encoder = C[:, :T, :]  # [B, T, K]
        processed_hidden = C[:, T:, :]  # [B, I, K]

        # Cast back to original dtypes to match the original API
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden