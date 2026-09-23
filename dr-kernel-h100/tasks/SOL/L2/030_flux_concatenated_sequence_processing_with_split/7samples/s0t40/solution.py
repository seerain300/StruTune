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
    stride_ob, stride_om, stride_ok,
    stride_x1b, stride_x1t, stride_x1k,
    stride_x2b, stride_x2i, stride_x2k,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along sequence dim (T + I)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)  # along hidden dim

    # Masks for bounds
    mask_m = m_offsets < (T + I)
    mask_k = k_offsets < K

    # Determine source tensor: x1 for m < T, x2 for m >= T
    use_x1 = m_offsets < T  # vector bool

    # Build base offsets for out, x1, x2
    out_base = pid_b * stride_ob + m_offsets[:, None] * stride_om + k_offsets[None, :] * stride_ok

    # Load from x1 where applicable
    x1_ptrs = x1_ptr + pid_b * stride_x1b + m_offsets[:, None] * stride_x1t + k_offsets[None, :] * stride_x1k
    x1_mask = mask_m[:, None] & mask_k[None, :]
    x1_vals = tl.load(x1_ptrs, mask=x1_mask, other=0.0)

    # Load from x2 where applicable
    x2_ptrs = x2_ptr + pid_b * stride_x2b + (m_offsets[:, None] - T) * stride_x2i + k_offsets[None, :] * stride_x2k
    # For m >= T, m_offsets - T >= 0; for m < T, (m_offsets - T) < 0 -> masked out by use_x1
    x2_mask = (mask_m[:, None] & mask_k[None, :] & (~use_x1) & (m_offsets[:, None] >= T))
    x2_vals = tl.load(x2_ptrs, mask=x2_mask, other=0.0)

    # Select values based on use_x1
    # Where use_x1 is True, x2_vals may be all zeros; we must not use it. To avoid dtype ambiguity, construct via where.
    vals = tl.where(use_x1[:, None], x1_vals, tl.zeros_like(x1_vals)) + tl.where((~use_x1)[:, None], x2_vals, tl.zeros_like(x2_vals))

    # Store to out
    store_mask = mask_m[:, None] & mask_k[None, :]
    tl.store(out_ptr + out_base, vals, mask=store_mask)


@triton.jit
def batched_matmul_kernel(
    C_ptr, A_ptr, W_ptr,
    B, M, K, N,  # here N=K, but keep generic
    stride_cb, stride_cm, stride_cn,
    stride_ab, stride_am, stride_ak,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, tiles along M, tiles along N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Pointers for A tile: [BM, BK]
        A_ptrs = A_ptr + pid_b * stride_ab + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)  # shape [BM, BK], float32

        # Pointers for W tile: [BK, BN]
        W_ptrs = W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn
        W_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        W_tile = tl.load(W_ptrs, mask=W_mask, other=0.0)  # shape [BK, BN], float32

        # Accumulate
        acc += tl.dot(A_tile, W_tile)

    # Store results
    C_ptrs = C_ptr + pid_b * stride_cb + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, K]
        encoder_hidden_states: [B, T, K]
        process_weight: [K, K]
        Returns:
            processed_encoder: [B, T, K]
            processed_hidden: [B, I, K]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]
        M = T + I

        # Ensure contiguous inputs and cast to fp32 for stable accumulation in Triton
        x1 = encoder_hidden_states.contiguous().to(torch.float32)  # [B, T, K]
        x2 = hidden_states.contiguous().to(torch.float32)         # [B, I, K]
        # Allocate output for concatenation A_out: [B, M, K]
        A_out = torch.empty((B, M, K), dtype=torch.float32, device=x1.device)

        # Launch concat kernel
        # Choose tiles; keep moderate sizes to avoid register pressure
        BLOCK_M = 64
        BLOCK_K = 64
        grid_concat = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
        concat_seq_dim1_kernel[grid_concat](
            A_out, x1, x2,
            B=B, T=T, I=I, K=K,
            stride_ob=A_out.stride(0), stride_om=A_out.stride(1), stride_ok=A_out.stride(2),
            stride_x1b=x1.stride(0), stride_x1t=x1.stride(1), stride_x1k=x1.stride(2),
            stride_x2b=x2.stride(0), stride_x2i=x2.stride(1), stride_x2k=x2.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

        # Prepare W = process_weight.T (already [K, K])
        W = process_weight.contiguous().to(torch.float32)  # [K, K]

        # Allocate output C: [B, M, K]
        C = torch.empty((B, M, K), dtype=torch.float32, device=x1.device)

        # Launch GEMM kernel
        BLOCK_M_G = 64
        BLOCK_N_G = 64
        BLOCK_K_G = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M_G), triton.cdiv(K, BLOCK_N_G))
        batched_matmul_kernel[grid_gemm](
            C, A_out, W,
            B, M, K, K,
            C.stride(0), C.stride(1), C.stride(2),
            A_out.stride(0), A_out.stride(1), A_out.stride(2),
            W.stride(0), W.stride(1),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
        )

        # Split outputs: processed_encoder = C[:, :T, :], processed_hidden = C[:, T:, :]
        processed_encoder = C[:, :T, :]              # [B, T, K]
        processed_hidden = C[:, T:, :]               # [B, I, K]

        # Cast back to original input dtypes to match original API
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
