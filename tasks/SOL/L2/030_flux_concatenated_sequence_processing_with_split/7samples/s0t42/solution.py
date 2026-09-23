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
    # Grid: (B, tiles over M, tiles over K)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    k_block = tl.program_id(2)

    M = T + I

    m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = m < M
    mask_k = k < K

    # Determine which source to load from: encoder (x1) if m < T, else image (x2)
    src_x1 = m < T

    # Compute addresses
    out_offsets = b * stride_ob + m[:, None] * stride_om + k[None, :] * stride_ok
    x1_offsets = b * stride_x1b + m[:, None] * stride_x1t + k[None, :] * stride_x1k
    x2_offsets = b * stride_x2b + (m[:, None] - T) * stride_x2i + k[None, :] * stride_x2k

    # Masks for loads
    mask_x1 = mask_m[:, None] & mask_k[None, :] & src_x1[:, None]
    mask_x2 = mask_m[:, None] & mask_k[None, :] & (~src_x1)[:, None]

    # Load values
    a1 = tl.load(x1_ptr + x1_offsets, mask=mask_x1, other=0.0)
    a2 = tl.load(x2_ptr + x2_offsets, mask=mask_x2, other=0.0)
    a = a1 + a2  # a2 contributes where m >= T, a1 where m < T; masked loads ensure out-of-range are 0

    # Store to output
    tl.store(out_ptr + out_offsets, a, mask=mask_m[:, None] & mask_k[None, :])


@triton.jit
def batched_matmul_kernel(
    C_ptr,          # *fp32, output [B, M, N] where N=K
    A_ptr,          # *fp32, input [B, M, K]
    W_ptr,          # *fp32, weight [K, K] (note: W is process_weight.T)
    B: tl.constexpr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    stride_cb, stride_cm, stride_cn,
    stride_ab, stride_am, stride_ak,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, tiles along M, tiles along N)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, K, BLOCK_K):
        kk = k_block + tl.arange(0, BLOCK_K)

        # Load A tile: [BM, BK]
        A_offsets = b * stride_ab + m[:, None] * stride_am + kk[None, :] * stride_ak
        A_mask = (m[:, None] < M) & (kk[None, :] < K)
        A_tile = tl.load(A_ptr + A_offsets, mask=A_mask, other=0.0)  # [BM, BK]

        # Load W tile: [BK, BN]
        W_offsets = kk[:, None] * stride_wk + n[None, :] * stride_wn
        W_mask = (kk[:, None] < K) & (n[None, :] < N)
        W_tile = tl.load(W_ptr + W_offsets, mask=W_mask, other=0.0)  # [BK, BN]

        # Accumulate
        acc += tl.dot(A_tile, W_tile)  # [BM, BN]

    # Store result
    C_offsets = b * stride_cb + m[:, None] * stride_cm + n[None, :] * stride_cn
    C_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(C_ptr + C_offsets, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]

        # Make inputs contiguous and float32 for kernel computation
        x1 = encoder_hidden_states.contiguous().to(torch.float32)
        x2 = hidden_states.contiguous().to(torch.float32)
        W = process_weight.contiguous().to(torch.float32)  # [K, K]

        M = T + I

        # 1) Concatenate sequences into A [B, M, K] using Triton
        A = torch.empty((B, M, K), device=x1.device, dtype=torch.float32)

        BLOCK_M = 64
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

        # 2) Compute processed = A @ W using Triton GEMM. Here N=K, output [B, M, K]
        C = torch.empty((B, M, K), device=x1.device, dtype=torch.float32)

        BLOCK_MG = 64
        BLOCK_NG = 64
        BLOCK_KG = 64

        grid_gemm = (B, triton.cdiv(M, BLOCK_MG), triton.cdiv(K, BLOCK_NG))
        batched_matmul_kernel[grid_gemm](
            C, A, W,
            B, M, K, K,  # N=K
            C.stride(0), C.stride(1), C.stride(2),
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1),
            BLOCK_M=BLOCK_MG, BLOCK_N=BLOCK_NG, BLOCK_K=BLOCK_KG,
            num_warps=4, num_stages=2,
        )

        # 3) Split outputs back into separate streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtypes
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
