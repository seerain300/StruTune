import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_dim1_kernel(
    out_ptr,        # *fp32, output A: [B, M, K], M = T + I
    x1_ptr,         # *fp32, encoder_hidden_states: [B, T, K]
    x2_ptr,         # *fp32, hidden_states: [B, I, K]
    B: tl.constexpr,  # batch size
    T: tl.constexpr,  # text_seq_len
    I: tl.constexpr,  # img_seq_len
    K: tl.constexpr,  # hidden dim
    stride_out_b, stride_out_m, stride_out_k,
    stride_x1_b, stride_x1_t, stride_x1_k,
    stride_x2_b, stride_x2_i, stride_x2_k,
):
    # Grid over (B, tiles along M, tiles along K)
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    M = T + I

    # Offsets
    offs_m = pid_m * 64 + tl.arange(0, 64)
    offs_k = pid_k * 64 + tl.arange(0, 64)
    mask_m = offs_m < M
    mask_k = offs_k < K

    # Determine source: x1 (encoder) if offs_m < T, else x2 (image)
    is_x1 = offs_m < T  # shape [64], boolean

    # For each m, compute pointers
    # Note: tl.where operates elementwise on vector.
    base_out = b * stride_out_b
    ptr_out = out_ptr + base_out + (offs_m[:, None] * stride_out_m) + (offs_k[None, :] * stride_out_k)

    # For x1 and x2 pointers
    # We'll compute row indices (t or i) based on whether m comes from encoder (offs_m < T)
    # When m comes from encoder: row index = offs_m; when from image: row index = offs_m - T
    row_idx = tl.where(is_x1, offs_m, offs_m - T)  # [64]
    # Build pointer for x1 and x2: for m < T: use x1[b, row_idx, k]; else: use x2[b, row_idx, k]
    ptr_x1 = x1_ptr + (b * stride_x1_b) + (row_idx[:, None] * stride_x1_t) + (offs_k[None, :] * stride_x1_k)
    ptr_x2 = x2_ptr + (b * stride_x2_b) + (row_idx[:, None] * stride_x2_i) + (offs_k[None, :] * stride_x2_k)
    # Select source based on is_x1
    # Triton doesn't support dynamic pointer selection directly; we do masked loads:
    # Load from x1 where is_x1, else 0; load from x2 where not is_x1, else 0.
    # Create mask for x1 and x2: valid if m < M and k < K, and is_x1 or not is_x1 respectively.
    mask_x1 = (mask_m[:, None]) & (mask_k[None, :]) & (is_x1[:, None])
    mask_x2 = (mask_m[:, None]) & (mask_k[None, :]) & (~is_x1[:, None])
    val_x1 = tl.load(ptr_x1, mask=mask_x1, other=0.0)
    val_x2 = tl.load(ptr_x2, mask=mask_x2, other=0.0)
    val = val_x1 + val_x2
    tl.store(ptr_out, val, mask=(mask_m[:, None] & mask_k[None, :]))


@triton.jit
def batched_matmul_kernel(
    C_ptr, A_ptr, W_ptr,
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_C_b, stride_C_m, stride_C_n,
    stride_A_b, stride_A_m, stride_A_k,
    stride_W_k, stride_W_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid over (B, tiles along M, tiles along N)
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Pointers for A[b, m, k] and W[k, n]
        ptr_A = A_ptr + (b * stride_A_b) + (offs_m[:, None] * stride_A_m) + (offs_k[None, :] * stride_A_k)
        ptr_W = W_ptr + (offs_k[:, None] * stride_W_k) + (offs_n[None, :] * stride_W_n)

        a = tl.load(ptr_A, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)  # [BM, BK]
        w = tl.load(ptr_W, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)  # [BK, BN]

        # Accumulate
        acc += tl.dot(a, w)

    # Store result to C[b, m, n]
    ptr_C = C_ptr + (b * stride_C_b) + (offs_m[:, None] * stride_C_m) + (offs_n[None, :] * stride_C_n)
    tl.store(ptr_C, acc, mask=(mask_m[:, None] & mask_n[None, :]))


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, I, K]
        encoder_hidden_states: torch.Tensor,  # [B, T, K]
        process_weight: torch.Tensor,  # [K, K]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure tensors are on CUDA and contiguous; compute in float32 for stability
        device = encoder_hidden_states.device
        B, T, K = encoder_hidden_states.shape
        B2, I, K2 = hidden_states.shape
        assert B == B2 and K == K2, "Batch size and hidden dim must match for concatenation."
        # Make contiguous and float32
        x1 = encoder_hidden_states.contiguous().to(torch.float32)
        x2 = hidden_states.contiguous().to(torch.float32)
        W = process_weight.contiguous().to(torch.float32)  # shape [K, K]
        M = T + I

        # Allocate output A [B, M, K] in fp32
        A = torch.empty((B, M, K), device=device, dtype=torch.float32)

        # Launch concat kernel: grid over (B, tiles along M, tiles along K)
        grid_concat = (B, triton.cdiv(M, 64), triton.cdiv(K, 64))
        concat_seq_dim1_kernel[grid_concat](
            A, x1, x2,
            B, T, I, K,
            A.stride(0), A.stride(1), A.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            x2.stride(0), x2.stride(1), x2.stride(2),
            num_warps=4, num_stages=2,
        )

        # Allocate output C [B, M, K] in fp32
        C = torch.empty((B, M, K), device=device, dtype=torch.float32)

        # Launch batched matmul kernel: grid over (B, tiles along M, tiles along N=K)
        grid_gemm = (B, triton.cdiv(M, 64), triton.cdiv(K, 64))
        batched_matmul_kernel[grid_gemm](
            C, A, W,
            B, M, K, K,
            C.stride(0), C.stride(1), C.stride(2),
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # Split outputs back into encoder and hidden streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original input dtypes to match the original API
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
