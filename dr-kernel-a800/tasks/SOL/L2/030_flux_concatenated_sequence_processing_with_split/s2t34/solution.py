import torch
import triton
import triton.language as tl


# Triton kernel: batched matmul computing C[b, m, n] = sum_k A[b, m, k] * W_T[k, n]
# A: [B, M, K], W_T: [K, N], C: [B, M, N]
@triton.jit
def _batched_mm_kernel(
    A_ptr,  # *fp32, [B, M, K]
    WT_ptr, # *fp32, [K, N]
    C_ptr,  # *fp32, [B, M, N]
    B: tl.int32,
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    stride_ab, stride_am, stride_ak,
    stride_wtk, stride_wtn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Compute the tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    k = 0
    while k < K:
        k_idx = k + offs_k  # [BLOCK_K]

        # Pointers for A[b, m, k]
        a_ptrs = A_ptr + pid_b * stride_ab + (offs_m[:, None] * stride_am) + (k_idx[None, :] * stride_ak)
        # Masks for A
        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)

        # Load A tile
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Pointers for W_T[k, n]
        w_ptrs = WT_ptr + (k_idx[:, None] * stride_wtk) + (offs_n[None, :] * stride_wtn)
        # Masks for W_T
        w_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)

        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, w)
        k += BLOCK_K

    # Store results to C[b, m, n]
    c_ptrs = C_ptr + pid_b * stride_cb + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_batched_mm(A: torch.Tensor, WT: torch.Tensor, output: torch.Tensor):
    """
    A: [B, M, K]
    WT: [K, N]
    output: [B, M, N]
    All tensors must be float32 and contiguous.
    """
    assert A.dtype == torch.float32 and WT.dtype == torch.float32 and output.dtype == torch.float32
    assert A.is_cuda and WT.is_cuda and output.is_cuda
    assert A.is_contiguous() and WT.is_contiguous() and output.is_contiguous()
    B, M, K = A.shape
    K_wt, N = WT.shape
    assert K == K_wt, f"K mismatch: {K} vs {K_wt}"

    # Compute grid
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _batched_mm_kernel[grid](
        A, WT, output,
        B, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        WT.stride(0), WT.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return output


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward that performs:
          processed_encoder = encoder_hidden_states @ process_weight.T   -> [B, T, H]
          processed_hidden   = hidden_states @ process_weight.T         -> [B, I, H]
        without using torch.cat or torch.matmul in the heavy path.
        """
        # Ensure everything is on the same device and contiguous, using float32 for numerical stability
        device = hidden_states.device
        # Make inputs contiguous and float32
        encoder = encoder_hidden_states.contiguous().float()
        hidden = hidden_states.contiguous().float()
        WT = process_weight.t().contiguous().float()  # [H, H]

        B, T, H = encoder.shape
        B2, I, H2 = hidden.shape
        assert B == B2 and H == H2, "Batch size and hidden_dim must match"

        # Outputs
        processed_encoder = torch.empty((B, T, H), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=device, dtype=torch.float32)

        # Launch Triton kernels per batch (since each is [B, *])
        # We launch per-batch copies to keep kernels simple; B is usually small in these configs.
        for b in range(B):
            # Prepare per-batch views
            A_b = encoder[b]  # [T, H]
            WT_b = WT  # [H, H], shared across batch
            C_b = processed_encoder[b]  # [T, H]

            # Dimensions
            M = A_b.shape[0]
            K = A_b.shape[1]
            N = WT_b.shape[1]

            # Compute grid for this batch
            BLOCK_M = 64
            BLOCK_N = 64
            BLOCK_K = 64
            grid = (1, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

            _batched_mm_kernel[grid](
                A_b, WT_b, C_b,
                1, M, N, K,
                A_b.stride(0), A_b.stride(1),  # strides for [T, H]
                WT_b.stride(0), WT_b.stride(1),  # strides for [H, H]
                C_b.stride(0), C_b.stride(1),    # strides for [T, H]
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # Do the same for hidden states
        for b in range(B):
            A_b = hidden[b]     # [I, H]
            WT_b = WT           # [H, H]
            C_b = processed_hidden[b]  # [I, H]
            M = A_b.shape[0]
            K = A_b.shape[1]
            N = WT_b.shape[1]

            BLOCK_M = 64
            BLOCK_N = 64
            BLOCK_K = 64
            grid = (1, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

            _batched_mm_kernel[grid](
                A_b, WT_b, C_b,
                1, M, N, K,
                A_b.stride(0), A_b.stride(1),
                WT_b.stride(0), WT_b.stride(1),
                C_b.stride(0), C_b.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
