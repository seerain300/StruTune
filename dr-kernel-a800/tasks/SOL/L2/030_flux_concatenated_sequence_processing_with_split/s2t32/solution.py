import torch
import triton
import triton.language as tl


@triton.jit
def batched_mm(
    A_ptr,         # *float32, [B, M, K]
    WT_ptr,        # *float32, [K, N]
    C_ptr,         # *float32, [B, M, N]
    B: tl.constexpr,  # batch size (runtime ok; masks guard bounds)
    M, N, K,       # int
    stride_Ab, stride_Am, stride_Ak,
    stride_WTk, stride_WTn,
    stride_Cb, stride_Cm, stride_Cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, tiles_m, tiles_n)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A[b, m, k] and WT[k, n]
        a_ptrs = A_ptr + pid_b * stride_Ab + offs_m[:, None] * stride_Am + offs_k[None, :] * stride_Ak
        w_ptrs = WT_ptr + offs_k[:, None] * stride_WTk + offs_n[None, :] * stride_WTn

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(a, w)

    # Store results to C[b, m, n]
    c_ptrs = C_ptr + pid_b * stride_Cb + offs_m[:, None] * stride_Cm + offs_n[None, :] * stride_Cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_batched_mm(A, WT, C, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3):
    """
    A: [B, M, K] float32
    WT: [K, N] float32
    C: [B, M, N] float32 (output)
    """
    assert A.is_cuda and WT.is_cuda and C.is_cuda, "Tensors must be on CUDA for Triton"
    assert A.dtype == torch.float32 and WT.dtype == torch.float32 and C.dtype == torch.float32, "Use float32 for Triton kernel"
    B, M, K = A.shape
    K_wt, N = WT.shape
    assert K == K_wt, "Incompatible shapes for batched matmul"
    tiles_m = triton.cdiv(M, BLOCK_M)
    tiles_n = triton.cdiv(N, BLOCK_N)
    grid = (B, tiles_m, tiles_n)

    stride_Ab, stride_Am, stride_Ak = A.stride()
    stride_WTk, stride_WTn = WT.stride()
    stride_Cb, stride_Cm, stride_Cn = C.stride()

    batched_mm[grid](
        A, WT, C,
        B, M, N, K,
        stride_Ab, stride_Am, stride_Ak,
        stride_WTk, stride_WTn,
        stride_Cb, stride_Cm, stride_Cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
          - processed_encoder = encoder_hidden_states @ process_weight.T -> [B, T, H]
          - processed_hidden = hidden_states @ process_weight.T -> [B, I, H]
        No torch.cat or torch.matmul is used for heavy computation. All math is done in Triton kernels.
        """
        device = encoder_hidden_states.device
        # Ensure CUDA tensors and float32
        encoder = encoder_hidden_states.contiguous().float()
        hidden = hidden_states.contiguous().float()
        WT = process_weight.t().contiguous().float()  # [H, H]

        B_e, T, H = encoder.shape
        B_h, I, H2 = hidden.shape
        assert B_e == B_h, "Batch size must match for encoder_hidden_states and hidden_states"
        assert H == H2, "Hidden dimension must match"
        assert WT.shape[0] == H and WT.shape[1] == H, "process_weight must be [H, H]"

        # Allocate outputs
        processed_encoder = torch.empty((B_e, T, H), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B_h, I, H), device=device, dtype=torch.float32)

        # Launch Triton batched GEMMs for each split
        triton_batched_mm(encoder, WT, processed_encoder, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3)
        triton_batched_mm(hidden, WT, processed_hidden, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
