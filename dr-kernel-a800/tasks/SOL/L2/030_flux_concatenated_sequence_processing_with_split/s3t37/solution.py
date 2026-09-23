import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_blocked(
    A_ptr,          # input A: [B, M, K], row-major [b, m, k]
    WT_ptr,         # weight transpose: [K, N]
    C_ptr,          # output C: [B, M, N]
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_Ab, stride_Am, stride_Ak,   # strides for A
    stride_WTk, stride_WTn,            # strides for WT
    stride_Cb, stride_Cm, stride_Cn,   # strides for C
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs: batch, M-tile, N-tile
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A tile: [BM, BK]
        A_ptrs = A_ptr + pid_b * stride_Ab + offs_m[:, None] * stride_Am + offs_k[None, :] * stride_Ak
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Pointers for WT tile: [BK, BN]
        WT_ptrs = WT_ptr + offs_k[:, None] * stride_WTk + offs_n[None, :] * stride_WTn
        WT_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        WT_tile = tl.load(WT_ptrs, mask=WT_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, WT_tile)

    # Store result C tile: [BM, BN]
    C_ptrs = C_ptr + pid_b * stride_Cb + offs_m[:, None] * stride_Cm + offs_n[None, :] * stride_Cn
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


def _triton_gemm(A: torch.Tensor, WT: torch.Tensor, out: torch.Tensor):
    """
    Perform batched GEMM:
        A: [B, M, K]
        WT: [K, N] (process_weight.T)
        out: [B, M, N]
    """
    assert A.is_cuda and WT.is_cuda and out.is_cuda, "Tensors must be CUDA for Triton."
    B = A.shape[0]
    M = A.shape[1]
    K = A.shape[2]
    N = WT.shape[1]

    # Heuristic block sizes
    BLOCK_M = 64 if M >= 64 else 32
    BLOCK_N = 128 if N >= 128 else 64
    BLOCK_K = 64

    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _gemm_blocked[grid](
        A, WT, out,
        B, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        WT.stride(0), WT.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
          - Do not use torch.cat or torch.matmul in forward.
          - Compute the two streams directly via Triton GEMM:
              processed_encoder = encoder_hidden_states @ process_weight.T  -> [B, T, D]
              processed_hidden  = hidden_states @ process_weight.T         -> [B, I, D]
        """
        # Ensure CUDA
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors."

        # Make tensors contiguous for predictable strides
        enc = encoder_hidden_states.contiguous()  # [B, T, D]
        hst = hidden_states.contiguous()          # [B, I, D]
        WT = process_weight.t().contiguous()      # [D, D]

        B = enc.shape[0]
        T = enc.shape[1]
        I = hst.shape[1]
        D = enc.shape[2]

        # Allocate outputs
        processed_encoder = torch.empty((B, T, D), dtype=enc.dtype, device=enc.device)
        processed_hidden = torch.empty((B, I, D), dtype=hst.dtype, device=hst.device)

        # Compute encoder stream
        _triton_gemm(enc, WT, processed_encoder)

        # Compute hidden stream
        _triton_gemm(hst, WT, processed_hidden)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
