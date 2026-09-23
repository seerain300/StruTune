import torch
import triton
import triton.language as tl


@triton.jit
def _batched_matmul_kernel(
    A_ptr, WT_ptr, C_ptr,
    B, M, N, K,
    stride_A_b, stride_A_m, stride_A_k,
    stride_WT_k, stride_WT_n,
    stride_C_b, stride_C_m, stride_C_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch: (batch, M-tiles, N-tiles)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in A/C (sequence length L)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols in WT/C (hidden dim H)

    # Masks for valid rows/cols
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in blocks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # reduction dim
        mask_k = offs_k < K

        # Load A tile: [BLOCK_M, BLOCK_K] -> A[b, m, k]
        A_ptrs = A_ptr + pid_b * stride_A_b + offs_m[:, None] * stride_A_m + offs_k[None, :] * stride_A_k
        A_tile = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load W^T tile: [BLOCK_K, BLOCK_N] -> WT[k, n]
        WT_ptrs = WT_ptr + offs_k[:, None] * stride_WT_k + offs_n[None, :] * stride_WT_n
        WT_tile = tl.load(WT_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(A_tile.to(tl.float32), WT_tile.to(tl.float32))

    # Store results to C: [B, M, N]
    C_ptrs = C_ptr + pid_b * stride_C_b + offs_m[:, None] * stride_C_m + offs_n[None, :] * stride_C_n
    tl.store(C_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,         # [B, I, H]
        encoder_hidden_states: torch.Tensor, # [B, T, H]
        process_weight: torch.Tensor,        # [H, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenate along sequence dimension using torch (data movement): A_cat [B, L, H], L = T + I
        - Triton batched GEMM: A_cat @ process_weight.T -> processed [B, L, H]
        - Split into encoder and hidden streams: [:T, :] and [T:, :]
        """
        # Ensure CUDA tensors and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA"
        encoder = encoder_hidden_states.contiguous()   # [B, T, H]
        hidden = hidden_states.contiguous()            # [B, I, H]
        WT = process_weight.t().contiguous()           # [H, H]

        # Concatenate along sequence dimension to get A_cat [B, L, H]
        A_cat = torch.cat([encoder, hidden], dim=1)    # [B, L, H], L = T + I

        # Prepare shapes
        B, L, H = A_cat.shape
        assert WT.shape == (H, H), "process_weight must be [H, H]"

        # Output for processed (fp32 accumulation is fine)
        processed = torch.empty((B, L, H), device=A_cat.device, dtype=torch.float32)

        # Strides
        stride_A_b, stride_A_m, stride_A_k = A_cat.stride()         # for A_cat
        stride_WT_k, stride_WT_n = WT.stride()                       # for WT
        stride_C_b, stride_C_m, stride_C_n = processed.stride()      # for output

        # Choose tile sizes; masks handle edge cases
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        # Grid over (batch, tiles along L, tiles along H)
        grid = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_N))
        _batched_matmul_kernel[grid](
            A_cat, WT, processed,
            B, L, H, H,  # K == H
            stride_A_b, stride_A_m, stride_A_k,
            stride_WT_k, stride_WT_n,
            stride_C_b, stride_C_m, stride_C_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Split into streams
        processed_encoder = processed[:, :L - hidden.shape[1], :]   # first T rows
        processed_hidden = processed[:, L - hidden.shape[1]:, :]    # remaining I rows

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
