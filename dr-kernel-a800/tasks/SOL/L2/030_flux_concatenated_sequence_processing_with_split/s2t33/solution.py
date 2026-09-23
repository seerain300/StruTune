import torch
import triton
import triton.language as tl


@triton.jit
def batched_mm_kernel(
    A_ptr,        # *float32, pointer to A with shape [B, M, K]
    WT_ptr,       # *float32, pointer to W_T with shape [K, N]
    C_ptr,        # *float32, pointer to C with shape [B, M, N]
    B,            # int32, batch size (not used in indexing since we use 3D grid per batch)
    M,            # int32, number of rows (sequence length for this split)
    N,            # int32, number of cols (hidden size = output features)
    K,            # int32, hidden size (input features)
    BLOCK_M: tl.constexpr,  # tile size along M
    BLOCK_N: tl.constexpr,  # tile size along N
    BLOCK_K: tl.constexpr,  # tile size along K
):
    # 3D grid: (batch, tiles along M, tiles along N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Compute row/col indices for this program
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Create accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in blocks
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A[b, m, k] as a (BLOCK_M, BLOCK_K) tile
        # A layout: [B, M, K] contiguous
        # Addressing: A_ptr + b*(M*K) + m[:, None]*K + k[None, :]
        A_row_ptrs = A_ptr + pid_b * (M * K) + m[:, None] * K + k[None, :]
        A_mask = (m[:, None] < M) & (k[None, :] < K)
        A_tile = tl.load(A_row_ptrs, mask=A_mask, other=0.0)

        # Load WT[k, n] as a (BLOCK_K, BLOCK_N) tile
        # WT layout: [K, N] contiguous
        # Addressing: WT_ptr + k[:, None]*N + n[None, :]
        WT_row_ptrs = WT_ptr + k[:, None] * N + n[None, :]
        WT_mask = (k[:, None] < K) & (n[None, :] < N)
        WT_tile = tl.load(WT_row_ptrs, mask=WT_mask, other=0.0)

        # Accumulate: acc += A_tile @ WT_tile
        # A_tile: [BLOCK_M, BLOCK_K], WT_tile: [BLOCK_K, BLOCK_N]
        acc += tl.dot(A_tile, WT_tile)

    # Store acc into C[b, m, n]
    C_ptrs = C_ptr + pid_b * (M * N) + m[:, None] * N + n[None, :]
    C_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


def triton_batched_mm(A: torch.Tensor, WT: torch.Tensor, C: torch.Tensor):
    """
    Compute C = A @ WT using Triton, where:
      - A: [B, M, K] (input batch, sequence, hidden)
      - WT: [K, N] (weight transposed, hidden -> output features)
      - C: [B, M, N] (output)
    All tensors must be float32 and contiguous.
    """
    assert A.dim() == 3, "A must be [B, M, K]"
    assert WT.dim() == 2, "WT must be [K, N]"
    assert C.dim() == 3, "C must be [B, M, N]"
    assert A.is_contiguous() and WT.is_contiguous() and C.is_contiguous(), "Tensors must be contiguous"

    B, M, K = A.shape
    K_wt, N = WT.shape
    assert K == K_wt, "WT's K must match A's K"

    # Triton grid: (batch, tiles along M, tiles along N)
    # Use moderate tile sizes; these are good defaults and robust for varied shapes.
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        B,
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    batched_mm_kernel[grid](
        A, WT, C,
        B, M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward that performs:
          concatenated = cat([encoder_hidden_states, hidden_states], dim=1)
          processed = concatenated @ process_weight.T
          return processed_encoder, processed_hidden
        All numerical computation is done via Triton kernels (no torch.matmul/cat).
        """
        # Extract shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]  # text_seq_len
        I = hidden_states.shape[1]          # img_seq_len
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H
        assert process_weight.shape[0] == H and process_weight.shape[1] == H

        # Ensure everything is on the same device and contiguous
        device = hidden_states.device

        # Prepare inputs and weights for Triton. Use float32 for numerical stability.
        encoder = encoder_hidden_states.contiguous().float()
        hidden = hidden_states.contiguous().float()
        WT = process_weight.t().contiguous().float()  # [H, H]

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=device, dtype=torch.float32)

        # Launch Triton kernels: compute each split directly
        triton_batched_mm(encoder, WT, processed_encoder)  # [B, T, H]
        triton_batched_mm(hidden, WT, processed_hidden)    # [B, I, H]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
