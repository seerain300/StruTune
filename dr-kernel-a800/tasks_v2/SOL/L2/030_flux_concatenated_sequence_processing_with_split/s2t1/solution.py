import torch
import triton
import triton.language as tl


@triton.jit
def _batched_matmul_kernel(
    A_ptr,  # [B, L, K]
    B_ptr,  # [K, N]
    C_ptr,  # [B, L, N]
    B, L, K, N,
    A_stride_b, A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_b, C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids: batch, sequence-tile, output-column-tile
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    # tile offsets
    m_start = m_block * BLOCK_M
    n_start = n_block * BLOCK_N
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # reduction over K (hidden_dim)
    k0 = 0
    while k0 < K:
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # pointers for tiles
        A_tile_ptrs = A_ptr + b * A_stride_b + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        B_tile_ptrs = B_ptr + k_offsets[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n

        # bounds masks
        a_mask = (m_offsets[:, None] < L) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # load tiles
        A_tile = tl.load(A_tile_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        B_tile = tl.load(B_tile_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # accumulate
        acc += tl.dot(A_tile, B_tile)

        k0 += BLOCK_K

    # store result
    C_ptrs = C_ptr + b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    c_mask = (m_offsets[:, None] < L) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


def _launch_triton_matmul(A_cat: torch.Tensor, B_mat: torch.Tensor, out: torch.Tensor):
    """
    A_cat: [B, L, K] contiguous, CUDA tensor
    B_mat: [K, N] contiguous, CUDA tensor (process_weight.T)
    out:   [B, L, N] allocated, float32
    """
    assert A_cat.is_cuda and B_mat.is_cuda and out.is_cuda
    B, L, K = A_cat.shape
    K_mat, N = B_mat.shape
    assert K_mat == K, "A_cat last dim must match B_mat first dim"

    # Strides
    A_stride_b, A_stride_m, A_stride_k = A_cat.stride(0), A_cat.stride(1), A_cat.stride(2)
    B_stride_k, B_stride_n = B_mat.stride(0), B_mat.stride(1)
    C_stride_b, C_stride_m, C_stride_n = out.stride(0), out.stride(1), out.stride(2)

    # Heuristic tile sizes for robust grid coverage
    # Use smaller BLOCK_M when L is large to ensure multiple tiles
    BLOCK_M = 128 if L >= 1024 else 64
    BLOCK_N = 128 if N >= 128 else 64
    BLOCK_K = 64 if K >= 64 else 32

    # Grid over (B, tiles of L, tiles of N)
    grid = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _batched_matmul_kernel[grid](
        A_cat, B_mat, out,
        B, L, K, N,
        A_stride_b, A_stride_m, A_stride_k,
        B_stride_k, B_stride_n,
        C_stride_b, C_stride_m, C_stride_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
          - Concatenate encoder_hidden_states and hidden_states along sequence dim (L = T + I)
          - Compute out = cat @ process_weight.T via Triton GEMM
          - Split out into (processed_encoder, processed_hidden)
        """
        # Ensure CUDA tensors; compute in float32 for stability
        device = hidden_states.device
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"

        # Cast to float32 for kernel (if not already)
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
        if encoder_hidden_states.dtype != torch.float32:
            encoder_hidden_states = encoder_hidden_states.float()
        if process_weight.dtype != torch.float32:
            process_weight = process_weight.float()

        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        assert hidden_states.shape[2] == H, "hidden and encoder must have same hidden_dim"
        assert process_weight.shape == (H, H), "process_weight must be [H, H]"

        # Concatenate along sequence dimension: [B, T + I, H]
        A_cat = torch.cat([encoder_hidden_states, hidden_states], dim=1).contiguous()

        # Prepare B = process_weight.T contiguous [H, H]
        B_mat = process_weight.t().contiguous()

        # Allocate output [B, L, H], float32
        L = T + I
        out = torch.empty((B, L, H), device=device, dtype=torch.float32)

        # Launch Triton kernel
        _launch_triton_matmul(A_cat, B_mat, out)

        # Split outputs
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
