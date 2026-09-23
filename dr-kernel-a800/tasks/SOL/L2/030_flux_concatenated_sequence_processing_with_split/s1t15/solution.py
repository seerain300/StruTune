import torch
import triton
import triton.language as tl

# Triton GEMM kernel: A[M, K] @ W[K, K] -> C[M, K], where M = B*(T+P), N = K
@triton.jit
def _matmul_kernel(
    A_ptr,              # *f32 [M, K]
    W_ptr,              # *f32 [K, K]
    C_ptr,              # *f32 [M, K]
    B: tl.constexpr,    # number of batches (for grid only)
    M: tl.constexpr,    # int = B*(T+P)
    N: tl.constexpr,    # int = K
    Kdim: tl.constexpr, # int = K
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, tiles over M, tiles over N)
    b = tl.program_id(0)
    m_tile = tl.program_id(1)
    n_tile = tl.program_id(2)
    m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m < M
    mask_n = n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks
    for k_start in range(0, Kdim, BLOCK_K):
        kk = k_start + tl.arange(0, BLOCK_K)
        mask_k = kk < Kdim

        # Load A[m, kk]
        a_ptrs = A_ptr + m[:, None] * Kdim + kk[None, :]
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load W[kk, n]
        w_ptrs = W_ptr + kk[:, None] * Kdim + n[None, :]
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(a, w)

    # Store C[m, n]
    c_ptrs = C_ptr + m[:, None] * N + n[None, :]
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]

        device = hidden_states.device

        # Ensure contiguity and dtype for Triton
        hidden = hidden_states.contiguous().to(torch.float32)      # [B, P, K]
        encoder = encoder_hidden_states.contiguous().to(torch.float32)  # [B, T, K]
        weight = process_weight.contiguous().to(torch.float32)      # [K, K]

        # Concatenate along sequence dimension (data movement, not heavy compute)
        Acat = torch.cat([encoder, hidden], dim=1)  # [B, T+P, K]

        # GEMM: Acat [M, K] @ W [K, K] -> C [M, K], M = B*(T+P)
        M = B * (T + P)
        C_flat = torch.empty((M, K), device=device, dtype=torch.float32)

        # Launch Triton GEMM with 3D grid
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_kernel[grid](
            Acat.view(M, K), weight.t().contiguous(),  # W.T is [K, K]
            C_flat,
            B, M, K, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Reshape back to [B, T+P, K]
        C = C_flat.view(B, T + P, K)

        # Split outputs
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
