import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_cat_kernel(
    X_ptr,        # *ptr to X: [B, P, D] (concatenated input)
    WT_ptr,       # *ptr to WT: [D, D] (process_weight.T)
    Y_ptr,        # *ptr to Y: [B, P, D] (output processed)
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile size along M (P)
    BLOCK_N: tl.constexpr,  # tile size along N (D)
    BLOCK_K: tl.constexpr,  # reduction tile size along K (D)
):
    # Grid: (B, ceil(P / BLOCK_M), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # along P
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_m = m_offsets < P
    mask_n = n_offsets < D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Load A tile [BM, BK]: X[b, m, k]
        a_ptrs = X_ptr + b * P * D + m_offsets[:, None] * D + k_offsets[None, :]
        a_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load B tile [BK, BN]: WT[k, n]
        b_ptrs = WT_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        b_tile = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a_tile, b_tile)

    # Store result
    y_ptrs = Y_ptr + b * P * D + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # 1) Concatenate along sequence using PyTorch (allowed by Triton-only constraint on matmul)
        P = encoder_hidden_states.shape[1] + hidden_states.shape[1]
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, P, D]

        # Ensure contiguous
        concatenated = concatenated.contiguous()
        WT = process_weight.t().contiguous()  # [D, D]

        B, P, D = concatenated.shape

        # Allocate output in float32 for numerical robustness
        Y = torch.empty((B, P, D), device=concatenated.device, dtype=torch.float32)

        # Tile sizes (tunable)
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 64

        # Launch Triton GEMM kernel
        grid = (B, triton.cdiv(P, BLOCK_M), triton.cdiv(D, BLOCK_N))
        _gemm_cat_kernel[grid](
            concatenated, WT, Y,
            B, P, D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Split back into separate streams using PyTorch (no Triton required here)
        processed_encoder = Y[:, :encoder_hidden_states.shape[1], :]
        processed_hidden = Y[:, encoder_hidden_states.shape[1]:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
