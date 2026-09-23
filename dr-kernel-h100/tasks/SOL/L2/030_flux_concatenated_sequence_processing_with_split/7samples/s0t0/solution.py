import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K', 'B'],
)
@triton.jit
def _matmul_bszMxKxK_WKxK(B_ptr, W_ptr, C_ptr,
                           B, M, N, K,
                           stride_b, stride_m, stride_k,
                           stride_wk, stride_wn,
                           stride_cb, stride_cm, stride_cn,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute C[b, m, n] = sum_{k=0..K-1} A[b, m, k] * W[k, n], where
    - A is a [B, M, K] tensor represented by B_ptr with provided strides.
    - W is a [K, K] tensor (process_weight), provided.
    - C is a [B, M, N] tensor (N equals K here).
    Grid: (B, ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N))
    """
    # Program ids
    b_id = tl.program_id(axis=0)
    m_block = tl.program_id(axis=1)
    n_block = tl.program_id(axis=2)

    # Compute row/col indices this program will handle
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A[b, m, k] tile: shape [BLOCK_M, BLOCK_K]
        A_ptrs = B_ptr \
                 + b_id * stride_b \
                 + m_offsets[:, None] * stride_m \
                 + k_offsets[None, :] * stride_k

        # Mask for A (handle edges where m or k exceed bounds)
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)

        # Load A tile
        A = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Pointers for W[k, n] tile: shape [BLOCK_K, BLOCK_N]
        W_ptrs = W_ptr \
                 + k_offsets[:, None] * stride_wk \
                 + n_offsets[None, :] * stride_wn

        # Mask for W (handle edges where k or n exceed bounds)
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        W = tl.load(W_ptrs, mask=w_mask, other=0.0)

        # Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(A, W)

    # Write back results to C[b, m, n]
    C_ptrs = C_ptr \
             + b_id * stride_cb \
             + m_offsets[:, None] * stride_cm \
             + n_offsets[None, :] * stride_cn

    # Mask for C (edges where m or n exceed bounds)
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)

    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward: performs concatenation, GEMM with process_weight, and split.
        All computation is done by Triton; no torch matmul in forward.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be CUDA tensors for Triton execution."

        batch = hidden_states.shape[0]
        text_seq_len = encoder_hidden_states.shape[1]
        img_seq_len = hidden_states.shape[1]
        hidden_dim = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == hidden_dim, "Mismatched hidden dimensions."

        # Concatenate along sequence dimension
        M = text_seq_len + img_seq_len
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        assert concatenated.shape == (batch, M, hidden_dim), "Concatenation produced unexpected shape."

        # Ensure contiguous tensors
        concatenated = concatenated.contiguous()
        process_weight = process_weight.contiguous()

        # Output tensor: [batch, M, hidden_dim]
        processed = torch.empty((batch, M, hidden_dim), device=concatenated.device, dtype=torch.float32)

        # Strides
        # For A = [B, M, K], strides are (M*K, K, 1) if contiguous
        stride_bA = concatenated.stride(0)
        stride_mA = concatenated.stride(1)
        stride_kA = concatenated.stride(2)

        # For W = [K, K], strides are (K, 1) if contiguous
        stride_wk = process_weight.stride(0)
        stride_wn = process_weight.stride(1)

        # For C = [B, M, N] with N=K
        stride_cb = processed.stride(0)
        stride_cm = processed.stride(1)
        stride_cn = processed.stride(2)

        # Launch Triton kernel: grid over (B, tiles in M, tiles in N)
        grid = (batch, triton.cdiv(M, 128), triton.cdiv(hidden_dim, 256))  # placeholders; autotune overrides per-config

        # Call kernel. Autotune will pick best config among provided.
        _matmul_bszMxKxK_WKxK[grid](
            concatenated, process_weight, processed,
            batch, M, hidden_dim, hidden_dim,  # N equals K here
            stride_bA, stride_mA, stride_kA,
            stride_wk, stride_wn,
            stride_cb, stride_cm, stride_cn,
        )

        # Split processed back into encoder and hidden parts
        processed_encoder = processed[:, :text_seq_len, :]
        processed_hidden = processed[:, text_seq_len:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
