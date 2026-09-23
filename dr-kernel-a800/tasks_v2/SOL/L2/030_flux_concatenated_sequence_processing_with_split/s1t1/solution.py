import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_flat_kernel(
    A_ptr, W_ptr, C_ptr,
    M, N, K,
    stride_A_m, stride_A_k,
    stride_W_k, stride_W_n,
    stride_C_m, stride_C_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ W, where:
      - A is [M, K] (flattened concatenated sequence).
      - W is [K, N] (process_weight.T contiguous).
      - C is [M, N].
    Each Triton program computes a [BLOCK_M x BLOCK_N] tile of C.
    """
    pid_m = tl.program_id(0)  # tile id along M
    pid_n = tl.program_id(1)  # tile id along N

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduce over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_A_m + k_offsets[None, :] * stride_A_k
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W tile: [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_W_k + n_offsets[None, :] * stride_W_n
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, w)

    # Store results to C
    c_ptrs = C_ptr + m_offsets[:, None] * stride_C_m + n_offsets[None, :] * stride_C_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
          - Concatenate along sequence dimension.
          - Perform matmul via Triton kernel (flattened M over batch and sequence).
          - Reshape and split outputs.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "Tensors must be on CUDA for Triton execution."
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3, \
            "Inputs must be 3D [B, len, K]."
        assert process_weight.ndim == 2, "process_weight must be 2D [K, K]."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]  # text_seq_len
        P = hidden_states.shape[1]          # img_seq_len
        K = hidden_states.shape[2]          # hidden_dim

        # Ensure contiguity
        A1 = encoder_hidden_states.contiguous()  # [B, T, K]
        A2 = hidden_states.contiguous()          # [B, P, K]
        W = process_weight.contiguous()          # [K, K]

        # Concatenate along sequence dimension: [B, T+P, K]
        total_L = T + P
        Acat = torch.cat([A1, A2], dim=1)  # [B, T+P, K]

        # Flatten Acat to [M, K] for Triton matmul
        M = B * total_L
        Aflat = Acat.reshape(M, K).contiguous()  # [M, K]

        # Prepare W_T: [K, K] contiguous
        Wt = W.t().contiguous()  # [K, K], corresponds to process_weight.T

        # Allocate output C [M, K] (N=K)
        N = K
        C = torch.empty((M, N), device=Acat.device, dtype=torch.float32)

        # Launch Triton kernel
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_flat_kernel[grid](
            Aflat, Wt, C,
            M, N, K,
            Aflat.stride(0), Aflat.stride(1),  # strides for [M, K]
            Wt.stride(0), Wt.stride(1),        # strides for [K, K]
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Reshape back to [B, T+P, K]
        C3 = C.reshape(B, total_L, K)

        # Split into encoder and image streams
        processed_encoder = C3[:, :T, :]
        processed_hidden = C3[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
