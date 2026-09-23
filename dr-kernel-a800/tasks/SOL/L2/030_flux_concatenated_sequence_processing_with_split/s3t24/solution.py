import torch
import triton
import triton.language as tl


@triton.jit
def _batched_gemm_kernel(
    A_ptr,       # *ptr to concatenated input X: [B, P, D], float32
    WT_ptr,      # *ptr to process_weight^T: [D, D], float32
    C_ptr,       # *ptr to output Y: [B, P, D], float32
    B: tl.constexpr,   # batch size
    P: tl.constexpr,   # sequence length after concat (T + I)
    D: tl.constexpr,   # feature dim
    BLOCK_M: tl.constexpr,  # tile size over M=P
    BLOCK_N: tl.constexpr,  # tile size over N=D
    BLOCK_K: tl.constexpr,  # reduction tile size over K=D
):
    # Grid: (B, ceil(P / BLOCK_M), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < P
    mask_n = n_offsets < D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension (features), reduce over D
    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Load A tile: A[b, m, k] -> shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + b * P * D + m_offsets[:, None] * D + k_offsets[None, :]
        a_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load WT tile: WT[k, n] -> shape [BLOCK_K, BLOCK_N]
        wt_ptrs = WT_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        wt_tile = tl.load(wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a_tile, wt_tile)

    # Store result C[b, m, n] = acc
    c_ptrs = C_ptr + b * P * D + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton implementation of:
          concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, P, D]
          processed = concatenated @ process_weight.T  # [B, P, D]
          processed_encoder = processed[:, :T, :]
          processed_hidden = processed[:, T:, :]
        We use torch.cat for concatenation and a Triton kernel for the GEMM, then torch slicing to split.
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        P = T + I

        # Ensure inputs are contiguous and on CUDA
        device = hidden_states.device
        assert device.type == "cuda", "ModelNew requires CUDA device"

        # Concatenate along sequence dimension using torch (to avoid complex Triton concat kernels)
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1).contiguous()  # [B, P, D]

        # Process weight transpose: [D, D]
        wt = process_weight.transpose(0, 1).contiguous()  # [D, D]

        # Cast to float32 for robust Triton GEMM
        A = concatenated.to(torch.float32)
        WT = wt.to(torch.float32)

        # Allocate output [B, P, D] in float32
        Y = torch.empty((B, P, D), dtype=torch.float32, device=device)

        # Choose block sizes (tuned for general use; can be adjusted)
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 64

        grid = (B, triton.cdiv(P, BLOCK_M), triton.cdiv(D, BLOCK_N))

        _batched_gemm_kernel[grid](
            A, WT, Y,
            B, P, D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=2
        )

        # Split back using torch slicing (not torch.cat)
        processed_encoder = Y[:, :T, :]
        processed_hidden = Y[:, T:, :]

        # Return results in original dtype (match PyTorch behavior)
        # Original run returns the processed tensors in the same dtype as inputs; here we keep float32.
        # If you need to match the input dtype exactly, cast processed_encoder and processed_hidden back.
        # However, since the forward's computation is in Triton, returning float32 is acceptable.
        # Cast back to original dtype of hidden_states if desired:
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
