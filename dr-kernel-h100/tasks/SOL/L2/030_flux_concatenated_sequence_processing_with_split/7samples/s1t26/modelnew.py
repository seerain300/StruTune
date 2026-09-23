import torch
import triton
import triton.language as tl

@triton.jit
def _batched_row_gemm_kernel(
    X_ptr,       # input [B, M, D], where M = T or I depending on call
    Wt_ptr,      # weight transposed [D, D]
    Y_ptr,       # output [B, M, D]
    B, M, D,     # meta-parameters
    X_b_stride, X_m_stride, X_d_stride,
    Wt_d_stride, Wt_k_stride,  # Wt strides: (stride along N, stride along K) i.e., (N=0, K=1)
    Y_b_stride, Y_m_stride, Y_d_stride,
    BLOCK_K: tl.constexpr,     # tile size along input feature dim (K)
):
    # Grid: (B, M) -> each program computes one output row Y[b, m, :]
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Accumulator for the output row
    acc = tl.zeros((D,), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, D, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)  # vector of K indices
        k_mask = k_idx < D

        # Load X[b, m, k0:k0+BLOCK_K]
        x_ptrs = X_ptr + pid_b * X_b_stride + pid_m * X_m_stride + k_idx * X_d_stride
        x = tl.load(x_ptrs, mask=k_mask, other=0.0)

        # Load Wt[k0:k0+BLOCK_K, :] as a matrix of shape [BLOCK_K, D]
        w_ptrs = Wt_ptr + k_idx[:, None] * Wt_k_stride + tl.arange(0, D)[None, :] * Wt_d_stride
        w = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)  # [BLOCK_K, D]

        # Accumulate: acc += sum over K tile of x[k] * w[k, :]
        # Convert x to [BLOCK_K, 1] to allow broadcasting
        acc += tl.sum(x[:, None] * w, axis=0)

    # Store the accumulated row to Y[b, m, :]
    y_ptrs = Y_ptr + pid_b * Y_b_stride + pid_m * Y_m_stride + tl.arange(0, D) * Y_d_stride
    tl.store(y_ptrs, acc)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, I, D]
        encoder_hidden_states: torch.Tensor,  # [B, T, D]
        process_weight: torch.Tensor,          # [D, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Avoids concatenation and split.
        - Computes processed_encoder = encoder_hidden_states @ process_weight.T
          and processed_hidden = hidden_states @ process_weight.T
          using Triton kernels (no torch.matmul).
        Returns:
          processed_encoder: [B, T, D]
          processed_hidden: [B, I, D]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be on CUDA device for Triton kernels."
        assert hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32) and \
               encoder_hidden_states.dtype == hidden_states.dtype and \
               process_weight.dtype == hidden_states.dtype, \
            "All tensors should have matching dtypes and be supported by Triton."

        # Ensure contiguity and float32 for accumulation
        device = hidden_states.device
        dtype = torch.float32  # compute in float32 for correctness; if inputs are not float32, cast

        E = encoder_hidden_states.contiguous().to(dtype)
        H = hidden_states.contiguous().to(dtype)
        W = process_weight.t().contiguous().to(dtype)  # Wt: [D, D]

        B = E.shape[0]
        T = E.shape[1]
        I = H.shape[1]
        D = E.shape[2]

        # Output tensors
        processed_encoder = torch.empty((B, T, D), device=device, dtype=dtype)
        processed_hidden = torch.empty((B, I, D), device=device, dtype=dtype)

        # Choose tile size along K. Use 128 if D >= 128, else 64 or 32.
        BLOCK_K = 128 if D >= 128 else (64 if D >= 64 else 32)

        # Launch for encoder stream: Y = E @ Wt
        grid_E = (B, T)
        _batched_row_gemm_kernel[grid_E](
            E, W, processed_encoder,
            B, T, D,
            E.stride(0), E.stride(1), E.stride(2),
            W.stride(0), W.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Launch for hidden stream: Y = H @ Wt
        grid_H = (B, I)
        _batched_row_gemm_kernel[grid_H](
            H, W, processed_hidden,
            B, I, D,
            H.stride(0), H.stride(1), H.stride(2),
            W.stride(0), W.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # If inputs were not float32 originally, we can cast back to original dtype here
        # The original run returns float32 since process_weight and inputs are float32 by default.
        # We keep outputs in float32 to match reference behavior.
        return processed_encoder, processed_hidden