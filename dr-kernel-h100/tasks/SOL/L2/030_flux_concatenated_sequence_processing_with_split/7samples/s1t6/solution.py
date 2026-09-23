import torch
import triton
import triton.language as tl


@triton.jit
def _batched_matmul_rows_kernel(
    X_ptr,        # *const float, input [B, M, D]
    W_ptr,        # *const float, weight [D, D]
    Y_ptr,        # *float, output [B, M, D]
    B, M, D,      # int32 sizes
    stride_xb, stride_xm, stride_xd,
    stride_w0, stride_w1,
    stride_yb, stride_ym, stride_yd,
    BLOCK_K: tl.constexpr,  # tile size for D (hidden_dim)
):
    # Each program handles one (batch, row) pair: pid over M, axis 0 over B
    pid_m = tl.program_id(0)  # output row index within sequence
    pid_b = tl.program_id(1)  # batch index
    # Bounds check
    if pid_m >= M or pid_b >= B:
        return

    # Initialize accumulator for this (b, m) row across D hidden_dim
    acc = tl.zeros([D], dtype=tl.float32)

    # Loop over K (hidden_dim) in tiles of BLOCK_K
    # We use a simple while loop to handle arbitrary D
    k0 = 0
    while k0 < D:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < D

        # Load input row vector X[b, m, k_offsets]
        x_ptrs = X_ptr + pid_b * stride_xb + pid_m * stride_xm + k_offsets * stride_xd
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]

        # Load weight submatrix W[k_offsets, 0:D]
        # Note: We want W[k, n] with k in k_offsets, n in [0:D)
        n_offsets = tl.arange(0, D)
        w_ptrs = W_ptr + k_offsets[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_sub = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)  # [BLOCK_K, D]

        # Accumulate: acc[n] += sum_k x_vec[k] * w_sub[k, n]
        # Loop over k within tile to reduce; we can also use broadcasting and reduction
        # Here, we reduce explicitly to keep it clear:
        for kk in range(BLOCK_K):
            # guard kk within active k_offsets
            if (k0 + kk) < D:
                xk = x_vec[kk]
                # column vector for this kk across n
                w_col = w_sub[kk, :]  # [D]
                acc += xk * w_col

        k0 += BLOCK_K

    # Store the result Y[b, m, :] = acc
    y_ptrs = Y_ptr + pid_b * stride_yb + pid_m * stride_ym + tl.arange(0, D) * stride_yd
    tl.store(y_ptrs, acc)


def _largest_pow2_divisor(n: int, max_block: int) -> int:
    # Returns the largest power-of-two <= min(n, max_block) and <= n, or 1 if n == 0
    if n <= 0:
        return 1
    candidate = 1
    while (candidate << 1) <= n and (candidate << 1) <= max_block:
        candidate <<= 1
    return candidate


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of run. Computes:
        processed_encoder = encoder_hidden_states @ process_weight.T
        processed_hidden = hidden_states @ process_weight.T
        Returns (processed_encoder, processed_hidden) with shapes [B, T, D] and [B, I, D].
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3
        assert process_weight.dim() == 2
        B_h = hidden_states.shape[0]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        B_e = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        assert B_e == B_h, "Batch size must match for both inputs"

        # Ensure device is CUDA and dtype is float32 for Triton
        if not hidden_states.is_cuda or not encoder_hidden_states.is_cuda or not process_weight.is_cuda:
            # Fallback to PyTorch if tensors are not on CUDA
            concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            processed = torch.matmul(concatenated, process_weight.t())
            processed_encoder = processed[:, :T, :]
            processed_hidden = processed[:, T:, :]
            return processed_encoder, processed_hidden

        # Make tensors contiguous for predictable strides
        X_e = encoder_hidden_states.contiguous()
        X_h = hidden_states.contiguous()
        W = process_weight.contiguous()

        # Allocate outputs
        processed_encoder = torch.empty((B_e, T, D), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B_h, I, D), device=hidden_states.device, dtype=torch.float32)

        # Compute BLOCK_K as largest power-of-two divisor of D up to 1024
        BLOCK_K_e = _largest_pow2_divisor(D, 1024)
        BLOCK_K_h = _largest_pow2_divisor(D, 1024)

        # Launch kernel for encoder hidden states: output [B, T, D]
        grid_encoder = (T, B_e)
        _batched_matmul_rows_kernel[grid_encoder](
            X_e, W, processed_encoder,
            B_e, T, D,
            X_e.stride(0), X_e.stride(1), X_e.stride(2),
            W.stride(0), W.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_K=BLOCK_K_e,
            num_warps=4,
            num_stages=2,
        )

        # Launch kernel for image hidden states: output [B, I, D]
        grid_hidden = (I, B_h)
        _batched_matmul_rows_kernel[grid_hidden](
            X_h, W, processed_hidden,
            B_h, I, D,
            X_h.stride(0), X_h.stride(1), X_h.stride(2),
            W.stride(0), W.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_K=BLOCK_K_h,
            num_warps=4,
            num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
