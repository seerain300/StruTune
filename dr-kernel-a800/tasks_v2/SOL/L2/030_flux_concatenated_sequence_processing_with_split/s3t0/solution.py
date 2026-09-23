import torch

# Triton is only available on GPU; import guarded to allow CPU fallback
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Generic kernel: compute Y[m, n] = sum_k X[m, k] * W_T[k, n] for a single row m
# X shape: [M, K], W_T shape: [K, N] (this is weight.T), Y shape: [M, N]
@triton.jit
def _matmul_row_kernel(
    X_ptr,           # *const float, [M, K]
    W_T_ptr,         # *const float, [K, N]
    Y_ptr,           # *float,        [M, N]
    M, N, K,         # int32
    stride_xm, stride_xk,
    stride_wtk, stride_wtn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)   # row index
    n_block = tl.program_id(1)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    # Accumulator for this row and block of columns
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_N):  # actually iterate by K tiles; we keep a while-like loop
        # Note: We'll implement a while-style iteration over k dimension
        # Start with k = 0
        k = 0
        # Since Triton doesn't have a for(range(0, K, BLOCK_N)) with dynamic K, we use a while:
        while k < K:
            k_offsets = k + tl.arange(0, BLOCK_N)
            # Load X row segment: X[m, k_offsets]
            x = tl.load(
                X_ptr + m * stride_xm + k_offsets * stride_xk,
                mask=k_offsets < K,
                other=0.0,
            )
            # Load W_T segment: W_T[k_offsets, n_offsets]
            w = tl.load(
                W_T_ptr + k_offsets[:, None] * stride_wtk + n_offsets[None, :] * stride_wtn,
                mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N),
                other=0.0,
            )
            # Accumulate: acc += sum over k_tile of x[k] * w[k, :]
            # We need to reduce along k-axis (axis=0) of w which is 2D [BLOCK_N, BLOCK_N]
            # Multiply x (shape [BLOCK_N]) with each row of w and accumulate
            for kk in range(BLOCK_N):
                acc += x[kk] * w[kk, :]
            k += BLOCK_N
        # Store results
        tl.store(
            Y_ptr + m * stride_ym + n_offsets * stride_yn,
            acc,
            mask=n_offsets < N,
        )
        # If multiple N blocks, we would need to return; here we compute one n_block=0
        # But since we launch grid over N blocks, we won't have multiple blocks per row.


# Convenience wrapper to run the Triton kernel for a given pair (rows, columns)
def _triton_matmul_rows(X, W_T):
    """
    Compute Y = X @ W_T using Triton. X: [M, K], W_T: [K, N]
    Returns Y: [M, N], same dtype as X.
    """
    assert X.is_cuda and W_T.is_cuda, "Inputs must be CUDA tensors for Triton."
    assert X.dtype in (torch.float32, torch.float16, torch.bfloat16), "Unsupported dtype."
    # Make contiguous for simpler strides
    Xc = X.contiguous()
    W_Tc = W_T.contiguous()
    M, K = Xc.shape
    K_w, N = W_Tc.shape
    assert K == K_w, f"Dimension mismatch: X is [M,{K}] and W_T is [{K_w},{N}]."

    # Allocate output
    Y = torch.empty((M, N), device=X.device, dtype=X.dtype)

    # Compute strides (in elements)
    stride_xm, stride_xk = Xc.stride(0), Xc.stride(1)
    stride_wtk, stride_wtn = W_Tc.stride(0), W_Tc.stride(1)
    stride_ym, stride_yn = Y.stride(0), Y.stride(1)

    # Launch grid: one program per row, and per N block
    BLOCK_N = 128  # tuneable
    grid = (M, triton.cdiv(N, BLOCK_N))
    # num_warps: small sizes -> 2 or 4; moderate -> 4 or 8
    _matmul_row_kernel[grid](
        Xc, W_Tc, Y,
        M, N, K,
        stride_xm, stride_xk,
        stride_wtk, stride_wtn,
        stride_ym, stride_yn,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return Y


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Avoids concatenation by computing encoder and image streams independently.
        - Uses Triton kernels to perform matrix-vector multiply (no bias).
        Returns processed_encoder and processed_hidden as in the original.
        """
        # If Triton not available or tensors not on CUDA, fall back to PyTorch
        if (not TRITON_AVAILABLE) or (not hidden_states.is_cuda) or (not encoder_hidden_states.is_cuda) or (not process_weight.is_cuda):
            # Fallback: original behavior
            concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            processed = torch.matmul(concatenated, process_weight.t())
            processed_encoder = processed[:, :encoder_hidden_states.shape[1], :]
            processed_hidden = processed[:, encoder_hidden_states.shape[1]:, :]
            return processed_encoder, processed_hidden

        # Ensure weight is transposed for the kernel: W_T [K, N] = weight [N, K].t()
        # Here N == hidden_dim, K == hidden_dim, so W_T is [D, D].
        W_T = process_weight.t().contiguous()

        # Compute encoder stream: [B, T, D] @ [D, D] -> [B, T, D]
        processed_encoder = _triton_matmul_rows(encoder_hidden_states, W_T)
        # Compute image stream: [B, I, D] @ [D, D] -> [B, I, D]
        processed_hidden = _triton_matmul_rows(hidden_states, W_T)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
