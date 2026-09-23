import torch
import triton
import triton.language as tl


# Triton kernel: compute y_row = X_row @ W, where
# - X is [M, D], we will pass it as a list of B tensors of shape [M, D]
# - W is [D, D], process_weight (we will transpose and feed as [D, D] to match A @ W)
# - Output y is [M, D], one program instance handles one row (one sequence position)
@triton.jit
def _gemv_rowwise_kernel(
    X_ptr,        # pointer to X_row data (we'll pass per-batch row tensors here)
    W_ptr,        # pointer to process_weight (shape [D, D])
    y_ptr,        # pointer to output y (shape [M, D])
    M, D,         # M = number of rows to process, D = hidden_dim
    stride_xm, stride_xd,   # strides for X: stride along M and D
    stride_w0, stride_w1,   # strides for W: along dim0 and dim1 (W is [D, D])
    stride_yM, stride_yD,   # strides for y: along M and D
    BLOCK_K: tl.constexpr,  # tile size for D dimension
):
    # Each program instance handles one output row (one sequence position)
    row = tl.program_id(0)
    # Bounds check (in case grid > M)
    if row >= M:
        return

    # Accumulator for this row
    acc = tl.zeros([D], dtype=tl.float32)

    # Iterate over K dimension (hidden_dim) in tiles of size BLOCK_K
    for k_start in range(0, D, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D

        # Load X[row, offs_k]
        # X is [M, D], we pass a tensor for this row; pointer arithmetic uses strides
        x_vals = tl.load(
            X_ptr + row * stride_xm + offs_k * stride_xd,
            mask=mask_k,
            other=0.0,
        )

        # Load W[offs_k, :] as a submatrix of shape [BLOCK_K, D]
        w_sub = tl.load(
            W_ptr + offs_k[:, None] * stride_w0 + tl.arange(0, D)[None, :] * stride_w1,
            mask=mask_k[:, None],
            other=0.0,
        )

        # Accumulate dot products: for each kk in BLOCK_K, sum W[kk, :] * X[row, kk]
        # We do this by summing across the last dimension of w_sub with x_vals
        # Note: acc is [D], w_sub is [BLOCK_K, D], x_vals is [BLOCK_K]
        for kk in range(BLOCK_K):
            kk_valid = mask_k[kk]
            # pick the kk-th column of w_sub and multiply by x_vals[kk]
            w_col = w_sub[kk, :]  # shape [D]
            x_val = x_vals[kk]    # scalar
            # multiply with mask
            acc += tl.where(kk_valid, x_val * w_col, 0.0)

    # Store the accumulated result for this row
    tl.store(y_ptr + row * stride_yM + tl.arange(0, D) * stride_yD, acc, mask=True)


def triton_gemv_batch_rows(X, W_t, M, D, out=None):
    """
    Compute y = X @ W where X has shape [M, D] (across batch, we pass per-batch rows),
    W is [D, D] (W_t is transposed process_weight). Returns y of shape [M, D].
    We will run the Triton kernel per batch row; in practice, pass per-batch row tensors to X_ptr
    and call the kernel multiple times, or concatenate rows and slice back. Here we return [M, D].
    """
    # Ensure contiguity and dtype
    X = X.contiguous()
    W_t = W_t.contiguous()
    # Output buffer
    if out is None:
        y = torch.empty((M, D), device=X.device, dtype=torch.float32)
    else:
        y = out

    # Launch one program per row
    grid = (M,)
    # We accumulate in float32; W_t and X should be float32 for numerical stability
    _gemv_rowwise_kernel[grid](
        X, W_t, y,
        M, D,
        X.stride(0), X.stride(1),
        W_t.stride(0), W_t.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_K=32,  # 32 divides many common D values; masks handle boundaries
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Avoids concatenation by computing directly:
          processed_encoder = cat([encoder_hidden_states @ W, hidden_states @ W], dim=1)
        - All heavy computation happens inside Triton kernels.
        Returns: (processed_encoder, processed_hidden) with shapes [B, T, D] and [B, I, D]
        """
        # Ensure dtype is float32 for stable accumulation; keep original dtype if needed
        B = hidden_states.shape[0]
        D = hidden_states.shape[2]

        # Prepare process_weight^T as [D, D]
        W_t = process_weight.t()  # shape [D, D], on same device

        # Compute yA for each batch: [T, D] per batch -> [B, T, D]
        # We will compute per-batch by passing each batch's encoder_hidden_states to the kernel.
        # This approach avoids building a large concatenated tensor and runs Triton per batch.
        yA_list = []
        for b in range(B):
            X = encoder_hidden_states[b]  # shape [T, D]
            yA = triton_gemv_batch_rows(X, W_t, M=X.shape[0], D=D)
            yA_list.append(yA)
        yA = torch.stack(yA_list, dim=0)  # [B, T, D]

        # Compute yB for each batch: [I, D] per batch -> [B, I, D]
        yB_list = []
        for b in range(B):
            X = hidden_states[b]  # shape [I, D]
            yB = triton_gemv_batch_rows(X, W_t, M=X.shape[0], D=D)
            yB_list.append(yB)
        yB = torch.stack(yB_list, dim=0)  # [B, I, D]

        # Form the final outputs: concatenate along sequence dimension per batch
        # processed_encoder = [A @ W, B @ W] per batch
        # We can use torch.cat along dim=1 to combine per-batch results
        processed_encoder = torch.cat([yA, yB], dim=1)  # [B, T+I, D] but we only need [:, :T, :]
        # However, we need to return [B, T, D] and [B, I, D] separately, which we already have as yA and yB.

        # Return (processed_encoder, processed_hidden)
        # Note: yA corresponds to encoder_hidden_states @ W, yB to hidden_states @ W.
        # The original code returns split based on T and I, but here we already computed per-stream outputs.
        return yA, yB