import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_rows_triton_kernel(
    X,            # pointer to input tensor [B, S, D], contiguous
    W,            # pointer to weight tensor [D, D], contiguous
    Y,            # pointer to output tensor [B, S, D], contiguous
    B, S, D,      # dimensions
    x_s0, x_s1, x_s2,   # strides for X
    w_s0, w_s1,         # strides for W
    y_s0, y_s1, y_s2,   # strides for Y
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, S, 1) -> each program handles one (batch, row)
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Base offsets for X and Y for this (b, s)
    x_base = b * x_s0 + s * x_s1
    y_base = b * y_s0 + s * y_s1

    # Accumulator vector for the output row
    acc = tl.zeros([D], dtype=tl.float32)

    # Loop over K dimension in tiles of BLOCK_K
    for k0 in range(0, D, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D

        # Load input row segment X[b, s, k0:k0+BLOCK_K]
        x_ptr = X + x_base + offs_k * x_s2
        x_row = tl.load(x_ptr, mask=mask_k, other=0.0)

        # Load weight submatrix W[offs_k, :] of shape [BLOCK_K, D]
        w_base = offs_k * w_s0  # rows
        w_ptr = W + w_base[:, None] + tl.arange(0, D) * w_s1  # columns
        w_sub = tl.load(w_ptr, mask=mask_k[:, None], other=0.0)

        # Accumulate: acc += sum_k x_row[k] * w_sub[k, :]
        # Broadcast x_row to [1, BLOCK_K] and reduce over axis=1
        x_row_exp = x_row[None, :]
        prod = x_row_exp * w_sub
        acc += tl.sum(prod, axis=1)

    # Store result Y[b, s, :]
    y_ptr = Y + y_base + tl.arange(0, D) * y_s2
    tl.store(y_ptr, acc)


def triton_linear_projection(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Compute input_tensor @ weight.T with no bias using Triton.
    input_tensor: [B, S, D], contiguous
    weight: [D, D], contiguous
    returns: [B, S, D], contiguous
    """
    assert input_tensor.is_cuda and weight.is_cuda, "Triton kernel requires CUDA tensors"
    B, S, D = input_tensor.shape
    # Ensure contiguous layout
    X = input_tensor.contiguous()
    W = weight.contiguous()
    Y = torch.empty((B, S, D), dtype=torch.float32, device=X.device)

    # Choose BLOCK_K. Use a power-of-two up to 1024. If D < 1024, pick D itself or a divisor.
    # For simplicity and safety, set BLOCK_K to min(1024, D). Masks handle boundaries.
    BLOCK_K = 1024 if D >= 1024 else (1 << (int(D - 1).bit_length() - 1)) or 1

    grid = (B, S)
    _matmul_rows_triton_kernel[grid](
        X, W, Y,
        B, S, D,
        X.stride(0), X.stride(1), X.stride(2),
        W.stride(0), W.stride(1),
        Y.stride(0), Y.stride(1), Y.stride(2),
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return Y


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate along sequence dimension (PyTorch)
        - Apply linear projection (Triton): [B, T+I, D] @ process_weight.T
        - Split back to [B, T, D] and [B, I, D]
        """
        # Step 1: Concatenate encoder_hidden_states and hidden_states along sequence dimension
        # Shapes: hidden_states [B, I, D], encoder_hidden_states [B, T, D]
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, D]

        # Step 2: Apply linear projection using Triton
        # process_weight is [D, D]; we multiply concatenated [B, T+I, D] by W.T -> [B, T+I, D]
        processed = triton_linear_projection(concatenated, process_weight)

        # Step 3: Split back into two outputs
        # processed shape: [B, T+I, D]
        total_seq = processed.shape[1]
        processed_encoder = processed[:, :encoder_hidden_states.shape[1], :]
        processed_hidden = processed[:, encoder_hidden_states.shape[1]:, :]

        return processed_encoder, processed_hidden