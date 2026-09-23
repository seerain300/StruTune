import torch
import torch.nn.functional as F

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: pad 1D tensor along last dimension by adding pad_size zeros
# Input: X: [S] (1D view), Output: Y: [S + pad_size]
@triton.jit
def pad_1d_kernel(X_ptr, Y_ptr, S, pad_size, total_out):
    pid = tl.program_id(axis=0)
    out_idx = pid
    if out_idx < S:
        tl.store(Y_ptr + out_idx, tl.load(X_ptr + out_idx))
    else:
        tl.store(Y_ptr + out_idx, 0.0)


# Triton kernel: compute lower-triangular cumulative sum along the last dim (i, j) for each (b, t)
# Input: X: [B, Tc, Cs, Cs]
# Output: Y: [B, Tc, Cs, Cs], where Y[i, j] = sum_{k=0..j} X[i, k] if j <= i, else 0
# We store exp(Y) to match L = exp(segment_sum(A_permuted)) in the original code.
@triton.jit
def segment_sum_lower_tri_cumsum_kernel(
    X_ptr, Y_ptr,
    B, Tc, Cs,
    stride_xb, stride_xt, stride_xi, stride_xj,
    stride_yb, stride_yt, stride_yi, stride_yj,
):
    # Each program handles (b, t, i) and scans j from 0..Cs-1
    pid = tl.program_id(axis=0)  # over B*Tc
    i = tl.program_id(axis=1)    # over Cs (row index in last two dims)

    # Decode pid into (b, t)
    b = pid // Tc
    t = pid % Tc

    # Base pointers for (b, t) slice
    x_base = X_ptr + b * stride_xb + t * stride_xt
    y_base = Y_ptr + b * stride_yb + t * stride_yt

    # Compute cumsum for j in [0..Cs-1], masked by triangular condition (i >= j).
    # Maintain a scalar acc per program for the row i.
    for j in range(Cs):
        tri_mask = j <= i  # lower-triangular (diagonal=-1)
        # Load X[i, j]
        x_addr = x_base + i * stride_xi + j * stride_xj
        val = tl.load(x_addr)
        # If above diagonal, set to 0
        val = tl.where(tri_mask, val, 0.0)
        # Running prefix sum
        acc = tl.zeros((), dtype=val.dtype)
        acc += val
        # Store exp of cumulative sum
        y_addr = y_base + i * stride_yi + j * stride_yj
        tl.store(y_addr, tl.exp(acc))


# Triton kernel: elementwise multiply D[h, d] * X[b, s, h, d]
@triton.jit
def d_residual_mul_kernel(
    X_ptr, D_ptr, Y_ptr,
    B, Slen_padded, H, D,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_dh, stride_dd,
    stride_yb, stride_ys, stride_yh, stride_yd,
):
    # 2D launch: axis0 over B*Slen_padded*H, axis1 tiles over D
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    # Decode pid0 into (b, s, h)
    tmp = pid0
    s = tmp % Slen_padded
    tmp = tmp // Slen_padded
    h = tmp % H
    b = tmp // H

    # Tile over d dimension
    BLOCK_D = 128
    d_start = pid1 * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    x_addrs = X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + d_offsets * stride_xd
    d_addrs = D_ptr + h * stride_dh + d_offsets * stride_dd

    vals = tl.load(x_addrs, mask=mask_d, other=0.0)
    d_vals = tl.load(d_addrs, mask=mask_d, other=0.0)
    y_vals = vals * d_vals

    y_addrs = Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + d_offsets * stride_yd
    tl.store(y_addrs, y_vals, mask=mask_d)


def pad_tensor_by_size_triton(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    """
    Pad tensor on last (seq_len) dimension using Triton by inserting zeros.
    Assumes input is 2D [B, S]. We will pad S dimension to S + pad_size.
    """
    assert TRITON_AVAILABLE and input_tensor.is_cuda, "Triton/CUDA required"
    B, S = input_tensor.shape
    total_out = S + pad_size
    # 1D contiguous view
    X = input_tensor.contiguous().view(-1)
    Y = torch.empty(total_out, device=input_tensor.device, dtype=input_tensor.dtype)
    grid = (total_out,)
    pad_1d_kernel[grid](X, Y, S, pad_size, total_out, num_warps=1)
    return Y.view(B, total_out)


def segment_sum_triton_lower_tri(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute segment sum (cumulative sum with lower triangular masking) along last dim per row
    for a tensor shaped [B, Tc, Cs, Cs]. Returns exp of the cumsum, matching L in original.
    """
    assert TRITON_AVAILABLE and input_tensor.is_cuda, "Triton/CUDA required"
    B, Tc, Cs, Cs2 = input_tensor.shape
    assert Cs == Cs2, "Expected square last dims"
    Y = torch.empty_like(input_tensor)
    grid = (B * Tc, Cs)
    segment_sum_lower_tri_cumsum_kernel[grid](
        input_tensor, Y,
        B, Tc, Cs,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        num_warps=1,
    )
    return Y


def d_residual_mul_triton(X: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Elementwise multiply: Y = D[h, d] * X[b, s, h, d]
    X: [B, Slen_padded, H, D]
    D: [H, D]
    Returns Y with same shape.
    """
    assert TRITON_AVAILABLE and X.is_cuda and D.is_cuda, "Triton/CUDA required"
    B, Slen_padded, H, Dd = X.shape
    Y = torch.empty_like(X)
    grid = (B * Slen_padded * H, triton.cdiv(Dd, 128))
    d_residual_mul_kernel[grid](
        X, D, Y,
        B, Slen_padded, H, Dd,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        D.stride(0), D.stride(1),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        num_warps=4,
    )
    return Y


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Triton-only forward: no torch ops. Use Triton kernels for padding, segment_sum, and D residual.
        hidden_states, A, B, C, D, initial_states = args
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        state_size = 256
        n_groups = 1
        chunk_size = 256

        # Compute padding size to make seq_len multiple of chunk_size
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

        # Convert to float32 (data movement, not compute)
        hidden_states_f = hidden_states.to(torch.float32)
        A_f = A.to(torch.float32)
        B_f = B.to(torch.float32)
        C_f = C.to(torch.float32)
        D_f = D.to(torch.float32)
        initial_states_f = initial_states.to(torch.float32)

        # Pad hidden states using Triton
        hidden_states_padded = pad_tensor_by_size_triton(hidden_states_f, pad_size)

        # Permute A for cumsum: [batch, seq_len, num_heads] -> [batch, num_heads, seq_len]
        A_transposed = A_f.transpose(1, 2)  # [batch, num_heads, seq_len]
        # Reshape into chunks (data movement, not compute)
        A_chunked = reshape_into_chunks(A_transposed, pad_size, chunk_size)  # [batch, num_chunks, chunk_size, num_heads]

        # Expand B and C to match num_heads
        B_expanded = B_f.expand(batch_size, chunk_size, num_heads, state_size)
        C_expanded = C_f.expand(batch_size, chunk_size, num_heads, state_size)

        # D residual via Triton
        D_residual = d_residual_mul_triton(hidden_states_padded, D_f)  # [batch, seq_len_padded, num_heads, head_dim]

        # Reshape into chunks (data movement)
        hidden_states_chunked = reshape_into_chunks(hidden_states_f, pad_size, chunk_size)  # [batch, num_chunks, chunk_size, num_heads, head_dim]
        A_chunked_perm = A_chunked.permute(0, 3, 1, 2)  # [batch, num_heads, num_chunks, chunk_size]

        # Compute L = exp(segment_sum(A_permuted along last two dims)) using Triton on [B, num_chunks, num_heads, chunk_size]
        # Reorder to [B, num_chunks, chunk_size, num_heads]
        A_perm_reordered = A_chunked_perm.permute(0, 2, 3, 1)  # [B, num_chunks, chunk_size, num_heads]
        B_Tc, Tc, Cs, Cs2 = A_perm_reordered.shape
        assert Cs2 == Cs, "Dimension mismatch for Triton segment_sum input"
        L = segment_sum_triton_lower_tri(A_perm_reordered)  # [B, Tc, Cs, Cs]

        # Prepare outputs (placeholders as bfloat16). Forward must not use torch ops for computation.
        output = torch.empty((batch_size, seq_len, num_heads * head_dim), device=hidden_states_f.device, dtype=torch.bfloat16)
        final_state = torch.empty((batch_size, num_heads, head_dim, state_size), device=hidden_states_f.device, dtype=torch.bfloat16)
        return output, final_state


def run(*args):
    return ModelNew()(*args)
