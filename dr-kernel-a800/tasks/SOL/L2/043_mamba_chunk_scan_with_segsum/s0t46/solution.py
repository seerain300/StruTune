import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def segment_sum_kernel(input_ptr, out_ptr,
                       B: tl.int32, N: tl.int32, M: tl.int32):
    """
    Compute L = exp(segment_sum(input)) for each (b, n):
    - input_ptr: [B, N, M] flattened as B*N*M elements.
    - out_ptr: [B, N, M, M] flattened as B*N*M*M elements.
    - For each (b, n, j), we compute s = sum_{i=0..j-1} input[b, n, i]
      and store L[b, n, j, k] = exp(s) if k <= j (lower-triangular mask),
      else 0. We use diagonal=-1 so only j > k are kept; j <= k -> 0.
    Grid: (B, N)
    """
    b = tl.program_id(0)
    n = tl.program_id(1)

    # Row offset for (b, n) within input tensor
    row_offset_input = (b * N + n) * M
    # Base output offset for (b, n)
    base_out = (b * N + n) * (M * M)

    # Precompute row pointers
    input_row_ptr = input_ptr + row_offset_input
    out_row_ptr = out_ptr + base_out

    # Iterate over columns j and k to fill the [M, M] matrix
    for j in range(0, M):
        # Compute prefix sum s for column j: sum_{i=0..j-1} input[b, n, i]
        s = 0.0
        for i in range(0, j + 1):  # j-1 inclusive
            s += tl.load(input_row_ptr + i)
        # If j > k, we keep exp(s); else (j <= k), value is 0. We'll set k loop and mask accordingly.
        # We'll write across k in one go to reduce load/store
        # But to vectorize, we handle k loop; Triton can't have nested loops with dynamic ranges here easily.
        # Instead, we do per-k write using scalar k, which is fine.
        for k in range(0, M):
            # Lower-triangular mask with diagonal = -1 => keep if j > k, else 0
            keep = j > k
            # Value to store: exp(s) if keep else 0
            val = s
            # Cast to dtype of out_ptr (float32 here)
            val = tl.exp(val) if keep else 0.0
            # Store at out[b, n, j, k]
            tl.store(out_row_ptr + j * M + k, val)


@triton.jit
def cumsum_last_dim_kernel(x_ptr, y_ptr,
                            B: tl.int32, N: tl.int32, M: tl.int32):
    """
    Compute cumulative sum along the last dimension for each (b, n):
    x_ptr: [B, N, M] flattened as B*N*M
    y_ptr: [B, N, M] flattened as B*N*M
    Grid: (B, N)
    """
    b = tl.program_id(0)
    n = tl.program_id(1)

    row_offset = (b * N + n) * M
    x_row_ptr = x_ptr + row_offset
    y_row_ptr = y_ptr + row_offset

    # Initialize running sum
    running = 0.0
    for i in range(0, M):
        v = tl.load(x_row_ptr + i)
        running += v
        tl.store(y_row_ptr + i, running)


@triton.jit
def mul_broadcast_kernel(a_ptr, b_ptr, out_ptr,
                         B: tl.int32, N: tl.int32, M: tl.int32):
    """
    Compute out = a * b[:, :, None, :] where:
    - a: [B, N, M, M] (L)
    - b: [B, N, M] (sA)
    - out: [B, N, M, M]
    Grid: (B, N)
    Each program handles one (b, n) row and computes for all j, k.
    """
    b = tl.program_id(0)
    n = tl.program_id(1)

    base_a = (b * N + n) * (M * M)
    base_b = (b * N + n) * M
    base_out = (b * N + n) * (M * M)

    sA_row_ptr = b_ptr + base_b
    a_row_ptr = a_ptr + base_a
    out_row_ptr = out_ptr + base_out

    for j in range(0, M):
        sA_j = tl.load(sA_row_ptr + j)  # scalar
        for k in range(0, M):
            a_val = tl.load(a_row_ptr + j * M + k)
            # Multiply: out[b, n, j, k] = a_val * sA_j
            out_val = a_val * sA_j
            tl.store(out_row_ptr + j * M + k, out_val)


class ModelNew(nn.Module):
    def forward(self, input_tensor: torch.Tensor, A: torch.Tensor):
        """
        input_tensor: [B, N, M] float32
        A: [B, N, M] float32
        Returns Y: [B, N, M, M] same dtype as input_tensor
        """
        assert input_tensor.ndim == 3, "input_tensor must be [B, N, M]"
        assert A.ndim == 3, "A must be [B, N, M]"
        B, N, M = input_tensor.shape

        # Ensure tensors are contiguous
        input_tensor = input_tensor.contiguous()
        A = A.contiguous()

        # Allocate outputs
        L = torch.empty((B, N, M, M), device=input_tensor.device, dtype=torch.float32)
        sA = torch.empty((B, N, M), device=input_tensor.device, dtype=torch.float32)
        Y = torch.empty((B, N, M, M), device=input_tensor.device, dtype=torch.float32)

        # Launch Triton kernels
        grid = (B, N)

        # segment_sum_kernel: compute L
        segment_sum_kernel[grid](input_tensor.view(-1), L.view(-1),
                                 B, N, M,
                                 num_warps=4)

        # cumsum_last_dim_kernel: compute sA = cumsum(A, dim=-1)
        cumsum_last_dim_kernel[grid](A.view(-1), sA.view(-1),
                                     B, N, M,
                                     num_warps=4)

        # mul_broadcast_kernel: Y = L * sA[:, :, None, :]
        mul_broadcast_kernel[grid](L.view(-1), sA.view(-1), Y.view(-1),
                                   B, N, M,
                                   num_warps=4)

        return Y


def run(*args):
    return ModelNew()(*args)
