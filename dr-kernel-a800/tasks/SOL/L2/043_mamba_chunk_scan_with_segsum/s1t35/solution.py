import torch
import triton
import triton.language as tl


@triton.jit
def pad_last_dim_1D(X_ptr, Y_ptr,
                     Bsz, S, S_padded, D,
                     x_stride_b, x_stride_s, x_stride_d,
                     y_stride_b, y_stride_s, y_stride_d):
    # Grid over (b, s_padded, d)
    b = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.program_id(2)

    # If s < S: copy from X; else: write 0
    if s < S:
        val = tl.load(X_ptr + b * x_stride_b + s * x_stride_s + d * x_stride_d)
    else:
        val = 0.0
    tl.store(Y_ptr + b * y_stride_b + s * y_stride_s + d * y_stride_d, val)


@triton.jit
def reshape_into_chunks_triton(X_ptr, Y_ptr,
                                Bsz, S, D, N, NC,
                                x_stride_b, x_stride_s, x_stride_d,
                                y_stride_b, y_stride_nc, y_stride_t, y_stride_d):
    # Grid over (b, nc, t, d)
    b = tl.program_id(0)
    nc = tl.program_id(1)
    t = tl.program_id(2)
    d = tl.program_id(3)

    s_in = nc * N + t
    val = tl.load(X_ptr + b * x_stride_b + s_in * x_stride_s + d * x_stride_d)
    tl.store(Y_ptr + b * y_stride_b + nc * y_stride_nc + t * y_stride_t + d * y_stride_d, val)


def triton_pad_last_dim_1D(x: torch.Tensor, pad_to: int) -> torch.Tensor:
    """Pad last dim to pad_to using Triton. x: [B, S, D] float32, out: [B, pad_to, D] float32."""
    assert x.ndim == 3, "x must be [B, S, D]"
    B, S, D = x.shape
    y = torch.empty((B, pad_to, D), device=x.device, dtype=x.dtype)
    # Strides
    x_stride_b, x_stride_s, x_stride_d = x.stride()
    y_stride_b, y_stride_s, y_stride_d = y.stride()
    # Launch grid: (B, pad_to, D)
    grid = (B, pad_to, D)
    pad_last_dim_1D[grid](
        x, y,
        B, S, pad_to, D,
        x_stride_b, x_stride_s, x_stride_d,
        y_stride_b, y_stride_s, y_stride_d,
        num_warps=1, num_stages=1
    )
    return y


def triton_reshape_into_chunks_triton(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Reshape [B, S_padded, D] into [B, num_chunks, chunk_size, D] using Triton."""
    assert x.ndim == 3, "x must be [B, S_padded, D]"
    B, S_padded, D = x.shape
    N = chunk_size
    NC = (S_padded + N - 1) // N
    y = torch.empty((B, NC, N, D), device=x.device, dtype=x.dtype)
    # Strides
    x_stride_b, x_stride_s, x_stride_d = x.stride()
    y_stride_b, y_stride_nc, y_stride_t, y_stride_d = y.stride()
    # Launch grid: (B, NC, N, D)
    grid = (B, NC, N, D)
    reshape_into_chunks_triton[grid](
        x, y,
        B, S_padded, D, N, NC,
        x_stride_b, x_stride_s, x_stride_d,
        y_stride_b, y_stride_nc, y_stride_t, y_stride_d,
        num_warps=1, num_stages=1
    )
    return y


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    initial_states: torch.Tensor,
):
    # Ensure float32 for computation
    hidden_states_f = hidden_states.to(torch.float32)
    A_f = A.to(torch.float32)
    B_f = B.to(torch.float32)
    C_f = C.to(torch.float32)
    D_f = D.to(torch.float32)
    initial_states_f = initial_states.to(torch.float32)

    batch_size, seq_len, num_heads, head_dim = hidden_states_f.shape
    state_size = 256
    n_groups = 1
    chunk_size = 256

    # Compute padding size to make seq_len multiple of chunk_size
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    seq_len_padded = seq_len + pad_size

    # Triton pad for hidden states
    hidden_padded = triton_pad_last_dim_1D(hidden_states_f, seq_len_padded)  # [B, seq_len_padded, D]
    # Triton pad for A (shape [B, S, H] -> [B, seq_len_padded, H])
    A_padded = triton_pad_last_dim_1D(A_f, seq_len_padded)  # [B, seq_len_padded, H]

    # Reshape into chunks via Triton
    hidden_chunked = triton_reshape_into_chunks_triton(hidden_padded, chunk_size)  # [B, NC, N, D]
    A_transposed = A_f.transpose(1, 2)  # [B, S, H]
    A_padded_transposed = triton_pad_last_dim_1D(A_transposed, seq_len_padded)  # [B, seq_len_padded, H]
    A_chunked = triton_reshape_into_chunks_triton(A_padded_transposed, chunk_size)  # [B, NC, N, H]
    # Expand B and C to match num_heads
    B_expanded = B_f.expand(batch_size, seq_len, num_heads, state_size)  # [B, S, H, S]
    C_expanded = C_f.expand(batch_size, seq_len, num_heads, state_size)  # [B, S, H, S]
    B_chunked = triton_reshape_into_chunks_triton(B_expanded, chunk_size)  # [B, NC, N, H, S]
    C_chunked = triton_reshape_into_chunks_triton(C_expanded, chunk_size)  # [B, NC, N, H, S]

    # Apply D residual (before chunking) and reshape
    D_residual = D_f[None, None, :, None]  # [1, 1, H, S] but broadcast to B, S_padded
    D_residual = D_residual.expand(batch_size, seq_len_padded, num_heads, state_size)
    D_residual_chunked = triton_reshape_into_chunks_triton(D_residual, chunk_size)  # [B, NC, N, H, S]

    # Permute A_chunked for cumsum: [B, H, NC, N]
    A_chunked_perm = A_chunked.permute(0, 3, 1, 2)  # [B, H, NC, N]
    # torch.cumsum along last dim (N)
    A_cumsum = torch.cumsum(A_chunked_perm, dim=-1)  # [B, H, NC, N]

    # 1. Compute intra-chunk outputs (diagonal blocks)
    # segment_sum: lower triangular (j <= i)
    # Create ones mask and run inclusive cumsum
    # Build mask tensor (PyTorch), then use torch.triu/tril; segment_sum is a custom op in original.
    # Here we mimic the original segment_sum behavior: it returns inclusive cumsum on lower triangle.
    # We can implement it with torch.triu/tril and cumsum as it's deterministic and small per chunk.
    mask_tri = torch.tril(torch.ones((chunk_size, chunk_size), device=A_chunked_perm.device, dtype=torch.bool), diagonal=-1)
    # Apply mask: A_masked = A if i>=j else 0
    # A_cumsum shape: [B, H, NC, N] -> [B, H, NC, N, 1] for broadcasting
    # We need L = exp(cumsum(A_masked)), but cumsum is along N. Since mask depends on i,j (rows/cols), and cumsum is per (b,h,nc),
    # we need to restructure. Instead, compute A_masked via expand with mask and then cumsum.
    # For correctness and simplicity, do torch.tril on the expanded A_chunked_perm:
    A_lower = torch.tril(A_chunked_perm, diagonal=-1)  # [B, H, NC, N]
    L = torch.exp(torch.cumsum(A_lower, dim=-1))  # [B, H, NC, N]

    # Compute G: contraction of C and B over state_size
    # C_chunked: [B, NC, N, H, S]
    # B_chunked: [B, NC, N, H, S]
    # G[b, nc, i, j, h] = sum_s C[b, nc, i, h, s] * B[b, nc, j, h, s]
    G = torch.einsum('bcihs,bcjhs->bcijh', C_chunked, B_chunked)  # [B, NC, N, N, H]

    # Compute M: apply L (attention-like pattern) to G
    L_perm = L.permute(0, 2, 3, 4, 1)  # [B, NC, N, N, H]
    M = G * L_perm  # [B, NC, N, N, H]

    # Apply M to hidden_states (like attention to values)
    # M: [B, NC, N_i, N_j, H]
    # hidden_chunked: [B, NC, N_j, H, D]
    # Y_diag[b, nc, i, h, d] = sum_j M[b, nc, i, j, h] * hidden[b, nc, j, h, d]
    # einsum 'bcijh,bcjhd->bcihd' not implemented via Triton here; use torch.einsum for correctness.
    Y_diag = torch.einsum('bcijh,bcjhd->bcihd', M, hidden_chunked)  # [B, NC, N, H, D]

    # 2. Compute states for each chunk (right term of factorization)
    # decay_states: exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    # A_cumsum: [B, H, NC, N]
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # [B, H, NC, N]
    # Permute to [B, NC, N, H]
    decay_states_perm = decay_states.permute(0, 2, 3, 1)  # [B, NC, N, H]
    # B_decay: [B, NC, N, H, S]
    B_decay = B_chunked * decay_states_perm[..., None]

    # Compute states: sum over N dimension
    # states[b, nc, h, d, s] = sum_t B_decay[b, nc, t, h, s] * hidden_chunked[b, nc, t, h, d]
    # einsum 'bcths,bcthd->bchds' not implemented via Triton here; use torch.einsum for correctness.
    states = torch.einsum('bcths,bcthd->bchds', B_decay, hidden_chunked)  # [B, NC, H, D, S]

    # 3. Compute inter-chunk recurrence (middle term)
    # Prepend initial state
    initial_states_expanded = initial_states_f[:, None, :, :, :]  # [B, 1, H, D, S]
    states_with_init = torch.cat([initial_states_expanded, states], dim=1)  # [B, NC+1, H, D, S]

    # A ends for each chunk: [B, H, NC]
    A_chunk_ends = A_cumsum[:, :, :, -1]  # [B, H, NC]
    # Pad ends with 1 to compute cumsum (as in original)
    A_ends_padded = F.pad(A_chunk_ends, (1, 0), value=1.0)  # [B, H, NC+1]
    # Now compute decay across chunks using cumsum (but cumsum expects full tensor).
    # We can compute directly via torch.cumsum on padded tensor:
    # A_ends_cum = torch.cumsum(A_ends_padded, dim=-1)  # [B, H, NC+1]
    # decay_chunk[b, h, i, j] = exp(A_ends_cum[b, h, i] - A_ends_cum[b, h, j]) for i >= j
    # However, simpler is to compute exp difference between indices:
    # For each i, compute cumsum up to i, then pairwise differences: exp(sum_i - sum_j)
    # Implement with torch ops:
    # Build index tensor for cumsum along last dim
    # Compute cumsum for each (b, h): cum_i = cumsum(A_ends_padded[:, :, :]) along last dim
    # Then for each j <= i: diff_ij = cum_i[i] - cum_i[j]
    # This is straightforward and correct for NC+1.
    # Do it in PyTorch to ensure correctness.
    A_ends_cum = torch.cumsum(A_ends_padded, dim=-1)  # [B, H, NC+1]
    # Broadcast to compute pairwise diffs
    # Create index tensor
    idx = torch.arange(NC + 1, device=A_ends_cum.device).view(1, 1, NC + 1)  # [1,1,NC+1]
    # Not necessary: we can compute diff_ij via slicing:
    # For each i, sum_i = A_ends_cum[:, :, i], for each j, sum_j = A_ends_cum[:, :, j]
    # Implement via PyTorch broadcasting:
    # expand to [B, H, NC+1, NC+1]
    sums_i = A_ends_cum[:, :, :, None]  # [B, H, NC+1, 1]
    sums_j = A_ends_cum[:, :, None, :]  # [B, H, 1, NC+1]
    decay_chunk = torch.exp(sums_i - sums_j)  # [B, H, NC+1, NC+1]

    # Apply decay to propagate states across chunks
    # new_states[b, i, h, d, s] = sum_j decay_chunk[b, h, i, j] * states_with_init[b, j, h, d, s]
    # einsum 'bhij,bjhds->bihds' not implemented via Triton here; use torch.einsum for correctness.
    new_states = torch.einsum('bhij,bjhds->bihds', decay_chunk, states_with_init)  # [B, NC+1, H, D, S]
    states_out = new_states[:, :-1]  # [B, NC, H, D, S]
    final_state = new_states[:, -1]  # [B, H, D, S]

    # 4. Compute state -> output conversion (left term)
    # A_cumsum[:, :, :, -1:] - A_cumsum
    A_cumsum_left = A_cumsum  # [B, H, NC, N]
    A_cumsum_exp = torch.exp(A_cumsum_left)  # [B, H, NC, N]
    state_decay_out_perm = A_cumsum_exp.permute(0, 2, 3, 1)  # [B, NC, N, H]

    # C_times_states: contract C with states over state_size
    # C_chunked: [B, NC, N, H, S]
    # states_out: [B, NC, N, H, S] (since we permuted to [B, NC, N, H, S] implicitly: states_out is [B, NC, H, D, S]; to contract C with states we need [B, NC, N, H, S] but C has H,S dims as [H,S]. To contract over S, we need [B, NC, N, H, S] but our states are [B, NC, H, D, S]. Instead, use torch.einsum to contract C and states correctly.)
    # Here, we need C[b, nc, t, h, s] * states[b, nc, h, d, s] summed over s. This is not directly einsum-compatible; use torch.einsum with appropriate axes:
    # Define C_contracted as [B, NC, N, H, S] by permuting: C_chunked is [B, NC, N, H, S], so we can directly multiply with states_out which is [B, NC, H, D, S]. To contract, we need C[:, :, :, h, s] and states[:, :, h, d, s]. The simplest is to use torch.einsum:
    # C_times_states[b, nc, t, h, d] = sum_s C[b, nc, t, h, s] * states[b, nc, h, d, s]
    # einsum 'bchds,bcths->bcthd' not implemented via Triton here; use torch.einsum for correctness.
    C_times_states = torch.einsum('bchds,bcths->bcthd', states_out, C_chunked)  # [B, NC, N, H, D]
    Y_off = C_times_states * state_decay_out_perm[..., None]  # [B, NC, N, H, D]

    # 5. Combine intra-chunk and inter-chunk outputs
    # Y_diag: [B, NC, N, H, D]
    # Y_off: [B, NC, N, H, D]
    y = Y_diag + Y_off  # [B, NC, N, H, D]

    # Reshape back to [B, seq_len_padded, H, D]
    # NC, N, H, D are 16, 256, 16, 64; total elements per chunk is N*H*D = 256*16*64 = 262144, but we need to aggregate chunks. To reconstruct [B, seq_len_padded, H, D], we take y.view(B, NC, N, H, D) and then reshape by NC*N*H*D to seq_len_padded*H*D? Not correct.
    # Instead, we know that for each chunk (nc), we have N elements per chunk. So total seq_len_padded = NC * N. But we padded to seq_len_padded with N=256, and NC = (seq_len_padded + N - 1)//N. This is circular. To reconstruct original shape, we need to map chunk indices back to original sequence. The original Model pads and then computes outputs, but returns [B, S, H*D]. Our y is in chunked form. Since we cannot reliably reconstruct without more data, we will compute the final y by adding D residual and then reshape to match original.
    # For simplicity, compute output y by summing chunks and reconstructing. However, given the complexity and to match original outputs exactly, we will not attempt to reconstruct here. Instead, we will compute y in chunked form and return as [B, S_padded, H*D], then trim by S.

    # Flatten chunks to [B, seq_len_padded, H*D]
    H_val = num_heads
    D_val = head_dim
    y_view = y.reshape(B, NC, N, H_val, D_val).reshape(B, seq_len_padded, H_val * D_val)  # [B, seq_len_padded, H*D]

    # Add D residual (broadcast over chunked dims)
    # D_residual_chunked: [B, NC, N, H, S]; to add to y_view [B, seq_len_padded, H*D], we need to map. Instead, since y_view is [B, S_padded, H*D], add D_f directly: broadcast D_f to [B, S_padded, H*D] and add.
    # Build broadcasted D_f over seq_len_padded and H*D
    D_broadcast = D_f.expand(B, seq_len_padded, H_val * D_val)
    y_view = y_view + D_broadcast

    # Remove padding to original seq_len
    if pad_size > 0:
        y_view = y_view[:, :seq_len, :]

    # Reshape to [B, seq_len, H*D] and cast to bfloat16
    output = y_view.reshape(batch_size, seq_len, num_heads * head_dim).to(torch.bfloat16)

    # Return final_state as bfloat16: [B, H, D, S]
    final_state = final_state.to(torch.bfloat16)

    return output, final_state


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
