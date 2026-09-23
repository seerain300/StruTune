import torch
import triton
import triton.language as tl


# Triton kernel to compute G = einsum('bcihs,bcjhs->bcijh'):
# G[b, nc, i, j, h] = sum_s C[b, nc, i, h, s] * B[b, nc, j, h, s]
@triton.jit
def outer_product_bcijh_kernel(
    C_ptr,    # *float32, shape [B, N_chunks, Chunk, H, State]
    B_ptr,    # *float32, shape [B, N_chunks, Chunk, H, State]
    G_ptr,    # *float32, shape [B, N_chunks, Chunk, Chunk, H]
    B_sz: tl.constexpr,          # batch size
    N_chunks: tl.constexpr,      # number of chunks per (batch,h)
    Chunk: tl.constexpr,         # chunk_size (time steps)
    H: tl.constexpr,             # num_heads
    State: tl.constexpr,         # state_size (256)
    S_stride0, S_stride1, S_stride2, S_stride3, S_stride4,  # strides for G
    C_stride0, C_stride1, C_stride2, C_stride3, C_stride4,  # strides for C
    B_stride0, B_stride1, B_stride2, B_stride3, B_stride4,  # strides for B
):
    # Grid: one program per (b, nc, i, j, h)
    pid = tl.program_id(axis=0)
    total = N_chunks * Chunk * Chunk * H
    b = pid // total
    rem = pid % total
    nc = rem // (Chunk * Chunk * H)
    i = rem % (Chunk * Chunk * H) // (Chunk * H)
    j = (rem % (Chunk * H)) // H
    h = rem % H

    # Accumulate G[i,j,h] over s in tiles of size 1 (State=256 in original), since we use scalar accumulation
    # We keep it simple: for each s in 0..State-1, compute and store directly. If State were large, we would tile.
    for s in range(0, State):
        # Load C and B at (b, nc, i, h, s) and (b, nc, j, h, s)
        c_idx = b * C_stride0 + nc * C_stride1 + i * C_stride2 + h * C_stride3 + s * C_stride4
        b_idx = b * B_stride0 + nc * B_stride1 + j * B_stride2 + h * B_stride3 + s * B_stride4
        c_val = tl.load(C_ptr + c_idx)
        b_val = tl.load(B_ptr + b_idx)

        # G address for (b, nc, i, j, h) and s-th element: last dim is H, so strides S_stride4 corresponds to H,
        # S_stride3 corresponds to Chunk, S_stride2 to Chunk, S_stride1 to N_chunks, S_stride0 to B.
        g_addr = b * S_stride0 + nc * S_stride1 + i * S_stride2 + j * S_stride3 + h * S_stride4
        # Note: G has shape [B, N_chunks, Chunk, Chunk, H] with strides S_stride0..S_stride4.
        # We store into G at fixed (i,j) for this (b,nc,h) across s.
        # To generalize, we store the scalar result at G[b, nc, i, j, h, s] using s*S_stride4.
        tl.store(G_ptr + g_addr + s * S_stride4, c_val * b_val)


# Triton kernel to compute S = einsum('bcths,bcthd->bchds'):
# S[b, nc, h, s] = sum_{t in chunk_size} sum_{d in head_dim} B[b, nc, t, h, s] * hidden[b, nc, t, h, d]
@triton.jit
def outer_product_bchds_kernel(
    B_ptr,      # *float32, shape [B, N_chunks, Chunk, H, State]
    hidden_ptr, # *float32, shape [B, N_chunks, Chunk, H, Dim]
    S_ptr,      # *float32, shape [B, N_chunks, H, State]
    B_sz: tl.constexpr,          # batch size
    N_chunks: tl.constexpr,      # number of chunks per (batch,h)
    Chunk: tl.constexpr,         # chunk_size
    H: tl.constexpr,             # num_heads
    State: tl.constexpr,         # state_size
    Dim: tl.constexpr,           # head_dim
    S_stride0, S_stride1, S_stride2, S_stride3,  # strides for S
    B_stride0, B_stride1, B_stride2, B_stride3, B_stride4,  # strides for B
    hidden_stride0, hidden_stride1, hidden_stride2, hidden_stride3, hidden_stride4,  # strides for hidden
):
    # Grid: one program per (b, nc, h)
    pid = tl.program_id(axis=0)
    total = N_chunks * H
    b = pid // total
    rem = pid % total
    nc = rem // H
    h = rem % H

    # Accumulate per s
    for s in range(0, State):
        acc = tl.zeros([1], dtype=tl.float32)
        # Loop over t (chunk_size) and d (head_dim)
        for t in range(0, Chunk):
            for d in range(0, Dim):
                b_val = tl.load(B_ptr + b * B_stride0 + nc * B_stride1 + t * B_stride2 + h * B_stride3 + s * B_stride4)
                hid_val = tl.load(hidden_ptr + b * hidden_stride0 + nc * hidden_stride1 + t * hidden_stride2 + h * hidden_stride3 + d * hidden_stride4)
                acc += b_val * hid_val
        # Store acc to S[b, nc, h, s]
        S_addr = b * S_stride0 + nc * S_stride1 + h * S_stride2 + s * S_stride3
        tl.store(S_ptr + S_addr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                initial_states: torch.Tensor):
        # Cast to float32 for numerical stability
        hidden_states_f = hidden_states.to(torch.float32)
        A_f = A.to(torch.float32)
        B_f = B.to(torch.float32)
        C_f = C.to(torch.float32)
        D_f = D.to(torch.float32)
        initial_states_f = initial_states.to(torch.float32)

        # Dimensions
        batch_size, seq_len, num_heads, head_dim = hidden_states_f.shape
        state_size = 256
        chunk_size = 256
        n_groups = 1  # not used directly

        # Compute padding size to make seq_len multiple of chunk_size
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        seq_len_padded = seq_len + pad_size

        # Pad hidden with torch (metadata), keep Triton for heavy reductions
        hidden_padded = F.pad(hidden_states_f, (0, pad_size), mode='constant', value=0).contiguous()  # [B, seq_len_padded, H, D]

        # Reshape into chunks: [batch, num_chunks, chunk_size, num_heads, head_dim]
        num_chunks = (seq_len_padded + chunk_size - 1) // chunk_size
        hidden_chunked = hidden_padded.reshape(batch_size, num_chunks, chunk_size, num_heads, head_dim).contiguous()

        # Transpose A: [batch, seq_len, num_heads] -> [batch, num_heads, seq_len]
        A_transposed = A_f.transpose(1, 2).contiguous()  # [B, H, seq_len]
        # Pad A to multiple of chunk_size for chunking
        seq_len_A = A_transposed.shape[2]
        pad_size_A = (chunk_size - seq_len_A % chunk_size) % chunk_size
        seq_len_padded_A = seq_len_A + pad_size_A
        A_padded = F.pad(A_transposed, (0, 0, 0, pad_size_A), mode='constant', value=0)  # [B, H, seq_len_padded_A]
        A_chunked = A_padded.reshape(batch_size, num_chunks, chunk_size, num_heads).contiguous()  # [B, N, Chunk, H]

        # Expand B and C to match num_heads (from n_groups=1 to num_heads)
        B_expanded = B_f.unsqueeze(-1).expand(batch_size, seq_len_padded, num_heads, state_size).contiguous()  # [B, seq_len_padded, H, S]
        C_expanded = C_f.unsqueeze(-1).expand(batch_size, seq_len_padded, num_heads, state_size).contiguous()  # [B, seq_len_padded, H, S]

        # Reshape B_expanded and C_expanded into chunks: [B, N, Chunk, H, S]
        B_chunked = B_expanded.reshape(batch_size, num_chunks, chunk_size, num_heads, state_size).contiguous()
        C_chunked = C_expanded.reshape(batch_size, num_chunks, chunk_size, num_heads, state_size).contiguous()

        # Compute G = einsum('bcihs,bcjhs->bcijh') using Triton
        G = torch.empty((batch_size, num_chunks, chunk_size, chunk_size, num_heads), dtype=torch.float32, device=A_chunked.device)

        grid_G = (batch_size * num_chunks * chunk_size * chunk_size * num_heads,)
        outer_product_bcijh_kernel[grid_G](
            C_chunked, B_chunked, G,
            B_sz=batch_size, N_chunks=num_chunks, Chunk=chunk_size, H=num_heads, State=state_size,
            S_stride0=G.stride(0), S_stride1=G.stride(1), S_stride2=G.stride(2), S_stride3=G.stride(3), S_stride4=G.stride(4),
            C_stride0=C_chunked.stride(0), C_stride1=C_chunked.stride(1), C_stride2=C_chunked.stride(2), C_stride3=C_chunked.stride(3), C_stride4=C_chunked.stride(4),
            B_stride0=B_chunked.stride(0), B_stride1=B_chunked.stride(1), B_stride2=B_chunked.stride(2), B_stride3=B_chunked.stride(3), B_stride4=B_chunked.stride(4),
        )

        # Compute S = einsum('bcths,bcthd->bchds') using Triton
        hidden_chunked = hidden_padded.reshape(batch_size, num_chunks, chunk_size, num_heads, head_dim).contiguous()  # [B, N, Chunk, H, Dim]
        S = torch.empty((batch_size, num_chunks, num_heads, state_size), dtype=torch.float32, device=A_chunked.device)

        grid_S = (batch_size * num_chunks * num_heads,)
        outer_product_bchds_kernel[grid_S](
            B_chunked, hidden_chunked, S,
            B_sz=batch_size, N_chunks=num_chunks, Chunk=chunk_size, H=num_heads, State=state_size, Dim=head_dim,
            S_stride0=S.stride(0), S_stride1=S.stride(1), S_stride2=S.stride(2), S_stride3=S.stride(3),
            B_stride0=B_chunked.stride(0), B_stride1=B_chunked.stride(1), B_stride2=B_chunked.stride(2), B_stride3=B_chunked.stride(3), B_stride4=B_chunked.stride(4),
            hidden_stride0=hidden_chunked.stride(0), hidden_stride1=hidden_chunked.stride(1), hidden_stride2=hidden_chunked.stride(2), hidden_stride3=hidden_chunked.stride(3), hidden_stride4=hidden_chunked.stride(4),
        )

        # Continue with the original logic for L, decays, and final assembly.
        # Note: The original uses torch.cumsum, torch.tril, torch.exp, etc. We keep those in torch for correctness
        # and because they are either small or involve triangular masks. The heavy einsum-like reductions have been moved to Triton.

        # Compute A_cumsum and A_chunk_ends_padded (cumsum along chunk dimension for each (b, h))
        # A_chunked is [B, N, Chunk, H]; flatten (N, Chunk) to 1D
        A_cumsum_flat = torch.cumsum(A_chunked.reshape(batch_size, num_chunks * chunk_size, num_heads), dim=1)  # [B, N*Chunk, H]
        A_cumsum = A_cumsum_flat.reshape(batch_size, num_chunks, chunk_size, num_heads)

        A_ends = A_cumsum[:, :, -1, :]  # [B, N, H]
        A_ends_padded = F.pad(A_ends, (1, 0))  # [B, N+1, H]

        # Build L = exp(cumsum(A_perm)) and apply lower-triangular mask with diagonal=-1
        # A_perm = A_cumsum.permute(0, 2, 1, 3) -> [B, Chunk, N, H]
        A_perm = A_cumsum.permute(0, 2, 1, 3)  # [B, Chunk, N, H]
        L_cumsum_flat = torch.cumsum(A_perm.reshape(batch_size, chunk_size, num_chunks * num_heads), dim=-1)  # [B, Chunk, N*H]
        L_cumsum = L_cumsum_flat.reshape(batch_size, chunk_size, num_chunks, num_heads)
        L = torch.exp(L_cumsum)  # [B, Chunk, N, H]
        # Apply tril mask: lower-triangular per (Chunk, Chunk) for each (b,h)
        mask_int = torch.empty((chunk_size, chunk_size), dtype=torch.int8, device=L.device)
        tril_mask_kernel[(chunk_size * chunk_size,)](mask_int, chunk_size, diagonal=-1)
        # Broadcast mask to [B, N, H, Chunk, Chunk] is not needed; we can mask L with torch.tril for simplicity:
        L_lower = torch.tril(L, diagonal=-1)  # [B, Chunk, N, H]

        # Compute G_scaled =


def run(*args):
    return ModelNew()(*args)
