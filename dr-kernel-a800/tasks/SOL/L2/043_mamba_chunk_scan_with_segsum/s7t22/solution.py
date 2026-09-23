import torch
import triton
import triton.language as tl


# Triton kernel: pad last dimension (constant 0) - 1D copy with mask
@triton.jit
def pad_last_dim_kernel(
    inp_ptr,      # *float32, input flattened
    out_ptr,      # *float32, output flattened
    n_in: tl.constexpr,   # number of valid elements
    out_len: tl.constexpr,  # total number of elements
    pad: tl.constexpr,    # pad size (in number of elements added)
):
    i = tl.program_id(axis=0)
    if i < n_in:
        val = tl.load(inp_ptr + i)
        tl.store(out_ptr + i, val)
    else:
        tl.store(out_ptr + i, 0.0)


# Triton kernel: inclusive cumsum along 1D, 2D launch: axis=0 iterates blocks, axis=1 threads in block
@triton.jit
def cumsum_1d_kernel(
    in_ptr,       # *float32, input flattened
    out_ptr,      # *float32, output flattened
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_block = tl.program_id(axis=0)
    pid_lane = tl.program_id(axis=1)
    start = pid_block * BLOCK + pid_lane
    running = tl.zeros([1], dtype=tl.float32)
    for i in range(0, BLOCK):
        idx = start + i
        if idx < n_elements:
            x = tl.load(in_ptr + idx)
            running += x
            tl.store(out_ptr + idx, running)


# Triton kernel: elementwise exp over 1D array
@triton.jit
def exp_kernel(
    in_ptr,       # *float32
    out_ptr,      # *float32
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_block = tl.program_id(axis=0)
    pid_lane = tl.program_id(axis=1)
    start = pid_block * BLOCK + pid_lane
    for i in range(0, BLOCK):
        idx = start + i
        if idx < n_elements:
            x = tl.load(in_ptr + idx)
            y = tl.exp(x)
            tl.store(out_ptr + idx, y)


# Triton kernel: compute G[b, nc, i, j, h] = sum_s C[b, nc, i, h, s] * B[b, nc, j, h, s]
# Inputs: C_flat: [B, N, Chunk, H, S] flattened by last dims s first, then others
#         B_flat: [B, N, Chunk, H, S] flattened similarly
# Grid dims: (B, N, H, Chunk), with constexpr S_BLOCK for tiling over s
@triton.jit
def dense_reduce_G_kernel(
    C_flat_ptr,   # *float32, flattened with s last
    B_flat_ptr,   # *float32, flattened with s last
    G_out_ptr,    # *float32, output G [B, N, Chunk, Chunk, H]
    Bsz: tl.constexpr,
    N: tl.constexpr,
    Chunk: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    S_BLOCK: tl.constexpr,
):
    b = tl.program_id(axis=0)
    nc = tl.program_id(axis=1)
    i = tl.program_id(axis=2)
    j = tl.program_id(axis=3)
    h = tl.program_id(axis=4)

    # Accumulator for G[i, j, h]
    acc = tl.zeros([1], dtype=tl.float32)

    # Iterate over s in blocks
    for s0 in range(0, S, S_BLOCK):
        s_idx = s0 + tl.arange(0, S_BLOCK)  # vector of s indices
        mask_s = s_idx < S

        # Build offsets for C and B: layout [B, N, Chunk, H, S] flattened
        # Base for (b, nc, i, h) then add s
        base_b = b * (N * Chunk * H * S)
        base_nc = nc * (Chunk * H * S)
        base_i = i * (H * S)
        base_h = h * S

        C_offsets = base_b + nc * (Chunk * H * S) + i * (H * S) + h * S + s_idx
        B_offsets = base_b + nc * (Chunk * H * S) + j * (H * S) + h * S + s_idx

        C_vec = tl.load(C_flat_ptr + C_offsets, mask=mask_s, other=0.0)  # [S_BLOCK]
        B_vec = tl.load(B_flat_ptr + B_offsets, mask=mask_s, other=0.0)  # [S_BLOCK]

        acc += tl.sum(C_vec * B_vec, axis=0)

    # Store G[b, nc, i, j, h]
    # G is laid out as [B, N, Chunk, Chunk, H] flattened with last dim H
    # For fixed (b, nc, i, j), G[i, j, h] at h is contiguous
    G_offset = (((b * N + nc) * Chunk + i) * Chunk + j) * H + h
    tl.store(G_out_ptr + G_offset, acc)


# Triton kernel: compute S[b, nc, h, s] = sum_{t,d} B_decay[b, nc, t, h, s] * hidden[b, nc, t, h, d]
# Inputs:
#   B_decay_flat: [B, N, Chunk, H, S] flattened as in dense_reduce_G (s last)
#   hidden_flat:  [B, N, Chunk, H, D] flattened as [B, N, Chunk, H, D] with D last
# Outputs:
#   S_out_flat:   [B, N, H, D, S] flattened as [B, N, H, D, S] with S last
# Grid dims: (B, N, H, S), loop over t and d in tiles
@triton.jit
def dense_reduce_S_kernel(
    B_decay_ptr,   # *float32, flattened [B, N, Chunk, H, S]
    hidden_ptr,    # *float32, flattened [B, N, Chunk, H, D]
    S_out_ptr,     # *float32, flattened [B, N, H, D, S]
    Bsz: tl.constexpr,
    N: tl.constexpr,
    Chunk: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    T_BLOCK: tl.constexpr,
    D_BLOCK: tl.constexpr,
):
    b = tl.program_id(axis=0)
    nc = tl.program_id(axis=1)
    h = tl.program_id(axis=2)
    s = tl.program_id(axis=3)

    acc = tl.zeros([1], dtype=tl.float32)

    # loop over chunk_size t in blocks
    for t0 in range(0, Chunk, T_BLOCK):
        t_idx = t0 + tl.arange(0, T_BLOCK)
        mask_t = t_idx < Chunk

        # loop over head_dim d in blocks
        for d0 in range(0, D, D_BLOCK):
            d_idx = d0 + tl.arange(0, D_BLOCK)
            mask_d = d_idx < D

            # Load B_decay[b, nc, t, h, s] vector over t for fixed h, s
            base_bd = b * (N * Chunk * H * S) + nc * (Chunk * H * S) + h * S + s
            B_offsets = base_bd + t_idx * (H * S)  # t varies, s fixed
            B_vec = tl.load(B_decay_ptr + B_offsets, mask=mask_t, other=0.0)  # [T_BLOCK]

            # Load hidden[b, nc, t, h, d] matrix over (t, d)
            base_h = b * (N * Chunk * H * D) + nc * (Chunk * H * D) + h * (D * Chunk)
            hidden_offsets = base_h + t_idx[:, None] * D + d_idx[None, :]  # shape [T_BLOCK, D_BLOCK]
            mask = mask_t[:, None] & mask_d[None, :]
            hidden_mat = tl.load(hidden_ptr + hidden_offsets, mask=mask, other=0.0)  # [T_BLOCK, D_BLOCK]

            # sum over d -> [T_BLOCK], then dot with B_vec
            col_sums = tl.sum(hidden_mat, axis=1)  # sum over D_BLOCK
            acc += tl.sum(B_vec * col_sums, axis=0)

    # Store to S_out[b, nc, h, s]
    # S_out is [B, N, H, D, S] flattened with S last
    base_so = b * (N * H * D * S) + nc * (H * D * S) + h * (D * S) + s
    tl.store(S_out_ptr + base_so, acc)


# ModelNew entry point
class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        initial_states: torch.Tensor,
    ):
        # Cast to float32
        hidden_states_f = hidden_states.to(torch.float32)
        A_f = A.to(torch.float32)
        B_f = B.to(torch.float32)
        C_f = C.to(torch.float32)
        D_f = D.to(torch.float32)
        initial_states_f = initial_states.to(torch.float32)

        # Dimensions
        batch_size, seq_len, num_heads, head_dim = hidden_states_f.shape
        state_size = 256
        n_groups = 1
        chunk_size = 256

        # Padding to make seq_len multiple of chunk_size
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        seq_len_padded = seq_len + pad_size

        # 1) Pad hidden states along last dimension (constant 0)
        hidden_flat_in = hidden_states_f.reshape(-1)                  # [B*S*num_heads*head_dim]
        hidden_flat_out = torch.empty(seq_len_padded * batch_size * num_heads * head_dim,
                                      dtype=torch.float32, device=hidden_states_f.device)
        out_len = hidden_flat_out.numel()
        n_in = hidden_flat_in.numel()
        pad_elems = pad_size * batch_size * num_heads * head_dim
        grid_pad = (out_len,)
        pad_last_dim_kernel[grid_pad](hidden_flat_in, hidden_flat_out, n_in, out_len, pad_elems)
        hidden_states_padded = hidden_flat_out.reshape(batch_size, seq_len_padded, num_heads, head_dim)

        # 2) Compute A_transposed = A.transpose(1, 2) -> [B, S, H], then inclusive cumsum along S for each (B, H)
        A_t = A_f.transpose(1, 2)  # [B, S, H]
        A_cumsum_flat = torch.empty(A_t.numel(), dtype=torch.float32, device=A_t.device)
        rows = batch_size * num_heads
        BLOCK = 1024  # constexpr block size for cumsum
        for row_id in range(rows):
            b = row_id // num_heads
            h = row_id % num_heads
            row_start = row_id * A_t.shape[1]
            grid = (triton.cdiv(A_t.shape[1], BLOCK), BLOCK)
            cumsum_1d_kernel[grid](A_t.reshape(-1) + row_start, A_cumsum_flat + row_id * A_t.shape[1], A_t.shape[1], BLOCK)
        A_cumsum = A_cumsum_flat.reshape(batch_size, seq_len, num_heads)  # inclusive cumsum along S

        # 3) Elementwise exp of A_cumsum (per (B,S,H))
        N_exp = A_cumsum.numel()
        exp_out = torch.empty(N_exp, dtype=torch.float32, device=A_cumsum.device)
        grid_exp = (triton.cdiv(N_exp, BLOCK), BLOCK)
        exp_kernel[grid_exp](A_cumsum.reshape(-1), exp_out, N_exp, BLOCK)
        exp_A = exp_out.reshape(batch_size, seq_len, num_heads)

        # 4) Expand B and C to match num_heads: [B, S, H, S] with H=num_heads (here 1 group)
        B_expanded = B_f.expand(batch_size, seq_len, num_heads, state_size)  # [B, S, H, S]
        C_expanded = C_f.expand(batch_size, seq_len, num_heads, state_size)  # [B, S, H, S]

        # 5) Compute D residual after padding
        D_flat = D_f.reshape(-1)  # [B*S*num_heads*head_dim]
        D_out = torch.empty(seq_len_padded * batch_size * num_heads * head_dim,
                            dtype=torch.float32, device=D_f.device)
        out_len_D = D_out.numel()
        n_in_D = D_flat.numel()
        pad_D = pad_size * batch_size * num_heads * head_dim
        grid_D = (out_len_D,)
        pad_last_dim_kernel[grid_D](D_flat, D_out, n_in_D, out_len_D, pad_D)
        D_padded = D_out.reshape(batch_size, seq_len_padded, num_heads, head_dim)

        D_residual = D_padded * hidden_states_padded  # [B, S', H, D]

        # 6) Reshape into chunks: [B, N, Chunk, H, D], N = seq_len_padded // chunk_size
        hidden_states_chunked = hidden_states_padded.reshape(batch_size, -1, chunk_size, num_heads, head_dim)  # [B, N, Chunk, H, D]
        # B and C expanded to chunked form
        # Note: B_expanded, C_expanded have shape [B, S, H, S]; we need [B, N, Chunk, H, S].
        # We can construct by using the padded sequence indices within each chunk. Since we've padded hidden, we can create B_chunked and C_chunked as reshaped views across N and Chunk dims.
        # In practice, we can view B_expanded/C_expanded as [B, N, Chunk, H, S] by repeating along N/Chunk via expand+reshape.
        # Here, we explicitly construct by indexing padded sequence positions within each chunk. For simplicity, we expand and reshape:
        # We need to map padded sequence positions to chunk indices. Since seq_len_padded is divisible by chunk_size, we can safely view:
        # Build B_chunked


def run(*args):
    return ModelNew()(*args)
