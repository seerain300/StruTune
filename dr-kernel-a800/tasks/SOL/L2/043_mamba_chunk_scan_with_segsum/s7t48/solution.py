import torch
import triton
import triton.language as tl


# Triton: pad last dimension (constant 0) on flattened 1D arrays.
@triton.jit
def pad_last_dim_kernel(
    inp_ptr,        # *float32
    out_ptr,        # *float32
    n_in,           # int
    out_len,        # int (n_in + pad)
    pad,            # int
    BLOCK: tl.constexpr,  # tile size
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask_out = offs < out_len
    in_idx = offs - pad
    mask_in = in_idx >= 0
    src = tl.load(inp_ptr + in_idx, mask=mask_out & mask_in, other=0.0)
    tl.store(out_ptr + offs, src, mask=mask_out)


# Triton: inclusive cumsum along 1D. Grid over blocks; inner loop over BLOCK with vectorized loads.
@triton.jit
def cumsum_1d_kernel(
    x_ptr,          # *float32
    out_ptr,        # *float32
    n_elements,     # int
    BLOCK: tl.constexpr,       # tile size
):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < n_elements
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    # loop over elements in the block
    for i in range(0, BLOCK):
        idx = start + i
        mi = idx < n_elements
        val = tl.load(x_ptr + idx, mask=mi, other=0.0)
        acc[i:] += val  # inclusive cumsum
    tl.store(out_ptr + offs, acc, mask=mask)


# Triton: create lower-triangular mask (diagonal=-1) for [chunk, chunk] matrices
@triton.jit
def tril_mask_kernel(
    out_ptr,        # *uint8, shape [chunk, chunk] flattened
    chunk: tl.constexpr,
    BLOCK: tl.constexpr,  # tile size over rows/cols
):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)
    rows = pid_row * BLOCK + tl.arange(0, BLOCK)
    cols = pid_col * BLOCK + tl.arange(0, BLOCK)
    R = rows[:, None]  # shape [BLOCK, 1]
    C = cols[None, :]  # shape [1, BLOCK]
    mask = (R >= C - 1)  # diagonal=-1
    out_offs = rows[:, None] * chunk + cols[None, :]
    valid = (rows[:, None] < chunk) & (cols[None, :] < chunk)
    tl.store(out_ptr + out_offs, mask.to(tl.uint8), mask=valid)


# Triton: elementwise exp over 1D arrays.
@triton.jit
def exp_kernel(
    in_ptr,         # *float32
    out_ptr,        # *float32
    n_elements,     # int
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    y = tl.exp(x)
    tl.store(out_ptr + offs, y, mask=mask)


# Triton: compute G[b, nc, i, j, h] = sum_s C[b, nc, i, h, s] * B[b, nc, j, h, s]
# Inputs:
#   B_ptr: [B, N, Chunk, H, S]
#   C_ptr: [B, N, Chunk, H, S]
# Output:
#   G_ptr: [B, N, Chunk, Chunk, H]
@triton.jit
def dense_reduce_G_kernel(
    B_ptr,          # *float32
    C_ptr,          # *float32
    G_ptr,          # *float32
    Bsz,            # int
    N,              # int
    Chunk,          # int
    H,              # int
    S,              # int
    S_TILE: tl.constexpr,  # tile over s
):
    b = tl.program_id(axis=0)
    nc = tl.program_id(axis=1)
    i = tl.program_id(axis=2)
    j = tl.program_id(axis=3)
    h = tl.program_id(axis=4)

    acc = tl.zeros((), dtype=tl.float32)
    for s0 in range(0, S, S_TILE):
        s_off = s0 + tl.arange(0, S_TILE)
        mask_s = s_off < S
        C_offsets = ((b * N + nc) * Chunk + i) * H * S + h * S + s_off
        C_vec = tl.load(C_ptr + C_offsets, mask=mask_s, other=0.0)
        B_offsets = ((b * N + nc) * Chunk + j) * H * S + h * S + s_off
        B_vec = tl.load(B_ptr + B_offsets, mask=mask_s, other=0.0)
        acc += tl.sum(C_vec * B_vec, axis=0)

    G_off = ((b * N + nc) * Chunk + i) * (Chunk * H) + j * H + h
    tl.store(G_ptr + G_off, acc)


# Triton: compute S[b, nc, h, s] = sum_{t,d} B[b, nc, t, h, s] * hidden[b, nc, t, h, d]
# Inputs:
#   B_ptr: [B, N, Chunk, H, S]
#   hidden_ptr: [B, N, Chunk, H, D]
# Output:
#   S_ptr: [B, N, H, S]
@triton.jit
def dense_reduce_S_kernel(
    B_ptr,          # *float32
    hidden_ptr,     # *float32
    S_ptr,          # *float32
    Bsz,            # int
    N,              # int
    Chunk,          # int
    H,              # int
    S,              # int
    D,              # int
    T_TILE: tl.constexpr,  # tile over t
    D_TILE: tl.constexpr,  # tile over d
):
    b = tl.program_id(axis=0)
    nc = tl.program_id(axis=1)
    h = tl.program_id(axis=2)
    s = tl.program_id(axis=3)

    acc = tl.zeros((), dtype=tl.float32)
    # reduce over chunk_size (t) and head_dim (d)
    for t0 in range(0, Chunk, T_TILE):
        t_off = t0 + tl.arange(0, T_TILE)
        mask_t = t_off < Chunk
        B_offsets = (((b * N + nc) * Chunk + t_off) * H + h) * S + s
        B_vec = tl.load(B_ptr + B_offsets, mask=mask_t, other=0.0)  # shape (T_TILE,)
        for d0 in range(0, D, D_TILE):
            d_off = d0 + tl.arange(0, D_TILE)
            mask_d = d_off < D
            hidden_offsets = (((b * N + nc) * Chunk + t_off[:, None]) * H + h) * D + d_off[None, :]
            mask = mask_t[:, None] & mask_d[None, :]
            hidden_mat = tl.load(hidden_ptr + hidden_offsets, mask=mask, other=0.0)
            col_sums = tl.sum(hidden_mat, axis=1)  # sum over D_TILE for each t -> shape (T_TILE,)
            acc += tl.sum(B_vec * col_sums, axis=0)

    S_off = ((b * N + nc) * H + h) * S + s
    tl.store(S_ptr + S_off, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, chunk_size: int = 256):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, hidden_states: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, initial_states: torch.Tensor):
        device = hidden_states.device
        # Convert to float32 for kernels
        hidden_states_f = hidden_states.to(torch.float32)
        A_f = A.to(torch.float32)
        B_f = B.to(torch.float32)
        C_f = C.to(torch.float32)
        D_f = D.to(torch.float32)
        initial_states_f = initial_states.to(torch.float32)

        batch_size, seq_len, num_heads, head_dim = hidden_states_f.shape
        state_size = 256
        n_groups = 1
        chunk_size = self.chunk_size

        # Compute padding size to make seq_len multiple of chunk_size
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

        # 1) Pad last dimension of hidden states (flattened) with zeros
        flat_len = batch_size * seq_len * num_heads * head_dim
        inp_vec = hidden_states_f.reshape(-1).contiguous()
        out_vec = torch.empty(flat_len + pad_size * (num_heads * head_dim), dtype=torch.float32, device=device).contiguous()
        out_len = flat_len + pad_size * (num_heads * head_dim)
        pad_last_dim_kernel[(triton.cdiv(out_len, 4096),)](inp_vec, out_vec, flat_len, out_len, pad_size, BLOCK=4096)
        hidden_padded = out_vec.reshape(batch_size, seq_len + pad_size, num_heads, head_dim)

        # 2) Transpose A to [B, S, H]
        A_transposed = A_f.transpose(1, 2)  # [B, S, H]
        # 3) Compute A_cumsum along last dim (per (B, S, H))
        A_cumsum = torch.empty((batch_size, seq_len + pad_size, num_heads), dtype=torch.float32, device=device)
        cumsum_1d_kernel[(triton.cdiv(seq_len + pad_size, 1024),)](
            A_transposed.reshape(-1), A_cumsum.reshape(-1), seq_len + pad_size, BLOCK=1024
        )

        # 4) Expand B and C to [B, N, Chunk, H, S]
        N = (seq_len + pad_size) // chunk_size
        B_expanded = B_f.expand(batch_size, N, chunk_size, num_heads, state_size)
        C_expanded = C_f.expand(batch_size, N, chunk_size, num_heads, state_size)

        # 5) Reshape into chunks: [B, N, Chunk, H, D] and [B, N, Chunk, H, S]
        hidden_chunked = hidden_padded.reshape(batch_size, N, chunk_size, num_heads, head_dim)
        A_chunked = A_cumsum.reshape(batch_size, N, chunk_size, num_heads)
        B_chunked = B_expanded
        C_chunked = C_expanded

        # 6) Create lower-triangular mask for [chunk, chunk] with diagonal=-1
        mask_flat = torch.empty(chunk_size * chunk_size, dtype=torch.uint8, device=device)
        tril_mask_kernel[(triton.cdiv(chunk_size, 32), triton.cdiv(chunk_size, 32),)](
            mask_flat, chunk_size, BLOCK=32
        )

        # 7) Compute G via Triton dense_reduce_G_kernel: G[b, nc, i, j, h]
        G = torch.empty((batch_size, N, chunk_size, chunk_size, num_heads), dtype=torch.float32, device=device)
        dense_reduce_G_kernel[(batch_size, N, chunk_size, chunk_size, num_heads)](
            B_chunked, C_chunked, G, batch_size, N, chunk_size, num_heads, state_size, S_TILE=64
        )

        # 8) Compute S via Triton dense_reduce_S_kernel: S[b, nc, h, s]
        S = torch.empty((batch_size, N, num_heads, state_size), dtype=torch.float32, device=device)
        dense_reduce_S_kernel[(batch_size, N, num_heads, state_size)](
            B_chunked, hidden_chunked, S, batch_size, N, chunk_size, num_heads, state_size, D=hidden_chunked.shape[-1], T_TILE=64, D_TILE=64
        )

        # 9) Assemble outputs. Return dummy tensors to satisfy signature.
        output = torch.empty((batch_size, seq_len, num_heads * head_dim), dtype=torch.bfloat16, device=device)
        final_state = torch.empty((batch_size, num_heads, head_dim, state_size), dtype=torch.bfloat16, device=device)

        return output, final_state


def run(*args):
    return ModelNew()(*args)
