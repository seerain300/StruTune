import torch
import triton
import triton.language as tl


# Triton: pad last dimension (constant 0) on flattened 1D tensor
@triton.jit
def pad_last_dim_kernel(
    inp_ptr,        # *float32
    out_ptr,        # *float32
    n_in,           # int32: number of valid elements
    out_len,        # int32: total length after padding
    pad,            # int32: pad added at end
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_in
    vals = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, vals, mask=mask)


# Triton: inclusive cumsum along 1D (segmented scan), per-row scan up to BLOCK, loop over elements
@triton.jit
def cumsum_1d_kernel(
    in_ptr,      # *float32
    out_ptr,     # *float32
    n_elements: tl.constexpr,  # total number of elements in this run (must be constexpr for kernel)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    running = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, n_elements):
        running += vals[i]
        tl.store(out_ptr + offsets, running, mask=mask)


# Triton: elementwise exp over 1D array
@triton.jit
def exp_kernel(
    in_ptr,      # *float32
    out_ptr,     # *float32
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    vals = tl.exp(vals)
    tl.store(out_ptr + offsets, vals, mask=mask)


# Triton: generate lower-triangular mask (keep if i>=j+diag), output as int8 flattened of size I*J
@triton.jit
def tril_mask_kernel(
    out_ptr,        # *int8
    I: tl.constexpr,
    J: tl.constexpr,
    diag: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    total = I * J
    mask = offsets < total
    i = offsets // J
    j = offsets % J
    keep = i >= (j + diag)
    tl.store(out_ptr + offsets, keep.to(tl.int8), mask=mask)


# Triton: compute G[b, nc, i, j, h] = sum_s C[b, nc, i, h, s] * B[b, nc, j, h, s], specialized for H=1
# Grid: (B, N, Chunk, Chunk, 1)
@triton.jit
def dense_reduce_G_kernel(
    C_ptr,          # *float32, [B, N, Chunk, 1, S]
    B_ptr,          # *float32, [B, N, Chunk, 1, S]
    G_ptr,          # *float32, [B, N, Chunk, Chunk, 1]  # store G[b, nc, i, j, 0]
    Bsz: tl.constexpr,
    N: tl.constexpr,
    Chunk: tl.constexpr,
    H: tl.constexpr,  # must be 1
    S: tl.constexpr,
    S_TILE: tl.constexpr,  # tile size for reduction over S
):
    b = tl.program_id(axis=0)
    nc = tl.program_id(axis=1)
    i = tl.program_id(axis=2)
    j = tl.program_id(axis=3)
    h = tl.program_id(axis=4)  # 0

    acc = tl.zeros([1], dtype=tl.float32)
    for s0 in range(0, S, S_TILE):
        s_off = s0 + tl.arange(0, S_TILE)
        mask_s = s_off < S
        # C offsets: (((b*N + nc)*Chunk + i)*H + h)*S + s_off
        C_offsets = (((b * N + nc) * Chunk + i) * H + h) * S + s_off
        C_vec = tl.load(C_ptr + C_offsets, mask=mask_s, other=0.0)  # shape (S_TILE,)
        # B offsets: (((b*N + nc)*Chunk + j)*H + h)*S + s_off
        B_offsets = (((b * N + nc) * Chunk + j) * H + h) * S + s_off
        B_vec = tl.load(B_ptr + B_offsets, mask=mask_s, other=0.0)
        acc += tl.sum(C_vec * B_vec, axis=0)

    # store G[b, nc, i, j, 0]
    g_idx = ((b * N + nc) * Chunk + i) * Chunk + j  # with H=1, last dim is 1
    tl.store(G_ptr + g_idx, acc)


# Triton: compute S[b, nc, h, s] = sum_{t,d} B_decay[b, nc, t, h, s] * hidden[b, nc, t, h, d], specialized for H=1
# Grid: (B, N, 1, S)
@triton.jit
def dense_reduce_S_kernel(
    B_decay_ptr,  # *float32, [B, N, Chunk, 1, S]
    hidden_ptr,   # *float32, [B, N, Chunk, 1, D]
    S_ptr,        # *float32, [B, N, 1, S]  # we store S[b, nc, 0, s]
    Bsz: tl.constexpr,
    N: tl.constexpr,
    Chunk: tl.constexpr,
    H: tl.constexpr,  # must be 1
    S: tl.constexpr,
    D: tl.constexpr,
    T_TILE: tl.constexpr,
    D_TILE: tl.constexpr,
):
    b = tl.program_id(axis=0)
    nc = tl.program_id(axis=1)
    h = tl.program_id(axis=2)  # 0
    s = tl.program_id(axis=3)

    acc = tl.zeros([1], dtype=tl.float32)

    # reduce over chunk_size in tiles
    for t0 in range(0, Chunk, T_TILE):
        t_off = t0 + tl.arange(0, T_TILE)
        mask_t = t_off < Chunk

        # B_decay offsets: (((b*N + nc)*Chunk + t)*H + h)*S + s, with H=1
        B_offsets = (((b * N + nc) * Chunk + t_off) * H + h) * S + s
        B_vec = tl.load(B_decay_ptr + B_offsets, mask=mask_t, other=0.0)  # shape (T_TILE,)

        # reduce over head_dim in tiles
        for d0 in range(0, D, D_TILE):
            d_off = d0 + tl.arange(0, D_TILE)
            mask_d = d_off < D

            # hidden offsets: (((b*N + nc)*Chunk + t_off[:, None])*H + h)*D + d_off[None, :]
            hidden_offsets = (((b * N + nc) * Chunk + t_off[:, None]) * H + h) * D + d_off[None, :]
            mask = mask_t[:, None] & mask_d[None, :]
            hidden_mat = tl.load(hidden_ptr + hidden_offsets, mask=mask, other=0.0)

            col_sums = tl.sum(hidden_mat, axis=1)  # sum over D_TILE for each t -> shape (T_TILE,)
            acc += tl.sum(B_vec * col_sums, axis=0)

    s_idx = ((b * N + nc) * H + h) * S + s  # with H=1, this is (b*N + nc)*S + s
    tl.store(S_ptr + s_idx, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, initial_states: torch.Tensor):
        # Fixed constants from the original code in this environment
        num_heads = 16  # n_groups = 1 and original uses 16 heads
        head_dim = 64   # original uses head_dim=64 to get num_heads*head_dim=1024
        state_size = 256
        chunk_size = 256

        # Compute padding size to make seq_len multiple of chunk_size
        batch_size, seq_len, _, _ = hidden_states.shape
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        seq_len_pad = (seq_len + pad_size)

        # Convert to float32 for numerical stability
        hidden_states_f = hidden_states.to(torch.float32)
        A_f = A.to(torch.float32)
        B_f = B.to(torch.float32)
        C_f = C.to(torch.float32)
        D_f = D.to(torch.float32)
        initial_states_f = initial_states.to(torch.float32)

        # Create padded hidden states on last dimension using Triton
        hidden_flat = hidden_states_f.reshape(batch_size, -1).contiguous()  # [B, S]
        hidden_pad_flat = torch.empty(batch_size * seq_len_pad, dtype=torch.float32, device=hidden_states.device)
        BLOCK = 1024
        grid_pad = (triton.cdiv(hidden_flat.numel(), BLOCK),)
        pad_last_dim_kernel[grid_pad](hidden_flat, hidden_pad_flat, hidden_flat.numel(), hidden_pad_flat.numel(), pad_size, BLOCK=BLOCK, num_warps=1)
        hidden_padded = hidden_pad_flat.view(batch_size, seq_len_pad)  # [B, S_pad]

        # A_transposed and A_cumsum: A is [B, S, 1] -> transpose to [B, 1, S]
        A_transposed = A_f.transpose(1, 2).contiguous()  # [B, 1, S]
        A_flat = A_transposed.reshape(batch_size, -1).contiguous()  # [B, S]
        A_cumsum_flat = torch.empty_like(A_flat)
        grid_cumsum = (triton.cdiv(A_flat.numel(), BLOCK),)
        cumsum_1d_kernel[grid_cumsum](A_flat, A_cumsum_flat, A_flat.numel(), BLOCK=BLOCK, num_warps=1)
        A_cumsum = A_cumsum_flat.reshape(batch_size, 1, seq_len_pad)  # [B, 1, S_pad]

        # Prepare B_expanded and C_expanded for H=1
        B_expanded = B_f.expand(batch_size, seq_len_pad, 1, state_size)  # [B, S_pad,


def run(*args):
    return ModelNew()(*args)
