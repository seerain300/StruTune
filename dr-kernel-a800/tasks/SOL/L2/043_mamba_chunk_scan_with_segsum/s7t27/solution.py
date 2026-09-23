import torch
import triton
import triton.language as tl


# Triton: pad last dimension (constant 0) on a flattened 1D tensor
@triton.jit
def pad_last_dim_kernel(
    inp_ptr,         # *float32, input flattened
    out_ptr,         # *float32, output flattened
    n_in,            # int, number of valid elements in input
    out_len,         # int, total number of elements in output
    pad,             # int, pad size added at the end
):
    i = tl.program_id(axis=0)
    if i < n_in:
        val = tl.load(inp_ptr + i)
        tl.store(out_ptr + i, val)
    else:
        tl.store(out_ptr + i, 0.0)


# Triton: 1D inclusive cumsum (used to form A_cumsum per (b, s, h))
@triton.jit
def cumsum_1d_kernel(
    in_ptr,          # *float32
    out_ptr,         # *float32
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    running = tl.zeros((), dtype=tl.float32)
    out_vals = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(BLOCK):
        idx = start + k
        if idx < n_elements:
            running += vals[k]
            out_vals[k] = running
    tl.store(out_ptr + offsets, out_vals, mask=mask)


# Triton: elementwise exp over 1D array
@triton.jit
def exp_elemwise_kernel(
    in_ptr,          # *float32
    out_ptr,         # *float32
    n_elements: tl.constexpr,
):
    i = tl.program_id(axis=0)
    val = tl.load(in_ptr + i)
    val = tl.exp(val)
    tl.store(out_ptr + i, val)


# Triton: compute G = sum_s C[b, nc, i, h, s] * B[b, nc, j, h, s]
# Inputs:
#   C_ptr: [B, N, Chunk, H, S]
#   B_ptr: [B, N, Chunk, H, S]
# Output:
#   G_ptr: [B, N, Chunk, Chunk, H] (H=1)
@triton.jit
def dense_reduce_G_kernel(
    C_ptr,           # *float32
    B_ptr,           # *float32
    G_ptr,           # *float32
    Bsz: tl.constexpr,
    N: tl.constexpr,
    Chunk: tl.constexpr,
    H: tl.constexpr,  # expected H=1
    S: tl.constexpr,
    S_TILE: tl.constexpr,
):
    b = tl.program_id(axis=0)
    nc = tl.program_id(axis=1)
    i = tl.program_id(axis=2)  # in [0, Chunk)
    j = tl.program_id(axis=3)  # in [0, Chunk)
    h = tl.program_id(axis=4)  # in [0, H)

    acc = tl.zeros([1], dtype=tl.float32)

    # Reduce over state_size s in tiles
    for s0 in range(0, S, S_TILE):
        s_off = s0 + tl.arange(0, S_TILE)
        mask_s = s_off < S

        C_offsets = (((b * N + nc) * Chunk + i) * H + h) * S + s_off
        C_vec = tl.load(C_ptr + C_offsets, mask=mask_s, other=0.0)  # (S_TILE,)

        B_offsets = (((b * N + nc) * Chunk + j) * H + h) * S + s_off
        B_vec = tl.load(B_ptr + B_offsets, mask=mask_s, other=0.0)  # (S_TILE,)

        acc += tl.sum(C_vec * B_vec, axis=0)

    out_idx = ((b * N + nc) * Chunk + i) * Chunk * H + j * H + h
    tl.store(G_ptr + out_idx, acc)


# Triton: compute S = sum_{t,d} B_decay[b, nc, t, h, s] * hidden[b, nc, t, h, d], for h=0 (since H=16, but here H=1 input)
# Inputs:
#   B_decay_ptr: [B, N, Chunk, 1, S]
#   hidden_ptr:  [B, N, Chunk, 1, D]
# Output:
#   S_ptr:       [B, N, 1, S]  (we store S[b, nc, 0, s])
@triton.jit
def dense_reduce_S_kernel(
    B_decay_ptr,     # *float32
    hidden_ptr,      # *float32
    S_ptr,           # *float32
    Bsz: tl.constexpr,
    N: tl.constexpr,
    Chunk: tl.constexpr,
    H: tl.constexpr,  # must be 1 in this usage
    S: tl.constexpr,
    D: tl.constexpr,
    T_TILE: tl.constexpr,
    D_TILE: tl.constexpr,
):
    b = tl.program_id(axis=0)
    nc = tl.program_id(axis=1)
    h = tl.program_id(axis=2)  # 0
    s = tl.program_id(axis=3)  # s in [0, S)

    acc = tl.zeros([1], dtype=tl.float32)

    # reduce over chunk_size in tiles
    for t0 in range(0, Chunk, T_TILE):
        t_off = t0 + tl.arange(0, T_TILE)
        mask_t = t_off < Chunk

        # B_decay offsets: (((b*N + nc)*Chunk + t)*H + h)*S + s, with H=1
        B_offsets = (((b * N + nc) * Chunk + t_off) * H + h) * S + s
        B_vec = tl.load(B_decay_ptr + B_offsets, mask=mask_t, other=0.0)  # shape (T_TILE,)

        # reduce over head_dim in tiles (here D corresponds to head_dim)
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
        # Constants from original
        self.chunk_size = 256
        self.n_groups = 1  # not used, but kept for consistency
        self.state_size = 256
        self.head_dim = 64
        self.num_heads = 16  # original sets num_heads=16

    def forward(self, hidden_states: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                C: torch.Tensor, D: torch.Tensor, initial_states: torch.Tensor):
        """
        hidden_states: [B, S, H, D]
        A: [B, S, H]
        B: [B, S, H, S]
        C: [B, S, H, S]
        D: [B, S, H]
        initial_states: [B, H, D, S]
        Returns (output: [B, S, H*D] in bfloat16, final_state: None)
        """

        # Convert to float32 for compute
        hidden_f32 = hidden_states.float()
        A_f32 = A.float()
        B_f32 = B.float()
        C_f32 = C.float()
        D_f32 = D.float()
        init_f32 = initial_states.float()

        # Shapes
        Bsz, S, H, D = hidden_f32.shape
        # For this benchmark, original code uses num_heads=16; we will return H*D=1024 per step.
        # Compute padding to make S multiple of chunk_size
        pad = (self.chunk_size - S % self.chunk_size) % self.chunk_size
        S_padded = S + pad
        N = (S_padded + self.chunk_size - 1) // self.chunk_size

        # 1) Pad hidden states on last dimension to S_padded and reshape into chunks
        hidden_pad_len = Bsz * S_padded * H * D
        hidden_pad = torch.empty(hidden_pad_len, dtype=torch.float32, device=hidden_f32.device)
        inp_hidden_flat = hidden_f32.reshape(-1)
        pad_last_dim_kernel[(hidden_pad_len,)](inp_hidden_flat, hidden_pad, inp_hidden_flat.numel(), hidden_pad_len, pad)

        # Reshape padded hidden to [B, S_padded, H, D], then to [B, N, Chunk, H, D]
        hidden_padded_4d = hidden_pad.reshape(Bsz, S_padded, H, D)
        hidden_chunked = hidden_padded_4d.view(Bsz, N, self.chunk_size, H, D)

        # 2) Compute A_cumsum per (b, s, h) using Triton 1D cumsum over flatten A and elementwise exp
        A_flat = A_f32.reshape(Bsz, S, H).reshape(-1)  # (B*S*H,)
        A_cumsum = torch.empty_like(A_flat)
        cumsum_1d_kernel[(triton.cdiv(A_flat.numel(), 1024),)](A_flat, A_cumsum, A_flat.numel(), 1024)
        A_cumsum = A_cumsum.reshape(Bsz, S, H)
        # Elementwise exp of A_cumsum using Triton
        A_cumsum_exp = torch.empty_like(A_cumsum)
        exp_elemwise_kernel[(A_cumsum.numel(),)](A_cumsum.reshape(-1), A_cumsum_exp.reshape(-1), A_cumsum_exp.numel())

        # 3) Expand B and C to [B, S, H, S]; we'll reshape them to chunked dims in next steps
        B_expanded = B_f32  # [B, S, H, S]
        C_expanded = C_f32  # [B, S, H, S]

        # 4) Prepare chunked tensors
        Chunk = self.chunk_size
        # Reshape B_expanded and C_expanded to [B, N, Chunk, H, S]
        # Since original code sets H=1 implicitly, we can expand to [B, N, Chunk, 1, S] by slicing h=0 and expanding to H=16 later.
        # However, to keep num_heads


def run(*args):
    return ModelNew()(*args)
