import torch
import triton
import triton.language as tl


@triton.jit
def build_lower_tri_causal_kernel(
    A_ptr,          # *float32, [B, C, L, H]
    L_ptr,          # *float32, [B, C, 128, 128, H]
    B_size: tl.constexpr,  # batch_size
    C_size: tl.constexpr,  # num_chunks
    H_size: tl.constexpr,  # num_heads
    L_len,            # actual chunk_size (runtime int)
):
    # Grid: (B, C, H, i, j)
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    i = tl.program_id(3)  # row in 128
    j = tl.program_id(4)  # col in 128

    # Compute cumulative sum over k in 0..L_len-1 of A[b, c, k, h]
    total = 0.0
    k = 0
    while k < L_len:
        addr = b * (C_size * L_len * H_size) + c * (L_len * H_size) + k * H_size + h
        a_val = tl.load(A_ptr + addr)
        total += a_val
        k += 1

    # Set L[i, j] = exp(total) if j <= i else 0
    if j <= i:
        l_val = tl.exp(total)
    else:
        l_val = 0.0

    # Store to L[b, c, i, j, h]
    L_addr = b * (C_size * 128 * 128 * H_size) + c * (128 * 128 * H_size) + i * (128 * H_size) + j * H_size + h
    tl.store(L_ptr + L_addr, l_val)


@triton.jit
def expand_groups_repeat_interleave(
    in_ptr,          # *float32, input tensor with groups dim
    out_ptr,         # *float32, output tensor with expanded heads
    B_size: tl.constexpr,  # batch_size
    C_size: tl.constexpr,  # num_chunks
    L_len: tl.constexpr,   # chunk_size
    H_size: tl.constexpr,  # num_heads
    S_size: tl.constexpr,  # state_size
    N_GROUPS: tl.constexpr,
    GROUP_EXPAND: tl.constexpr,
    group_dim: tl.constexpr,  # which group dimension (3 for B/C)
):
    # Grid: (B, C, i, h, s)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # i along L
    h = tl.program_id(3)  # head index
    s = tl.program_id(4)  # state index

    # Compute original group index for this expanded head h
    # Mapping: h -> g; we expand groups into heads by repeating each group GROUP_EXPAND times.
    # g = h // GROUP_EXPAND
    g = h // GROUP_EXPAND

    # Address in input tensor: [B, C, L, groups, S]
    in_addr = b * (C_size * L_len * N_GROUPS * S_size) + c * (L_len * N_GROUPS * S_size) + i * (N_GROUPS * S_size) + g * S_size + s
    val = tl.load(in_ptr + in_addr)

    # Address in output tensor: [B, C, L, H, S]
    out_h = h
    out_addr = b * (C_size * L_len * H_size * S_size) + c * (L_len * H_size * S_size) + i * (H_size * S_size) + out_h * S_size + s
    tl.store(out_ptr + out_addr, val)


@triton.jit
def g_outer_kernel(
    C_exp_ptr,      # *float32, [B, C, L, H, S]
    B_exp_ptr,      # *float32, [B, C, L, H, S]
    G_ptr,          # *float32, [B, C, L, L, H]
    B_size: tl.constexpr,
    C_size: tl.constexpr,
    L_len: tl.constexpr,
    H_size: tl.constexpr,
    S_size: tl.constexpr,
):
    # Grid: (B, C, i, j, h)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    g_val = 0.0
    s = 0
    while s < S_size:
        C_addr = b * (C_size * L_len * H_size * S_size) + c * (L_len * H_size * S_size) + i * (H_size * S_size) + h * S_size + s
        B_addr = b * (C_size * L_len * H_size * S_size) + c * (L_len * H_size * S_size) + j * (H_size * S_size) + h * S_size + s
        C_val = tl.load(C_exp_ptr + C_addr)
        B_val = tl.load(B_exp_ptr + B_addr)
        g_val += C_val * B_val
        s += 1

    G_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
    tl.store(G_ptr + G_addr, g_val)


@triton.jit
def multiply_mask_kernel(
    G_ptr,          # *float32, [B, C, L, L, H]
    L_ptr,          # *float32, [B, C, 128, 128, H]
    M_ptr,          # *float32, [B, C, L, L, H]
    B_size: tl.constexpr,
    C_size: tl.constexpr,
    L_len: tl.constexpr,
    H_size: tl.constexpr,
):
    # Grid: (B, C, i, j, h)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    G_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
    G_val = tl.load(G_ptr + G_addr)

    # L index uses 128 regardless of L_len; j must be < L_len and i < L_len
    L_addr = b * (C_size * 128 * 128 * H_size) + c * (128 * 128 * H_size) + i * (128 * H_size) + j * H_size + h
    L_val = tl.load(L_ptr + L_addr)

    M_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
    tl.store(M_ptr + M_addr, G_val * L_val)


@triton.jit
def y_diag_reduce_kernel(
    M_ptr,          # *float32, [B, C, L, L, H]
    hidden_ptr,     # *float32, [B, C, L, H, D]
    out_ptr,        # *float32, [B, C, L, H, D]
    B_size: tl.constexpr,
    C_size: tl.constexpr,
    L_len: tl.constexpr,
    H_size: tl.constexpr,
    D_size: tl.constexpr,
):
    # Grid: (B, C, i, h)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    # Accumulator over D dimension
    acc = tl.zeros((D_size,), dtype=tl.float32)

    j = 0
    while j < L_len:
        M_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
        M_val = tl.load(M_ptr + M_addr)

        hidden_addr = b * (C_size * L_len * H_size * D_size) + c * (L_len * H_size * D_size) + j * (H_size * D_size) + h * D_size + tl.arange(0, D_size)
        hidden_vec = tl.load(hidden_ptr + hidden_addr)  # vector over D

        acc += M_val * hidden_vec
        j += 1

    # Store result Y[b, c, i, h, :]
    out_addr = b * (C_size * L_len * H_size * D_size) + c * (L_len * H_size * D_size) + i * (H_size * D_size) + h * D_size + tl.arange(0, D_size)
    tl.store(out_ptr + out_addr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, num_heads=32, n_groups=8, group_expand=4):
        super().__init__()
        self.num_heads = num_heads
        self.n_groups = n_groups
        self.group_expand = group_expand

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        # Shapes
        batch_size, num_chunks, chunk_size, num_heads, head_dim = hidden_states.shape

        # Constants (match original code)
        NUM_HEADS = self.num_heads
        N_GROUPS = self.n_groups
        GROUP_EXPAND = self.group_expand
        L_mat_size = 128  # as in original code, mask size

        # Cast to float32 for kernel computations
        hidden_f32 = hidden_states.to(torch.float32)
        A_f32 = A_cumsum.to(torch.float32)
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)

        # 1) Build L matrix [B, C, 128, 128, H] in Triton
        L = torch.empty((batch_size, num_chunks, L_mat_size, L_mat_size, num_heads), dtype=torch.float32, device=hidden_f32.device)

        grid_L = (batch_size, num_chunks, num_heads, L_mat_size, L_mat_size)
        build_lower_tri_causal_kernel[grid_L](
            A_f32, L, batch_size, num_chunks, num_heads, chunk_size
        )

        # 2) Expand B and C from groups to heads in Triton
        S_size = C_f32.shape[-1]
        B_exp = torch.empty((batch_size, num_chunks, chunk_size, num_heads, S_size), dtype=torch.float32, device=hidden_f32.device)
        C_exp = torch.empty((batch_size, num_chunks, chunk_size, num_heads, S_size), dtype=torch.float32, device=hidden_f32.device)

        grid_expand = (batch_size, num_chunks, chunk_size, num_heads, S_size)
        expand_groups_repeat_interleave[grid_expand](
            B_f32, B_exp, batch_size, num_chunks, chunk_size, num_heads, S_size, N_GROUPS, GROUP_EXPAND, 3
        )
        expand_groups_repeat_interleave[grid_expand](
            C_f32, C_exp, batch_size, num_chunks, chunk_size, num_heads, S_size, N_GROUPS, GROUP_EXPAND, 3
        )

        # 3) Compute G[b, c, i, j, h] = sum_s C_exp[b, c, i, h, s] * B_exp[b, c, j, h, s] in Triton
        G = torch.empty((batch_size, num_chunks, chunk_size, chunk_size, num_heads), dtype=torch.float32, device=hidden_f32.device)

        grid_G = (batch_size, num_chunks, chunk_size, chunk_size, num_heads)
        g_outer_kernel[grid_G](
            C_exp, B_exp, G, batch_size, num_chunks, chunk_size, num_heads, S_size
        )

        # 4) Apply mask: M = G * L (element-wise) in Triton
        M = torch.empty_like(G)
        grid_M = (batch_size, num_chunks, chunk_size, chunk_size, num_heads)
        multiply_mask_kernel[grid_M](
            G, L, M, batch_size, num_chunks, chunk_size, num_heads
        )

        # 5) Compute Y_diag by contraction over j in Triton: Y[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, d]
        Y = torch.empty((batch_size, num_chunks, chunk_size, num_heads, head_dim), dtype=torch.float32, device=hidden_f32.device)

        grid_Y = (batch_size, num_chunks, chunk_size, num_heads)
        y_diag_reduce_kernel[grid_Y](
            M, hidden_f32, Y, batch_size, num_chunks, chunk_size, num_heads, head_dim
        )

        return Y.to(torch.bfloat16)


# Example usage:
# m = ModelNew(num_heads=32, n_groups=8, group_expand=4)
# out = m(hidden_states, A_cumsum, B, C)


def run(*args):
    return ModelNew()(*args)
