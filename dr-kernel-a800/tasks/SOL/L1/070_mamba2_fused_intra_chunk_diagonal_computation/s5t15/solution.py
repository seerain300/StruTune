import torch
import triton
import triton.language as tl


# Kernel 1: Build L matrix for causal mask: 128x128, diagonal=-1, based on A_cumsum
# L_ptr: [B, C, 128, 128, H] float32
# A_ptr: [B, C, L, H] float32
@triton.jit
def build_lower_tri_causal_kernel(
    A_ptr,            # *float32, [B, C, L, H]
    L_ptr,            # *float32, [B, C, 128, 128, H]
    B_size,           # int
    C_size,           # int
    H_size,           # int
    L_len,            # int (actual chunk length, hidden_states.shape[2])
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # Prepare index vectors for rows and cols (0..127)
    i = tl.arange(0, 128)
    j = tl.arange(0, 128)

    # Compute cumulative sum for row i over actual L_len: total = sum_{k=0..L_len-1} A[b, c, k, h]
    total = 0.0
    k = 0
    while k < L_len:
        a_addr = b * (C_size * L_len * H_size) + c * (L_len * H_size) + k * H_size + h
        a_val = tl.load(A_ptr + a_addr)
        total += a_val
        k += 1

    # For each row i, write lower-triangular entries: if j <= i, L[i, j] = exp(total); else 0
    i_idx = 0
    while i_idx < 128:
        # broadcast i_idx for vectorized comparison with j
        # We only need to write j <= i_idx
        mask_j = j <= i_idx
        l_vals = tl.where(mask_j, tl.exp(total), 0.0)

        # Compute L addresses for this row i_idx
        L_base = b * (C_size * 128 * 128 * H_size) + c * (128 * 128 * H_size) + i_idx * (128 * H_size)
        L_addrs = L_base + j * H_size + h  # j in [0,127], h is the last dim
        # Store vector l_vals to L_addrs (j is the fast dimension)
        tl.store(L_ptr + L_addrs, l_vals, mask=mask_j)
        i_idx += 1


# Kernel 2: Expand groups to heads via repeat_interleave along group dimension
# B: [B, C, L, G, S] float32 -> B_exp: [B, C, L, H, S] float32
# Mapping: h_in = h // 4; g_src = h % 4
@triton.jit
def expand_groups_repeat_interleave_B(
    B_ptr,            # *float32, [B, C, L, G, S]
    B_exp_ptr,        # *float32, [B, C, L, H, S]
    B_size,           # int
    C_size,           # int
    L_len,            # int
    G_size,           # int = 8
    H_size,           # int = 32
    S_size,           # int
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    l = tl.program_id(2)
    h = tl.program_id(3)
    s = tl.program_id(4)

    g_src = h % 4  # since GROUP_EXPAND=4, one group maps to 4 heads
    h_in = h // 4

    B_addr = b * (C_size * L_len * G_size * S_size) + c * (L_len * G_size * S_size) + l * (G_size * S_size) + g_src * S_size + s
    B_val = tl.load(B_ptr + B_addr)

    B_exp_addr = b * (C_size * L_len * H_size * S_size) + c * (L_len * H_size * S_size) + l * (H_size * S_size) + h * S_size + s
    tl.store(B_exp_ptr + B_exp_addr, B_val)


# Kernel 3: Same for C
@triton.jit
def expand_groups_repeat_interleave_C(
    C_ptr,            # *float32, [B, C, L, G, S]
    C_exp_ptr,        # *float32, [B, C, L, H, S]
    B_size,           # int
    C_size,           # int
    L_len,            # int
    G_size,           # int = 8
    H_size,           # int = 32
    S_size,           # int
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    l = tl.program_id(2)
    h = tl.program_id(3)
    s = tl.program_id(4)

    g_src = h % 4
    h_in = h // 4

    C_addr = b * (C_size * L_len * G_size * S_size) + c * (L_len * G_size * S_size) + l * (G_size * S_size) + g_src * S_size + s
    C_val = tl.load(C_ptr + C_addr)

    C_exp_addr = b * (C_size * L_len * H_size * S_size) + c * (L_len * H_size * S_size) + l * (H_size * S_size) + h * S_size + s
    tl.store(C_exp_ptr + C_exp_addr, C_val)


# Kernel 4: Compute G[b, c, i, j, h] = sum_s C_exp[b, c, i, h, s] * B_exp[b, c, j, h, s]
# G_ptr: [B, C, L, L, H] float32
@triton.jit
def g_outer_kernel(
    C_exp_ptr,        # *float32, [B, C, L, H, S]
    B_exp_ptr,        # *float32, [B, C, L, H, S]
    G_ptr,            # *float32, [B, C, L, L, H]
    B_size,           # int
    C_size,           # int
    L_len,            # int
    H_size,           # int
    S_size,           # int
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # row index in L
    j = tl.program_id(3)  # col index in L
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


# Kernel 5: Element-wise apply mask L to G: M = G * L
# G: [B, C, L, L, H], L: [B, C, 128, 128, H], M: [B, C, L, L, H]
# We assume L has been zero-padded to 128x128 with values 0 for i >= 128, j >= 128.
@triton.jit
def apply_mask_kernel(
    G_ptr,            # *float32, [B, C, L, L, H]
    L_ptr,            # *float32, [B, C, 128, 128, H]
    M_ptr,            # *float32, [B, C, L, L, H]
    B_size,           # int
    C_size,           # int
    L_len,            # int
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # row index in L
    j = tl.program_id(3)  # col index in L
    h = tl.program_id(4)

    # Load G
    G_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
    G_val = tl.load(G_ptr + G_addr)

    # Load corresponding L at (min(i, 127), min(j, 127))
    iL = tl.minimum(i, 127)
    jL = tl.minimum(j, 127)
    L_addr = b * (C_size * 128 * 128 * H_size) + c * (128 * 128 * H_size) + iL * (128 * H_size) + jL * H_size + h
    L_val = tl.load(L_ptr + L_addr)

    M_val = G_val * L_val
    M_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
    tl.store(M_ptr + M_addr, M_val)


# Kernel 6: Compute Y_diag contraction over j: Y[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden_states[b, c, j, h, d]
# hidden_states: [B, C, L, H, D], M: [B, C, L, L, H], Y: [B, C, L, H, D]
@triton.jit
def y_diag_reduce_kernel(
    M_ptr,            # *float32, [B, C, L, L, H]
    hidden_ptr,       # *float32, [B, C, L, H, D]
    Y_ptr,            # *float32, [B, C, L, H, D]
    B_size,           # int
    C_size,           # int
    L_len,            # int
    H_size,           # int
    D_size,           # int
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # row index in L
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    j = 0
    while j < L_len:
        M_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
        M_val = tl.load(M_ptr + M_addr)

        hs_addr = b * (C_size * L_len * H_size * D_size) + c * (L_len * H_size * D_size) + j * (H_size * D_size) + h * D_size + d
        hs_val = tl.load(hidden_ptr + hs_addr)

        acc += M_val * hs_val
        j += 1

    Y_addr = b * (C_size * L_len * H_size * D_size) + c * (L_len * H_size * D_size) + i * (H_size * D_size) + h * D_size + d
    tl.store(Y_ptr + Y_addr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward that mirrors the original PyTorch computation but performs all heavy math in Triton.
        Returns Y_diag with dtype bfloat16, matching original behavior.
        """
        # Shapes from inputs
        batch_size, num_chunks, chunk_size, num_heads, head_dim = hidden_states.shape
        # Constants
        NUM_HEADS = 32
        N_GROUPS = 8
        GROUP_EXPAND = 4
        L_mat_size = 128  # reference implementation uses 128x128 mask

        # Ensure dtype float32 for kernels
        hidden_f32 = hidden_states.to(torch.float32)
        A_f32 = A_cumsum.to(torch.float32)
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)

        # Output tensor: Y_diag [B, C, L, H, D] float32, later cast to bfloat16
        Y = torch.empty((batch_size, num_chunks, chunk_size, num_heads, head_dim), dtype=torch.float32, device=hidden_f32.device)

        # 1) Build L matrix [B, C, 128, 128, H] in Triton
        L_ptr = torch.empty((batch_size, num_chunks, L_mat_size, L_mat_size, num_heads), dtype=torch.float32, device=hidden_f32.device)
        grid_L = (batch_size, num_chunks, num_heads)
        build_lower_tri_causal_kernel[grid_L](
            A_f32, L_ptr, batch_size, num_chunks, num_heads, chunk_size
        )

        # 2) Expand B and C from groups to heads in Triton
        B_exp = torch.empty((batch_size, num_chunks, chunk_size, num_heads, B_f32.shape[-1]), dtype=torch.float32, device=hidden_f32.device)
        C_exp = torch.empty((batch_size, num_chunks, chunk_size, num_heads, C_f32.shape[-1]), dtype=torch.float32, device=hidden_f32.device)

        # Determine S size from C (assume C and B have same S)
        S_size = C_f32.shape[-1]

        grid_expand = (batch_size, num_chunks, chunk_size, num_heads, S_size)
        expand_groups_repeat_interleave_B[grid_expand](
            B_f32, B_exp, batch_size, num_chunks, chunk_size, N_GROUPS, num_heads, S_size
        )
        expand_groups_repeat_interleave_C[grid_expand](
            C_f32, C_exp, batch_size, num_chunks, chunk_size, N_GROUPS, num_heads, S_size
        )

        # 3) Compute G[b, c, i, j, h] = sum_s C_exp[b, c, i, h, s] * B_exp[b, c, j, h, s] in Triton
        G = torch.empty((batch_size, num_chunks, chunk_size, chunk_size, num_heads), dtype=torch.float32, device=hidden_f32.device)

        grid_g = (batch_size, num_chunks, chunk_size, chunk_size, num_heads)
        g_outer_kernel[grid_g](
            C_exp, B_exp, G, batch_size, num_chunks, chunk_size, num_heads, S_size
        )

        # 4) Apply mask L element-wise: M = G * L
        M = torch.empty_like(G)
        grid_mask = (batch_size, num_chunks, chunk_size, chunk_size, num_heads)
        apply_mask_kernel[grid_mask](
            G, L_ptr, M, batch_size, num_chunks, chunk_size
        )

        # 5) Compute Y_diag by contraction over j: Y[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, d]
        grid_y = (batch_size, num_chunks, chunk_size, num_heads, head_dim)
        y_diag_reduce_kernel[grid_y](
            M, hidden_f32, Y, batch_size, num_chunks, chunk_size, num_heads, head_dim
        )

        # Return cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
