import torch
import triton
import triton.language as tl


# Kernel 1: build lower-triangular 128x128 causal L per (b, c, h)
# L[i, j] = exp(sum_k A[b, c, k, h]) if j <= i else 0.0
@triton.jit
def build_lower_tri_causal_kernel(
    A_ptr,            # *const float32, shape (B, C, L, H)
    L_ptr,            # *float32, shape (B, C, 128, 128, H)
    B: tl.constexpr,  # int: batch size
    C: tl.constexpr,  # int: num_chunks
    H: tl.constexpr,  # int: num_heads
    L_dim: tl.constexpr,  # int: hidden size along chunk dimension
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # Compute total = sum_{k=0..L_dim-1} A[b, c, k, h]
    total = 0.0
    for k in range(L_dim):
        idx = b * (C * H * L_dim) + c * (H * L_dim) + k * H + h
        val = tl.load(A_ptr + idx)
        total += val

    # Fill 128x128 lower-triangular L: if j <= i, store exp(total), else 0.0
    for i in range(128):
        for j in range(128):
            out_idx = b * (C * 128 * 128 * H) + c * (128 * 128 * H) + i * (128 * H) + j * H + h
            if j <= i:
                tl.store(L_ptr + out_idx, tl.exp(total))
            else:
                tl.store(L_ptr + out_idx, 0.0)


# Kernel 2: expand groups -> heads for B and C with repeat_interleave (repeats=4)
# in_ptr: (B, C, L, G, S); out_ptr: (B, C, L, H, S)
@triton.jit
def expand_groups_repeat_interleave(
    in_ptr,           # *const float32
    out_ptr,          # *float32
    B: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    G: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    repeats: tl.constexpr,  # 4
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # i in [0, L)
    h = tl.program_id(3)  # h in [0, H)
    s = tl.program_id(4)  # s in [0, S)

    group_size = H // G
    g = h // group_size  # which group this head maps to

    in_idx = (((((b * C) + c) * L) + i) * G + g) * S + s
    out_idx = (((((b * C) + c) * L) + i) * H + h) * S + s

    val = tl.load(in_ptr + in_idx)
    tl.store(out_ptr + out_idx, val)


# Kernel 3: compute G[i, j, h] = sum_s B_exp[j, s] * C_exp[i, s] over s in [0, S)
@triton.jit
def compute_G_kernel(
    B_exp_ptr,        # *const float32, shape (B, C, L, H, S)
    C_exp_ptr,        # *const float32, shape (B, C, L, H, S)
    G_ptr,            # *float32, shape (B, C, L, L, H)
    Bsz: tl.constexpr,
    Csz: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # row i
    j = tl.program_id(3)  # col j

    for h in range(H):
        acc = 0.0
        for s in range(S):
            B_val = tl.load(B_exp_ptr + (((((b * Csz) + c) * L) + j) * H + h) * S + s)
            C_val = tl.load(C_exp_ptr + (((((b * Csz) + c) * L) + i) * H + h) * S + s)
            acc += B_val * C_val

        # store G[b, c, i, j, h]
        out_idx = (((((b * Csz) + c) * L) + i) * (L * H) + (j * H) + h)
        tl.store(G_ptr + out_idx, acc)


# Kernel 4: apply mask L to G element-wise: M = G * L
# G: (B, C, L, L, H), L: (B, C, 128, 128, H)
@triton.jit
def apply_mask_L_to_G_kernel(
    G_ptr,            # *float32, shape (B, C, L, L, H)
    L_ptr,            # *float32, shape (B, C, 128, 128, H)
    M_ptr,            # *float32, shape (B, C, L, L, H)
    Bsz: tl.constexpr,
    Csz: tl.constexpr,
    Ldim: tl.constexpr,
    H: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # i in [0, Ldim)
    j = tl.program_id(3)  # j in [0, Ldim)
    h = tl.program_id(4)  # h in [0, H)

    g_val = tl.load(G_ptr + (((((b * Csz) + c) * Ldim) + i) * (Ldim * H) + (j * H) + h))
    l_val = tl.load(L_ptr + (((((b * Csz) + c) * 128 + i) * 128 + j) * H + h))
    m_val = g_val * l_val
    tl.store(M_ptr + (((((b * Csz) + c) * Ldim) + i) * (Ldim * H) + (j * H) + h), m_val)


# Kernel 5: compute Y_diag[i, h, d] = sum_j M[i, j, h] * hidden[j, h, d]
@triton.jit
def compute_Y_diag_kernel(
    M_ptr,            # *float32, shape (B, C, L, L, H)
    hidden_ptr,       # *const float32, shape (B, C, L, H, D)
    Y_ptr,            # *float32, shape (B, C, L, H, D)
    Bsz: tl.constexpr,
    Csz: tl.constexpr,
    Ldim: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # i in [0, Ldim)
    h = tl.program_id(3)  # h in [0, H)

    for d in range(D):
        acc = 0.0
        for j in range(Ldim):
            m = tl.load(M_ptr + (((((b * Csz) + c) * Ldim) + i) * (Ldim * H) + (j * H) + h))
            hid_idx = (((((b * Csz) + c) * Ldim) + j) * H + h) * D + d
            hid = tl.load(hidden_ptr + hid_idx)
            acc += m * hid

        out_idx = (((((b * Csz) + c) * Ldim) + i) * (H * D) + (h * D) + d)
        tl.store(Y_ptr + out_idx, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
        # Shapes
        batch_size, num_chunks, chunk_size, num_heads, head_dim = hidden_states.shape

        # Cast to float32 (no torch ops in forward)
        hidden_f32 = hidden_states.to(torch.float32)
        A_f32 = A_cumsum.to(torch.float32)
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)

        # 1) Build L (B, C, 128, 128, H) in Triton
        L_mat = torch.empty((batch_size, num_chunks, 128, 128, num_heads), dtype=torch.float32, device=hidden_f32.device)
        grid_L = (batch_size, num_chunks, num_heads)
        build_lower_tri_causal_kernel[grid_L](
            A_f32, L_mat, batch_size, num_chunks, num_heads, chunk_size
        )

        # 2) Expand B and C from groups (N_GROUPS=8) to heads (NUM_HEADS=32) using repeat_interleave (repeats=4)
        S_size = C_f32.shape[-1]
        B_exp = torch.empty((batch_size, num_chunks, chunk_size, num_heads, S_size), dtype=torch.float32, device=hidden_f32.device)
        C_exp = torch.empty((batch_size, num_chunks, chunk_size, num_heads, S_size), dtype=torch.float32, device=hidden_f32.device)

        grid_expand = (batch_size, num_chunks, chunk_size, num_heads, S_size)
        expand_groups_repeat_interleave[grid_expand](
            B_f32, B_exp, batch_size, num_chunks, chunk_size, 8, num_heads, S_size, repeats=4
        )
        expand_groups_repeat_interleave[grid_expand](
            C_f32, C_exp, batch_size, num_chunks, chunk_size, 8, num_heads, S_size, repeats=4
        )

        # 3) Compute G (B, C, L, L, H) in Triton
        G = torch.empty((batch_size, num_chunks, chunk_size, chunk_size, num_heads), dtype=torch.float32, device=hidden_f32.device)
        grid_G = (batch_size, num_chunks, chunk_size, chunk_size)
        compute_G_kernel[grid_G](
            B_exp, C_exp, G, batch_size, num_chunks, chunk_size, num_heads, S_size
        )

        # 4) Apply mask L to G: M = G * L
        M = torch.empty_like(G, dtype=torch.float32, device=hidden_f32.device)
        grid_apply = (batch_size, num_chunks, chunk_size, chunk_size, num_heads)
        apply_mask_L_to_G_kernel[grid_apply](
            G, L_mat, M, batch_size, num_chunks, chunk_size, num_heads
        )

        # 5) Compute Y_diag (B, C, L, H, D) in Triton
        Y = torch.empty((batch_size, num_chunks, chunk_size, num_heads, head_dim), dtype=torch.float32, device=hidden_f32.device)
        grid_Y = (batch_size, num_chunks, chunk_size, num_heads)
        compute_Y_diag_kernel[grid_Y](
            M, hidden_f32, Y, batch_size, num_chunks, chunk_size, num_heads, head_dim
        )

        # Return in original expected dtype (bf16 per original run signature)
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
