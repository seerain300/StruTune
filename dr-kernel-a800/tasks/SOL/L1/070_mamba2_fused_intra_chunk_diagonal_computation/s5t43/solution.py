import torch
import triton
import triton.language as tl


@triton.jit
def build_lower_tri_causal_kernel(
    A_ptr,  # *const float, shape (B, C, L, H)
    L_ptr,  # *float, shape (B, C, 128, 128, H)
    B, C, L, H
):
    # program_id over (b, c, h)
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # total = sum over k in 0..L-1 of A[b, c, k, h]
    total = 0.0
    for k in range(0, L):
        idx = (((b * C) + c) * (L * H)) + (k * H) + h
        val = tl.load(A_ptr + idx)
        total += val

    # fill L[i, j] = exp(total) if j <= i else 0, for i,j in [0, 127]
    for i in range(0, 128):
        for j in range(0, 128):
            out_idx = (((((b * C) + c) * (128 * 128)) + (i * 128) + j) * H) + h
            if j <= i:
                tl.store(L_ptr + out_idx, tl.exp(total))
            else:
                tl.store(L_ptr + out_idx, 0.0)


@triton.jit
def expand_groups_repeat_interleave(
    in_ptr,          # *const float, shape (B, C, L, G, S)
    out_ptr,         # *float, shape (B, C, L, H, S)
    B, C, L, G, H, S,
    repeats: tl.constexpr,
):
    # program_id over (b, c, i, h, s)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # 0..L-1
    h = tl.program_id(3)  # 0..H-1
    s = tl.program_id(4)  # 0..S-1

    # map group g to head index: h = g * repeats + r, r in [0, repeats)
    group_size = H // G
    g = h // group_size
    r = h % group_size
    # valid only if r < repeats (repeats is constexpr), we ensure repeats * group_size == H
    if r < repeats:
        in_idx = (((((b * C) + c) * L) + i) * (G * S) + (g * S + s))
        out_idx = (((((b * C) + c) * L) + i) * (H * S) + (h * S + s))
        val = tl.load(in_ptr + in_idx)
        tl.store(out_ptr + out_idx, val)


@triton.jit
def compute_G_kernel(
    Bexp_ptr,  # *const float, shape (B, C, L, H, S)
    Cexp_ptr,  # *const float, shape (B, C, L, H, S)
    G_ptr,     # *float, shape (B, C, L, L, H)
    B, C, L, H, S
):
    # grid over (b, c, i, j)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)

    # Accumulate G[i, j, h] over s
    for h in range(0, H):
        G_val = 0.0
        for s in range(0, S):
            Bval = tl.load(Bexp_ptr + (((((b * C) + c) * L) + i) * (H * S) + (h * S + s)))
            Cval = tl.load(Cexp_ptr + (((((b * C) + c) * L) + j) * (H * S) + (h * S + s)))
            G_val += Bval * Cval
        # store G_val into G[b, c, i, j, h]
        G_idx = (((((b * C) + c) * L) + i) * (L * H)) + (j * H) + h
        tl.store(G_ptr + G_idx, G_val)


@triton.jit
def apply_mask_L_to_G_kernel(
    G_ptr,   # *const float, shape (B, C, L, L, H)
    L_ptr,   # *const float, shape (B, C, 128, 128, H)
    M_ptr,   # *float, shape (B, C, L, L, H)
    B, C, L, H
):
    # grid over (b, c, i, j, h)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    G_val = tl.load(G_ptr + (((((b * C) + c) * L) + i) * (L * H) + (j * H) + h))
    # L is 128x128, but we only need L[i, j] because i and j are within [0, L-1] and we index it as L[i, j]
    # In original, L[i, j] is defined via 128x128 mask and uses cumsum along k. Here G and L have same (i, j) range.
    # We assume L matrix is precomputed with correct values in build_lower_tri_causal_kernel.
    L_val = tl.load(L_ptr + (((((b * C) + c) * (128 * 128)) + (i * 128) + j) * H + h))
    M_val = G_val * L_val
    M_idx = (((((b * C) + c) * L) + i) * (L * H)) + (j * H) + h
    tl.store(M_ptr + M_idx, M_val)


@triton.jit
def compute_Y_diag_kernel(
    M_ptr,    # *const float, shape (B, C, L, L, H)
    Hs_ptr,   # *const float, shape (B, C, L, H, D)
    Y_ptr,    # *float, shape (B, C, L, H, D)
    B, C, L, H, D
):
    # grid over (b, c, i, h)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    # Accumulate over j
    for j in range(0, L):
        M_val = tl.load(M_ptr + (((((b * C) + c) * L) + i) * (L * H) + (j * H) + h))
        for d in range(0, D):
            hs_val = tl.load(Hs_ptr + (((((b * C) + c) * L) + j) * (H * D) + (h * D) + d))
            Y_idx = (((((b * C) + c) * L) + i) * (H * D)) + (h * D) + d
            if d == 0:
                tl.store(Y_ptr + Y_idx, M_val * hs_val)
            else:
                curr = tl.load(Y_ptr + Y_idx)
                tl.store(Y_ptr + Y_idx, curr + (M_val * hs_val))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 32
        self.n_groups = 8
        self.group_expand = 4  # repeats per group to reach 32 heads

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
        # Extract shapes
        batch_size, num_chunks, chunk_size, num_heads, head_dim = hidden_states.shape

        # Cast inputs to float32 for compute
        hidden_f32 = hidden_states.to(torch.float32)
        A_f32 = A_cumsum.to(torch.float32)
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)

        # 1) Build L: (B, C, 128, 128, H)
        L_mat = torch.empty((batch_size, num_chunks, 128, 128, num_heads), dtype=torch.float32, device=hidden_f32.device)
        grid_L = (batch_size, num_chunks, num_heads)
        build_lower_tri_causal_kernel[grid_L](
            A_f32, L_mat, batch_size, num_chunks, chunk_size, num_heads
        )

        # 2) Expand B and C from groups to heads
        # Shapes: B: [B, C, L, G, S]; C: [B, C, L, G, S]
        B_exp = torch.empty((batch_size, num_chunks, chunk_size, num_heads, B_f32.shape[-1]), dtype=torch.float32, device=hidden_f32.device)
        C_exp = torch.empty((batch_size, num_chunks, chunk_size, num_heads, C_f32.shape[-1]), dtype=torch.float32, device=hidden_f32.device)

        # Determine S from B/C last dim (should be same for both)
        S_size = B_f32.shape[-1]

        grid_expand = (batch_size, num_chunks, chunk_size, num_heads, S_size)
        expand_groups_repeat_interleave[grid_expand](
            B_f32, B_exp, batch_size, num_chunks, chunk_size, self.n_groups, num_heads, S_size, repeats=self.group_expand
        )
        expand_groups_repeat_interleave[grid_expand](
            C_f32, C_exp, batch_size, num_chunks, chunk_size, self.n_groups, num_heads, S_size, repeats=self.group_expand
        )

        # 3) Compute G: [B, C, L, L, H]
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

        # 5) Compute Y_diag: [B, C, L, H, D]
        Y = torch.empty((batch_size, num_chunks, chunk_size, num_heads, head_dim), dtype=torch.float32, device=hidden_f32.device)
        grid_Y = (batch_size, num_chunks, chunk_size, num_heads)
        compute_Y_diag_kernel[grid_Y](
            M, hidden_f32, Y, batch_size, num_chunks, chunk_size, num_heads, head_dim
        )

        # Return in bfloat16 (as in original run signature)
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
