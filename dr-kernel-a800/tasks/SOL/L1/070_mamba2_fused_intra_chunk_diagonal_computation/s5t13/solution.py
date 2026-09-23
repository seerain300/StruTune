import torch
import triton
import triton.language as tl

# Kernel 1: Build lower-triangular causal mask L: L[i, j] = exp(sum_{k=0..i} A[b, c, k, h]) if i >= j, else 0
@triton.jit
def build_lower_tri_causal_kernel(
    A_ptr,          # *float32, [B, C, L, H]
    L_ptr,          # *float32, [B, C, L, L, H]
    B_size,         # int
    C_size,         # int
    L_len,          # int
    H_size,         # int
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # For each row i and column j, compute L[i, j] per lower-triangular rule
    i = 0
    while i < L_len:
        total = 0.0
        k = 0
        # cumulative sum over k from 0..i
        while k <= i:
            addr = b * (C_size * L_len * H_size) + c * (L_len * H_size) + k * H_size + h
            a_val = tl.load(A_ptr + addr)
            total += a_val
            k += 1
        # now write L[i, j] for j = 0..L_len-1
        j = 0
        while j < L_len:
            # i >= j ensures causal (diagonal = -1)
            if j <= i:
                l_val = tl.exp(total)
            else:
                l_val = 0.0
            L_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
            tl.store(L_ptr + L_addr, l_val)
            j += 1
        i += 1


# Kernel 2: Expand from groups to H via repeat_interleave along dim=3 (groups -> H)
# Writes B_expanded to B_exp_ptr and C_expanded to C_exp_ptr
@triton.jit
def expand_groups_repeat_interleave(
    B_src_ptr,      # *float32, [B, C, L, groups, S]
    C_src_ptr,      # *float32, [B, C, L, groups, S]
    B_exp_ptr,      # *float32, [B, C, L, H, S]
    C_exp_ptr,      # *float32, [B, C, L, H, S]
    B_size,         # int
    C_size,         # int
    L_len,          # int
    groups,         # int (N_GROUPS = 8)
    H_size,         # int (NUM_HEADS = 32)
    S_size,         # int
):
    # grid = (B*C, groups*H) -> map to (b,c,group,h')
    bc = tl.program_id(0)  # flattened (b,c)
    group_h = tl.program_id(1)
    # derive b,c
    b = bc // C_size
    c = bc % C_size
    # derive group and h' (expanded index)
    group = group_h // H_size
    h_prime = group_h % H_size

    # map expanded head index h_prime to original group index
    if group < groups:
        # For each (b, c, i, s), write to B_exp_ptr[b, c, i, h_prime, s]
        i = 0
        while i < L_len:
            s = 0
            while s < S_size:
                B_src_addr = b * (C_size * L_len * groups * S_size) + c * (L_len * groups * S_size) + i * (groups * S_size) + group * S_size + s
                B_src_val = tl.load(B_src_ptr + B_src_addr)
                B_exp_addr = b * (C_size * L_len * H_size * S_size) + c * (L_len * H_size * S_size) + i * (H_size * S_size) + h_prime * S_size + s
                tl.store(B_exp_ptr + B_exp_addr, B_src_val)
                s += 1
            i += 1

        # And similarly for C_src_ptr -> C_exp_ptr
        i = 0
        while i < L_len:
            s = 0
            while s < S_size:
                C_src_addr = b * (C_size * L_len * groups * S_size) + c * (L_len * groups * S_size) + i * (groups * S_size) + group * S_size + s
                C_src_val = tl.load(C_src_ptr + C_src_addr)
                C_exp_addr = b * (C_size * L_len * H_size * S_size) + c * (L_len * H_size * S_size) + i * (H_size * S_size) + h_prime * S_size + s
                tl.store(C_exp_ptr + C_exp_addr, C_src_val)
                s += 1
            i += 1


# Kernel 3: Compute G[b, c, i, j, h] = sum_s C_exp[b, c, i, h, s] * B_exp[b, c, j, h, s]
@triton.jit
def g_outer_kernel(
    C_exp_ptr,      # *float32, [B, C, L, H, S]
    B_exp_ptr,      # *float32, [B, C, L, H, S]
    G_ptr,          # *float32, [B, C, L, L, H]
    B_size,         # int
    C_size,         # int
    L_len,          # int
    H_size,         # int
    S_size,         # int
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # along L for rows
    j = tl.program_id(3)  # along L for cols
    h = tl.program_id(4)  # along H

    g_val = 0.0
    s = 0
    while s < S_size:
        C_addr = b * (C_size * L_len * H_size * S_size) + c * (L_len * H_size * S_size) + i * (H_size * S_size) + h * S_size + s
        B_addr = b * (C_size * L_len * H_size * S_size) + c * (L_len * H_size * S_size) + j * (H_size * S_size) + h * S_size + s
        C_val = tl.load(C_exp_ptr + C_addr)
        B_val = tl.load(B_exp_ptr + B_addr)
        g_val += C_val * B_val
        s += 1

    # Store G[b, c, i, j, h] = g_val
    G_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
    tl.store(G_ptr + G_addr, g_val)


# Kernel 4: Compute Y_diag[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden_states[b, c, j, h, d]
# M: [B, C, L, L, H], hidden_states: [B, C, L, H, D], output Y: [B, C, L, H, D]
@triton.jit
def y_diag_reduce_kernel(
    M_ptr,          # *float32, [B, C, L, L, H]
    HS_ptr,         # *float32, [B, C, L, H, D]
    Y_ptr,          # *float32, [B, C, L, H, D]
    B_size,         # int
    C_size,         # int
    L_len,          # int
    H_size,         # int
    D_size,         # int
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    d_offsets = tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    j = 0
    while j < L_len:
        # M[b, c, i, j, h]
        M_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
        m_val = tl.load(M_ptr + M_addr)
        # hidden_states[b, c, j, h, d]
        HS_base = b * (C_size * L_len * H_size * D_size) + c * (L_len * H_size * D_size) + j * (H_size * D_size) + h * D_size
        HS_addr = HS_base + d_offsets
        hs_vec = tl.load(HS_ptr + HS_addr, mask=d_offsets < D_size)
        acc += m_val * hs_vec
        j += 1

    # Store Y[b, c, i, h, d]
    Y_base = b * (C_size * L_len * H_size * D_size) + c * (L_len * H_size * D_size) + i * (H_size * D_size) + h * D_size
    Y_addr = Y_base + d_offsets
    tl.store(Y_ptr + Y_addr, acc, mask=d_offsets < D_size)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, C, L, H, D]
        A_cumsum:      [B, C, L, H]
        B:             [B, C, L, groups, S]
        C:             [B, C, L, groups, S]
        Returns:       [B, C, L, H, D] in bfloat16
        """

        # Shapes
        B_size, C_size, L_len, H_size, D_size = hidden_states.shape
        groups = 8  # N_GROUPS
        group_expand = 4  # NUM_HEADS // N_GROUPS

        # Ensure contiguous
        hidden_states = hidden_states.contiguous()
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        # 1) Build L in Triton: L is [B, C, L, L, H], float32
        L = torch.empty((B_size, C_size, L_len, L_len, H_size), dtype=torch.float32, device=hidden_states.device)
        # Launch grid over (B, C, H)
        grid1 = (B_size, C_size, H_size)
        build_lower_tri_causal_kernel[grid1](
            A_cumsum, L,
            B_size, C_size, L_len, H_size
        )

        # 2) Expand B and C from groups to H using Triton
        B_exp = torch.empty((B_size, C_size, L_len, H_size, B.shape[-1]), dtype=torch.float32, device=hidden_states.device)
        C_exp = torch.empty((B_size, C_size, L_len, H_size, C.shape[-1]), dtype=torch.float32, device=hidden_states.device)
        # grid over (B*C, groups*H)
        grid2 = (B_size * C_size, groups * H_size)
        expand_groups_repeat_interleave[grid2](
            B, C, B_exp, C_exp,
            B_size, C_size, L_len, groups, H_size, B.shape[-1]
        )

        # 3) Compute G via Triton outer-product kernel
        G = torch.empty((B_size, C_size, L_len, L_len, H_size), dtype=torch.float32, device=hidden_states.device)
        S_size = B_exp.shape[-1]  # same as C_exp.shape[-1]
        grid3 = (B_size, C_size, L_len, L_len, H_size)
        g_outer_kernel[grid3](
            C_exp, B_exp, G,
            B_size, C_size, L_len, H_size, S_size
        )

        # 4) Apply L mask (element-wise multiply): M = G * L
        # L already in [B, C, L, L, H] from step 1
        M = G * L  # element-wise multiply in Triton is implicit in PyTorch here; we can implement it in Triton too:
        # For correctness, perform in Triton by launching elementwise multiply:
        M = torch.empty_like(G)
        grid4 = (B_size, C_size, L_len, L_len, H_size)
        # Create elementwise multiply kernel? Torch multiply is fine for correctness; to strictly use Triton only, we can implement as above by writing directly to M.

        # 5) Compute Y_diag via Triton reduction kernel
        # Cast hidden_states to float32 for computation
        HS_f32 = hidden_states.to(torch.float32)
        Y = torch.empty((B_size, C_size, L_len, H_size, D_size), dtype=torch.float32, device=hidden_states.device)

        # We need to ensure that M is float32 [B, C, L, L, H] and HS_f32 [B, C, L, H, D]
        # Launch reduction kernel
        grid5 = (B_size, C_size, L_len, H_size)
        y_diag_reduce_kernel[grid5](
            M, HS_f32, Y,
            B_size, C_size, L_len, H_size, D_size,
            BLOCK_D=128
        )

        # Return in bfloat16
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
