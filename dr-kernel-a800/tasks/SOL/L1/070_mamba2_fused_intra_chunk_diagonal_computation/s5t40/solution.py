import torch
import triton
import triton.language as tl


@triton.jit
def build_lower_tri_causal_5d_kernel(
    A_ptr,  # *float32
    L_ptr,  # *float32
    B: tl.constexpr,  # batch size
    C: tl.constexpr,  # num_chunks
    H: tl.constexpr,  # num_heads
    L_size: tl.constexpr,  # actual chunk_size of hidden_states
    L_mat: tl.constexpr,   # 128
    stride_A_b, stride_A_c, stride_A_l, stride_A_h,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
):
    # Grid: (B, C, H, L_mat, L_mat)
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    i = tl.program_id(3)
    j = tl.program_id(4)

    # Bounds check for i,j in [0, L_mat)
    if (i < 0) or (i >= L_mat) or (j < 0) or (j >= L_mat):
        return

    # Compute total = sum_{k=0..L_size-1} A[b, c, k, h]
    total = 0.0
    for k in range(L_size):
        # Address for A[b, c, k, h]
        addr_A = A_ptr + b * stride_A_b + c * stride_A_c + k * stride_A_l + h * stride_A_h
        a_val = tl.load(addr_A)
        total += a_val

    # If j <= i: L[i, j] = exp(total), else 0
    val = 0.0
    if j <= i:
        val = tl.exp(total)

    # Address for L[b, c, i, j, h]
    addr_L = L_ptr + b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
    tl.store(addr_L, val)


@triton.jit
def expand_groups_repeat_interleave(
    X_ptr,  # *float32, input with shape (B, C, L, N_GROUPS, S)
    Y_ptr,  # *float32, output with shape (B, C, L, NUM_HEADS, S)
    B: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    N_GROUPS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    S: tl.constexpr,
    GROUP_EXPAND: tl.constexpr,
    stride_X_b, stride_X_c, stride_X_l, stride_X_g, stride_X_s,
    stride_Y_b, stride_Y_c, stride_Y_l, stride_Y_h, stride_Y_s,
):
    # Grid: (B, C, L, NUM_HEADS, S)
    b = tl.program_id(0)
    c = tl.program_id(1)
    l = tl.program_id(2)
    h = tl.program_id(3)
    s = tl.program_id(4)

    # Compute group index g corresponding to head h
    # h in [0, NUM_HEADS); for each group, h spans GROUP_EXPAND consecutive heads.
    group = h // GROUP_EXPAND
    if group >= N_GROUPS:
        return
    # Address for X[b, c, l, group, s]
    addr_X = X_ptr + b * stride_X_b + c * stride_X_c + l * stride_X_l + group * stride_X_g + s * stride_X_s
    x_val = tl.load(addr_X)
    # Address for Y[b, c, l, h, s]
    addr_Y = Y_ptr + b * stride_Y_b + c * stride_Y_c + l * stride_Y_l + h * stride_Y_h + s * stride_Y_s
    tl.store(addr_Y, x_val)


@triton.jit
def compute_G_5d_kernel(
    B_exp_ptr,  # *float32, shape (B, C, L, H, S)
    C_exp_ptr,  # *float32, shape (B, C, L, H, S)
    G_ptr,      # *float32, shape (B, C, L_i, L_j, H)
    Bsz: tl.constexpr,
    Csz: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    stride_B_b, stride_B_c, stride_B_li, stride_B_h, stride_B_s,
    stride_C_b, stride_C_c, stride_C_lj, stride_C_h, stride_C_s,
    stride_G_b, stride_G_c, stride_G_li, stride_G_lj, stride_G_h,
):
    # Grid: (B, C, H, L, L) - over (b, c, h, li, lj)
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    li = tl.program_id(3)
    lj = tl.program_id(4)

    if (li < 0) or (li >= L) or (lj < 0) or (lj >= L):
        return

    # Accumulate G[b, c, li, lj, h]
    acc = 0.0
    for s in range(S):
        addr_B = B_exp_ptr + b * stride_B_b + c * stride_B_c + li * stride_B_li + h * stride_B_h + s * stride_B_s
        addr_C = C_exp_ptr + b * stride_C_b + c * stride_C_c + lj * stride_C_lj + h * stride_C_h + s * stride_C_s
        b_val = tl.load(addr_B)
        c_val = tl.load(addr_C)
        acc += b_val * c_val

    addr_G = G_ptr + b * stride_G_b + c * stride_G_c + li * stride_G_li + lj * stride_G_lj + h * stride_G_h
    tl.store(addr_G, acc)


@triton.jit
def reduce_Y_diag_kernel(
    M_ptr,        # *float32, shape (B, C, L_i, L_j, H)
    HS_ptr,       # *float32, hidden_states (B, C, L_j, H, D)
    Y_ptr,        # *float32, output (B, C, L_i, H, D)
    Bsz: tl.constexpr,
    Csz: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    stride_M_b, stride_M_c, stride_M_li, stride_M_lj, stride_M_h,
    stride_HS_b, stride_HS_c, stride_HS_lj, stride_HS_h, stride_HS_d,
    stride_Y_b, stride_Y_c, stride_Y_li, stride_Y_h, stride_Y_d,
):
    # Grid: (B, C, L, H, D) - over (b, c, li, h, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    li = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    if (li < 0) or (li >= L) or (h < 0) or (h >= H) or (d < 0) or (d >= D):
        return

    # Compute Y[b, c, li, h, d] = sum_{lj=0..L-1} M[b, c, li, lj, h] * HS[b, c, lj, h, d]
    acc = 0.0
    for lj in range(L):
        addr_M = M_ptr + b * stride_M_b + c * stride_M_c + li * stride_M_li + lj * stride_M_lj + h * stride_M_h
        m_val = tl.load(addr_M)
        addr_HS = HS_ptr + b * stride_HS_b + c * stride_HS_c + lj * stride_HS_lj + h * stride_HS_h + d * stride_HS_d
        hs_val = tl.load(addr_HS)
        acc += m_val * hs_val

    addr_Y = Y_ptr + b * stride_Y_b + c * stride_Y_c + li * stride_Y_li + h * stride_Y_h + d * stride_Y_d
    tl.store(addr_Y, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized version of the original run function.
        Computes Y_diag as defined in the original PyTorch implementation,
        but using Triton kernels for all heavy computations.

        Shapes:
          - hidden_states: [B, C, L, H, D]
          - A_cumsum:      [B, C, L, H]
          - B:             [B, C, L, N_GROUPS, S]
          - C:             [B, C, L, N_GROUPS, S]
        Constants:
          - NUM_HEADS = 32
          - N_GROUPS  = 8
          - GROUP_EXPAND = 4 (since 32 // 8 = 4)
          - L_mat_size = 128 (as in original code for mask size)
        """
        # Extract shapes
        Bsz, Csz, L, H, D = hidden_states.shape
        assert H == 32, "num_heads must be 32 in this Triton implementation"
        assert B.ndim == 4 and C.ndim == 4, "B and C must be 4D tensors with group dimension"
        Bsz_A, Csz_A, L_A, H_A = A_cumsum.shape
        assert Bsz_A == Bsz and Csz_A == Csz and H_A == H, "Shape mismatch between A_cumsum, B, C"
        N_GROUPS = 8
        S = C.shape[-1]  # state_size

        # Output tensor (will be cast to bfloat16 at the end)
        Y = torch.empty((Bsz, Csz, L, H, D), dtype=torch.float32, device=hidden_states.device)

        # 1) Build L matrix [B, C, 128, 128, H] in Triton with diagonal=-1
        L_mat = 128
        L_t = torch.empty((Bsz, Csz, L_mat, L_mat, H), dtype=torch.float32, device=hidden_states.device)

        # Launch grid must match 5D: (B, C, H, 128, 128)
        grid_L = (Bsz, Csz, H, L_mat, L_mat)
        build_lower_tri_causal_5d_kernel[grid_L](
            A_cumsum.to(torch.float32), L_t,
            Bsz, Csz, H, L, L_mat,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L_t.stride(0), L_t.stride(1), L_t.stride(2), L_t.stride(3), L_t.stride(4),
            num_warps=4, num_stages=2
        )

        # 2) Expand B and C from groups to heads in Triton: B_exp [B, C, L, H, S], C_exp [B, C, L, H, S]
        B_exp = torch.empty((Bsz, Csz, L, H, S), dtype=torch.float32, device=hidden_states.device)
        C_exp = torch.empty((Bsz, Csz, L, H, S), dtype=torch.float32, device=hidden_states.device)

        grid_expand = (Bsz, Csz, L, H, S)
        expand_groups_repeat_interleave[grid_expand](
            B.to(torch.float32), B_exp, Bsz, Csz, L, N_GROUPS, H, S, 4,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            num_warps=4, num_stages=2
        )
        expand_groups_repeat_interleave[grid_expand](
            C.to(torch.float32), C_exp, Bsz, Csz, L, N_GROUPS, H, S, 4,
            C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            num_warps=4, num_stages=2
        )

        # 3) Compute G[b, c, li, lj, h] = sum_s B_exp[b, c, lj, h, s] * C_exp[b, c, li, h, s] in Triton
        G = torch.empty((Bsz, Csz, L, L, H), dtype=torch.float32, device=hidden_states.device)

        grid_G = (Bsz, Csz, H, L, L)
        compute_G_5d_kernel[grid_G](
            B_exp, C_exp, G,
            Bsz, Csz, L, H, S,
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            num_warps=4, num_stages=2
        )

        # 4) Apply mask L to G: M = G * L (element-wise). Here, L is already computed in Triton.
        #    We just need to load L and multiply. However, since we already have L_t in GPU, we can
        #    elementwise multiply in Triton. Define an elementwise multiply kernel for 5D tensors:
        M = torch.empty_like(G)

        @triton.jit
        def elementwise_mul_5d_kernel(X_ptr, Y_ptr, Z_ptr, B: tl.constexpr, C: tl.constexpr, L: tl.constexpr, H: tl.constexpr, stride_X_b, stride_X_c, stride_X_li, stride_X_lj, stride_X_h, stride_Y_b, stride_Y_c, stride_Y_li, stride_Y_lj, stride_Y_h, stride_Z_b, stride_Z_c, stride_Z_li, stride_Z_lj, stride_Z_h):
            b = tl.program_id(0); c = tl.program_id(1); li = tl.program_id(2); lj = tl.program_id(3); h = tl.program_id(4)
            if (li < 0 or li >= L) or (lj < 0 or lj >= L) or (h < 0 or h >= H) or (b < 0 or b >= B) or (c < 0 or c >= C):
                return
            addr_X = X_ptr + b * stride_X_b + c * stride_X_c + li * stride_X_li + lj * stride_X_lj + h * stride_X_h
            x = tl.load(addr_X)
            addr_Y = Y_ptr + b * stride_Y_b + c * stride_Y_c + li * stride_Y_li + lj * stride_Y_lj + h * stride_Y_h
            y = tl.load(addr_Y)
            z = x * y
            addr_Z = Z_ptr + b * stride_Z_b + c * stride_Z_c + li * stride_Z_li + lj * stride_Z_lj + h * stride_Z_h
            tl.store(addr_Z, z)

        grid_mul = (Bsz, Csz, L, L, H)
        elementwise_mul_5d_kernel[grid_mul](
            G, L_t, M,
            Bsz, Csz, L, H,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L_t.stride(0), L_t.stride(1), L_t.stride(2), L_t.stride(3), L_t.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=4, num_stages=2
        )

        # 5) Compute Y_diag by contracting over lj: Y[b, c, li, h, d] = sum_{lj} M[b, c, li, lj, h] * hidden_states[b, c, lj, h, d]
        HS = hidden_states.to(torch.float32)
        Y = torch.empty((Bsz, Csz, L, H, D), dtype=torch.float32, device=hidden_states.device)

        grid_reduce = (Bsz, Csz, L, H, D)
        reduce_Y_diag_kernel[grid_reduce](
            M, HS, Y,
            Bsz, Csz, L, H, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            HS.stride(0), HS.stride(1), HS.stride(2), HS.stride(3), HS.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=4, num_stages=2
        )

        # Cast to bfloat16 to match original return dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
