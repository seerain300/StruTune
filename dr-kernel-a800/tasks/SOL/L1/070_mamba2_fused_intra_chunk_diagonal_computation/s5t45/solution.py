import torch
import triton
import triton.language as tl


# Kernel 1: Build L matrix [B, C, 128, 128, H] with lower-triangular (diagonal=-1), L[i, j] = exp(sum_{k=0..L-1} A[b,c,k,h]) if j <= i, else 0
@triton.jit
def build_L_kernel(A_ptr, L_ptr,
                   B, C, H,
                   stride_A_b, stride_A_c, stride_A_l, stride_A_h,
                   stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
                   L_len):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    i = tl.program_id(3)  # row index
    j = tl.program_id(4)  # col index

    # bounds check: we launch grid exactly 128x128, so no out-of-bounds needed
    total = 0.0
    # cumsum over actual L_len
    # Note: we assume L_len <= 128; if larger, we can't represent with 128x128. The original code uses 128 in mask; here we pad L_len to 128 effectively by using L_len as sum length.
    for k in range(0, L_len):
        a_val = tl.load(A_ptr + b * stride_A_b + c * stride_A_c + k * stride_A_l + h * stride_A_h)
        total += a_val

    apply = j <= i  # diagonal = -1
    exp_val = tl.exp(total)
    L_val = tl.where(apply, exp_val, 0.0)

    tl.store(L_ptr + b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h, L_val)


# Kernel 2: Expand B from groups to heads (repeat_interleave along group dim)
# B_exp[b, c, i, h, s] = B[b, c, i, g, s], where h = g * GROUP_EXPAND
@triton.jit
def expand_groups_repeat_interleave_B(B_ptr, B_exp_ptr,
                                      B_stride_b, B_stride_c, B_stride_l, B_stride_g, B_stride_s,
                                      Bexp_stride_b, Bexp_stride_c, Bexp_stride_l, Bexp_stride_h, Bexp_stride_s,
                                      G, S, GROUP_EXPAND):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # but we will ignore i and use loop over i; here we treat as scalar program over (b,c)
    h = tl.program_id(3)
    s = tl.program_id(4)  # but we will ignore s and use loop over s; here we treat as scalar program over (b,c)

    # We will loop over i and s inside the kernel to avoid large grids
    # For clarity and safety, we launch with grid (B, C, 1, 1, 1) and iterate.
    # However Triton requires 1D grid dims. So we rewrite as:
    # We launch with grid (B, C) and loop over i and s using tl.range, but Triton JIT needs explicit loops per program.
    # Better: launch with grid (B, C, L, H, S). That's fine. So we restate:
    # We will use 5D grid (b, c, i, h, s) with provided ranges.

    # Load from B and store to B_exp at expanded head index
    # We will use current program ids directly: (b, c, i, h, s)
    g = h // GROUP_EXPAND  # map expanded head back to group index
    val = tl.load(B_ptr + b * B_stride_b + c * B_stride_c + i * B_stride_l + g * B_stride_g + s * B_stride_s)
    tl.store(B_exp_ptr + b * Bexp_stride_b + c * Bexp_stride_c + i * Bexp_stride_l + h * Bexp_stride_h + s * Bexp_stride_s, val)


# Kernel 3: Expand C from groups to heads similarly
@triton.jit
def expand_groups_repeat_interleave_C(C_ptr, C_exp_ptr,
                                      C_stride_b, C_stride_c, C_stride_l, C_stride_g, C_stride_s,
                                      Cexp_stride_b, Cexp_stride_c, Cexp_stride_l, Cexp_stride_h, Cexp_stride_s,
                                      G, S, GROUP_EXPAND):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    s = tl.program_id(4)
    g = h // GROUP_EXPAND
    val = tl.load(C_ptr + b * C_stride_b + c * C_stride_c + i * C_stride_l + g * C_stride_g + s * C_stride_s)
    tl.store(C_exp_ptr + b * Cexp_stride_b + c * Cexp_stride_c + i * Cexp_stride_l + h * Cexp_stride_h + s * Cexp_stride_s, val)


# Kernel 4: Compute G[b, c, i, j, h] = sum_s C_exp[b, c, i, h, s] * B_exp[b, c, j, h, s]
@triton.jit
def compute_G_kernel(B_exp_ptr, C_exp_ptr, G_ptr,
                     Be_stride_b, Be_stride_c, Be_stride_l, Be_stride_h, Be_stride_s,
                     Ce_stride_b, Ce_stride_c, Ce_stride_l, Ce_stride_h, Ce_stride_s,
                     G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_h,
                     L, S):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    acc = 0.0
    for s in range(0, S):
        bval = tl.load(B_exp_ptr + b * Be_stride_b + c * Be_stride_c + i * Be_stride_l + h * Be_stride_h + s * Be_stride_s)
        cval = tl.load(C_exp_ptr + b * Ce_stride_b + c * Ce_stride_c + i * Ce_stride_l + h * Ce_stride_h + s * Ce_stride_s)
        acc += bval * cval
    tl.store(G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + h * G_stride_h, acc)


# Kernel 5: Compute Y_diag[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, d]
@triton.jit
def compute_Y_kernel(M_ptr, hidden_ptr, Y_ptr,
                     M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_h,
                     hidden_stride_b, hidden_stride_c, hidden_stride_l, hidden_stride_h, hidden_stride_d,
                     Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_h, Y_stride_d,
                     L, D):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in range(0, L):
        mval = tl.load(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + h * M_stride_h)
        hval = tl.load(hidden_ptr + b * hidden_stride_b + c * hidden_stride_c + j * hidden_stride_l + h * hidden_stride_h + d * hidden_stride_d)
        acc += mval * hval
    tl.store(Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + h * Y_stride_h + d * Y_stride_d, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward implementing the original logic.
        - Launches Triton kernels for L, B-expand, C-expand, G, and Y.
        - No torch ops in forward for heavy math.
        """
        # Constants
        NUM_HEADS = 32
        N_GROUPS = 8
        GROUP_EXPAND = 4
        L_mat_size = 128  # original code uses 128x128 mask

        # Input shapes
        B, C, L, H, D = hidden_states.shape

        # Prepare tensors
        hidden_f32 = hidden_states.to(torch.float32)
        A_f32 = A_cumsum.to(torch.float32)
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)

        # 1) Build L matrix [B, C, 128, 128, H] via Triton
        L_ptr = torch.empty((B, C, L_mat_size, L_mat_size, H), dtype=torch.float32, device=hidden_f32.device)

        # Launch grid over (b, c, h, i, j)
        grid_L = (B, C, H, L_mat_size, L_mat_size)
        build_L_kernel[grid_L](
            A_f32, L_ptr,
            B, C, H,
            A_f32.stride(0), A_f32.stride(1), A_f32.stride(2), A_f32.stride(3),
            L_ptr.stride(0), L_ptr.stride(1), L_ptr.stride(2), L_ptr.stride(3), L_ptr.stride(4),
            L,  # L_len
        )

        # 2) Expand B and C from groups to heads in Triton
        S = B_f32.shape[-1]
        B_exp = torch.empty((B, C, L, NUM_HEADS, S), dtype=torch.float32, device=hidden_f32.device)
        C_exp = torch.empty((B, C, L, NUM_HEADS, S), dtype=torch.float32, device=hidden_f32.device)

        B_stride_b, B_stride_c, B_stride_l, B_stride_g, B_stride_s = B_f32.stride()
        Bexp_stride_b, Bexp_stride_c, Bexp_stride_l, Bexp_stride_h, Bexp_stride_s = B_exp.stride()

        C_stride_b, C_stride_c, C_stride_l, C_stride_g, C_stride_s = C_f32.stride()
        Cexp_stride_b, Cexp_stride_c, Cexp_stride_l, Cexp_stride_h, Cexp_stride_s = C_exp.stride()

        grid_expand = (B, C, L, NUM_HEADS, S)
        expand_groups_repeat_interleave_B[grid_expand](
            B_f32, B_exp,
            B_stride_b, B_stride_c, B_stride_l, B_stride_g, B_stride_s,
            Bexp_stride_b, Bexp_stride_c, Bexp_stride_l, Bexp_stride_h, Bexp_stride_s,
            N_GROUPS, S, GROUP_EXPAND
        )

        expand_groups_repeat_interleave_C[grid_expand](
            C_f32, C_exp,
            C_stride_b, C_stride_c, C_stride_l, C_stride_g, C_stride_s,
            Cexp_stride_b, Cexp_stride_c, Cexp_stride_l, Cexp_stride_h, Cexp_stride_s,
            N_GROUPS, S, GROUP_EXPAND
        )

        # 3) Compute G = contraction over S in Triton
        G = torch.empty((B, C, L, L, H), dtype=torch.float32, device=hidden_f32.device)
        Be_stride_b, Be_stride_c, Be_stride_l, Be_stride_h, Be_stride_s = B_exp.stride()
        Ce_stride_b, Ce_stride_c, Ce_stride_l, Ce_stride_h, Ce_stride_s = C_exp.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_h = G.stride()

        grid_G = (B, C, L, L, H)
        compute_G_kernel[grid_G](
            B_exp, C_exp, G,
            Be_stride_b, Be_stride_c, Be_stride_l, Be_stride_h, Be_stride_s,
            Ce_stride_b, Ce_stride_c, Ce_stride_l, Ce_stride_h, Ce_stride_s,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_h,
            L, S
        )

        # 4) Apply causal mask L: M = G * L
        # We implement elementwise multiply in Triton: M[b, c, i, j, h] = G[b, c, i, j, h] * L[b, c, i, j, h]
        M = torch.empty((B, C, L, L, H), dtype=torch.float32, device=hidden_f32.device)
        G_ptr = G  # we reuse G as source
        L_strides = L_ptr.stride()
        M_strides = M.stride()
        G_strides = G.stride()

        grid_M = (B, C, L, L, H)
        # Note: Triton requires pointer, so we pass G and L as inputs; we'll compute M using a similar kernel if needed.
        # For simplicity, perform torch.mul here since sizes are small; or implement kernel:
        # We implement a simple elementwise kernel using precomputed G and L.
        # We'll create a kernel that multiplies two tensors element-wise.
        @triton.jit
        def elementwise_mul_kernel(A_ptr, B_ptr, C_ptr,
                                    A_stride0, A_stride1, A_stride2, A_stride3, A_stride4,
                                    B_stride0, B_stride1, B_stride2, B_stride3, B_stride4,
                                    C_stride0, C_stride1, C_stride2, C_stride3, C_stride4,
                                    size):
            b = tl.program_id(0)
            c = tl.program_id(1)
            i = tl.program_id(2)
            j = tl.program_id(3)
            h = tl.program_id(4)
            a = tl.load(A_ptr + b * A_stride0 + c * A_stride1 + i * A_stride2 + j * A_stride3 + h * A_stride4)
            bval = tl.load(B_ptr + b * B_stride0 + c * B_stride1 + i * B_stride2 + j * B_stride3 + h * B_stride4)
            cval = a * bval
            tl.store(C_ptr + b * C_stride0 + c * C_stride1 + i * C_stride2 + j * C_stride3 + h * C_stride4, cval)

        elementwise_mul_kernel[grid_M](
            G, L_ptr, M,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_h,
            L_ptr.stride(0), L_ptr.stride(1), L_ptr.stride(2), L_ptr.stride(3), L_ptr.stride(4),
            M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_h,
            B, C, L, L, H
        )

        # 5) Compute Y_diag: reduction over j
        Y = torch.empty((B, C, L, H, D), dtype=torch.float32, device=hidden_f32.device)
        M_strides = M.stride()
        hidden_strides = hidden_f32.stride()
        Y_strides = Y.stride()

        grid_Y = (B, C, L, H, D)
        compute_Y_kernel[grid_Y](
            M, hidden_f32, Y,
            M_strides[0], M_strides[1], M_strides[2], M_strides[3], M_strides[4],
            hidden_strides[0], hidden_strides[1], hidden_strides[2], hidden_strides[3], hidden_strides[4],
            Y_strides[0], Y_strides[1], Y_strides[2], Y_strides[3], Y_strides[4],
            L, D
        )

        # Return in bfloat16 as original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
