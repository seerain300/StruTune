import torch
import triton
import triton.language as tl


@triton.jit
def build_L_kernel(A_ptr, L_ptr,
                   A_stride_b, A_stride_c, A_stride_l, A_stride_h,
                   L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_h,
                   L_len: tl.int32):
    # program ids: (b, c, h, i, j)
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    i = tl.program_id(3)
    j = tl.program_id(4)

    # Initialize cumsum for row i
    total = 0.0
    # Sum over k in [0..L_len-1]
    for k in range(0, L_len):
        a_val = tl.load(A_ptr + b * A_stride_b + c * A_stride_c + k * A_stride_l + h * A_stride_h)
        total += a_val

    # If j <= i, set L[i, j] = exp(total), else 0
    exp_val = tl.exp(total)
    val = tl.where(j <= i, exp_val, 0.0)

    # Store to L[b, c, i, j, h]
    tl.store(L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + h * L_stride_h, val)


@triton.jit
def expand_groups_repeat_interleave(A_ptr, B_ptr,
                                     A_stride_b, A_stride_c, A_stride_l, A_stride_g, A_stride_s,
                                     B_stride_b, B_stride_c, B_stride_l, B_stride_h, B_stride_s,
                                     Bsz: tl.int32, Csz: tl.int32, L: tl.int32, H: tl.int32, G: tl.int32, S: tl.int32, GROUP_EXPAND: tl.int32):
    # program ids: (b, c, l, h, s)
    b = tl.program_id(0)
    c = tl.program_id(1)
    l = tl.program_id(2)
    h = tl.program_id(3)
    s = tl.program_id(4)

    # Compute source group index g for this head position h
    g = h // GROUP_EXPAND  # since H = G * GROUP_EXPAND

    # Load from A_ptr
    a_val = tl.load(A_ptr + b * A_stride_b + c * A_stride_c + l * A_stride_l + g * A_stride_g + s * A_stride_s)

    # Store to B_ptr at corresponding head position
    tl.store(B_ptr + b * B_stride_b + c * B_stride_c + l * B_stride_l + h * B_stride_h + s * B_stride_s, a_val)


@triton.jit
def compute_G_kernel(B_exp_ptr, C_exp_ptr, G_ptr,
                     B_stride_b, B_stride_c, B_stride_l, B_stride_h, B_stride_s,
                     C_stride_b, C_stride_c, C_stride_l, C_stride_h, C_stride_s,
                     G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_h,
                     Bsz: tl.int32, Csz: tl.int32, L: tl.int32, H: tl.int32, S: tl.int32):
    # program ids: (b, c, i, j, h)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    acc = 0.0
    # Loop over S dimension to compute outer product contraction
    for s in range(0, S):
        bval = tl.load(B_exp_ptr + b * B_stride_b + c * B_stride_c + i * B_stride_l + h * B_stride_h + s * B_stride_s)
        cval = tl.load(C_exp_ptr + b * C_stride_b + c * C_stride_c + j * C_stride_l + h * C_stride_h + s * C_stride_s)
        acc += bval * cval

    # Store G[b, c, i, j, h]
    tl.store(G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + h * G_stride_h, acc)


@triton.jit
def elementwise_M_kernel(G_ptr, L_ptr, M_ptr,
                          G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_h,
                          L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_h,
                          M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_h,
                          Bsz: tl.int32, Csz: tl.int32, L_len: tl.int32, H: tl.int32):
    # program ids: (b, c, i, j, h)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    gval = tl.load(G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + h * G_stride_h)
    lval = tl.load(L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + h * L_stride_h)
    mval = gval * lval
    tl.store(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + h * M_stride_h, mval)


@triton.jit
def compute_Y_kernel(M_ptr, hidden_ptr, Y_ptr,
                     M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_h,
                     hidden_stride_b, hidden_stride_c, hidden_stride_l, hidden_stride_h, hidden_stride_d,
                     Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_h, Y_stride_d,
                     L_len: tl.int32):
    # program ids: (b, c, i, h, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in range(0, L_len):
        mval = tl.load(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + h * M_stride_h)
        hval = tl.load(hidden_ptr + b * hidden_stride_b + c * hidden_stride_c + j * hidden_stride_l + h * hidden_stride_h + d * hidden_stride_d)
        acc += mval * hval
    tl.store(Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + h * Y_stride_h + d * Y_stride_d, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward implementing the original logic.
        Launches Triton kernels for L, B-expand, C-expand, G, elementwise M, and Y.
        Returns output in bfloat16.
        """
        # Shapes
        Bsz, Csz, L, H, D = hidden_states.shape
        # Constants
        NUM_HEADS = 32
        N_GROUPS = 8
        GROUP_EXPAND = 4  # NUM_HEADS // N_GROUPS

        # Dtype: original returns bfloat16, but we compute in float32 for stability
        hidden_f32 = hidden_states.to(torch.float32)
        A_f32 = A_cumsum.to(torch.float32)
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)

        # Output tensor: Y_diag [B, C, L, H, D] float32
        Y = torch.empty((Bsz, Csz, L, H, D), dtype=torch.float32, device=hidden_f32.device)

        # 1) Build L matrix [B, C, 128, 128, H] in Triton
        L_mat_size = 128
        L_ptr = torch.empty((Bsz, Csz, L_mat_size, L_mat_size, H), dtype=torch.float32, device=hidden_f32.device)

        grid_L = (Bsz, Csz, H, L_mat_size, L_mat_size)
        build_L_kernel[grid_L](
            A_f32, L_ptr,
            A_f32.stride(0), A_f32.stride(1), A_f32.stride(2), A_f32.stride(3),
            L_ptr.stride(0), L_ptr.stride(1), L_ptr.stride(2), L_ptr.stride(3), L_ptr.stride(4),
            L
        )

        # 2) Expand B and C from groups to heads in Triton (S is last dim)
        # Compute S size from B or C
        S = B_f32.shape[-1]

        # Allocate expanded B and C
        B_exp = torch.empty((Bsz, Csz, L, NUM_HEADS, S), dtype=torch.float32, device=hidden_f32.device)
        C_exp = torch.empty((Bsz, Csz, L, NUM_HEADS, S), dtype=torch.float32, device=hidden_f32.device)

        grid_expand = (Bsz, Csz, L, NUM_HEADS, S)
        expand_groups_repeat_interleave[grid_expand](
            B_f32, B_exp,
            B_f32.stride(0), B_f32.stride(1), B_f32.stride(2), B_f32.stride(3), B_f32.stride(4),
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            Bsz, Csz, L, H, N_GROUPS, S, GROUP_EXPAND
        )
        expand_groups_repeat_interleave[grid_expand](
            C_f32, C_exp,
            C_f32.stride(0), C_f32.stride(1), C_f32.stride(2), C_f32.stride(3), C_f32.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            Bsz, Csz, L, H, N_GROUPS, S, GROUP_EXPAND
        )

        # 3) Compute G[b, c, i, j, h] = sum_s C_exp[b, c, i, h, s] * B_exp[b, c, j, h, s] in Triton
        G = torch.empty((Bsz, Csz, L, L, H), dtype=torch.float32, device=hidden_f32.device)
        grid_G = (Bsz, Csz, L, L, H)
        compute_G_kernel[grid_G](
            B_exp, C_exp, G,
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            Bsz, Csz, L, H, S
        )

        # 4) Apply causal mask L: M = G * L element-wise in Triton
        M = torch.empty((Bsz, Csz, L, L, H), dtype=torch.float32, device=hidden_f32.device)
        grid_M = (Bsz, Csz, L, L, H)
        elementwise_M_kernel[grid_M](
            G, L_ptr, M,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L_ptr.stride(0), L_ptr.stride(1), L_ptr.stride(2), L_ptr.stride(3), L_ptr.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            Bsz, Csz, L, H
        )

        # 5) Compute Y_diag[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, d] in Triton
        grid_Y = (Bsz, Csz, L, H, D)
        compute_Y_kernel[grid_Y](
            M, hidden_f32, Y,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_f32.stride(0), hidden_f32.stride(1), hidden_f32.stride(2), hidden_f32.stride(3), hidden_f32.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            L
        )

        # Return in bfloat16 as in original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
