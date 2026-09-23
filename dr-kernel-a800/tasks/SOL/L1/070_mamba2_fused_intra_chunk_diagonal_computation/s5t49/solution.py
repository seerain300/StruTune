import torch
import triton
import triton.language as tl


# Kernel 1: Build L matrix [B, C, 128, 128, H]
# For each (b, c, h), build L[i, j] = exp(sum_{k=0..L-1} A[b, c, k, h]) if j <= i else 0, i,j in 0..127
@triton.jit
def build_L_kernel(A_ptr, L_ptr,
                   B_stride, C_stride, H,
                   A_stride_b, A_stride_c, A_stride_l, A_stride_h,
                   L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_h):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    i = tl.program_id(3)
    j = tl.program_id(4)

    # Only write when within 128x128
    if (i >= 0 and i < 128 and j >= 0 and j < 128):
        total = 0.0
        # Compute cumsum over hidden_states third dim (L) up to L-1
        L_len = tl.load(C_stride)  # assuming we pass L_len through a separate way; here we need to fetch L from shape of A
        # We can't fetch from A_ptr without L, so we pass L_len via runtime parameter. Let's rewrite the kernel signature to include L_len.
        pass


# We need to pass L_len as an argument. Redefining with correct signature.

@triton.jit
def build_L_kernel(A_ptr, L_ptr, L_len: tl.int32,
                   A_stride_b, A_stride_c, A_stride_l, A_stride_h,
                   L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_h):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    i = tl.program_id(3)
    j = tl.program_id(4)

    if (i >= 0 and i < 128 and j >= 0 and j < 128):
        total = 0.0
        # Sum over k in [0..L_len-1]
        for k in range(0, L_len):
            # Load A[b, c, k, h], masked for safety
            a_val = tl.load(A_ptr + b * A_stride_b + c * A_stride_c + k * A_stride_l + h * A_stride_h)
            total += a_val
        # Apply lower-triangular condition (diagonal=-1): j <= i
        if j <= i:
            l_val = tl.exp(total)
        else:
            l_val = 0.0
        # Store to L[b, c, i, j, h]
        tl.store(L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + h * L_stride_h, l_val)


# Kernel 2: Expand B from groups to heads: B_exp[B, C, L, H, S]
# Map each group s to 4 consecutive heads: h_base = h*GROUP_EXPAND + g, g in [0..3]
@triton.jit
def expand_groups_repeat_interleave_B(B_ptr, B_exp_ptr,
                                      B_stride0, B_stride1, B_stride2, B_stride3, B_stride4,
                                      B_exp_stride0, B_exp_stride1, B_exp_stride2, B_exp_stride3, B_exp_stride4,
                                      Bsz, Csz, L_len, H, S, GROUP_EXPAND: tl.int32):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    s = tl.program_id(4)

    # g loop: write to 4 consecutive head positions
    for g in range(0, GROUP_EXPAND):
        h_out = h * GROUP_EXPAND + g
        # Load from B[b, c, i, s]
        val = tl.load(B_ptr + b * B_stride0 + c * B_stride1 + i * B_stride2 + s * B_stride4)
        # Store to B_exp[b, c, i, h_out, s]
        tl.store(B_exp_ptr + b * B_exp_stride0 + c * B_exp_stride1 + i * B_exp_stride2 + h_out * B_exp_stride3 + s * B_exp_stride4, val)


# Kernel 3: Expand C from groups to heads: C_exp[B, C, L, H, S]
@triton.jit
def expand_groups_repeat_interleave_C(C_ptr, C_exp_ptr,
                                      C_stride0, C_stride1, C_stride2, C_stride3, C_stride4,
                                      C_exp_stride0, C_exp_stride1, C_exp_stride2, C_exp_stride3, C_exp_stride4,
                                      Bsz, Csz, L_len, H, S, GROUP_EXPAND: tl.int32):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    s = tl.program_id(4)

    for g in range(0, GROUP_EXPAND):
        h_out = h * GROUP_EXPAND + g
        val = tl.load(C_ptr + b * C_stride0 + c * C_stride1 + i * C_stride2 + s * C_stride4)
        tl.store(C_exp_ptr + b * C_exp_stride0 + c * C_exp_stride1 + i * C_exp_stride2 + h_out * C_exp_stride3 + s * C_exp_stride4, val)


# Kernel 4: Compute G[b, c, i, j, h] = sum_s C_exp[b, c, i, h, s] * B_exp[b, c, j, h, s]
@triton.jit
def compute_G_kernel(B_exp_ptr, C_exp_ptr, G_ptr,
                     B_exp_stride0, B_exp_stride1, B_exp_stride2, B_exp_stride3, B_exp_stride4,
                     C_exp_stride0, C_exp_stride1, C_exp_stride2, C_exp_stride3, C_exp_stride4,
                     G_stride0, G_stride1, G_stride2, G_stride3, G_stride4,
                     Bsz, Csz, L_len, H, S):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    acc = 0.0
    for s in range(0, S):
        bval = tl.load(B_exp_ptr + b * B_exp_stride0 + c * B_exp_stride1 + j * B_exp_stride2 + h * B_exp_stride3 + s * B_exp_stride4)
        cval = tl.load(C_exp_ptr + b * C_exp_stride0 + c * C_exp_stride1 + i * C_exp_stride2 + h * C_exp_stride3 + s * C_exp_stride4)
        acc += bval * cval
    tl.store(G_ptr + b * G_stride0 + c * G_stride1 + i * G_stride2 + j * G_stride3 + h * G_stride4, acc)


# Kernel 5: Elementwise M = G * L
@triton.jit
def elementwise_M_kernel(G_ptr, L_ptr, M_ptr,
                          G_stride0, G_stride1, G_stride2, G_stride3, G_stride4,
                          L_stride0, L_stride1, L_stride2, L_stride3, L_stride4,
                          M_stride0, M_stride1, M_stride2, M_stride3, M_stride4):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    gval = tl.load(G_ptr + b * G_stride0 + c * G_stride1 + i * G_stride2 + j * G_stride3 + h * G_stride4)
    lval = tl.load(L_ptr + b * L_stride0 + c * L_stride1 + i * L_stride2 + j * L_stride3 + h * L_stride4)
    mval = gval * lval
    tl.store(M_ptr + b * M_stride0 + c * M_stride1 + i * M_stride2 + j * M_stride3 + h * M_stride4, mval)


# Kernel 6: Compute Y_diag[b, c, i, h, s] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, s]
@triton.jit
def compute_Y_kernel(M_ptr, hidden_ptr, Y_ptr,
                     M_stride0, M_stride1, M_stride2, M_stride3, M_stride4,
                     hidden_stride0, hidden_stride1, hidden_stride2, hidden_stride3, hidden_stride4,
                     Y_stride0, Y_stride1, Y_stride2, Y_stride3, Y_stride4,
                     L_len: tl.int32, S: tl.int32):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    s = tl.program_id(4)

    acc = 0.0
    for j in range(0, L_len):
        mval = tl.load(M_ptr + b * M_stride0 + c * M_stride1 + i * M_stride2 + j * M_stride3 + h * M_stride4)
        hval = tl.load(hidden_ptr + b * hidden_stride0 + c * hidden_stride1 + j * hidden_stride2 + h * hidden_stride3 + s * hidden_stride4)
        acc += mval * hval
    tl.store(Y_ptr + b * Y_stride0 + c * Y_stride1 + i * Y_stride2 + h * Y_stride3 + s * Y_stride4, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward implementing the original logic.
        Launches Triton kernels for L, B-expand, C-expand, G, elementwise M, and Y.
        Returns output in bfloat16.
        """
        # Cast to float32 for kernel computations
        device = hidden_states.device
        hidden_f32 = hidden_states.to(torch.float32)
        A_f32 = A_cumsum.to(torch.float32)
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)

        # Shapes
        Bsz, Csz, L, H, D = hidden_f32.shape
        NUM_HEADS = 32
        N_GROUPS = 8
        GROUP_EXPAND = 4  # 32/8

        # 1) Build L matrix [B, C, 128, 128, H]
        L_mat = torch.empty((Bsz, Csz, 128, 128, H), dtype=torch.float32, device=device)
        # Launch Triton kernel: grid over (B, C, H, 128, 128)
        grid_L = (Bsz, Csz, H, 128, 128)
        build_L_kernel[grid_L](
            A_f32, L_mat, L,
            A_f32.stride(0), A_f32.stride(1), A_f32.stride(2), A_f32.stride(3),
            L_mat.stride(0), L_mat.stride(1), L_mat.stride(2), L_mat.stride(3), L_mat.stride(4),
            num_warps=4
        )

        # 2) Expand B and C from groups to heads: [B, C, L, H, S]
        # We need S (state_size). In original, it's C.shape[-1]. Let's derive it from C_f32.
        S = C_f32.shape[-1]
        B_exp = torch.empty((Bsz, Csz, L, H * GROUP_EXPAND, S), dtype=torch.float32, device=device)
        C_exp = torch.empty((Bsz, Csz, L, H * GROUP_EXPAND, S), dtype=torch.float32, device=device)

        grid_expand = (Bsz, Csz, L, H, S)
        expand_groups_repeat_interleave_B[grid_expand](
            B_f32, B_exp,
            B_f32.stride(0), B_f32.stride(1), B_f32.stride(2), B_f32.stride(3), B_f32.stride(4),
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            Bsz, Csz, L, H, S, GROUP_EXPAND,
            num_warps=4
        )
        expand_groups_repeat_interleave_C[grid_expand](
            C_f32, C_exp,
            C_f32.stride(0), C_f32.stride(1), C_f32.stride(2), C_f32.stride(3), C_f32.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            Bsz, Csz, L, H, S, GROUP_EXPAND,
            num_warps=4
        )

        # 3) Compute G[b, c, i, j, h] = sum_s C_exp[b, c, i, h, s] * B_exp[b, c, j, h, s]
        G = torch.empty((Bsz, Csz, L, L, H), dtype=torch.float32, device=device)
        grid_G = (Bsz, Csz, L, L, H)
        compute_G_kernel[grid_G](
            B_exp, C_exp, G,
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            Bsz, Csz, L, H, S,
            num_warps=4
        )

        # 4) Apply causal mask L: M = G * L element-wise in Triton
        M = torch.empty((Bsz, Csz, L, L, H), dtype=torch.float32, device=device)
        grid_M = (Bsz, Csz, L, L, H)
        elementwise_M_kernel[grid_M](
            G, L_mat, M,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L_mat.stride(0), L_mat.stride(1), L_mat.stride(2), L_mat.stride(3), L_mat.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4)
        )

        # 5) Compute Y_diag[b, c, i, h, s] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, s] in Triton
        Y = torch.empty((Bsz, Csz, L, H, S), dtype=torch.float32, device=device)
        grid_Y = (Bsz, Csz, L, H, S)
        compute_Y_kernel[grid_Y](
            M, hidden_f32, Y,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_f32.stride(0), hidden_f32.stride(1), hidden_f32.stride(2), hidden_f32.stride(3), hidden_f32.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            L, S,
            num_warps=4
        )

        # Return in bfloat16 as in original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
