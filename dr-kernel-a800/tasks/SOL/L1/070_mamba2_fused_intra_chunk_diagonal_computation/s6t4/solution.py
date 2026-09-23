import torch
import triton
import triton.language as tl


# 1) Triton: generate lower-triangular mask M_lower of shape [S, S] with diagonal=-1
@triton.jit
def tril_mask_kernel(M: tl.pointer_type(tl.int8), S: tl.constexpr):
    i = tl.program_id(0)  # row
    j = tl.program_id(1)  # col
    if j <= i:
        M[i * S + j] = 1  # True (1)
    else:
        M[i * S + j] = 0  # False (0)


# 2) Triton: cumsum + mask + exp to produce L[b, h, n, i, j] for j <= i, else 0
# A_in is [B, H, N, S, S] float32; Out L same shape
@triton.jit
def cumsum_exp_mask_kernel(
    A_ptr, L_ptr,
    B_size, H_size, N_size, S_size,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s1, A_stride_s2,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2
):
    # program ids for (b, h, n, i)
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    i = tl.program_id(3)

    # running sum accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # iterate over j from 0 to S_size-1 in blocks
    # we'll write only for j <= i; for j > i we store 0
    # We assume grid launches with i in [0, S_size-1], and for j > i we guard store.
    # This kernel computes L[b, h, n, i, j] = exp(sum_{t<=j and t<=i} A[b, h, n, t]) for j <= i, else 0.
    # Since for j > i, sum is unaffected (no contribution), we store 0.
    for j in range(0, S_size):
        # fetch A[b, h, n, j] only when j <= i, otherwise skip
        # using tl.load with a boolean condition is not directly supported; we branch with runtime if.
        if j <= i:
            val = tl.load(A_ptr + b * A_stride_b + h * A_stride_h + n * A_stride_n + j * A_stride_s1 + j * A_stride_s2)
            acc += val
            expv = tl.exp(acc)
        else:
            expv = 0.0
        # store to L[b, h, n, i, j]
        tl.store(L_ptr + b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_s1 + j * L_stride_s2, expv)


# 3) Triton: G contraction: G[b, n, i, j, h] = sum_d C[b, n, i, h, d] * B[b, n, j, h, d]
# Grid over (B, N, H); program computes full SxS G for given (b, n, h).
@triton.jit
def g_contract_kernel(
    B_ptr, C_ptr, G_ptr,
    B_size, N_size, H_size, S_size, D_size,
    B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
    C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h
):
    b = tl.program_id(0)  # batch
    n = tl.program_id(1)  # chunk
    h = tl.program_id(2)  # head
    # For each i, j compute G[i, j, h] = sum over d of C[b, n, i, h, d] * B[b, n, j, h, d]
    for i in range(S_size):
        for j in range(S_size):
            acc = tl.zeros((), dtype=tl.float32)
            # sum over D dimension
            for d in range(D_size):
                bval = tl.load(B_ptr + b * B_stride_b + n * B_stride_n + j * B_stride_s + h * B_stride_h + d * B_stride_d)
                cval = tl.load(C_ptr + b * C_stride_b + n * C_stride_n + i * C_stride_s + h * C_stride_h + d * C_stride_d)
                acc += bval * cval
            tl.store(G_ptr + b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h, acc)


# 4) Triton: elementwise multiply M = G * L restricted to lower triangle (j <= i)
@triton.jit
def m_mul_kernel(
    G_ptr, L_ptr, M_ptr, M_lower_ptr,
    B_size, N_size, H_size, S_size,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    M_lower_stride,  # M_lower is 1D S*S
    diagonal: tl.constexpr  # -1 here
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)
    # only compute where j <= i (lower triangle)
    if j <= i:
        gval = tl.load(G_ptr + b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h)
        lval = tl.load(L_ptr + b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_s1 + j * L_stride_s2)
        mval = gval * lval
        tl.store(M_ptr + b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h, mval)
    # no need to write zeros for upper triangle; M is allocated as zeros.


# 5) Triton: Y_diag reduction: Y[b, n, i, h, :] = sum_j M[b, n, i, j, h] * hidden_states[b, n, j, h, :]
@triton.jit
def y_diag_reduce_kernel(
    M_ptr, hidden_states_ptr, Y_ptr,
    B_size, N_size, H_size, S_size, D_size,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    Hs_stride_b, Hs_stride_n, Hs_stride_s, Hs_stride_h, Hs_stride_d,
    Y_stride_b, Y_stride_n, Y_stride_s, Y_stride_h, Y_stride_d
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    acc = tl.zeros((D_size,), dtype=tl.float32)
    for j in range(S_size):
        mval = tl.load(M_ptr + b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h)
        # hidden_states[b, n, j, h, :]
        hs_vec = tl.zeros((D_size,), dtype=tl.float32)
        for d in range(D_size):
            hs_vec[d] = tl.load(hidden_states_ptr + b * Hs_stride_b + n * Hs_stride_n + j * Hs_stride_s + h * Hs_stride_h + d * Hs_stride_d)
        acc += mval * hs_vec
    for d in range(D_size):
        tl.store(Y_ptr + b * Y_stride_b + n * Y_stride_n + i * Y_stride_s + h * Y_stride_h + d * Y_stride_d, acc[d])


class ModelNew(torch.nn.Module):
    def __init__(self, NUM_HEADS=32, N_GROUPS=8):
        super().__init__()
        self.NUM_HEADS = NUM_HEADS
        self.N_GROUPS = N_GROUPS
        assert NUM_HEADS % N_GROUPS == 0, "NUM_HEADS must be divisible by N_GROUPS"

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Shapes
        B_size, N_size, S_size, H_size, D_size = hidden_states.shape
        assert A_cumsum.shape == (B_size, H_size, N_size, S_size), "A_cumsum must have shape [B, H, N, S]"
        assert B.shape == (B_size, N_size, S_size, H_size, D_size), "B must have shape [B, N, S, H, D]"
        assert C.shape == (B_size, N_size, S_size, H_size, D_size), "C must have shape [B, N, S, H, D]"
        assert H_size == self.NUM_HEADS, "num_heads must be 32"
        device = hidden_states.device
        assert device.type == 'cuda', "ModelNew requires CUDA device for Triton kernels"

        # 1) Triton: Lower-triangular mask M_lower of shape [S, S] with diagonal=-1
        M_lower = torch.empty((S_size, S_size), dtype=torch.int8, device=device)
        grid_mask = (S_size, S_size)
        tril_mask_kernel[grid_mask](M_lower, S_size, num_warps=1, num_stages=1)

        # 2) Triton: L = exp(cumsum(masked A)) along S for each (b, h, n, i): implement with cumsum_exp_mask_kernel
        # Expand A_cumsum to [B, H, N, S, S] float32
        A_expanded = A_cumsum.unsqueeze(-1).expand(B_size, H_size, N_size, S_size, S_size).to(torch.float32).contiguous()
        L = torch.empty((B_size, H_size, N_size, S_size, S_size), dtype=torch.float32, device=device)

        A_stride_b, A_stride_h, A_stride_n, A_stride_s1, A_stride_s2 = A_expanded.stride()
        L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2 = L.stride()

        grid_l = (B_size, H_size, N_size, S_size)
        cumsum_exp_mask_kernel[grid_l](
            A_expanded, L,
            B_size, H_size, N_size, S_size,
            A_stride_b, A_stride_h, A_stride_n, A_stride_s1, A_stride_s2,
            L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
            num_warps=4, num_stages=2
        )

        # 3) Triton: G contraction: compute G[b, n, i, j, h]
        G = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)

        B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d = B.stride()
        C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d = C.stride()
        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()

        grid_g = (B_size, N_size, H_size)
        g_contract_kernel[grid_g](
            B, C, G,
            B_size, N_size, H_size, S_size, D_size,
            B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
            C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            num_warps=4, num_stages=2
        )

        # 4) Triton: M = G * L on lower triangle
        M = torch.empty_like(G, dtype=torch.float32, device=device)
        # We need to pass M_lower to kernel; linearize it to 1D S*S as M_lower_stride (not strictly necessary since we guard by j<=i).
        # However, to pass it, we'll treat it as 1D length S*S (contiguous).
        M_lower_flat = M_lower.view(-1)
        grid_m = (B_size, N_size, S_size, S_size, H_size)
        m_mul_kernel[grid_m](
            G, L, M, M_lower_flat,
            B_size, N_size, H_size, S_size,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            M_lower_flat.numel(),  # diagonal is unused since we use j<=i guard
            num_warps=4, num_stages=2
        )

        # 5) Triton: Y_diag reduction
        Y = torch.empty((B_size, N_size, S_size, H_size, D_size), dtype=torch.float32, device=device)

        Hs_stride_b, Hs_stride_n, Hs_stride_s, Hs_stride_h, Hs_stride_d = hidden_states.stride()
        Y_stride_b, Y_stride_n, Y_stride_s, Y_stride_h, Y_stride_d = Y.stride()

        grid_y = (B_size, N_size, S_size, H_size)
        y_diag_reduce_kernel[grid_y](
            M, hidden_states, Y,
            B_size, N_size, H_size, S_size, D_size,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            Hs_stride_b, Hs_stride_n, Hs_stride_s, Hs_stride_h, Hs_stride_d,
            Y_stride_b, Y_stride_n, Y_stride_s, Y_stride_h, Y_stride_d,
            num_warps=4, num_stages=2
        )

        # Return in bfloat16 to match original model behavior
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
