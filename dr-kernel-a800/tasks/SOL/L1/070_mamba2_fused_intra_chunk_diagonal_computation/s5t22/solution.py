import torch
import triton
import triton.language as tl

# Constants as in the original model
NUM_HEADS = 32
N_GROUPS = 8
GROUP_EXPAND = 4
L_MAT_SIZE = 128  # hardcoded as in original code

@triton.jit
def build_L_kernel(
    A_ptr,  # [B, C, L, H]
    L_ptr,  # [B, C, 128, 128, H]
    B_SIZE: tl.constexpr,
    C_SIZE: tl.constexpr,
    L_DIM: tl.constexpr,  # actual chunk size per (b, c)
    H_SIZE: tl.constexpr,
):
    # Each program handles one (b, c, h): fill L[i, j] for i,j < 128
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Compute total = sum over k in 0..L_DIM-1 of A[b, c, k, h]
    total = tl.zeros((), dtype=tl.float32)
    for k in range(0, L_DIM):
        a_off = (
            pid_b * A_ptr.stride(0) +
            pid_c * A_ptr.stride(1) +
            k * A_ptr.stride(2) +
            pid_h * A_ptr.stride(3)
        )
        a_val = tl.load(A_ptr, offsets=a_off)
        total += a_val

    # Fill L[i, j] for i,j < 128: if j <= i, L[i,j] = exp(total); else 0
    for ii in range(0, L_MAT_SIZE):
        for jj in range(0, L_MAT_SIZE):
            cond = jj <= ii
            l_val = tl.exp(total) if cond else 0.0
            off = (
                pid_b * L_ptr.stride(0) +
                pid_c * L_ptr.stride(1) +
                ii * L_ptr.stride(2) +
                jj * L_ptr.stride(3) +
                pid_h * L_ptr.stride(4)
            )
            tl.store(L_ptr, l_val, offsets=off)


@triton.jit
def expand_groups_kernel(
    B_ptr,  # [B, C, L, G, S]
    B_exp_ptr,  # [B, C, L, H, S]
    C_ptr,  # [B, C, L, G, S]
    C_exp_ptr,  # [B, C, L, H, S]
    B_SIZE: tl.constexpr,
    C_SIZE: tl.constexpr,
    L_DIM: tl.constexpr,
    G_SIZE: tl.constexpr,
    S_SIZE: tl.constexpr,
    H_SIZE: tl.constexpr,
):
    # Grid is (B, C, L, H, S). Each program handles one (b, c, l, h, s).
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_s = tl.program_id(4)

    # For expansion: h maps to group g by repeat_interleave with GROUP_EXPAND=4
    g = pid_h // GROUP_EXPAND  # since H = G * GROUP_EXPAND

    # Load from B/C at group g
    b_off = (
        pid_b * B_ptr.stride(0) +
        pid_c * B_ptr.stride(1) +
        pid_l * B_ptr.stride(2) +
        g * B_ptr.stride(3) +
        pid_s * B_ptr.stride(4)
    )
    c_off = (
        pid_b * C_ptr.stride(0) +
        pid_c * C_ptr.stride(1) +
        pid_l * C_ptr.stride(2) +
        g * C_ptr.stride(3) +
        pid_s * C_ptr.stride(4)
    )
    b_val = tl.load(B_ptr, offsets=b_off)
    c_val = tl.load(C_ptr, offsets=c_off)

    # Store into B_exp/C_exp at head h
    be_off = (
        pid_b * B_exp_ptr.stride(0) +
        pid_c * B_exp_ptr.stride(1) +
        pid_l * B_exp_ptr.stride(2) +
        pid_h * B_exp_ptr.stride(3) +
        pid_s * B_exp_ptr.stride(4)
    )
    ce_off = (
        pid_b * C_exp_ptr.stride(0) +
        pid_c * C_exp_ptr.stride(1) +
        pid_l * C_exp_ptr.stride(2) +
        pid_h * C_exp_ptr.stride(3) +
        pid_s * C_exp_ptr.stride(4)
    )
    tl.store(B_exp_ptr, b_val, offsets=be_off)
    tl.store(C_exp_ptr, c_val, offsets=ce_off)


@triton.jit
def compute_G_kernel(
    B_exp_ptr,  # [B, C, L, H, S]
    C_exp_ptr,  # [B, C, L, H, S]
    G_ptr,      # [B, C, L, L, H]
    B_SIZE: tl.constexpr,
    C_SIZE: tl.constexpr,
    L_DIM: tl.constexpr,
    H_SIZE: tl.constexpr,
    S_SIZE: tl.constexpr,
):
    # Grid: (B, C, H, L, L) => each program computes one G[i, j, h] for given (b, c)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_i = tl.program_id(3)
    pid_j = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    for s in range(0, S_SIZE):
        be_off = (
            pid_b * B_exp_ptr.stride(0) +
            pid_c * B_exp_ptr.stride(1) +
            pid_i * B_exp_ptr.stride(2) +
            pid_h * B_exp_ptr.stride(3) +
            s * B_exp_ptr.stride(4)
        )
        bval = tl.load(B_exp_ptr, offsets=be_off)

        ce_off = (
            pid_b * C_exp_ptr.stride(0) +
            pid_c * C_exp_ptr.stride(1) +
            pid_j * C_exp_ptr.stride(2) +
            pid_h * C_exp_ptr.stride(3) +
            s * C_exp_ptr.stride(4)
        )
        cval = tl.load(C_exp_ptr, offsets=ce_off)

        acc += bval * cval

    # Store G[b, c, i, j, h] = acc
    g_off = (
        pid_b * G_ptr.stride(0) +
        pid_c * G_ptr.stride(1) +
        pid_i * G_ptr.stride(2) +
        pid_j * G_ptr.stride(3) +
        pid_h * G_ptr.stride(4)
    )
    tl.store(G_ptr, acc, offsets=g_off)


@triton.jit
def elementwise_mul_5d_kernel(
    G_ptr,  # [B, C, L, L, H]
    L_ptr,  # [B, C, 128, 128, H]
    M_ptr,  # [B, C, L, L, H]
    B_SIZE: tl.constexpr,
    C_SIZE: tl.constexpr,
    L_DIM: tl.constexpr,
    H_SIZE: tl.constexpr,
):
    # Grid: (B, C, L, L, H)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    g_off = (
        pid_b * G_ptr.stride(0) +
        pid_c * G_ptr.stride(1) +
        pid_i * G_ptr.stride(2) +
        pid_j * G_ptr.stride(3) +
        pid_h * G_ptr.stride(4)
    )
    g_val = tl.load(G_ptr, offsets=g_off)

    l_off = (
        pid_b * L_ptr.stride(0) +
        pid_c * L_ptr.stride(1) +
        pid_i * L_ptr.stride(2) +
        pid_j * L_ptr.stride(3) +
        pid_h * L_ptr.stride(4)
    )
    l_val = tl.load(L_ptr, offsets=l_off)

    m_val = g_val * l_val
    m_off = (
        pid_b * M_ptr.stride(0) +
        pid_c * M_ptr.stride(1) +
        pid_i * M_ptr.stride(2) +
        pid_j * M_ptr.stride(3) +
        pid_h * M_ptr.stride(4)
    )
    tl.store(M_ptr, m_val, offsets=m_off)


@triton.jit
def reduce_j_kernel(
    M_ptr,        # [B, C, L, L, H]
    hidden_ptr,   # [B, C, L, H, D]
    Y_ptr,        # [B, C, L, H, D]
    B_SIZE: tl.constexpr,
    C_SIZE: tl.constexpr,
    L_DIM: tl.constexpr,
    H_SIZE: tl.constexpr,
    D_SIZE: tl.constexpr,
):
    # Grid: (B, C, L, H, D). Each program computes Y[b, c, i, h, d] over j loop.
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_d = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    for j in range(0, L_DIM):
        m_off = (
            pid_b * M_ptr.stride(0) +
            pid_c * M_ptr.stride(1) +
            pid_i * M_ptr.stride(2) +
            j * M_ptr.stride(3) +
            pid_h * M_ptr.stride(4)
        )
        m_val = tl.load(M_ptr, offsets=m_off)

        hs_off = (
            pid_b * hidden_ptr.stride(0) +
            pid_c * hidden_ptr.stride(1) +
            j * hidden_ptr.stride(2) +
            pid_h * hidden_ptr.stride(3) +
            pid_d * hidden_ptr.stride(4)
        )
        hs_val = tl.load(hidden_ptr, offsets=hs_off)

        acc += m_val * hs_val

    y_off = (
        pid_b * Y_ptr.stride(0) +
        pid_c * Y_ptr.stride(1) +
        pid_i * Y_ptr.stride(2) +
        pid_h * Y_ptr.stride(3) +
        pid_d * Y_ptr.stride(4)
    )
    tl.store(Y_ptr, acc, offsets=y_off)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
        """
        hidden_states: [B, C, L, H, D]
        A_cumsum:      [B, C, L, H]
        B:             [B, C, L, G, S]
        C:             [B, C, L, G, S]
        Returns:       [B, C, L, H, D]
        """
        B_SIZE, C_SIZE, L_DIM, H_SIZE, D_SIZE = hidden_states.shape
        assert H_SIZE == NUM_HEADS, "H_SIZE must equal NUM_HEADS=32"
        device = hidden_states.device

        # 1) Build L matrix [B, C, 128, 128, H] via Triton kernel
        A_f32 = A_cumsum.to(torch.float32)
        L = torch.empty((B_SIZE, C_SIZE, L_MAT_SIZE, L_MAT_SIZE, H_SIZE), dtype=torch.float32, device=device)

        grid_L = (B_SIZE, C_SIZE, H_SIZE)
        build_L_kernel[grid_L](
            A_f32, L,
            B_SIZE=B_SIZE, C_SIZE=C_SIZE, L_DIM=L_DIM, H_SIZE=H_SIZE,
            num_warps=1, num_stages=1
        )

        # 2) Expand B and C from groups to heads in Triton
        B_ptr = B.to(torch.float32)
        C_ptr = C.to(torch.float32)
        S_SIZE = B_ptr.shape[-1]
        G_SIZE = B_ptr.shape[3]
        assert G_SIZE == N_GROUPS and G_SIZE == C_ptr.shape[3], "Group size must equal N_GROUPS=8"

        B_exp = torch.empty((B_SIZE, C_SIZE, L_DIM, H_SIZE, S_SIZE), dtype=torch.float32, device=device)
        C_exp = torch.empty((B_SIZE, C_SIZE, L_DIM, H_SIZE, S_SIZE), dtype=torch.float32, device=device)

        grid_expand = (B_SIZE, C_SIZE, L_DIM, H_SIZE, S_SIZE)
        expand_groups_kernel[grid_expand](
            B_ptr, B_exp, C_ptr, C_exp,
            B_SIZE=B_SIZE, C_SIZE=C_SIZE, L_DIM=L_DIM, G_SIZE=G_SIZE, S_SIZE=S_SIZE, H_SIZE=H_SIZE,
            num_warps=1, num_stages=1
        )

        # 3) Compute G[b, c, i, j, h] = sum_s B_exp[b, c, j, h, s] * C_exp[b, c, i, h, s]
        G = torch.empty((B_SIZE, C_SIZE, L_DIM, L_DIM, H_SIZE), dtype=torch.float32, device=device)

        grid_G = (B_SIZE, C_SIZE, H_SIZE, L_DIM, L_DIM)
        compute_G_kernel[grid_G](
            B_exp, C_exp, G,
            B_SIZE=B_SIZE, C_SIZE=C_SIZE, L_DIM=L_DIM, H_SIZE=H_SIZE, S_SIZE=S_SIZE,
            num_warps=1, num_stages=1
        )

        # 4) Apply mask: M = G * L
        M = torch.empty((B_SIZE, C_SIZE, L_DIM, L_DIM, H_SIZE), dtype=torch.float32, device=device)

        grid_M = (B_SIZE, C_SIZE, L_DIM, L_DIM, H_SIZE)
        elementwise_mul_5d_kernel[grid_M](
            G, L, M,
            B_SIZE=B_SIZE, C_SIZE=C_SIZE, L_DIM=L_DIM, H_SIZE=H_SIZE,
            num_warps=1, num_stages=1
        )

        # 5) Compute Y_diag by contraction over j
        hidden_f32 = hidden_states.to(torch.float32)
        Y = torch.empty((B_SIZE, C_SIZE, L_DIM, H_SIZE, D_SIZE), dtype=torch.float32, device=device)

        grid_reduce = (B_SIZE, C_SIZE, L_DIM, H_SIZE, D_SIZE)
        reduce_j_kernel[grid_reduce](
            M, hidden_f32, Y,
            B_SIZE=B_SIZE, C_SIZE=C_SIZE, L_DIM=L_DIM, H_SIZE=H_SIZE, D_SIZE=D_SIZE,
            num_warps=1, num_stages=1
        )

        return Y


def run(*args):
    return ModelNew()(*args)
