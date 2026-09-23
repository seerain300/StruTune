import torch
import triton
import triton.language as tl

# Constants (match original model)
NUM_HEADS = 32
N_GROUPS = 8
GROUP_EXPAND = 4
L_MAT_SIZE = 128  # mask matrix size as in original code


@triton.jit
def build_L_5d_kernel(
    A_ptr,            # [B, C, L, H], float32
    L_ptr,            # [B, C, 128, 128, H], float32
    B_SIZE: tl.constexpr,
    C_SIZE: tl.constexpr,
    L_DIM: tl.constexpr,  # actual sequence length per (b, c)
    H_SIZE: tl.constexpr,
):
    # Grid: (B, C, i, j, H) -> 5D
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)  # row in 128
    pid_j = tl.program_id(3)  # col in 128
    pid_h = tl.program_id(4)

    # Compute total = sum_{k=0..L_DIM-1} A[b, c, k, h]
    total = 0.0
    for k in range(0, L_DIM):
        a_off = (
            pid_b * A_ptr.stride(0) +
            pid_c * A_ptr.stride(1) +
            k * A_ptr.stride(2) +
            pid_h * A_ptr.stride(3)
        )
        a_val = tl.load(A_ptr, offsets=a_off)
        total += a_val

    # Set L[i, j] = exp(total) if j <= i else 0
    if pid_j <= pid_i:
        l_val = tl.exp(total)
    else:
        l_val = 0.0

    l_off = (
        pid_b * L_ptr.stride(0) +
        pid_c * L_ptr.stride(1) +
        pid_i * L_ptr.stride(2) +
        pid_j * L_ptr.stride(3) +
        pid_h * L_ptr.stride(4)
    )
    tl.store(L_ptr, l_val, offsets=l_off)


@triton.jit
def expand_groups_repeat_interleave(
    B_ptr,            # [B, C, L, H, S], float32
    B_out_ptr,        # [B, C, L, H, S], float32 (same shape, destination)
    C_ptr,            # [B, C, L, H, S], float32
    C_out_ptr,        # [B, C, L, H, S], float32 (same shape, destination)
    B_SIZE: tl.constexpr,
    C_SIZE: tl.constexpr,
    L_DIM: tl.constexpr,
    H_SIZE: tl.constexpr,
    S_SIZE: tl.constexpr,
    G_SIZE: tl.constexpr,
):
    # 5D grid: (B, C, l, h, s)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_s = tl.program_id(4)

    # Map h -> original group g, then write to 4 consecutive heads
    g = pid_h // G_SIZE
    # For each k in 0..GROUP_EXPAND-1, output head index h_out = g*G_SIZE + g_offset + k
    for k in range(0, GROUP_EXPAND):
        h_out = g * G_SIZE + k
        # Bounds check
        if h_out >= H_SIZE:
            continue
        b_in_off = (
            pid_b * B_ptr.stride(0) +
            pid_c * B_ptr.stride(1) +
            pid_l * B_ptr.stride(2) +
            pid_h * B_ptr.stride(3) +
            pid_s * B_ptr.stride(4)
        )
        b_val = tl.load(B_ptr, offsets=b_in_off)

        b_out_off = (
            pid_b * B_out_ptr.stride(0) +
            pid_c * B_out_ptr.stride(1) +
            pid_l * B_out_ptr.stride(2) +
            h_out * B_out_ptr.stride(3) +
            pid_s * B_out_ptr.stride(4)
        )
        tl.store(B_out_ptr, b_val, offsets=b_out_off)

        c_in_off = (
            pid_b * C_ptr.stride(0) +
            pid_c * C_ptr.stride(1) +
            pid_l * C_ptr.stride(2) +
            pid_h * C_ptr.stride(3) +
            pid_s * C_ptr.stride(4)
        )
        c_val = tl.load(C_ptr, offsets=c_in_off)

        c_out_off = (
            pid_b * C_out_ptr.stride(0) +
            pid_c * C_out_ptr.stride(1) +
            pid_l * C_out_ptr.stride(2) +
            h_out * C_out_ptr.stride(3) +
            pid_s * C_out_ptr.stride(4)
        )
        tl.store(C_out_ptr, c_val, offsets=c_out_off)


@triton.jit
def compute_G_reduce_s_kernel(
    B_exp_ptr,        # [B, C, L, H, S], float32
    C_exp_ptr,        # [B, C, L, H, S], float32
    G_ptr,            # [B, C, L, L, H], float32
    B_SIZE: tl.constexpr,
    C_SIZE: tl.constexpr,
    L_DIM: tl.constexpr,
    H_SIZE: tl.constexpr,
    S_SIZE: tl.constexpr,
):
    # Grid: (B, C, i, j, h) i.e., one program computes G[i, j, h] for a given (b, c, h)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    acc = 0.0
    for s in range(0, S_SIZE):
        b_off = (
            pid_b * B_exp_ptr.stride(0) +
            pid_c * B_exp_ptr.stride(1) +
            pid_j * B_exp_ptr.stride(2) +
            pid_h * B_exp_ptr.stride(3) +
            s * B_exp_ptr.stride(4)
        )
        b_val = tl.load(B_exp_ptr, offsets=b_off)

        c_off = (
            pid_b * C_exp_ptr.stride(0) +
            pid_c * C_exp_ptr.stride(1) +
            pid_i * C_exp_ptr.stride(2) +
            pid_h * C_exp_ptr.stride(3) +
            s * C_exp_ptr.stride(4)
        )
        c_val = tl.load(C_exp_ptr, offsets=c_off)

        acc += b_val * c_val

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
    G_ptr,            # [B, C, L, L, H], float32
    L_ptr,            # [B, C, 128, 128, H], float32
    M_ptr,            # [B, C, L, L, H], float32
    B_SIZE: tl.constexpr,
    C_SIZE: tl.constexpr,
    L_DIM: tl.constexpr,
    H_SIZE: tl.constexpr,
):
    # Grid: (B, C, i, j, H) i.e., one program computes M[i, j, h] for given (b, c, i, j, h)
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
    M_ptr,            # [B, C, L, L, H], float32
    hidden_ptr,       # [B, C, L, H, D], float32
    Y_ptr,            # [B, C, L, H, D], float32
    B_SIZE: tl.constexpr,
    C_SIZE: tl.constexpr,
    L_DIM: tl.constexpr,
    H_SIZE: tl.constexpr,
    D_SIZE: tl.constexpr,
):
    # Grid: (B, C, i, h, d)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_d = tl.program_id(4)

    acc = 0.0
    for j in range(0, L_DIM):
        m_off = (
            pid_b * M_ptr.stride(0) +
            pid_c * M_ptr.stride(1) +
            pid_i * M_ptr.stride(2) +
            j * M_ptr.stride(3) +
            pid_h * M_ptr.stride(4)
        )
        mval = tl.load(M_ptr, offsets=m_off)

        h_off = (
            pid_b * hidden_ptr.stride(0) +
            pid_c * hidden_ptr.stride(1) +
            j * hidden_ptr.stride(2) +
            pid_h * hidden_ptr.stride(3) +
            pid_d * hidden_ptr.stride(4)
        )
        hval = tl.load(hidden_ptr, offsets=h_off)

        acc += mval * hval

    y_off = (
        pid_b * Y_ptr.stride(0) +
        pid_c * Y_ptr.stride(1) +
        pid_i * Y_ptr.stride(2) +
        pid_h * Y_ptr.stride(3) +
        pid_d * Y_ptr.stride(4)
    )
    tl.store(Y_ptr, acc, offsets=y_off)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, C, L, H, D]
        # A_cumsum:      [B, C, L, H]
        # B:             [B, C, L, H, S]
        # C:             [B, C, L, H, S]
        assert hidden_states.dim() == 5, "hidden_states must be [B, C, L, H, D]"
        assert A_cumsum.dim() == 4, "A_cumsum must be [B, C, L, H]"
        assert B.dim() == 5 and C.dim() == 5, "B and C must be [B, C, L, H, S]"

        B_SIZE, C_SIZE, L_DIM, H_SIZE, D_SIZE = hidden_states.shape
        device = hidden_states.device

        # Prepare inputs as float32 for kernel computations
        hidden_f32 = hidden_states.to(torch.float32)
        A_f32 = A_cumsum.to(torch.float32)
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)

        # 1) Build L: [B, C, 128, 128, H]
        L = torch.empty((B_SIZE, C_SIZE, L_MAT_SIZE, L_MAT_SIZE, H_SIZE), dtype=torch.float32, device=device)
        grid_L = (B_SIZE, C_SIZE, L_MAT_SIZE, L_MAT_SIZE, H_SIZE)
        build_L_5d_kernel[grid_L](
            A_f32, L,
            B_SIZE=B_SIZE, C_SIZE=C_SIZE, L_DIM=L_DIM, H_SIZE=H_SIZE,
            num_warps=2, num_stages=1
        )

        # 2) Expand B and C from groups (N_GROUPS=8) to heads (NUM_HEADS=32) via repeat_interleave
        S_SIZE = B_f32.shape[-1]
        G_SIZE = NUM_HEADS // N_GROUPS  # = 4
        B_exp = torch.empty((B_SIZE, C_SIZE, L_DIM, H_SIZE, S_SIZE), dtype=torch.float32, device=device)
        C_exp = torch.empty((B_SIZE, C_SIZE, L_DIM, H_SIZE, S_SIZE), dtype=torch.float32, device=device)

        grid_expand = (B_SIZE, C_SIZE, L_DIM, H_SIZE, S_SIZE)
        expand_groups_repeat_interleave[grid_expand](
            B_f32, B_exp, C_f32, C_exp,
            B_SIZE=B_SIZE, C_SIZE=C_SIZE, L_DIM=L_DIM, H_SIZE=H_SIZE, S_SIZE=S_SIZE, G_SIZE=G_SIZE,
            num_warps=2, num_stages=1
        )

        # 3) Compute G[b, c, i, j, h] = sum_s B_exp[b, c, j, h, s] * C_exp[b, c, i, h, s]
        G = torch.empty((B_SIZE, C_SIZE, L_DIM, L_DIM, H_SIZE), dtype=torch.float32, device=device)
        grid_G = (B_SIZE, C_SIZE, L_DIM, L_DIM, H_SIZE)
        compute_G_reduce_s_kernel[grid_G](
            B_exp, C_exp, G,
            B_SIZE=B_SIZE, C_SIZE=C_SIZE, L_DIM=L_DIM, H_SIZE=H_SIZE, S_SIZE=S_SIZE,
            num_warps=2, num_stages=1
        )

        # 4) Apply mask: M = G * L
        M = torch.empty((B_SIZE, C_SIZE, L_DIM, L_DIM, H_SIZE), dtype=torch.float32, device=device)
        grid_M = (B_SIZE, C_SIZE, L_DIM, L_DIM, H_SIZE)
        elementwise_mul_5d_kernel[grid_M](
            G, L, M,
            B_SIZE=B_SIZE, C_SIZE=C_SIZE, L_DIM=L_DIM, H_SIZE=H_SIZE,
            num_warps=2, num_stages=1
        )

        # 5) Compute Y_diag by contraction over j: Y[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, d]
        Y = torch.empty((B_SIZE, C_SIZE, L_DIM, H_SIZE, D_SIZE), dtype=torch.float32, device=device)
        grid_reduce = (B_SIZE, C_SIZE, L_DIM, H_SIZE, D_SIZE)
        reduce_j_kernel[grid_reduce](
            M, hidden_f32, Y,
            B_SIZE=B_SIZE, C_SIZE=C_SIZE, L_DIM=L_DIM, H_SIZE=H_SIZE, D_SIZE=D_SIZE,
            num_warps=2, num_stages=1
        )

        return Y


def run(*args):
    return ModelNew()(*args)
