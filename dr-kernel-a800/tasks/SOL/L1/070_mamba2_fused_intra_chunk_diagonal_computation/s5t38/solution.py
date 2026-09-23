import torch
import triton
import triton.language as tl


@triton.jit
def build_L_kernel_3d(
    A_ptr, L_ptr,
    B, C, H,
    L_len,
    A_stride_b, A_stride_c, A_stride_k, A_stride_h,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_h,
    diag_offset,
):
    # program ids
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # offsets
    A_off = b * A_stride_b + c * A_stride_c
    # cumsum over k in 0..L_len-1
    total = 0.0
    for k in range(0, L_len):
        a_off = A_off + k * A_stride_k + h * A_stride_h
        val = tl.load(A_ptr + a_off)
        total += val
    # build L[i, j] = exp(total) if j <= i, else 0
    for i in range(0, L_len):
        for j in range(0, L_len):
            keep = (j <= i)  # diagonal=-1
            val_ij = total if keep else 0.0
            # store to L[b, c, i, j, h]
            L_off = b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + h * L_stride_h
            tl.store(L_ptr + L_off, val_ij)


@triton.jit
def expand_groups_repeat_interleave_B_3d(
    B_in_ptr, B_out_ptr,
    B_in_stride_b, B_in_stride_c, B_in_stride_k, B_in_stride_s,
    B_out_stride_b, B_out_stride_c, B_out_stride_k, B_out_stride_h, B_out_stride_s,
    N_GROUPS, GROUP_EXPAND,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    k = tl.program_id(2)
    h = tl.program_id(3)
    s = tl.program_id(4)

    # map output head index h to input group index g
    g = h // GROUP_EXPAND
    if g >= N_GROUPS:
        g = 0  # safe default
    # load from B_in: [B, C, L, S] at (b, c, k, s)
    in_off = b * B_in_stride_b + c * B_in_stride_c + k * B_in_stride_k + s * B_in_stride_s
    val = tl.load(B_in_ptr + in_off)
    # write to B_out: [B, C, L, H, S] at (b, c, k, h, s)
    out_off = b * B_out_stride_b + c * B_out_stride_c + k * B_out_stride_k + h * B_out_stride_h + s * B_out_stride_s
    tl.store(B_out_ptr + out_off, val)


@triton.jit
def compute_G_3d(
    C_exp_ptr, B_exp_ptr, G_ptr,
    C_stride_b, C_stride_c, C_stride_i, C_stride_h, C_stride_s,
    B_stride_b, B_stride_c, B_stride_j, B_stride_h, B_stride_s,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_h,
    S_len,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    total = 0.0
    for s in range(0, S_len):
        c_off = b * C_stride_b + c * C_stride_c + i * C_stride_i + h * C_stride_h + s * C_stride_s
        b_off = b * B_stride_b + c * B_stride_c + j * B_stride_j + h * B_stride_h + s * B_stride_s
        c_val = tl.load(C_exp_ptr + c_off)
        b_val = tl.load(B_exp_ptr + b_off)
        total += c_val * b_val

    # store G[b, c, i, j, h] = total
    g_off = b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + h * G_stride_h
    tl.store(G_ptr + g_off, total)


@triton.jit
def compute_M_3d(
    G_ptr, L_ptr, M_ptr,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_h,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_h,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_h,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    g_off = b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + h * G_stride_h
    l_off = b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + h * L_stride_h
    g_val = tl.load(G_ptr + g_off)
    l_val = tl.load(L_ptr + l_off)
    m_val = g_val * l_val
    m_off = b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + h * M_stride_h
    tl.store(M_ptr + m_off, m_val)


@triton.jit
def compute_Y_diag_3d(
    M_ptr, hidden_ptr, Y_ptr,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_h,
    hidden_stride_b, hidden_stride_c, hidden_stride_j, hidden_stride_h, hidden_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_h, Y_stride_d,
    J_len, D_len,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in range(0, J_len):
        m_off = b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + h * M_stride_h
        h_off = b * hidden_stride_b + c * hidden_stride_c + j * hidden_stride_j + h * hidden_stride_h + d * hidden_stride_d
        m_val = tl.load(M_ptr + m_off)
        h_val = tl.load(hidden_ptr + h_off)
        acc += m_val * h_val

    y_off = b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + h * Y_stride_h + d * Y_stride_d
    tl.store(Y_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Shapes from original signature: hidden_states [B, C, L, H, D], A_cumsum [B, C, L, H],
        # B [B, C, L, N_GROUPS, S], C [B, C, L, N_GROUPS, S]
        Bsz, Cnum, L_len, H, D = hidden_states.shape
        N_GROUPS = 8
        GROUP_EXPAND = 4
        NUM_HEADS = 32

        # Compute constants
        S_size = C.shape[-1]  # state_size

        # Cast to float32 for computations
        hidden_f32 = hidden_states.to(torch.float32)
        A_f32 = A_cumsum.to(torch.float32)
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)

        # Allocate L [B, C, L, L, H] as float32
        L = torch.empty((Bsz, Cnum, L_len, L_len, H), dtype=torch.float32, device=hidden_f32.device)

        # Launch build_L_kernel_3d
        grid_L = (Bsz, Cnum, H)
        build_L_kernel_3d[grid_L](
            A_f32, L,
            Bsz, Cnum, H,
            L_len,
            A_f32.stride(0), A_f32.stride(1), A_f32.stride(2), A_f32.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            -1,  # diagonal offset as in original
            num_warps=1, num_stages=1,
        )

        # Expand B and C from groups to heads
        B_exp = torch.empty((Bsz, Cnum, L_len, NUM_HEADS, S_size), dtype=torch.float32, device=hidden_f32.device)
        C_exp = torch.empty((Bsz, Cnum, L_len, NUM_HEADS, S_size), dtype=torch.float32, device=hidden_f32.device)

        grid_expand = (Bsz, Cnum, L_len, NUM_HEADS, S_size)
        expand_groups_repeat_interleave_B_3d[grid_expand](
            B_f32, B_exp,
            B_f32.stride(0), B_f32.stride(1), B_f32.stride(2), B_f32.stride(3),
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            N_GROUPS, GROUP_EXPAND,
            num_warps=1, num_stages=1,
        )

        grid_expand2 = (Bsz, Cnum, L_len, NUM_HEADS, S_size)
        expand_groups_repeat_interleave_B_3d[grid_expand2](
            C_f32, C_exp,
            C_f32.stride(0), C_f32.stride(1), C_f32.stride(2), C_f32.stride(3),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            N_GROUPS, GROUP_EXPAND,
            num_warps=1, num_stages=1,
        )

        # Compute G: [B, C, L, L, H]
        G = torch.empty((Bsz, Cnum, L_len, L_len, H), dtype=torch.float32, device=hidden_f32.device)

        grid_G = (Bsz, Cnum, L_len, L_len, H)
        compute_G_3d[grid_G](
            C_exp, B_exp, G,
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            S_size,
            num_warps=1, num_stages=1,
        )

        # Compute M = G * L
        M = torch.empty_like(G)

        grid_M = (Bsz, Cnum, L_len, L_len, H)
        compute_M_3d[grid_M](
            G, L, M,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=1, num_stages=1,
        )

        # Compute Y_diag: [B, C, L, H, D]
        Y = torch.empty((Bsz, Cnum, L_len, H, D), dtype=torch.float32, device=hidden_f32.device)

        grid_Y = (Bsz, Cnum, L_len, H, D)
        compute_Y_diag_3d[grid_Y](
            M, hidden_f32, Y,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_f32.stride(0), hidden_f32.stride(1), hidden_f32.stride(2), hidden_f32.stride(3), hidden_f32.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            L_len, D,
            num_warps=1, num_stages=1,
        )

        # Return in bfloat16 as original signature suggests
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
