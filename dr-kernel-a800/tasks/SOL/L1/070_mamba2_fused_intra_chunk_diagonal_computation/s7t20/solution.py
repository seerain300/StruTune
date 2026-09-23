import torch
import triton
import triton.language as tl

@triton.jit
def masked_cumsum_exp_kernel(
    A_ptr, L_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr,
    A_stride_b, A_stride_c, A_stride_s, A_stride_n,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
):
    # Grid: (Bsz, Csz, N, S)
    b = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2)
    i = tl.program_id(3)  # fixed i for this program

    running = 0.0
    for j in range(0, S):
        a = tl.load(A_ptr + b * A_stride_b + c * A_stride_c + j * A_stride_s + n * A_stride_n)
        include = j < i
        running += a * include
        l_val = tl.exp(running) if include else 0.0
        tl.store(L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n, l_val)

@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_d,
    C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_d,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
):
    # Grid: (Bsz, Csz, S, S, N)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    acc = 0.0
    for k in range(0, D):
        b_val = tl.load(B_ptr + b * B_stride_b + c * B_stride_c + j * B_stride_j + n * B_stride_n + k * B_stride_d)
        c_val = tl.load(C_ptr + b * C_stride_b + c * C_stride_c + i * C_stride_i + n * C_stride_n + k * C_stride_d)
        acc += b_val * c_val
    tl.store(G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n, acc)

@triton.jit
def elementwise_multiply_G_L_kernel(
    G_ptr, L_ptr, M_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
):
    # Grid: (Bsz, Csz, S, S, N)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    g = tl.load(G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n)
    l = tl.load(L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n)
    m = g * l
    tl.store(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n, m)

@triton.jit
def diag_contract_Y_kernel(
    M_ptr, HS_ptr, Y_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
):
    # Grid: (Bsz, Csz, S, N, D)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in range(0, S):
        m = tl.load(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n)
        hs = tl.load(HS_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d)
        acc += m * hs
    tl.store(Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d, acc)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Extract shapes: hidden_states: [B, C, S, N, D]
        Bsz, Csz, S, N, D = hidden_states.shape

        device = hidden_states.device

        # 1) Compute L via Triton: masked cumsum with diagonal=-1, then exp
        A = A_cumsum.to(torch.float32).contiguous()  # [B, C, S, N]
        L = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)  # [B, C, S, S, N]

        A_stride_b, A_stride_c, A_stride_s, A_stride_n = A.stride()
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = L.stride()

        grid_L = (Bsz, Csz, N, S)
        masked_cumsum_exp_kernel[grid_L](
            A, L,
            Bsz, Csz, S, N,
            A_stride_b, A_stride_c, A_stride_s, A_stride_n,
            L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
            num_warps=1, num_stages=1
        )

        # 2) Expand B and C from n_groups to num_heads (32) via repeat_interleave(4)
        B_expanded = B.to(torch.float32).repeat_interleave(4, dim=3).contiguous()  # [B, C, S, N, D]
        C_expanded = C.to(torch.float32).repeat_interleave(4, dim=3).contiguous()  # [B, C, S, N, D]

        # 3) Compute G = contraction of B and C: G[b, c, i, j, n] = sum_k B[b,c,j,n,k] * C[b,c,i,n,k]
        G = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        B_exp_stride_b, B_exp_stride_c, B_exp_stride_j, B_exp_stride_n, B_exp_stride_d = B_expanded.stride()
        C_exp_stride_b, C_exp_stride_c, C_exp_stride_i, C_exp_stride_n, C_exp_stride_d = C_expanded.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()

        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G_kernel[grid_G](
            B_expanded, C_expanded, G,
            Bsz, Csz, S, N, D,
            B_exp_stride_b, B_exp_stride_c, B_exp_stride_j, B_exp_stride_n, B_exp_stride_d,
            C_exp_stride_b, C_exp_stride_c, C_exp_stride_i, C_exp_stride_n, C_exp_stride_d,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            num_warps=1, num_stages=1
        )

        # 4) Elementwise M = G * L in Triton
        M = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        G_stride_g_b, G_stride_g_c, G_stride_g_i, G_stride_g_j, G_stride_g_n = G.stride()
        L_stride_m_b, L_stride_m_c, L_stride_m_i, L_stride_m_j, L_stride_m_n = L.stride()
        M_stride_m_b, M_stride_m_c, M_stride_m_i, M_stride_m_j, M_stride_m_n = M.stride()

        grid_M = (Bsz, Csz, S, S, N)
        elementwise_multiply_G_L_kernel[grid_M](
            G, L, M,
            Bsz, Csz, S, N,
            G_stride_g_b, G_stride_g_c, G_stride_g_i, G_stride_g_j, G_stride_g_n,
            L_stride_m_b, L_stride_m_c, L_stride_m_i, L_stride_m_j, L_stride_m_n,
            M_stride_m_b, M_stride_m_c, M_stride_m_i, M_stride_m_j, M_stride_m_n,
            num_warps=1, num_stages=1
        )

        # 5) Diagonal contraction to compute Y_diag: [B, C, S, N, D]
        hidden_states_f32 = hidden_states.to(torch.float32).contiguous()  # [B, C, S, N, D]
        Y = torch.empty((Bsz, Csz, S, N, D), device=device, dtype=torch.float32)

        M_stride_y_b, M_stride_y_c, M_stride_y_i, M_stride_y_j, M_stride_y_n = M.stride()
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = hidden_states_f32.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y_kernel[grid_Y](
            M, hidden_states_f32, Y,
            Bsz, Csz, S, N, D,
            M_stride_y_b, M_stride_y_c, M_stride_y_i, M_stride_y_j, M_stride_y_n,
            HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
            Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
