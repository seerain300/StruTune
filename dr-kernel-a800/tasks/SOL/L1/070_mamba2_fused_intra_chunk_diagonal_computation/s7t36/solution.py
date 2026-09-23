import torch
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_lower_exp_kernel(
    A_ptr, L_ptr,
    Bsz, Csz, S, N,
    A_stride_b, A_stride_c, A_stride_j, A_stride_n,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid: (B, C, N)
    b = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2)

    # running sum for each i in [0..S-1]
    running = tl.zeros((S,), dtype=tl.float32)

    # Compute cumsum along j with mask j < i (diagonal = -1)
    for i in range(0, S):
        sum_val = 0.0
        for j in range(0, S):
            if j < i:
                a_ptr = A_ptr + b * A_stride_b + c * A_stride_c + j * A_stride_j + n * A_stride_n
                val = tl.load(a_ptr)
                sum_val += val
        running[i] = sum_val

    # Apply exp; for j >= i, exp(cumsum) is 0 because original masks with -inf -> exp(-inf)=0
    for i in range(0, S):
        for j in range(0, S):
            if j < i:
                l_val = tl.exp(running[i])
            else:
                l_val = 0.0
            l_ptr = L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n
            tl.store(l_ptr, l_val)


@triton.jit
def contract_BC_to_G_kernel(
    B_exp_ptr, C_exp_ptr, G_ptr,
    Bsz, Csz, S, N, K,
    B_exp_stride_b, B_exp_stride_c, B_exp_stride_j, B_exp_stride_n, B_exp_stride_k,
    C_exp_stride_b, C_exp_stride_c, C_exp_stride_i, C_exp_stride_n, C_exp_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
    K_const: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid: (B, C, S, S, N)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # destination j index (row)
    j = tl.program_id(3)  # source i index (col)
    n = tl.program_id(4)

    acc = 0.0
    for k in range(0, K_const):
        C_ptr = C_exp_ptr + b * C_exp_stride_b + c * C_exp_stride_c + i * C_exp_stride_i + n * C_exp_stride_n + k * C_exp_stride_k
        B_ptr = B_exp_ptr + b * B_exp_stride_b + c * B_exp_stride_c + j * B_exp_stride_j + n * B_exp_stride_n + k * B_exp_stride_k
        c_val = tl.load(C_ptr)
        b_val = tl.load(B_ptr)
        acc += c_val * b_val

    G_store_ptr = G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    tl.store(G_store_ptr, acc)


@triton.jit
def elementwise_mul_LG_kernel(
    G_ptr, L_ptr, M_ptr,
    Bsz, Csz, S, N,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid: (B, C, S, S, N)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    g_addr = G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    l_addr = L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n
    g_val = tl.load(g_addr)
    l_val = tl.load(l_addr)
    m_val = g_val * l_val
    m_addr = M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n
    tl.store(m_addr, m_val)


@triton.jit
def diag_contract_Y_kernel(
    M_ptr, HS_ptr, Y_ptr,
    Bsz, Csz, S, N, D,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
    d_const: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid: (B, C, S, N, D) specialized per d_const
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # destination j index (row)
    n = tl.program_id(3)
    # d is fixed by d_const

    acc = 0.0
    for j in range(0, S):
        M_addr = M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n
        HS_addr = HS_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d_const * HS_stride_d
        m_val = tl.load(M_addr)
        hs_val = tl.load(HS_addr)
        acc += m_val * hs_val

    Y_addr = Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d_const * Y_stride_d
    tl.store(Y_addr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Extract shapes (NUM_HEADS=32, N_GROUPS=8 from original signature)
        Bsz = hidden_states.shape[0]
        Csz = hidden_states.shape[1]
        S = hidden_states.shape[2]            # chunk_size (must match A_cumsum S)
        N = hidden_states.shape[3]            # num_heads (must be NUM_HEADS // N_GROUPS = 4)
        D = hidden_states.shape[4]            # head_dim
        K = D  # state_size equals head_dim in this setup

        device = hidden_states.device

        # 1) Compute L via Triton: masked cumsum + exp
        # A_cumsum shape: [Bsz, Csz, S, N]
        A = A_cumsum
        # Allocate L: [Bsz, Csz, S, S, N], float32
        L = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        A_stride_b, A_stride_c, A_stride_j, A_stride_n = A.stride()
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = L.stride()

        grid_masked = (Bsz, Csz, N)
        masked_cumsum_lower_exp_kernel[grid_masked](
            A, L,
            Bsz, Csz, S, N,
            A_stride_b, A_stride_c, A_stride_j, A_stride_n,
            L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
            num_warps=2, num_stages=2
        )

        # 2) Expand B and C from n_groups to num_heads (NUM_HEADS=32)
        # Original code: expand to NUM_HEADS via repeat_interleave(NUM_HEADS // N_GROUPS) = 4
        B_expanded = B.repeat_interleave(4, dim=3).to(torch.float32)  # factor=4 to get 32 heads
        C_expanded = C.repeat_interleave(4, dim=3).to(torch.float32)  # factor=4

        # 3) Compute G[b, c, i, j, n] = sum over k of C_exp[b,c,i,n,k] * B_exp[b,c,j,n,k]
        G = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        B_exp_stride_b, B_exp_stride_c, B_exp_stride_j, B_exp_stride_n, B_exp_stride_k = B_expanded.stride()
        C_exp_stride_b, C_exp_stride_c, C_exp_stride_i, C_exp_stride_n, C_exp_stride_k = C_expanded.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()

        contract_BC_to_G_kernel[(Bsz, Csz, S, S, N)](
            B_expanded, C_expanded, G,
            Bsz, Csz, S, N, K,
            B_exp_stride_b, B_exp_stride_c, B_exp_stride_j, B_exp_stride_n, B_exp_stride_k,
            C_exp_stride_b, C_exp_stride_c, C_exp_stride_i, C_exp_stride_n, C_exp_stride_k,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            K_const=K,
            num_warps=2, num_stages=2
        )

        # 4) Compute M = G * L in Triton (elementwise multiply)
        M = torch.empty_like(G, dtype=torch.float32, device=device)
        elementwise_mul_LG_kernel[(Bsz, Csz, S, S, N)](
            G, L, M,
            Bsz, Csz, S, N,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
            M.stride()[0], M.stride()[1], M.stride()[2], M.stride()[3], M.stride()[4],
            num_warps=2, num_stages=2
        )

        # 5) Diagonal contraction to compute Y_diag: [Bsz, Csz, S, N, D]
        Y = torch.empty((Bsz, Csz, S, N, D), device=device, dtype=torch.float32)

        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = hidden_states.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        # Launch Triton kernel per d_const specialization
        for d in range(D):
            diag_contract_Y_kernel[(Bsz, Csz, S, N, 1)](
                M, hidden_states.to(torch.float32), Y,
                Bsz, Csz, S, N, D,
                M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
                HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
                Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
                d_const=d,
                num_warps=2, num_stages=2
            )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
