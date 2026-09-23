import torch
import triton
import triton.language as tl


@triton.jit
def _build_constant_L_from_A(
    A_ptr, L_ptr,
    N, H, T, L,
    A_s0, A_s1, A_s2, A_s3,
    L_s0, L_s1, L_s2, L_s3, L_s4,
):
    # Each program handles one (n, h, t), and writes L[n, h, t, i, j] = exp(A[n, h, t, i]) for all j
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    i = 0
    while i < L:
        a_off = pid_n * A_s0 + pid_h * A_s1 + pid_t * A_s2 + i * A_s3
        a_val = tl.load(A_ptr + a_off)
        exp_val = tl.exp(a_val)
        j = 0
        while j < L:
            l_off = pid_n * L_s0 + pid_h * L_s1 + pid_t * L_s2 + i * L_s3 + j * L_s4
            tl.store(L_ptr + l_off, exp_val)
            j += 1
        i += 1


@triton.jit
def _contract_BCToG(
    B_ptr, C_ptr, G_ptr,
    N, T, L, H, G_GROUPS, N_GROUPS, K,
    B_s0, B_s1, B_s2, B_s3, B_s4,
    C_s0, C_s1, C_s2, C_s3, C_s4,
    G_s0, G_s1, G_s2, G_s3, G_s4,
    BLOCK_K: tl.constexpr,
):
    # Grid over (n, t, i, j, h)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    g_off = pid_n * G_s0 + pid_t * G_s1 + pid_i * G_s2 + pid_j * G_s3 + pid_h * G_s4
    tl.store(G_ptr + g_off, 0.0)

    g_idx = 0
    while g_idx < G_GROUPS:
        k = 0
        while k < K:
            # h_global for this group g is h_global = g * N_GROUPS + h_local, where h_local == pid_h
            h_global = g_idx * N_GROUPS + pid_h
            # Validity: h_global must be in [0, H), but grid ensures pid_h < H for each group. We still guard.
            if h_global >= H:
                break

            b_off = pid_n * B_s0 + pid_t * B_s1 + pid_j * B_s2 + g_idx * B_s3 + k * B_s4
            c_off = pid_n * C_s0 + pid_t * C_s1 + pid_i * C_s2 + g_idx * C_s3 + k * C_s4
            b_val = tl.load(B_ptr + b_off)
            c_val = tl.load(C_ptr + c_off)
            g_val = tl.load(G_ptr + g_off) + (b_val * c_val)
            tl.store(G_ptr + g_off, g_val)
            k += 1
        g_idx += 1


@triton.jit
def _apply_mask_and_store_M(
    G_ptr, L_ptr, M_ptr,
    N, T, L, H,
    G_s0, G_s1, G_s2, G_s3, G_s4,
    L_s0, L_s1, L_s2, L_s3, L_s4,
    M_s0, M_s1, M_s2, M_s3, M_s4,
):
    # Grid over (n, t, i, j, h)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    g_off = pid_n * G_s0 + pid_t * G_s1 + pid_i * G_s2 + pid_j * G_s3 + pid_h * G_s4
    l_off = pid_n * L_s0 + pid_t * L_s1 + pid_i * L_s2 + pid_j * L_s3 + pid_h * L_s4
    m_off = pid_n * M_s0 + pid_t * M_s1 + pid_i * M_s2 + pid_j * M_s3 + pid_h * M_s4

    g_val = tl.load(G_ptr + g_off)
    # L_ptr is [N, H, T, L, 1] in practice, but we load per (i,j) and it's a constant for all j.
    l_val = tl.load(L_ptr + l_off)
    m_val = g_val * l_val
    tl.store(M_ptr + m_off, m_val)


@triton.jit
def _diag_matvec_sum_M_and_HS(
    M_ptr, HS_ptr, Y_ptr,
    N, T, L, H, D,
    M_s0, M_s1, M_s2, M_s3, M_s4,
    HS_s0, HS_s1, HS_s2, HS_s3, HS_s4,
    Y_s0, Y_s1, Y_s2, Y_s3, Y_s4,
):
    # Grid over (n, t, h)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)

    # For each i in [0..L-1], compute sum_j M[n, t, i, j, h] * HS[n, t, j, h, d] and store to Y[n, t, i, h, d]
    i = 0
    while i < L:
        acc = tl.zeros((D,), dtype=tl.float32)
        j = 0
        while j < L:
            m_off = pid_n * M_s0 + pid_t * M_s1 + i * M_s2 + j * M_s3 + pid_h * M_s4
            h_off = pid_n * HS_s0 + pid_t * HS_s1 + j * HS_s2 + pid_h * HS_s3
            m_val = tl.load(M_ptr + m_off)  # [D] is broadcast per element; here M is scalar per (i,j,h)
            # Note: M is a scalar per (i,j,h); HS is [D]. For diag matvec, we multiply M by HS[d].
            # We need to load HS scalar per d? Not correct. HS is [N, T, L, H, D]; our h_off points to j-th, h-th, and d-th is implicit.
            # We need per-d multiplication. Since HS is [L, D] at fixed (n,t,h), we iterate d.
            # Let's correct: HS is [N, T, L, H, D]; so h_off base should include d. We'll compute h_off + d * HS_s4.
            # But we need to load HS[n, t, j, h, d] for all d. Triton supports pointer math with d; however, it's clearer to compute per d loop.
            # We'll restructure: for each d, load HS scalar and accumulate.
            d = 0
            while d < D:
                hs_off = pid_n * HS_s0 + pid_t * HS_s1 + j * HS_s2 + pid_h * HS_s3 + d * HS_s4
                # M_ptr holds M per (i,j,h) as scalar; but output per d requires per-d weighting. Since M is scalar, we multiply by 1.
                # We need M per (i,j,h) value, which is already loaded via m_off. But for diag matvec, we need to read M for each d? No, M is scalar for (i,j,h).
                # The original operation is: sum over j of M[n, t, i, j, h] * HS[n, t, j, h, d]. M is scalar. We can pre-load and keep acc.
                # The previous code had a mistake: it tried to use hs_off with d * HS_s4, but M is scalar. We'll fix by loading HS for each d and multiplying by M scalar.
                m_scalar = tl.load(M_ptr + m_off)
                hs_scalar = tl.load(HS_ptr + hs_off)
                acc[d] += m_scalar * hs_scalar
                d += 1
            i += 1
        # Store acc for all d
        y_base = pid_n * Y_s0 + pid_t * Y_s1 + i * Y_s2 + pid_h * Y_s3
        d = 0
        while d < D:
            tl.store(Y_ptr + y_base + d * Y_s4, acc[d])
            d += 1
        i += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [N, T, L, H, D]
        A_cumsum: [N, H, T, L] (float32)
        B: [N, T, L, G, K] (float32)
        C: [N, T, L, G, K] (float32)
        Returns: Y_diag [N, T, L, H, D] in bfloat16
        """
        device = hidden_states.device

        # Shapes
        N, T, L_hs, H, D = hidden_states.shape
        assert A_cumsum.shape == (N, H, T, L_hs), "A_cumsum shape must be [N, H, T, L]"
        assert B.shape[3] == C.shape[3], "B and C must have the same G dimension"
        assert B.shape[2] == L_hs and C.shape[2] == L_hs, "B/C chunk size must match hidden_states"
        assert B.shape[4] == C.shape[4], "B/C K dimension must match"
        G = B.shape[3]
        K = B.shape[4]

        # Cast inputs to float32 for compute
        A_in = A_cumsum.contiguous().to(torch.float32)
        B_in = B.contiguous().to(torch.float32)
        C_in = C.contiguous().to(torch.float32)
        HS = hidden_states.contiguous().to(torch.float32)

        # 1) Build L as constant per-row exponential: L_out [N, H, T, L, L], but we can use [N, H, T, L, 1] conceptually.
        # We implement as [N, H, T, L, L] storing constant for all columns.
        L_out = torch.empty((N, H, T, L_hs, L_hs), device=device, dtype=torch.float32)
        grid_L = (N, H, T)
        _build_constant_L_from_A[grid_L](
            A_in, L_out,
            N, H, T, L_hs,
            A_in.stride(0), A_in.stride(1), A_in.stride(2), A_in.stride(3),
            L_out.stride(0), L_out.stride(1), L_out.stride(2), L_out.stride(3), L_out.stride(4),
            num_warps=1, num_stages=1,
        )

        # 2) Contract B @ C^T to form G: [N, T, L, L, H]
        Gout = torch.empty((N, T, L_hs, L_hs, H), device=device, dtype=torch.float32)
        grid_contract = (N, T, L_hs, L_hs, H)
        _contract_BCToG[grid_contract](
            B_in, C_in, Gout,
            N, T, L_hs, H, G, self.N_GROUPS, K,
            B_in.stride(0), B_in.stride(1), B_in.stride(2), B_in.stride(3), B_in.stride(4),
            C_in.stride(0), C_in.stride(1), C_in.stride(2), C_in.stride(3), C_in.stride(4),
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            BLOCK_K=K,
            num_warps=1, num_stages=1,
        )

        # 3) Apply mask (here, L is per-row constant, so M = G * L)
        M = torch.empty((N, T, L_hs, L_hs, H), device=device, dtype=torch.float32)
        grid_apply = (N, T, L_hs, L_hs, H)
        _apply_mask_and_store_M[grid_apply](
            Gout, L_out, M,
            N, T, L_hs, H,
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            L_out.stride(0), L_out.stride(1), L_out.stride(2), L_out.stride(3), L_out.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=1, num_stages=1,
        )

        # 4) Compute Y_diag: sum over j of M[..., j] * hidden_states[..., j] along chunk dimension
        Y = torch.empty((N, T, L_hs, H, D), device=device, dtype=torch.float32)
        grid_diag = (N, T, H)
        _diag_matvec_sum_M_and_HS[grid_diag](
            M, HS, Y,
            N, T, L_hs, H, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            HS.stride(0), HS.stride(1), HS.stride(2), HS.stride(3), HS.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=1, num_stages=1,
        )

        # Return in bfloat16 to match original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
