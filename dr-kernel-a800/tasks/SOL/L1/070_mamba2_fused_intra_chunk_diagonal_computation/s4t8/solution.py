import torch
import triton
import triton.language as tl


@triton.jit
def _build_lower_tri_exp(A_ptr, L_ptr,
                         N, H, T, L,
                         stride_A_n, stride_A_h, stride_A_t, stride_A_l,
                         stride_L_n, stride_L_h, stride_L_t, stride_L_i, stride_L_j):
    # Each program handles one (n, h, t)
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    # Loop over i (rows) and j (cols) up to L
    i = 0
    while i < L:
        seg = 0.0
        j = 0
        while j < L:
            m = 0
            while m <= j:
                # Load A[n, h, t, i] if i < L
                a_val = tl.load(A_ptr + pid_n * stride_A_n + pid_h * stride_A_h + pid_t * stride_A_t + i * stride_A_l,
                                mask=(i < L), other=0.0)
                seg += a_val
                m += 1
            # For i <= j: L[i, j] = exp(seg), else 0
            l_val = tl.exp(seg) if (i <= j) else 0.0
            tl.store(L_ptr + pid_n * stride_L_n + pid_h * stride_L_h + pid_t * stride_L_t + i * stride_L_i + j * stride_L_j, l_val)
            j += 1
        i += 1


@triton.jit
def _contract_bc_to_g(B_ptr, C_ptr, G_ptr,
                      N, T, L, G, K,
                      stride_B_n, stride_B_t, stride_B_l, stride_B_g, stride_B_k,
                      stride_C_n, stride_C_t, stride_C_l, stride_C_g, stride_C_k,
                      stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h):
    # Grid over (N, T, L, L, H)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    acc = 0.0
    # Sum over groups and K
    g = 0
    while g < G:
        k = 0
        while k < K:
            b_val = tl.load(B_ptr + pid_n * stride_B_n + pid_t * stride_B_t + pid_j * stride_B_l + g * stride_B_g + k * stride_B_k)
            c_val = tl.load(C_ptr + pid_n * stride_C_n + pid_t * stride_C_t + pid_i * stride_C_l + g * stride_C_g + k * stride_C_k)
            acc += b_val * c_val
            k += 1
        g += 1

    tl.store(G_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_i * stride_G_l_i + pid_j * stride_G_l_j + pid_h * stride_G_h, acc)


@triton.jit
def _apply_mask(G_ptr, L_ptr, M_ptr,
                N, T, L, H,
                stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h,
                stride_L_n, stride_L_h, stride_L_t, stride_L_i, stride_L_j,
                stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j, stride_M_h):
    # Grid over (N, T, L, L, H)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    g_val = tl.load(G_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_i * stride_G_l_i + pid_j * stride_G_l_j + pid_h * stride_G_h)
    l_val = tl.load(L_ptr + pid_n * stride_L_n + pid_h * stride_L_h + pid_t * stride_L_t + pid_i * stride_L_i + pid_j * stride_L_j)
    m_val = g_val * l_val
    tl.store(M_ptr + pid_n * stride_M_n + pid_t * stride_M_t + pid_i * stride_M_l_i + pid_j * stride_M_l_j + pid_h * stride_M_h, m_val)


@triton.jit
def _diag_matvec_sum(M_ptr, HS_ptr, Y_ptr,
                     N, T, L, H, D,
                     stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j, stride_M_h,
                     stride_HS_n, stride_HS_t, stride_HS_l, stride_HS_h, stride_HS_d,
                     stride_Y_n, stride_Y_t, stride_Y_l, stride_Y_h, stride_Y_d):
    # 5D grid over (n, t, i, h, d)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_d = tl.program_id(4)

    acc = 0.0
    j = 0
    while j < L:
        lower = pid_i >= j
        m_val = tl.load(M_ptr + pid_n * stride_M_n + pid_t * stride_M_t + pid_i * stride_M_l_i + j * stride_M_l_j + pid_h * stride_M_h)
        hs_val = tl.load(HS_ptr + pid_n * stride_HS_n + pid_t * stride_HS_t + j * stride_HS_l + pid_h * stride_HS_h + pid_d * stride_HS_d)
        # Apply lower-triangular mask: include only if i >= j
        if lower:
            acc += m_val * hs_val
        j += 1

    tl.store(Y_ptr + pid_n * stride_Y_n + pid_t * stride_Y_t + pid_i * stride_Y_l + pid_h * stride_Y_h + pid_d * stride_Y_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants implied by original code
        self.CHUNK_SIZE = 128
        self.NUM_HEADS = 32
        self.N_GROUPS = 8
        self.K = 32  # default state size, half of head_dim=64

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous tensors and compute in float32
        device = hidden_states.device
        A = A_cumsum.contiguous().to(torch.float32)
        Bt = B.contiguous().to(torch.float32)
        Ct = C.contiguous().to(torch.float32)
        HS = hidden_states.contiguous().to(torch.float32)

        N, T, L, H, D = HS.shape

        # 1) Build L in Triton: L [N, H, T, L, L]
        L_out = torch.empty((N, H, T, L, L), device=device, dtype=torch.float32)
        grid_L = (N, H, T)
        _build_lower_tri_exp[grid_L](
            A, L_out,
            N, H, T, L,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L_out.stride(0), L_out.stride(1), L_out.stride(2), L_out.stride(3), L_out.stride(4),
        )

        # 2) Compute G = B @ C^T: G [N, T, L, L, H]
        Gout = torch.empty((N, T, L, L, H), device=device, dtype=torch.float32)
        grid_G = (N, T, L, L, H)
        _contract_bc_to_g[grid_G](
            Bt, Ct, Gout,
            N, T, L, self.N_GROUPS, self.K,
            Bt.stride(0), Bt.stride(1), Bt.stride(2), Bt.stride(3), Bt.stride(4),
            Ct.stride(0), Ct.stride(1), Ct.stride(2), Ct.stride(3), Ct.stride(4),
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
        )

        # 3) Apply lower-triangular mask to G to form M: M [N, T, L, L, H]
        M = torch.empty((N, T, L, L, H), device=device, dtype=torch.float32)
        grid_apply = (N, T, L, L, H)
        _apply_mask[grid_apply](
            Gout, L_out, M,
            N, T, L, H,
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            L_out.stride(0), L_out.stride(1), L_out.stride(2), L_out.stride(3), L_out.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
        )

        # 4) Compute Y_diag: for each (n, t, i, h, d), Y[n, t, i, h, d] = sum_j M[n, t, i, j, h] * HS[n, t, j, h, d] with i >= j
        Y = torch.empty((N, T, L, H, D), device=device, dtype=torch.float32)
        grid_diag = (N, T, L, H, D)
        _diag_matvec_sum[grid_diag](
            M, HS, Y,
            N, T, L, H, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            HS.stride(0), HS.stride(1), HS.stride(2), HS.stride(3), HS.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
        )

        # Return in bfloat16 to match original function
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
