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

    # For each row i and each column j, compute segment_sum[i] up to j and set L[i, j] = exp(segment_sum) if i <= j else 0
    i = 0
    while i < L:
        segment_sum = 0.0
        j = 0
        while j < L:
            # Only include A_cumsum[n, h, t, i] when i <= j (lower-triangular)
            if i <= j:
                a_val = tl.load(A_ptr + pid_n * stride_A_n + pid_h * stride_A_h + pid_t * stride_A_t + i * stride_A_l)
                segment_sum += a_val
            j += 1
        j = 0
        while j < L:
            lower = i <= j
            val = tl.exp(segment_sum) if lower else 0.0
            tl.store(L_ptr + pid_n * stride_L_n + pid_h * stride_L_h + pid_t * stride_L_t + i * stride_L_i + j * stride_L_j, val)
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
        # Apply lower-triangular mask: i >= j
        lower = pid_i >= j
        m_val = tl.load(M_ptr + pid_n * stride_M_n + pid_t * stride_M_t + pid_i * stride_M_l_i + j * stride_M_l_j + pid_h * stride_M_h)
        hs_val = tl.load(HS_ptr + pid_n * stride_HS_n + pid_t * stride_HS_t + j * stride_HS_l + pid_h * stride_HS_h + pid_d * stride_HS_d)
        if lower:
            acc += m_val * hs_val
        j += 1

    tl.store(Y_ptr + pid_n * stride_Y_n + pid_t * stride_Y_t + pid_i * stride_Y_l + pid_h * stride_Y_h + pid_d * stride_Y_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants implied by original code
        self.CHUNK_SIZE = 128  # default; L taken from inputs
        self.NUM_HEADS = 32
        self.N_GROUPS = 8
        self.K = 32  # state size, half of head_dim

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous and float32 for compute
        device = hidden_states.device
        N, T, L, H, D = hidden_states.shape

        A_in = A_cumsum.contiguous().to(torch.float32)  # [N, H, T, L]
        B_in = B.contiguous().to(torch.float32)         # [N, T, L, G, K]
        C_in = C.contiguous().to(torch.float32)         # [N, T, L, G, K]
        HS = hidden_states.contiguous().to(torch.float32)  # [N, T, L, H, D]

        # 1) Build L lower-triangular exponential mask in Triton: L_out [N, H, T, L, L]
        L_out = torch.empty((N, H, T, L, L), device=device, dtype=torch.float32)
        grid_L = (N, H, T)
        _build_lower_tri_exp[grid_L](
            A_in, L_out,
            N, H, T, L,
            A_in.stride(0), A_in.stride(1), A_in.stride(2), A_in.stride(3),
            L_out.stride(0), L_out.stride(1), L_out.stride(2), L_out.stride(3), L_out.stride(4),
            num_warps=1, num_stages=1
        )

        # 2) Contract B @ C^T to form G: [N, T, L, L, H]
        Gout = torch.empty((N, T, L, L, H), device=device, dtype=torch.float32)
        grid_contract = (N, T, L, L, H)
        _contract_bc_to_g[grid_contract](
            B_in, C_in, Gout,
            N, T, L, self.N_GROUPS, self.K,
            B_in.stride(0), B_in.stride(1), B_in.stride(2), B_in.stride(3), B_in.stride(4),
            C_in.stride(0), C_in.stride(1), C_in.stride(2), C_in.stride(3), C_in.stride(4),
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            num_warps=1, num_stages=1
        )

        # 3) Compute Y_diag via diagonal matvec in Triton: [N, T, L, H, D]
        Y = torch.empty((N, T, L, H, D), device=device, dtype=torch.float32)
        grid_diag = (N, T, L, H, D)
        _diag_matvec_sum[grid_diag](
            Gout, HS, Y,
            N, T, L, H, D,
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            HS.stride(0), HS.stride(1), HS.stride(2), HS.stride(3), HS.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=1, num_stages=1
        )

        # Return in bfloat16 to match original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
