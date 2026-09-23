import torch
import triton
import triton.language as tl


# Kernel 1: Build L via segment_sum(A_cumsum) with lower-triangular mask (diagonal = -1), then exp.
# For each (n, h, t), compute segment_sum[i, j] = sum_{m=0..j} A_cumsum[n, h, t, i] if i <= j else 0.
# Then L[n, h, t, i, j] = exp(segment_sum[i, j]) for i <= j, else 0.
# Output L_out: [N, H, T, L, L] (float32)
@triton.jit
def _build_lower_tri_exp_mask_from_cumsum(
    A_cumsum_ptr, L_out_ptr,
    N, H, T, L,
    stride_A_n, stride_A_h, stride_A_t, stride_A_l,
    stride_L_n, stride_L_h, stride_L_t, stride_L_i, stride_L_j,
):
    n = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)

    # We will fill L_out[n, h, t, i, j] for all i, j in [0..L-1]
    for i in range(L):
        # maintain segment_sum scalar
        seg_sum = 0.0
        for j in range(L):
            if j >= i:
                a_ptr = A_cumsum_ptr + n * stride_A_n + h * stride_A_h + t * stride_A_t + i * stride_A_l
                a_val = tl.load(a_ptr)
                seg_sum += a_val
            l_val = tl.exp(seg_sum) if j >= i else 0.0
            l_ptr = L_out_ptr + n * stride_L_n + h * stride_L_h + t * stride_L_t + i * stride_L_i + j * stride_L_j
            tl.store(l_ptr, l_val)


# Kernel 2: Contract B @ C^T to form G: G[n, t, i, j, h] = sum over groups g and state K of C[n,t,i,g,k] * B[n,t,j,g,k]
# Inputs:
#   B_in: [N, T, L, G, K]
#   C_in: [N, T, L, G, K]
# Output:
#   Gout: [N, T, L, L, H] (float32)
# Grid: (N, T, L, L, H)
@triton.jit
def _contract_bc_to_g(
    B_in_ptr, C_in_ptr, Gout_ptr,
    N, T, L, H, G, K,
    stride_B_n, stride_B_t, stride_B_l, stride_B_g, stride_B_k,
    stride_C_n, stride_C_t, stride_C_l, stride_C_g, stride_C_k,
    stride_G_n, stride_G_t, stride_G_i, stride_G_j, stride_G_h,
):
    n = tl.program_id(0)
    t = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    acc = 0.0
    # Loop over groups and K (small sizes; G, K vary by axes)
    for g in range(G):
        for k in range(K):
            b_ptr = B_in_ptr + n * stride_B_n + t * stride_B_t + j * stride_B_l + g * stride_B_g + k * stride_B_k
            c_ptr = C_in_ptr + n * stride_C_n + t * stride_C_t + i * stride_C_l + g * stride_C_g + k * stride_C_k
            b = tl.load(b_ptr)
            c = tl.load(c_ptr)
            acc += c * b

    gout_ptr = Gout_ptr + n * stride_G_n + t * stride_G_t + i * stride_G_i + j * stride_G_j + h * stride_G_h
    tl.store(gout_ptr, acc)


# Kernel 3: Apply mask L to G: M[n, t, i, j, h] = G[n, t, i, j, h] * L[n, t, i, j, h]
# Inputs:
#   G: [N, T, L, L, H]
#   L_out: [N, H, T, L, L] (we pass the original L_out directly)
# Output:
#   M: [N, T, L, L, H] (float32)
@triton.jit
def _apply_mask_to_G(
    G_ptr, L_ptr, M_ptr,
    N, T, L, H,
    stride_G_n, stride_G_t, stride_G_i, stride_G_j, stride_G_h,
    stride_L_n, stride_L_h, stride_L_t, stride_L_i, stride_L_j,
    stride_M_n, stride_M_t, stride_M_i, stride_M_j, stride_M_h,
):
    n = tl.program_id(0)
    t = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    g_ptr = G_ptr + n * stride_G_n + t * stride_G_t + i * stride_G_i + j * stride_G_j + h * stride_G_h
    l_ptr = L_ptr + n * stride_L_n + t * stride_L_h + i * stride_L_i + j * stride_L_j + h * stride_L_j  # note: using h mapping
    # We need to map L's h index consistently. Since L is [N, H, T, L, L], and G is [N, T, L, L, H], the h in G
    # corresponds to the same h used in L. The grid ensures we access L[n, h, t, i, j].
    l_ptr = L_ptr + n * stride_L_n + h * stride_L_h + t * stride_L_t + i * stride_L_i + j * stride_L_j
    g_val = tl.load(g_ptr)
    l_val = tl.load(l_ptr)
    m_val = g_val * l_val
    m_ptr = M_ptr + n * stride_M_n + t * stride_M_t + i * stride_M_i + j * stride_M_j + h * stride_M_h
    tl.store(m_ptr, m_val)


# Kernel 4: Diagonal matvec sum over j: Y[n, t, i, h, d] = sum_j M[n, t, i, j, h] * HS[n, t, j, h, d]
# Inputs:
#   M: [N, T, L, L, H]
#   HS: [N, T, L, H, D]
# Output:
#   Y: [N, T, L, H, D] (float32)
# Grid: (N, T, H) -> loop over i and d inside
@triton.jit
def _diag_matvec_sum_M_and_HS_to_Y(
    M_ptr, HS_ptr, Y_ptr,
    N, T, L, H, D,
    stride_M_n, stride_M_t, stride_M_i, stride_M_j, stride_M_h,
    stride_HS_n, stride_HS_t, stride_HS_l, stride_HS_h, stride_HS_d,
    stride_Y_n, stride_Y_t, stride_Y_i, stride_Y_h, stride_Y_d,
):
    n = tl.program_id(0)
    t = tl.program_id(1)
    h = tl.program_id(2)

    # Accumulator for each d
    for i in range(L):
        for d_idx in range(D):
            sum_j = 0.0
            for j in range(L):
                m_ptr = M_ptr + n * stride_M_n + t * stride_M_t + i * stride_M_i + j * stride_M_j + h * stride_M_h
                m_val = tl.load(m_ptr)
                hs_ptr = HS_ptr + n * stride_HS_n + t * stride_HS_t + j * stride_HS_l + h * stride_HS_h + d_idx * stride_HS_d
                hs_val = tl.load(hs_ptr)
                sum_j += m_val * hs_val
            # Store to Y[n, t, i, h, d_idx]
            y_ptr = Y_ptr + n * stride_Y_n + t * stride_Y_t + i * stride_Y_i + h * stride_Y_h + d_idx * stride_Y_d
            tl.store(y_ptr, sum_j)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
          - hidden_states: [N, T, L, H, D]
          - A_cumsum: [N, H, T, L] float32
          - B: [N, T, L, G, K]
          - C: [N, T, L, G, K]
        Output:
          - Y_diag: [N, T, L, H, D] in bfloat16 (matching original)
        """
        device = hidden_states.device

        # Ensure inputs are contiguous and float32
        A_in = A_cumsum.contiguous().to(torch.float32)       # [N, H, T, L]
        B_in = B.contiguous().to(torch.float32)             # [N, T, L, G, K]
        C_in = C.contiguous().to(torch.float32)             # [N, T, L, G, K]
        HS = hidden_states.contiguous().to(torch.float32)   # [N, T, L, H, D]

        N, T, L_hs, H, D = HS.shape
        G = B_in.shape[3]
        K = B_in.shape[4]

        # 1) Build L: lower-triangular segment_sum(A) + exp
        L_out = torch.empty((N, H, T, L_hs, L_hs), device=device, dtype=torch.float32)
        grid_L = (N, H, T)
        _build_lower_tri_exp_mask_from_cumsum[grid_L](
            A_in, L_out,
            N, H, T, L_hs,
            A_in.stride(0), A_in.stride(1), A_in.stride(2), A_in.stride(3),
            L_out.stride(0), L_out.stride(1), L_out.stride(2), L_out.stride(3), L_out.stride(4),
            num_warps=4,
        )

        # 2) Contract B @ C^T to G: [N, T, L, L, H]
        Gout = torch.empty((N, T, L_hs, L_hs, H), device=device, dtype=torch.float32)
        grid_contract = (N, T, L_hs, L_hs, H)
        _contract_bc_to_g[grid_contract](
            B_in, C_in, Gout,
            N, T, L_hs, H, G, K,
            B_in.stride(0), B_in.stride(1), B_in.stride(2), B_in.stride(3), B_in.stride(4),
            C_in.stride(0), C_in.stride(1), C_in.stride(2), C_in.stride(3), C_in.stride(4),
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            num_warps=4,
        )

        # 3) Apply mask L to G to get M
        M = torch.empty((N, T, L_hs, L_hs, H), device=device, dtype=torch.float32)
        grid_apply = (N, T, L_hs, L_hs, H)
        _apply_mask_to_G[grid_apply](
            Gout, L_out, M,
            N, T, L_hs, H,
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            L_out.stride(0), L_out.stride(1), L_out.stride(2), L_out.stride(3), L_out.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=4,
        )

        # 4) Compute Y_diag: sum over j of M[..., j] * HS[..., j]
        Y = torch.empty((N, T, L_hs, H, D), device=device, dtype=torch.float32)
        grid_diag = (N, T, H)
        _diag_matvec_sum_M_and_HS_to_Y[grid_diag](
            M, HS, Y,
            N, T, L_hs, H, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            HS.stride(0), HS.stride(1), HS.stride(2), HS.stride(3), HS.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=4,
        )

        # Return in bfloat16 to match original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
