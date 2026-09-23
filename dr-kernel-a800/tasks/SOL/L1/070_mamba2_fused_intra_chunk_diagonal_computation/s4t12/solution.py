import torch
import triton
import triton.language as tl


# Kernel 1: Build segment_sum(A_cumsum) with lower-triangular mask (diagonal = -1),
# then exp to get L_out: [N, H, T, L, L] (float32)
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

    # Loop over i and j; compute segment_sum[i, j] with tril mask (j >= i), else 0
    for i in range(L):
        seg_sum = 0.0
        for j in range(L):
            # lower-triangular condition
            if j >= i:
                a_ptr = A_cumsum_ptr + n * stride_A_n + h * stride_A_h + t * stride_A_t + i * stride_A_l
                a_val = tl.load(a_ptr)
                seg_sum += a_val
                l_val = tl.exp(seg_sum)
            else:
                l_val = 0.0
            l_ptr = L_out_ptr + n * stride_L_n + h * stride_L_h + t * stride_L_t + i * stride_L_i + j * stride_L_j
            tl.store(l_ptr, l_val)


# Kernel 2: Contract B @ C^T to form G: G[n, t, i, j, h] = sum_{g,K} C[n,t,i,g,k] * B[n,t,j,g,k]
# Inputs:
#   B_in: [N, T, L, G, K]
#   C_in: [N, T, L, G, K]
# Output:
#   Gout: [N, T, L, L, H] (float32), but we won't store H in this kernel; we assume H is handled by host.
@triton.jit
def _contract_bc_to_g_single_h(
    B_in_ptr, C_in_ptr, Gout_ptr,
    N, T, L, G, K,
    H,  # number of heads (runtime)
    stride_B_n, stride_B_t, stride_B_l, stride_B_g, stride_B_k,
    stride_C_n, stride_C_t, stride_C_l, stride_C_g, stride_C_k,
    stride_G_n, stride_G_t, stride_G_i, stride_G_j, stride_G_h,
):
    # Grid over (n, t, i, j, h)
    n = tl.program_id(0)
    t = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    acc = 0.0
    for g in range(G):
        for k in range(K):
            b_ptr = B_in_ptr + n * stride_B_n + t * stride_B_t + j * stride_B_l + g * stride_B_g + k * stride_B_k
            c_ptr = C_in_ptr + n * stride_C_n + t * stride_C_t + i * stride_C_l + g * stride_C_g + k * stride_C_k
            b = tl.load(b_ptr)
            c = tl.load(c_ptr)
            acc += c * b

    # Store to Gout at [n, t, i, j, h]
    gout_ptr = Gout_ptr + n * stride_G_n + t * stride_G_t + i * stride_G_i + j * stride_G_j + h * stride_G_h
    tl.store(gout_ptr, acc)


# Kernel 3: Apply mask L (from L_out) to G to produce M: M = G * L
# Inputs:
#   G: [N, T, L, L, H]
#   L_perm: [N, T, L, L, H] (this is L_out permuted to [N, T, L, L, H])
# Output:
#   M: [N, T, L, L, H] (float32)
@triton.jit
def _apply_mask_to_G(
    G_ptr, L_perm_ptr, M_ptr,
    N, T, L, H,
    stride_G_n, stride_G_t, stride_G_i, stride_G_j, stride_G_h,
    stride_L_n, stride_L_t, stride_L_i, stride_L_j, stride_L_h,
    stride_M_n, stride_M_t, stride_M_i, stride_M_j, stride_M_h,
):
    n = tl.program_id(0)
    t = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    g_ptr = G_ptr + n * stride_G_n + t * stride_G_t + i * stride_G_i + j * stride_G_j + h * stride_G_h
    l_ptr = L_perm_ptr + n * stride_L_n + t * stride_L_t + i * stride_L_i + j * stride_L_j + h * stride_L_h
    m = tl.load(g_ptr) * tl.load(l_ptr)

    m_ptr = M_ptr + n * stride_M_n + t * stride_M_t + i * stride_M_i + j * stride_M_j + h * stride_M_h
    tl.store(m_ptr, m)


# Kernel 4: Compute final Y_diag: sum over j of M[n, t, i, j, h] * hidden_states[n, t, j, h, d]
# We need to produce output Y: [N, T, L, H, D] (float32) and return bfloat16.
# Host can handle this last step. Here we compute per (n,t,h), sum over i and j, and store into Y.
@triton.jit
def _diag_matvec_sum_store_Y(
    M_ptr, HS_ptr, Y_ptr,
    N, T, L, H, D,
    stride_M_n, stride_M_t, stride_M_i, stride_M_j, stride_M_h,
    stride_HS_n, stride_HS_t, stride_HS_l, stride_HS_h, stride_HS_d,
    stride_Y_n, stride_Y_t, stride_Y_i, stride_Y_h, stride_Y_d,
):
    n = tl.program_id(0)
    t = tl.program_id(1)
    h = tl.program_id(2)

    # For each d, compute Y[n, t, i, h, d] = sum_j M[n, t, i, j, h] * HS[n, t, j, h, d]
    for d in range(D):
        acc = 0.0
        for i in range(L):
            for j in range(L):
                m_ptr = M_ptr + n * stride_M_n + t * stride_M_t + i * stride_M_i + j * stride_M_j + h * stride_M_h
                hs_ptr = HS_ptr + n * stride_HS_n + t * stride_HS_t + j * stride_HS_l + h * stride_HS_h + d * stride_HS_d
                m = tl.load(m_ptr)
                hs = tl.load(hs_ptr)
                acc += m * hs
            # Store acc into Y[n, t, i, h, d]
            y_ptr = Y_ptr + n * stride_Y_n + t * stride_Y_t + i * stride_Y_i + h * stride_Y_h + d * stride_Y_d
            tl.store(y_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [N, T, L, H, D]
        A_cumsum:      [N, H, T, L]  (float32)
        B:             [N, T, L, G, K]
        C:             [N, T, L, G, K]
        Returns:       [N, T, L, H, D] bfloat16
        """
        device = hidden_states.device
        N, T, L_hs, H, D = hidden_states.shape

        # Ensure contiguous and float32 for compute
        A = A_cumsum.contiguous().to(torch.float32)
        Bc = B.contiguous().to(torch.float32)
        Cc = C.contiguous().to(torch.float32)
        HS = hidden_states.contiguous().to(torch.float32)

        # Build L_out: [N, H, T, L, L] in Triton (float32)
        L_out = torch.empty((N, H, T, L_hs, L_hs), device=device, dtype=torch.float32)
        grid_L = (N, H, T)
        _build_lower_tri_exp_mask_from_cumsum[grid_L](
            A, L_out,
            N, H, T, L_hs,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L_out.stride(0), L_out.stride(1), L_out.stride(2), L_out.stride(3), L_out.stride(4),
            num_warps=1, num_stages=1,
        )

        # Permute L to [N, T, L, L, H] for multiplication with G
        L_perm = L_out.permute(0, 2, 3, 4, 1).contiguous()

        # 2) Contract B @ C^T to form G: [N, T, L, L, H]
        # We need G and K; infer from B and C shapes.
        G = Bc.shape[3]
        K = Bc.shape[4]
        Gout = torch.empty((N, T, L_hs, L_hs, H), device=device, dtype=torch.float32)
        grid_contract = (N, T, L_hs, L_hs, H)
        _contract_bc_to_g_single_h[grid_contract](
            Bc, Cc, Gout,
            N, T, L_hs, G, K,
            H,
            Bc.stride(0), Bc.stride(1), Bc.stride(2), Bc.stride(3), Bc.stride(4),
            Cc.stride(0), Cc.stride(1), Cc.stride(2), Cc.stride(3), Cc.stride(4),
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            num_warps=1, num_stages=1,
        )

        # 3) Apply causal mask: M = G * L_perm
        M = torch.empty((N, T, L_hs, L_hs, H), device=device, dtype=torch.float32)
        grid_apply = (N, T, L_hs, L_hs, H)
        _apply_mask_to_G[grid_apply](
            Gout, L_perm, M,
            N, T, L_hs, H,
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            L_perm.stride(0), L_perm.stride(1), L_perm.stride(2), L_perm.stride(3), L_perm.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=1, num_stages=1,
        )

        # 4) Compute Y_diag: sum over j of M[..., j] * hidden_states[..., j] along chunk dimension
        # Output shape: [N, T, L, H, D] (float32), then cast to bfloat16
        Y = torch.empty((N, T, L_hs, H, D), device=device, dtype=torch.float32)
        grid_diag = (N, T, H)
        _diag_matvec_sum_store_Y[grid_diag](
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
