import torch
import triton
import triton.language as tl

# Kernel 1: Build lower-triangular exponential mask L from A_cumsum
# A: [N, H, T, L], L: [N, H, T, L, L]
@triton.jit
def _build_lower_tri_exp_mask(
    A_ptr, L_ptr,
    N, H, T, L,
    A_stride_n, A_stride_h, A_stride_t, A_stride_l,
    L_stride_n, L_stride_h, L_stride_t, L_stride_i, L_stride_j,
):
    # Program ids for (n, h, t)
    n = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.program_id(2)

    i = 0
    while i < L:
        # row-wise cumsum along j: segment_sum[i, j] = sum_{m=0..j} A[n,h,t,i] for m >= i (lower-tri)
        row_sum = 0.0
        j = 0
        while j < L:
            a_val = tl.load(A_ptr + n * A_stride_n + h * A_stride_h + t * A_stride_t + i * A_stride_l,
                            mask=(i < L) & (j < L), other=0.0)
            row_sum += a_val
            if i <= j:
                tl.store(L_ptr + n * L_stride_n + h * L_stride_h + t * L_stride_t + i * L_stride_i + j * L_stride_j,
                         tl.exp(row_sum))
            else:
                tl.store(L_ptr + n * L_stride_n + h * L_stride_h + t * L_stride_t + i * L_stride_i + j * L_stride_j,
                         0.0)
            j += 1
        i += 1

# Kernel 2: Contract B @ C^T to form G: G[n, t, i, j, h] = sum over groups g and K of C[n,t,i,g,k] * B[n,t,j,g,k]
# Assumes N_GROUPS=8, K=32. Output G: [N, T, L, L, H]
@triton.jit
def _contract_bc_to_g(
    B_ptr, C_ptr, G_ptr,
    N, T, L, N_GROUPS, K,
    B_stride_n, B_stride_t, B_stride_l, B_stride_g, B_stride_k,
    C_stride_n, C_stride_t, C_stride_l, C_stride_g, C_stride_k,
    G_stride_n, G_stride_t, G_stride_i, G_stride_j, G_stride_h,
    BLOCK_K: tl.constexpr,
):
    n = tl.program_id(0)
    t = tl.program_id(1)
    i = tl.program_id(2)  # source i
    j = tl.program_id(3)  # target j
    h = tl.program_id(4)  # head index

    acc = 0.0
    # Loop over groups g=0..7
    for g in range(0, N_GROUPS):
        # Loop over K in blocks
        for k0 in range(0, K, BLOCK_K):
            k_range = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_range < K
            b_vec = tl.load(B_ptr + n * B_stride_n + t * B_stride_t + j * B_stride_l + g * B_stride_g +
                            k_range * B_stride_k,
                            mask=k_mask, other=0.0)
            c_vec = tl.load(C_ptr + n * C_stride_n + t * C_stride_t + i * C_stride_l + g * C_stride_g +
                            k_range * C_stride_k,
                            mask=k_mask, other=0.0)
            acc += tl.sum(b_vec * c_vec, axis=0)
    tl.store(G_ptr + n * G_stride_n + t * G_stride_t + i * G_stride_i + j * G_stride_j + h * G_stride_h, acc)

# Kernel 3: Compute Y_diag = sum over j of M[n,t,i,j,h] * HS[n,t,j,h,d]
# We implement M on-the-fly as M[n, t, i, j, h] = G[n, t, i, j, h] * L_perm[n, t, i, j, h]
# Output Y: [N, T, L, H, D] (float32 compute, cast to bfloat16 later)
@triton.jit
def _diag_matvec_sum_with_L_and_HS(
    G_ptr, L_ptr, HS_ptr, Y_ptr,
    N, T, L, H, D,
    G_stride_n, G_stride_t, G_stride_i, G_stride_j, G_stride_h,
    L_stride_n, L_stride_t, L_stride_i, L_stride_j, L_stride_h,
    HS_stride_n, HS_stride_t, HS_stride_l, HS_stride_h, HS_stride_d,
    Y_stride_n, Y_stride_t, Y_stride_i, Y_stride_h, Y_stride_d,
):
    # Grid over (n, t, h). We loop over i and d inside the kernel.
    n = tl.program_id(0)
    t = tl.program_id(1)
    h = tl.program_id(2)

    d = 0
    while d < D:
        acc = 0.0
        i = 0
        while i < L:
            # Accumulate over j
            j = 0
            while j < L:
                # M[n, t, i, j, h] = G[n, t, i, j, h] * L[n, t, i, j, h]
                g_val = tl.load(G_ptr + n * G_stride_n + t * G_stride_t + i * G_stride_i + j * G_stride_j + h * G_stride_h)
                l_val = tl.load(L_ptr + n * L_stride_n + t * L_stride_t + i * L_stride_i + j * L_stride_j + h * L_stride_h)
                m_val = g_val * l_val
                # hidden_states[n, t, j, h, d]
                hs_val = tl.load(HS_ptr + n * HS_stride_n + t * HS_stride_t + j * HS_stride_l + h * HS_stride_h + d * HS_stride_d)
                acc += m_val * hs_val
                j += 1
            i += 1
        # Store acc into Y[n, t, i, h, d] for all i
        i = 0
        while i < L:
            tl.store(Y_ptr + n * Y_stride_n + t * Y_stride_t + i * Y_stride_i + h * Y_stride_h + d * Y_stride_d, acc)
            i += 1
        d += 1

class ModelNew(torch.nn.Module):
    def __init__(self, num_heads=32, n_groups=8):
        super().__init__()
        self.NUM_HEADS = num_heads
        self.N_GROUPS = n_groups
        # In the original, K = head_dim // 2, and head_dim is 64 (-> K=32).
        self.K = 32

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [N, T, L, H, D]
        A_cumsum: [N, H, T, L]
        B: [N, T, L, G, K]
        C: [N, T, L, G, K]
        Returns: Y_diag [N, T, L, H, D] in bfloat16 (to match original)
        """
        device = hidden_states.device
        N, T, L, H, D = hidden_states.shape
        assert H == self.NUM_HEADS, "num_heads must be 32"
        assert self.N_GROUPS == 8, "n_groups must be 8"

        # Ensure inputs are contiguous and float32 for compute
        A_in = A_cumsum.contiguous().to(torch.float32)        # [N, H, T, L]
        B_in = B.contiguous().to(torch.float32)              # [N, T, L, G, K]
        C_in = C.contiguous().to(torch.float32)              # [N, T, L, G, K]
        HS = hidden_states.contiguous().to(torch.float32)    # [N, T, L, H, D]

        # 1) Build L lower-triangular exponential mask in Triton: L_out [N, H, T, L, L] (float32)
        L_out = torch.empty((N, H, T, L, L), device=device, dtype=torch.float32)
        grid_L = (N, H, T)
        _build_lower_tri_exp_mask[grid_L](
            A_in, L_out,
            N, H, T, L,
            A_in.stride(0), A_in.stride(1), A_in.stride(2), A_in.stride(3),
            L_out.stride(0), L_out.stride(1), L_out.stride(2), L_out.stride(3), L_out.stride(4),
        )

        # 2) Contract B @ C^T to form G: [N, T, L, L, H] (float32)
        Gout = torch.empty((N, T, L, L, H), device=device, dtype=torch.float32)
        grid_contract = (N, T, L, L, H)
        _contract_bc_to_g[grid_contract](
            B_in, C_in, Gout,
            N, T, L, self.N_GROUPS, self.K,
            B_in.stride(0), B_in.stride(1), B_in.stride(2), B_in.stride(3), B_in.stride(4),
            C_in.stride(0), C_in.stride(1), C_in.stride(2), C_in.stride(3), C_in.stride(4),
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            BLOCK_K=self.K,
        )

        # 3) Permute L to [N, T, L, L, H] for multiplication with G
        L_perm = L_out.permute(0, 2, 3, 4, 1).contiguous()  # [N, T, L, L, H]

        # 4) Compute Y_diag directly in Triton: Y [N, T, L, H, D] (float32)
        Y = torch.empty((N, T, L, H, D), device=device, dtype=torch.float32)
        grid_diag = (N, T, H)
        _diag_matvec_sum_with_L_and_HS[grid_diag](
            Gout, L_perm, HS, Y,
            N, T, L, H, D,
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            L_perm.stride(0), L_perm.stride(1), L_perm.stride(2), L_perm.stride(3), L_perm.stride(4),
            HS.stride(0), HS.stride(1), HS.stride(2), HS.stride(3), HS.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
        )

        # Return in bfloat16 to match original function
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
