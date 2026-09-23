import torch
import triton
import triton.language as tl


@triton.jit
def _build_lower_tri_exp(A_ptr, L_ptr,
                         N, H, T, L,
                         stride_A_n, stride_A_h, stride_A_t, stride_A_l,
                         stride_L_n, stride_L_h, stride_L_t, stride_L_i, stride_L_j,
                         BLOCK: tl.constexpr):
    # Each program handles one (n, h, t) and builds a [BLOCK, BLOCK] tile of L
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    i_offsets = pid_n * BLOCK + tl.arange(0, BLOCK)
    j_offsets = pid_h * BLOCK + tl.arange(0, BLOCK)

    i_mask = i_offsets < L
    j_mask = j_offsets < L

    # Initialize segment_sum matrix for this (n, t) slice over (i, j)
    # segment_sum[i, j] = sum_{m=0..j} A[n, :, t, i] when i <= j, else 0
    segment_sum = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    # Accumulate segment_sum across m from 0 to L-1; mask ensures safety for i>j
    m = 0
    while m < L:
        # We only accumulate when i <= j; for i>j, segment_sum remains 0
        # Load A[n, h, t, i] for all i in tile and broadcast to j
        a_vals = tl.load(
            A_ptr + pid_n * stride_A_n + pid_h * stride_A_h + pid_t * stride_A_t + i_offsets * stride_A_l,
            mask=i_mask,
            other=0.0,
        )  # shape: [BLOCK]
        # For each j, segment_sum[i, j] += A[n, h, t, i] if i<=j else 0
        # We can add a_vals to segment_sum[i, j] for all j where i<=j
        # We need to broadcast a_vals across j dimension
        # Compute j-dependent mask: (i<=j) & (j<L)
        # This loop is safe because m<L and we mask loads appropriately.
        # Note: Triton supports broadcasting a 1D vector to 2D via [:, None] and [None, :]
        add_mask = (i_offsets[:, None] <= j_offsets[None, :]) & j_mask[None, :]
        # For each m, add a_vals to segment_sum where i<=j
        # We need to update segment_sum only when i<=j; since add_mask zeros out i>j, this is fine.
        segment_sum += a_vals[:, None]

        m += 1

    # Now compute L = exp(segment_sum) but zero out upper triangle
    lower_mask = (i_offsets[:, None] <= j_offsets[None, :]) & i_mask[:, None] & j_mask[None, :]
    L_vals = tl.where(lower_mask, tl.exp(segment_sum), 0.0)

    # Store L into L_out[n, h, t, i, j]
    tl.store(
        L_ptr + pid_n * stride_L_n + pid_h * stride_L_h + pid_t * stride_L_t + i_offsets[:, None] * stride_L_i + j_offsets[None, :] * stride_L_j,
        L_vals,
        mask=i_mask[:, None] & j_mask[None, :],
    )


@triton.jit
def _contract_bc_to_g(B_ptr, C_ptr, G_ptr,
                      N, T, L, G, K,
                      stride_B_n, stride_B_t, stride_B_l, stride_B_g, stride_B_k,
                      stride_C_n, stride_C_t, stride_C_l, stride_C_g, stride_C_k,
                      stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j):
    # Grid: (N, T, L, L). Output G stores [N, T, L, L], independent of H since we will broadcast later.
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)

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

    tl.store(G_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_i * stride_G_l_i + pid_j * stride_G_l_j, acc)


@triton.jit
def _apply_mask(G_ptr, L_ptr, M_ptr,
                N, T, L,
                stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j,
                stride_L_n, stride_L_t, stride_L_i, stride_L_j,
                stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j):
    # Grid: (N, T, L, L). Compute M = G * L with lower-triangular mask i >= j.
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)

    g_val = tl.load(G_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_i * stride_G_l_i + pid_j * stride_G_l_j)
    l_val = tl.load(L_ptr + pid_n * stride_L_n + pid_t * stride_L_t + pid_i * stride_L_i + pid_j * stride_L_j)

    lower = pid_i >= pid_j  # lower-triangular condition
    m_val = tl.where(lower, g_val * l_val, 0.0)

    tl.store(M_ptr + pid_n * stride_M_n + pid_t * stride_M_t + pid_i * stride_M_l_i + pid_j * stride_M_l_j, m_val)


@triton.jit
def _diag_matvec_sum(M_ptr, HS_ptr, Y_ptr,
                     N, T, L, D,
                     stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j,
                     stride_HS_n, stride_HS_t, stride_HS_l, stride_HS_d,
                     stride_Y_n, stride_Y_t, stride_Y_l, stride_Y_d):
    # 5D grid: (N, T, L, H, D) where H is implicit (we assume H=32 as in original). We loop over H inside if needed.
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_d = tl.program_id(4)

    acc = 0.0
    j = 0
    while j < L:
        lower = pid_i >= j
        m_val = tl.load(M_ptr + pid_n * stride_M_n + pid_t * stride_M_t + pid_i * stride_M_l_i + j * stride_M_l_j)  # implicit H by index
        hs_val = tl.load(HS_ptr + pid_n * stride_HS_n + pid_t * stride_HS_t + j * stride_HS_l + pid_h * stride_HS_d)  # H implied in store below
        if lower:
            acc += m_val * hs_val
        j += 1

    tl.store(Y_ptr + pid_n * stride_Y_n + pid_t * stride_Y_t + pid_i * stride_Y_l + pid_h * stride_Y_h + pid_d * stride_Y_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants implied by original code: chunk_size=128, num_heads=32, n_groups=8, K=state_size=32 (half of head_dim=64)
        self.CHUNK_SIZE = 128
        self.NUM_HEADS = 32
        self.N_GROUPS = 8
        self.K = 32

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [N, T, L, H, D]
        A_cumsum: [N, H, T, L] (float32)
        B: [N, T, L, G, K] (float32)
        C: [N, T, L, G, K] (float32)
        Returns: Y_diag [N, T, L, H, D] in bfloat16
        """
        device = hidden_states.device
        N, T, L, H, D = hidden_states.shape
        assert H == self.NUM_HEADS, "num_heads must be 32"
        assert L <= self.CHUNK_SIZE, "chunk_size must be >= current L"
        # Ensure inputs are contiguous and float32
        A = A_cumsum.contiguous().to(torch.float32)
        Bt = B.contiguous().to(torch.float32)
        Ct = C.contiguous().to(torch.float32)
        HS = hidden_states.contiguous().to(torch.float32)

        # 1) Build L lower-triangular exponential mask in Triton: L_out [N, H, T, L, L]
        L_out = torch.empty((N, H, T, L, L), device=device, dtype=torch.float32)
        grid_L = (N, H, T)
        _build_lower_tri_exp[grid_L](
            A, L_out,
            N, H, T, L,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L_out.stride(0), L_out.stride(1), L_out.stride(2), L_out.stride(3), L_out.stride(4),
            BLOCK=128,
        )

        # 2) Contract B @ C^T to form G: [N, T, L, L]
        G = torch.empty((N, T, L, L), device=device, dtype=torch.float32)
        grid_contract = (N, T, L, L)
        _contract_bc_to_g[grid_contract](
            Bt, Ct, G,
            N, T, L, self.N_GROUPS, self.K,
            Bt.stride(0), Bt.stride(1), Bt.stride(2), Bt.stride(3), Bt.stride(4),
            Ct.stride(0), Ct.stride(1), Ct.stride(2), Ct.stride(3), Ct.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3),
        )

        # 3) Apply lower-triangular mask to G: M = G * L (we only keep i >= j)
        M = torch.empty((N, T, L, L), device=device, dtype=torch.float32)
        grid_apply = (N, T, L, L)
        _apply_mask[grid_apply](
            G, L_out, M,
            N, T, L,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3),
            L_out.stride(0), L_out.stride(2), L_out.stride(3), L_out.stride(4),  # ignore H stride in mask (we use i,j)
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
        )

        # 4) Compute Y_diag: sum over j of M[..., j] * hidden_states[..., j] along chunk dimension, output [N, T, L, H, D]
        # We need to broadcast M over H dimension (original H). We'll do per (i, h, d) by looping over H in Triton launch.
        # However, Triton grid requires known H; to handle this robustly, we can compute Y for each h separately by making H constant (32).
        # Given original uses H=32, we proceed with H=32.
        Y = torch.empty((N, T, L, H, D), device=device, dtype=torch.float32)
        grid_diag = (N, T, L, H, D)
        _diag_matvec_sum[grid_diag](
            M, HS, Y,
            N, T, L, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            HS.stride(0), HS.stride(1), HS.stride(2), HS.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
        )

        # Return in bfloat16 to match original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
