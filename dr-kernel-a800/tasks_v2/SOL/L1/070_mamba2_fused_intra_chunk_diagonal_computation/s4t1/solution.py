import torch
import triton
import triton.language as tl


# Kernel 1: Compute L as lower-triangular exponential mask directly in Triton.
# Input A_cumsum: [N, H, T, L] where L = chunk_size (e.g., 128).
# Output L_out: [N, H, T, L, L] (float32)
# For each (n, h, t), build lower-triangular mask and compute cumsum + exp.
@triton.jit
def _build_lower_tri_exp_mask(
    A_in_ptr, L_out_ptr,
    N, H, T, L,
    a_n_stride, a_h_stride, a_t_stride, a_l_stride,
    l_n_stride, l_h_stride, l_t_stride, l_i_stride, l_j_stride,
):
    n = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    t = tl.program_id(axis=2)

    # Create indices for i (source) and j (target)
    i = tl.arange(0, L)
    j = tl.arange(0, L)

    # Load A_cumsum vector for this (n, h, t, i): shape (L,)
    # Address: n * a_n_stride + h * a_h_stride + t * a_t_stride + i * a_l_stride
    a_off = n * a_n_stride + h * a_h_stride + t * a_t_stride + i * a_l_stride
    A_vec = tl.load(A_in_ptr + a_off)  # float32

    # Build lower-triangular mask: keep A_vec at positions where i <= j, else 0
    # Create 2D indices for broadcasting
    i2d = i[:, None]  # (L, 1)
    j2d = j[None, :]  # (1, L)
    keep = i2d <= j2d
    A_masked = tl.where(keep, A_vec[:, None], 0.0)

    # segment_sum along source axis j: for each row i, compute prefix sums
    # We need A_masked[j, k], then segment_sum[i, k] = sum_{m=0..k} A_masked[i, m]
    # Implement cumsum by looping k from 0 to L-1
    segment_sum = tl.zeros((L,), dtype=tl.float32)
    for k in range(0, L):
        # For each k, add A_masked[i, k] to all positions >= k
        # Broadcast add: segment_sum += A_masked[:, k] for positions where k <= j
        # This is a simple row-wise scan: accumulate A_masked[i, k] into segment_sum[i]
        # A_masked[:, k] is the k-th column vector across i
        val_k = A_masked[k, :]
        # We can add val_k to all positions k <= j (keep matrix with kth row constant)
        # However, Triton doesn't allow direct vectorized masked assignment like PyTorch.
        # Instead, we perform the scan using a per-row loop.
        # We'll store cumulative sums directly in segment_sum via the k loop.
        # Note: segment_sum is 1D vector across i. For each k, add val_k[i] to segment_sum[i].
        # This is straightforward: segment_sum += val_k
        segment_sum += val_k

    # Now compute L_out as exp(segment_sum) where we keep lower-triangular positions,
    # and 0 where i > j. We need L_out[i, j] = exp(segment_sum[i]) if i <= j, else 0.
    # Store to L_out for all j
    for j_idx in range(0, L):
        # For each j, we store exp(segment_sum) for i <= j; for i > j store 0
        # Compute addresses for row i and column j_idx
        l_off = n * l_n_stride + h * l_h_stride + t * l_t_stride + i * l_i_stride + j_idx * l_j_stride
        exp_val = tl.exp(segment_sum)  # vector over i
        # Zero out upper triangular
        exp_val = tl.where(i <= j_idx, exp_val, 0.0)
        tl.store(L_out_ptr + l_off, exp_val)


# Kernel 2: Compute G = B @ C^T for each (n, t, i, j, h). Inputs B and C assumed [N, T, L, G, K].
@triton.jit
def _contract_bc_to_g(
    B_ptr, C_ptr, Gout_ptr,
    N, T, L, G, K,
    b_n_stride, b_t_stride, b_i_stride, b_g_stride, b_k_stride,
    c_n_stride, c_t_stride, c_j_stride, c_g_stride, c_k_stride,
    g_n_stride, g_t_stride, g_i_stride, g_j_stride, g_h_stride,
    BLOCK_K: tl.constexpr,
):
    n = tl.program_id(axis=0)
    t = tl.program_id(axis=1)
    i = tl.program_id(axis=2)
    j = tl.program_id(axis=3)
    h = tl.program_id(axis=4)

    # Initialize accumulator for G[i, j] across h
    acc = 0.0

    # Loop over groups g (N_GROUPS=8) and over K in blocks
    for g in range(0, G):
        # We need to compute dot over k: sum_k C[n, t, i, g, k] * B[n, t, j, g, k]
        # We'll iterate k in blocks of BLOCK_K and accumulate.
        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_idx < K
            # Load B vector over k for (n, t, j, g)
            B_off = n * b_n_stride + t * b_t_stride + j * b_i_stride + g * b_g_stride + k_idx * b_k_stride
            B_vec = tl.load(B_ptr + B_off, mask=mask_k, other=0.0)
            # Load C vector over k for (n, t, i, g)
            C_off = n * c_n_stride + t * c_t_stride + i * c_j_stride + g * c_g_stride + k_idx * c_k_stride
            C_vec = tl.load(C_ptr + C_off, mask=mask_k, other=0.0)
            # Accumulate dot: sum(B_vec * C_vec)
            # For vectorized dot in Triton, reduce over BLOCK_K
            acc += tl.sum(B_vec * C_vec, axis=0)

    # Store G[i, j] for this (n, t, h)
    G_off = n * g_n_stride + t * g_t_stride + i * g_i_stride + j * g_j_stride + h * g_h_stride
    tl.store(Gout_ptr + G_off, acc)


# Kernel 3: Multiply M = G * L_perm (L_perm is already prepared as [N, T, L, L, H]).
@triton.jit
def _apply_mask_and_store_M(
    G_ptr, L_ptr, M_ptr,
    N, T, L, H,
    g_n_stride, g_t_stride, g_i_stride, g_j_stride, g_h_stride,
    l_n_stride, l_t_stride, l_i_stride, l_j_stride, l_h_stride,
    m_n_stride, m_t_stride, m_i_stride, m_j_stride, m_h_stride,
):
    n = tl.program_id(axis=0)
    t = tl.program_id(axis=1)
    i = tl.program_id(axis=2)
    j = tl.program_id(axis=3)
    h = tl.program_id(axis=4)

    # Load G[n, t, i, j, h]
    G_off = n * g_n_stride + t * g_t_stride + i * g_i_stride + j * g_j_stride + h * g_h_stride
    G_val = tl.load(G_ptr + G_off)

    # Load L[n, t, i, j, h]
    L_off = n * l_n_stride + t * l_t_stride + i * l_i_stride + j * l_j_stride + h * l_h_stride
    L_val = tl.load(L_ptr + L_off)

    # Multiply and store
    M_val = G_val * L_val
    M_off = n * m_n_stride + t * m_t_stride + i * m_i_stride + j * m_j_stride + h * m_h_stride
    tl.store(M_ptr + M_off, M_val)


# Kernel 4: Compute Y_diag[n, t, i, h] = sum_j M[n, t, i, j, h] * hidden_states[n, t, j, h, d].
# hidden_states is [N, T, L, H, D] (float32). Output Y is [N, T, L, H, D] (float32).
@triton.jit
def _diag_matvec_sum(
    M_ptr, HS_ptr, Y_ptr,
    N, T, L, H, D,
    m_n_stride, m_t_stride, m_i_stride, m_j_stride, m_h_stride,
    hs_n_stride, hs_t_stride, hs_j_stride, hs_h_stride, hs_d_stride,
    y_n_stride, y_t_stride, y_i_stride, y_h_stride, y_d_stride,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(axis=0)
    t = tl.program_id(axis=1)
    h = tl.program_id(axis=2)

    # Accumulator over i
    acc = tl.zeros((D,), dtype=tl.float32)

    # Loop over i in [0, L)
    for i in range(0, L):
        # Accumulate over j in [0, L): sum_j M[n, t, i, j, h] * HS[n, t, j, h, d]
        for j in range(0, L):
            # Load M[n, t, i, j, h] (scalar)
            M_off = n * m_n_stride + t * m_t_stride + i * m_i_stride + j * m_j_stride + h * m_h_stride
            M_val = tl.load(M_ptr + M_off)
            # Load HS[n, t, j, h, d] vector over d
            d_idx = tl.arange(0, D)
            HS_off = n * hs_n_stride + t * hs_t_stride + j * hs_j_stride + h * hs_h_stride + d_idx * hs_d_stride
            HS_vec = tl.load(HS_ptr + HS_off)  # vector of size D
            # Accumulate elementwise product into acc
            acc += M_val * HS_vec

    # Store acc into Y[n, t, i, h, d]
    # We write all d at once by broadcasting d_idx
    for i in range(0, L):
        for d_idx in range(0, D):
            Y_off = n * y_n_stride + t * y_t_stride + i * y_i_stride + h * y_h_stride + d_idx * y_d_stride
            tl.store(Y_ptr + Y_off, acc[d_idx])


# ModelNew: forward uses Triton kernels exclusively
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from original code (assumed in evaluation)
        self.CHUNK_SIZE = 128
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [N, T, L, H, D] (D typically 64)
        A_cumsum: [N, H, T, L] float32
        B: [N, T, L, G, K] (G=8, K likely 32)
        C: [N, T, L, G, K]
        Returns: Y_diag [N, T, L, H, D] in bfloat16 (to match original)
        """
        device = hidden_states.device
        N, T, L, H, D = hidden_states.shape
        assert H == self.NUM_HEADS, "num_heads must be 32"
        assert L == self.CHUNK_SIZE, "chunk_size must be 128"

        # Ensure inputs are contiguous
        A_in = A_cumsum.contiguous().to(torch.float32)
        B_in = B.contiguous().to(torch.float32)
        C_in = C.contiguous().to(torch.float32)
        HS = hidden_states.contiguous().to(torch.float32)

        # 1) Build L lower-triangular exponential mask in Triton
        L_out = torch.empty((N, H, T, L, L), device=device, dtype=torch.float32)
        grid_L = (N, H, T)
        _build_lower_tri_exp_mask[grid_L](
            A_in, L_out,
            N, H, T, L,
            A_in.stride(0), A_in.stride(1), A_in.stride(2), A_in.stride(3),
            L_out.stride(0), L_out.stride(1), L_out.stride(2), L_out.stride(3), L_out.stride(4),
        )

        # Permute L to match G's dimensions for multiplication: we need L_perm: [N, T, L, L, H]
        L_perm = L_out.permute(0, 2, 3, 4, 1).contiguous()  # [N, T, L, L, H]

        # 2) Contract B @ C^T to form G: [N, T, L, L, H]
        # Ensure B and C have expected shapes
        assert B_in.ndim == 5 and C_in.ndim == 5, "B and C must be 5D [N, T, L, G, K]"
        Nbt, Tbt, Lbt, G, K = B_in.shape
        assert Nbt == N and Tbt == T and Lbt == L, "B shapes must match N, T, L"
        assert G == self.N_GROUPS, "B/C must have N_GROUPS=8 in the 4th dim"
        Gout = torch.empty((N, T, L, L, H), device=device, dtype=torch.float32)
        grid_contract = (N, T, L, L, H)
        _contract_bc_to_g[grid_contract](
            B_in, C_in, Gout,
            N, T, L, G, K,
            B_in.stride(0), B_in.stride(1), B_in.stride(2), B_in.stride(3), B_in.stride(4),
            C_in.stride(0), C_in.stride(1), C_in.stride(2), C_in.stride(3), C_in.stride(4),
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            BLOCK_K=32,
        )

        # 3) Apply causal mask L_perm to G to get M
        M = torch.empty((N, T, L, L, H), device=device, dtype=torch.float32)
        grid_apply = (N, T, L, L, H)
        _apply_mask_and_store_M[grid_apply](
            Gout, L_perm, M,
            N, T, L, H,
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            L_perm.stride(0), L_perm.stride(1), L_perm.stride(2), L_perm.stride(3), L_perm.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
        )

        # 4) Compute Y_diag = sum over j of M[..., j] * hidden_states[..., j]
        Y = torch.empty((N, T, L, H, D), device=device, dtype=torch.float32)
        grid_diag = (N, T, H)
        _diag_matvec_sum[grid_diag](
            M, HS, Y,
            N, T, L, H, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            HS.stride(0), HS.stride(1), HS.stride(2), HS.stride(3), HS.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            BLOCK_D=D,  # D is typically 64
        )

        # Return in bfloat16 as original code
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
