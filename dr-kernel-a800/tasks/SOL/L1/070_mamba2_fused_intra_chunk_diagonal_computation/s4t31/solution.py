import torch
import triton
import triton.language as tl


@triton.jit
def _build_lower_tri_exp(B_ptr, OUT_ptr,
                         N, H, T, L,
                         stride_B_n, stride_B_h, stride_B_t, stride_B_l,
                         stride_OUT_n, stride_OUT_h, stride_OUT_t, stride_OUT_i, stride_OUT_j,
                         num_heads_heads: tl.constexpr):
    # Grid: (N, H, T)
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    # Create 2D offsets for i, j in [0..L)
    # We use BLOCK size to tile if L > BLOCK, but here we keep it simple and assume L is small enough
    # to fit in BLOCK=128 for typical cases. If L > 128, we can re-launch with grid over j blocks,
    # but here we assume chunk_size is moderate (e.g., <= 128). Adjust if needed.
    BLOCK = 128
    i_offsets = tl.arange(0, BLOCK)
    j_offsets = tl.arange(0, BLOCK)

    # Masks to restrict to valid range
    i_valid = i_offsets < L
    j_valid = j_offsets < L

    # Load A_cumsum vector for this (n, h, t)
    # A_cumsum shape: [N, H, T, L]
    a_vals = tl.load(
        B_ptr + pid_n * stride_B_n + pid_h * stride_B_h + pid_t * stride_B_t + i_offsets * stride_B_l,
        mask=i_valid, other=0.0
    )  # shape [BLOCK]

    # Compute segment_sum[i, j] = sum_{m=0..j} a_vals[i] for i <= j; else 0
    # Initialize segment_sum
    segment_sum = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)

    # j is broadcasted column vector
    j_idx = j_offsets[None, :]  # shape [1, BLOCK]
    i_idx = i_offsets[:, None]  # shape [BLOCK, 1]

    # Lower-triangular condition: i <= j
    lower_mask = (i_idx <= j_idx) & (i_valid[:, None]) & (j_valid[None, :])

    # For each m from 0 to j, add a_vals[i] to segment_sum[i, j]
    # We implement a simple loop over j (runtime) and accumulate; since BLOCK is moderate, this is fine.
    # Note: Triton prefers static loops for performance; here L is runtime, but we keep the loop simple.
    for m in range(0, 128):  # up to BLOCK
        j_less_m = j_idx <= m  # shape [1, BLOCK]
        # Only update where lower_mask & j_less_m & j_valid
        update_mask = lower_mask & j_less_m & (j_valid[None, :])
        # a_vals[i] contributes to all columns j >= m
        segment_sum += tl.where(update_mask, a_vals[:, None], 0.0)

    # Exponential of segment_sum; set upper triangle (i > j) and diagonal to 0 per original tril(diagonal=-1)
    L_vals = tl.exp(segment_sum)  # shape [BLOCK, BLOCK]
    # i > j means upper triangle in this context; set to 0
    upper_mask = (i_idx > j_idx) & (i_valid[:, None]) & (j_valid[None, :])
    L_vals = tl.where(upper_mask, 0.0, L_vals)

    # Store L to output [N, H, T, i, j]
    # We store only valid i/j
    store_mask = (i_valid[:, None]) & (j_valid[None, :])
    tl.store(
        OUT_ptr + pid_n * stride_OUT_n + pid_h * stride_OUT_h + pid_t * stride_OUT_t
        + i_offsets[:, None] * stride_OUT_i + j_offsets[None, :] * stride_OUT_j,
        L_vals,
        mask=store_mask
    )


@triton.jit
def _contract_bc_to_g(B_ptr, C_ptr, G_ptr,
                      N, T, L, G, K,
                      stride_B_n, stride_B_t, stride_B_l, stride_B_g, stride_B_k,
                      stride_C_n, stride_C_t, stride_C_l, stride_C_g, stride_C_k,
                      stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h,
                      num_heads_heads: tl.constexpr):
    # Grid: (N, T, i, j, h)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    acc = 0.0
    # Sum over groups and K
    for g in range(0, G):
        # Map h to group-specific head index: original expands by repeat_interleave(NUM_HEADS//N_GROUPS) = 4
        # num_heads_heads = 4
        # h_global = g * num_heads_heads + pid_h
        h_global = g * num_heads_heads + pid_h

        for k in range(0, K):
            b_val = tl.load(
                B_ptr + pid_n * stride_B_n + pid_t * stride_B_t + pid_j * stride_B_l + g * stride_B_g + k * stride_B_k
            )
            c_val = tl.load(
                C_ptr + pid_n * stride_C_n + pid_t * stride_C_t + pid_i * stride_C_l + g * stride_C_g + k * stride_C_k
            )
            acc += b_val * c_val

    # Store G[n, t, i, j, h_global]
    tl.store(
        G_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_i * stride_G_l_i + pid_j * stride_G_l_j + h_global * stride_G_h,
        acc
    )


@triton.jit
def _apply_mask_and_store_M(G_ptr, L_ptr, M_ptr,
                            N, T, L, H,
                            stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h,
                            stride_L_n, stride_L_h, stride_L_t, stride_L_i, stride_L_j,
                            stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j, stride_M_h):
    # Grid: (N, T, i, j, h)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    g_val = tl.load(G_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_i * stride_G_l_i + pid_j * stride_G_l_j + pid_h * stride_G_h)
    l_val = tl.load(L_ptr + pid_n * stride_L_n + pid_t * stride_L_t + pid_i * stride_L_i + pid_j * stride_L_j)
    m_val = g_val * l_val
    tl.store(M_ptr + pid_n * stride_M_n + pid_t * stride_M_t + pid_i * stride_M_l_i + pid_j * stride_M_l_j + pid_h * stride_M_h, m_val)


@triton.jit
def _diag_matvec_sum_M_and_HS(M_ptr, HS_ptr, Y_ptr,
                              N, T, L, H, D,
                              stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j, stride_M_h,
                              stride_HS_n, stride_HS_t, stride_HS_l, stride_HS_h, stride_HS_d,
                              stride_Y_n, stride_Y_t, stride_Y_l, stride_Y_h, stride_Y_d):
    # Grid: (N, T, i, h, d) — we compute one d per program and loop over j
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_d = tl.program_id(4)

    acc = 0.0
    for j in range(0, L):
        m_val = tl.load(M_ptr + pid_n * stride_M_n + pid_t * stride_M_t + pid_i * stride_M_l_i + j * stride_M_l_j + pid_h * stride_M_h)
        hs_val = tl.load(HS_ptr + pid_n * stride_HS_n + pid_t * stride_HS_t + j * stride_HS_l + pid_h * stride_HS_h + pid_d * stride_HS_d)
        acc += m_val * hs_val

    tl.store(Y_ptr + pid_n * stride_Y_n + pid_t * stride_Y_t + pid_i * stride_Y_l + pid_h * stride_Y_h + pid_d * stride_Y_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        Triton-only implementation:
        - Build L mask in Triton
        - Contract B @ C^T to G in Triton
        - Apply mask in Triton
        - Compute diagonal matvec in Triton
        Return output in bfloat16 with shape [N, T, L, H, D].
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA for Triton kernels."

        device = hidden_states.device
        N, T, L, H, D = hidden_states.shape
        # A_cumsum: [N, H, T, L]
        N_A, H_A, T_A, L_A = A_cumsum.shape
        assert N_A == N and H_A == H and T_A == T and L_A == L, "A_cumsum shape must match N, H, T, L."

        # Prepare output buffers
        # 1) Build L mask [N, H, T, L, L]
        L_mask = torch.empty((N, H, T, L, L), device=device, dtype=torch.float32)
        # Strides for A_cumsum
        stride_B_n, stride_B_h, stride_B_t, stride_B_l = A_cumsum.stride()
        # Strides for L_mask
        stride_L_n, stride_L_h, stride_L_t, stride_L_i, stride_L_j = L_mask.stride()

        grid_L = (N, H, T)
        _build_lower_tri_exp[grid_L](
            A_cumsum, L_mask,
            N, H, T, L,
            stride_B_n, stride_B_h, stride_B_t, stride_B_l,
            stride_L_n, stride_L_h, stride_L_t, stride_L_i, stride_L_j,
            num_heads_heads=4,  # NUM_HEADS // N_GROUPS = 32 // 8 = 4
            num_warps=2, num_stages=2
        )

        # 2) Contract B @ C^T -> G[n, t, i, j, h] with h mapping via groups
        G = torch.empty((N, T, L, L, H * (4)), device=device, dtype=torch.float32)  # since groups=8, H_total = H*4
        stride_B_nB, stride_B_tB, stride_B_lB, stride_B_gB, stride_B_kB = B.stride()
        stride_C_nC, stride_C_tC, stride_C_lC, stride_C_gC, stride_C_kC = C.stride()
        stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h = G.stride()
        grid_G = (N, T, L, L, H * 4)
        _contract_bc_to_g[grid_G](
            B, C, G,
            N, T, L, 8, 32,
            stride_B_nB, stride_B_tB, stride_B_lB, stride_B_gB, stride_B_kB,
            stride_C_nC, stride_C_tC, stride_C_lC, stride_C_gC, stride_C_kC,
            stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h,
            num_heads_heads=4,
            num_warps=4, num_stages=2
        )

        # 3) Apply mask M = G * L
        M = torch.empty((N, T, L, L, H * 4), device=device, dtype=torch.float32)
        stride_G_nG, stride_G_tG, stride_G_l_iG, stride_G_l_jG, stride_G_hG = G.stride()
        stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j, stride_M_h = M.stride()
        grid_apply = (N, T, L, L, H * 4)
        _apply_mask_and_store_M[grid_apply](
            G, L_mask, M,
            N, T, L, H * 4,
            stride_G_nG, stride_G_tG, stride_G_l_iG, stride_G_l_jG, stride_G_hG,
            stride_L_n, stride_L_h, stride_L_t, stride_L_i, stride_L_j,
            stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j, stride_M_h,
            num_warps=4, num_stages=2
        )

        # 4) Diagonal matvec: Y[n, t, i, h, d] = sum_j M[n, t, i, j, h] * hidden_states[n, t, j, h, d]
        # hidden_states: [N, T, L, H, D]
        Y = torch.empty((N, T, L, H * 4, D), device=device, dtype=torch.float32)
        stride_HS_n, stride_HS_t, stride_HS_l, stride_HS_h, stride_HS_d = hidden_states.stride()
        stride_M_nM, stride_M_tM, stride_M_l_iM, stride_M_l_jM, stride_M_hM = M.stride()
        grid_diag = (N, T, L, H * 4, D)
        _diag_matvec_sum_M_and_HS[grid_diag](
            M, hidden_states, Y,
            N, T, L, H * 4, D,
            stride_M_nM, stride_M_tM, stride_M_l_iM, stride_M_l_jM, stride_M_hM,
            stride_HS_n, stride_HS_t, stride_HS_l, stride_HS_h, stride_HS_d,
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=4, num_stages=2
        )

        # Return in bfloat16 to match original function’s dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
