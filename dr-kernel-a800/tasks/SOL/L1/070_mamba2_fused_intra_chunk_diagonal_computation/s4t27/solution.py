import torch
import triton
import triton.language as tl


@triton.jit
def _build_lower_tri_exp_kernel(
    A_ptr, L_ptr,
    N, H, T, L_hs,
    stride_A_n, stride_A_h, stride_A_t, stride_A_l,
    stride_L_n, stride_L_h, stride_L_t, stride_L_i, stride_L_j,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # One program per (n, h, t)
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    # 2D offsets for i (rows) and j (cols) within chunk size
    i_offsets = tl.arange(0, 128)  # tile, will mask i<L_hs
    j_offsets = tl.arange(0, 128)  # tile, will mask j<L_hs
    i = i_offsets[None, :]  # shape (1, 128)
    j = j_offsets[:, None]  # shape (128, 1)

    mask_i = i < L_hs
    mask_j = j < L_hs

    # Accumulate segment_sum[i, j] = sum_{m=0..j} A[n, h, t, i] if i <= j else 0
    segment_sum = tl.zeros((1, 128), dtype=tl.float32)

    m = 0
    while m < 128:
        i_le_j = i <= j
        # Load A[n, h, t, i] for all i; mask i<L_hs, m<j (m+0<=j), and i<=j
        a_val = tl.load(
            A_ptr + pid_n * stride_A_n + pid_h * stride_A_h + pid_t * stride_A_t + i * stride_A_l,
            mask=mask_i & (m < L_hs) & (m <= (j - 0)) & i_le_j,
            other=0.0
        )
        segment_sum += a_val
        m += 1

    # L_out[i, j] = exp(segment_sum[i, j]) if i <= j else 0
    lower_mask = (i <= j) & mask_i & mask_j
    l_vals = tl.where(lower_mask, tl.exp(segment_sum), 0.0)

    # Store to L_out at (n, h, t, i, j)
    tl.store(
        L_ptr + pid_n * stride_L_n + pid_h * stride_L_h + pid_t * stride_L_t + i * stride_L_i + j * stride_L_j,
        l_vals,
        mask=mask_i[:, None] & mask_j[None, :]
    )


@triton.jit
def _contract_bc_to_g_kernel(
    B_ptr, C_ptr, G_ptr,
    N, T, L_hs, H, G_const, K,
    stride_B_n, stride_B_t, stride_B_l, stride_B_g, stride_B_k,
    stride_C_n, stride_C_t, stride_C_l, stride_C_g, stride_C_k,
    stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid over (N, T, i, j, h)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    acc = 0.0
    # Loop over groups and K
    g = 0
    while g < G_const:
        k = 0
        while k < K:
            b_val = tl.load(
                B_ptr + pid_n * stride_B_n + pid_t * stride_B_t + pid_j * stride_B_l + g * stride_B_g + k * stride_B_k
            )
            c_val = tl.load(
                C_ptr + pid_n * stride_C_n + pid_t * stride_C_t + pid_i * stride_C_l + g * stride_C_g + k * stride_C_k
            )
            acc += b_val * c_val
            k += 1
        g += 1

    tl.store(
        G_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_i * stride_G_l_i + pid_j * stride_G_l_j + pid_h * stride_G_h,
        acc
    )


@triton.jit
def _apply_mask_to_g_kernel(
    G_ptr, L_ptr, M_ptr,
    N, T, L_hs, H,
    stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h,
    stride_L_n, stride_L_h, stride_L_t, stride_L_i, stride_L_j,
    stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j, stride_M_h,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid over (N, T, i, j, h)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    g_val = tl.load(
        G_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_i * stride_G_l_i + pid_j * stride_G_l_j + pid_h * stride_G_h
    )
    lower = pid_i >= pid_j  # lower-triangular mask
    l_val = tl.load(
        L_ptr + pid_n * stride_L_n + pid_h * stride_L_h + pid_t * stride_L_t + pid_i * stride_L_i + pid_j * stride_L_j
    )
    m_val = g_val * l_val
    tl.store(
        M_ptr + pid_n * stride_M_n + pid_t * stride_M_t + pid_i * stride_M_l_i + pid_j * stride_M_l_j + pid_h * stride_M_h,
        m_val
    )


@triton.jit
def _diag_matvec_sum_kernel(
    M_ptr, HS_ptr, Y_ptr,
    N, T, L_hs, H, D,
    stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j, stride_M_h,
    stride_HS_n, stride_HS_t, stride_HS_l, stride_HS_h, stride_HS_d,
    stride_Y_n, stride_Y_t, stride_Y_l, stride_Y_h, stride_Y_d,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid over (N, T, i, h, d)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_d = tl.program_id(4)

    acc = 0.0
    j = 0
    while j < L_hs:
        # lower-triangular mask: i >= j
        if pid_i >= j:
            m_val = tl.load(
                M_ptr + pid_n * stride_M_n + pid_t * stride_M_t + pid_i * stride_M_l_i + j * stride_M_l_j + pid_h * stride_M_h
            )
            hs_val = tl.load(
                HS_ptr + pid_n * stride_HS_n + pid_t * stride_HS_t + j * stride_HS_l + pid_h * stride_HS_h + pid_d * stride_HS_d
            )
            acc += m_val * hs_val
        j += 1

    tl.store(
        Y_ptr + pid_n * stride_Y_n + pid_t * stride_Y_t + pid_i * stride_Y_l + pid_h * stride_Y_h + pid_d * stride_Y_d,
        acc
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute intra-chunk diagonal output Y_diag for Mamba2 SSD.

        Steps:
        1) Build L (lower-triangular exponential mask) from A_cumsum in Triton.
        2) Contract B and C to form G in Triton.
        3) Apply lower-triangular mask to G in Triton.
        4) Compute Y_diag by applying M to hidden_states in Triton.
        Returns: Y_diag (N, T, L, H, D) as torch.bfloat16.
        """

        device = hidden_states.device
        dtype = torch.float32

        N, T, L_hs, H, D = hidden_states.shape
        G_const = 8  # N_GROUPS as in original
        K = 32       # state_size as in original

        # 1) Build L in Triton
        L = torch.empty((N, H, T, L_hs, L_hs), device=device, dtype=dtype)
        grid_L = (N, H, T)
        _build_lower_tri_exp_kernel[grid_L](
            A_cumsum, L,
            N, H, T, L_hs,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=1, num_stages=1
        )

        # 2) Contract B @ C^T to G in Triton
        G = torch.empty((N, T, L_hs, L_hs, H), device=device, dtype=dtype)
        grid_G = (N, T, L_hs, L_hs, H)
        _contract_bc_to_g_kernel[grid_G](
            B, C, G,
            N, T, L_hs, H, G_const, K,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            num_warps=1, num_stages=1
        )

        # 3) Apply lower-triangular mask to G in Triton
        M = torch.empty((N, T, L_hs, L_hs, H), device=device, dtype=dtype)
        grid_M = (N, T, L_hs, L_hs, H)
        _apply_mask_to_g_kernel[grid_M](
            G, L, M,
            N, T, L_hs, H,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=1, num_stages=1
        )

        # 4) Compute Y_diag via diagonal matvec in Triton
        Y = torch.empty((N, T, L_hs, H, D), device=device, dtype=dtype)
        grid_Y = (N, T, L_hs, H, D)
        _diag_matvec_sum_kernel[grid_Y](
            M, hidden_states, Y,
            N, T, L_hs, H, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=1, num_stages=1
        )

        # Return in bfloat16 to match original behavior
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
