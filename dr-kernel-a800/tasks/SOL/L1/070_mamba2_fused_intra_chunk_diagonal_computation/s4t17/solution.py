import torch
import triton
import triton.language as tl

# Kernel 1: Build L (lower-triangular exponential mask) from A_cumsum
# A_cumsum: [N, H, T, L] float32
# Output L_out: [N, H, T, L, L] float32, where L[i,j] = exp(sum_{m=0..j} A[i]) for i <= j, else 0
@triton.jit
def _build_lower_tri_exp_kernel(
    A_ptr, L_ptr,
    N, H, T, L,
    stride_A_n, stride_A_h, stride_A_t, stride_A_l,
    stride_L_n, stride_L_h, stride_L_t, stride_L_i, stride_L_j,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    # Iterate over rows i and columns j using runtime loops to handle small L safely.
    i = 0
    while i < L:
        row_mask = i < L
        j = 0
        while j < L:
            # Accumulate segment_sum[i, j] = sum_{m=0..j} A[n, h, t, i] if i <= j else 0
            segment_sum = 0.0
            m = 0
            while m <= j:
                a_val = tl.load(
                    A_ptr + pid_n * stride_A_n + pid_h * stride_A_h + pid_t * stride_A_t + i * stride_A_l,
                    mask=row_mask,
                    other=0.0,
                )
                segment_sum += a_val
                m += 1

            # L[i, j] = exp(segment_sum) if i <= j else 0
            l_val = 0.0
            if i <= j:
                l_val = tl.exp(segment_sum)
            tl.store(
                L_ptr + pid_n * stride_L_n + pid_h * stride_L_h + pid_t * stride_L_t + i * stride_L_i + j * stride_L_j,
                l_val,
            )
            j += 1
        i += 1

# Kernel 2: Contract B @ C^T to form G
# B: [N, T, L, G, K] float32, here G=8, K=32
# C: [N, T, L, G, K] float32
# Output Gout: [N, T, L, L, H] float32, where H=32
@triton.jit
def _contract_bc_to_g_kernel(
    B_ptr, C_ptr, Gout_ptr,
    N, T, L, G, K, NUM_HEADS, N_GROUPS,
    stride_B_n, stride_B_t, stride_B_l, stride_B_g, stride_B_k,
    stride_C_n, stride_C_t, stride_C_l, stride_C_g, stride_C_k,
    stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    acc = 0.0
    # Sum over groups and K
    g = 0
    while g < G:
        # h index corresponds to group g: for each g, accumulate into H indices h in [g*N_GROUPS, (g+1)*N_GROUPS)
        h_local = pid_h - g * N_GROUPS
        in_group = (h_local >= 0) & (h_local < N_GROUPS)
        k = 0
        while k < K:
            b_val = tl.load(
                B_ptr + pid_n * stride_B_n + pid_t * stride_B_t + pid_j * stride_B_l + g * stride_B_g + k * stride_B_k,
                mask=in_group,
                other=0.0,
            )
            c_val = tl.load(
                C_ptr + pid_n * stride_C_n + pid_t * stride_C_t + pid_i * stride_C_l + g * stride_C_g + k * stride_C_k,
                mask=in_group,
                other=0.0,
            )
            acc += b_val * c_val
            k += 1
        g += 1

    tl.store(
        Gout_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_i * stride_G_l_i + pid_j * stride_G_l_j + pid_h * stride_G_h,
        acc,
    )

# Kernel 3: Apply lower-triangular mask to G
# G: [N, T, L, L, H] float32
# L: [N, H, T, L, L] float32
# Output M: [N, T, L, L, H] float32
@triton.jit
def _apply_mask_to_g_kernel(
    G_ptr, L_ptr, M_ptr,
    N, T, L, H,
    stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h,
    stride_L_n, stride_L_t, stride_L_i, stride_L_j, stride_L_h,
    stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j, stride_M_h,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    g_val = tl.load(G_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_i * stride_G_l_i + pid_j * stride_G_l_j + pid_h * stride_G_h)
    # Lower-triangular condition: i >= j
    lower = pid_i >= pid_j
    l_val = tl.load(
        L_ptr + pid_n * stride_L_n + pid_t * stride_L_t + pid_i * stride_L_i + pid_j * stride_L_j + pid_h * stride_L_h,
        mask=lower,
        other=1.0,
    )
    m_val = g_val * l_val
    tl.store(
        M_ptr + pid_n * stride_M_n + pid_t * stride_M_t + pid_i * stride_M_l_i + pid_j * stride_M_l_j + pid_h * stride_M_h,
        m_val,
    )

# Kernel 4: Diagonal matvec: Y[n, t, i, h, d] = sum_j M[n, t, i, j, h] * hidden_states[n, t, j, h, d]
# M: [N, T, L, L, H] float32
# hidden_states: [N, T, L, H, D] float32
# Output Y: [N, T, L, H, D] float32
@triton.jit
def _diag_matvec_sum_kernel(
    M_ptr, HS_ptr, Y_ptr,
    N, T, L, H, D,
    stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j, stride_M_h,
    stride_HS_n, stride_HS_t, stride_HS_l, stride_HS_h, stride_HS_d,
    stride_Y_n, stride_Y_t, stride_Y_l, stride_Y_h, stride_Y_d,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_d = tl.program_id(4)

    acc = 0.0
    j = 0
    while j < L:
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
        acc,
    )

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        Compute intra-chunk diagonal output Y_diag for Mamba2 SSD.

        hidden_states: [N, T, L, H, D]
        A_cumsum: [N, H, T, L] float32
        B: [N, T, L, G, K] float32 (G=8, K=32)
        C: [N, T, L, G, K] float32 (G=8, K=32)

        Output: Y_diag: [N, T, L, H, D] bfloat16
        """
        assert hidden_states.ndim == 5, "hidden_states must be [N, T, L, H, D]"
        assert A_cumsum.ndim == 4, "A_cumsum must be [N, H, T, L]"
        assert B.ndim == 5 and C.ndim == 5, "B and C must be [N, T, L, G, K]"
        N, T, L, H, D = hidden_states.shape
        N_GROUPS = 8
        NUM_HEADS = 32
        assert H == NUM_HEADS, "num_heads must be 32"
        assert B.shape == (N, T, L, 8, 32) and C.shape == (N, T, L, 8, 32), "B and C must have shape [N, T, L, 8, 32]"

        device = hidden_states.device

        # 1) Build L in Triton
        L_out = torch.empty((N, H, T, L, L), device=device, dtype=torch.float32)
        grid_L = (N, H, T)
        _build_lower_tri_exp_kernel[grid_L](
            A_cumsum, L_out,
            N, H, T, L,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L_out.stride(0), L_out.stride(1), L_out.stride(2), L_out.stride(3), L_out.stride(4),
            num_warps=1, num_stages=1,
        )

        # 2) Contract B @ C^T to G in Triton
        Gout = torch.empty((N, T, L, L, H), device=device, dtype=torch.float32)
        grid_G = (N, T, L, L, H)
        _contract_bc_to_g_kernel[grid_G](
            B, C, Gout,
            N, T, L, 8, 32, NUM_HEADS, N_GROUPS,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            num_warps=1, num_stages=1,
        )

        # 3) Apply mask L to G in Triton
        M = torch.empty((N, T, L, L, H), device=device, dtype=torch.float32)
        grid_M = (N, T, L, L, H)
        _apply_mask_to_g_kernel[grid_M](
            Gout, L_out, M,
            N, T, L, H,
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            L_out.stride(0), L_out.stride(1), L_out.stride(2), L_out.stride(3), L_out.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=1, num_stages=1,
        )

        # 4) Diagonal matvec to produce Y in Triton
        Y = torch.empty((N, T, L, H, D), device=device, dtype=torch.float32)
        grid_Y = (N, T, L, H, D)
        _diag_matvec_sum_kernel[grid_Y](
            M, hidden_states, Y,
            N, T, L, H, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=1, num_stages=1,
        )

        # Return in bfloat16 to match the original behavior
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
