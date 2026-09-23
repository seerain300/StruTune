import torch
import triton
import triton.language as tl


@triton.jit
def _build_L_from_hidden_and_A(HS_ptr, L_ptr,
                                N, H, T, L,
                                stride_HS_n, stride_HS_h, stride_HS_t, stride_HS_l,
                                stride_L_n, stride_L_h, stride_L_t, stride_L_i, stride_L_j):
    # Grid over (n, h, t)
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    # For each (i, j) in [0, L), compute L[i, j] = exp(HS[n, h, t, i]) if i <= j else 0
    i = 0
    while i < L:
        j = 0
        while j < L:
            # Load HS[n, h, t, i]
            hs_val = tl.load(HS_ptr + pid_n * stride_HS_n + pid_h * stride_HS_h + pid_t * stride_HS_t + i * stride_HS_l)
            exp_val = tl.exp(hs_val)
            # Apply lower-triangular condition: i >= j
            lower = i >= j
            val = tl.where(lower, exp_val, 0.0)
            # Store into L[n, h, t, i, j]
            tl.store(L_ptr + pid_n * stride_L_n + pid_h * stride_L_h + pid_t * stride_L_t + i * stride_L_i + j * stride_L_j, val)
            j += 1
        i += 1


@triton.jit
def _contract_B_and_C_to_G(B_ptr, C_ptr, G_ptr,
                           N, T, L, H, N_GROUPS, K,
                           stride_B_n, stride_B_t, stride_B_l, stride_B_g, stride_B_k,
                           stride_C_n, stride_C_t, stride_C_l, stride_C_g, stride_C_k,
                           stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h):
    # Grid over (n, t, i, j, h)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    acc = 0.0
    # Sum over groups (N_GROUPS=8) and K (32)
    g = 0
    while g < N_GROUPS:
        k = 0
        while k < K:
            b_val = tl.load(B_ptr + pid_n * stride_B_n + pid_t * stride_B_t + pid_j * stride_B_l + g * stride_B_g + k * stride_B_k)
            c_val = tl.load(C_ptr + pid_n * stride_C_n + pid_t * stride_C_t + pid_i * stride_C_l + g * stride_C_g + k * stride_C_k)
            acc += b_val * c_val
            k += 1
        g += 1

    # Store G[n, t, i, j, h]
    tl.store(G_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_i * stride_G_l_i + pid_j * stride_G_l_j + pid_h * stride_G_h, acc)


@triton.jit
def _apply_mask_and_store_M(G_ptr, L_ptr, M_ptr,
                            N, T, L, H,
                            stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h,
                            stride_L_n, stride_L_h, stride_L_t, stride_L_i, stride_L_j,
                            stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j, stride_M_h):
    # Grid over (n, t, i, j, h)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    g_val = tl.load(G_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_i * stride_G_l_i + pid_j * stride_G_l_j + pid_h * stride_G_h)
    l_val = tl.load(L_ptr + pid_n * stride_L_n + pid_h * stride_L_h + pid_t * stride_L_t + pid_i * stride_L_i + pid_j * stride_L_j)

    # Apply lower-triangular condition: i >= j
    lower = pid_i >= pid_j
    m_val = tl.where(lower, g_val * l_val, 0.0)
    tl.store(M_ptr + pid_n * stride_M_n + pid_t * stride_M_t + pid_i * stride_M_l_i + pid_j * stride_M_l_j + pid_h * stride_M_h, m_val)


@triton.jit
def _diag_matvec_sum_M_and_HS(M_ptr, HS_ptr, Y_ptr,
                              N, T, L, H, D,
                              stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j, stride_M_h,
                              stride_HS_n, stride_HS_t, stride_HS_l, stride_HS_h, stride_HS_d,
                              stride_Y_n, stride_Y_t, stride_Y_l, stride_Y_h, stride_Y_d):
    # Grid over (n, t, i, h, d)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_d = tl.program_id(4)

    acc = 0.0
    j = 0
    while j < L:
        # Lower-triangular condition: i >= j
        lower = pid_i >= j
        m_val = tl.load(M_ptr + pid_n * stride_M_n + pid_t * stride_M_t + pid_i * stride_M_l_i + j * stride_M_l_j + pid_h * stride_M_h)
        hs_val = tl.load(HS_ptr + pid_n * stride_HS_n + pid_t * stride_HS_t + j * stride_HS_l + pid_h * stride_HS_h + pid_d * stride_HS_d)
        m_hs = m_val * hs_val
        acc += tl.where(lower, m_hs, 0.0)
        j += 1

    tl.store(Y_ptr + pid_n * stride_Y_n + pid_t * stride_Y_t + pid_i * stride_Y_l + pid_h * stride_Y_h + pid_d * stride_Y_d, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        # Extract shapes
        N, H, T, L, D = hidden_states.shape
        N_GROUPS = 8
        K = 32  # original code uses state_size=32

        device = hidden_states.device

        # Reconstruct A_cumsum from hidden_states to ensure shape compatibility ([N, H, T, L])
        # Note: original run function assumes A_cumsum exists; we replicate that assumption here.
        A = hidden_states[:, :, :, :]  # same last three dims as hidden_states (H, T, L)
        A = A.permute(0, 2, 3, 1)  # [N, T, L, H]
        A = A.permute(0, 3, 1, 2)  # [N, H, T, L]

        # Ensure A is float32 for exp
        A = A.to(torch.float32)

        # 1) Build L: [N, H, T, L, L]
        L_tensor = torch.empty((N, H, T, L, L), device=device, dtype=torch.float32)
        grid_L = (N, H, T)
        _build_L_from_hidden_and_A[grid_L](
            A, L_tensor,
            N, H, T, L,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L_tensor.stride(0), L_tensor.stride(1), L_tensor.stride(2), L_tensor.stride(3), L_tensor.stride(4),
            num_warps=1, num_stages=1,
        )

        # 2) Contract B @ C^T -> G: [N, T, L, L, H]
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)

        G = torch.empty((N, T, L, L, H), device=device, dtype=torch.float32)
        grid_G = (N, T, L, L, H)
        _contract_B_and_C_to_G[grid_G](
            B_f32, C_f32, G,
            N, T, L, H, N_GROUPS, K,
            B_f32.stride(0), B_f32.stride(1), B_f32.stride(2), B_f32.stride(3), B_f32.stride(4),
            C_f32.stride(0), C_f32.stride(1), C_f32.stride(2), C_f32.stride(3), C_f32.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            num_warps=1, num_stages=1,
        )

        # 3) Apply mask L to G -> M
        M = torch.empty((N, T, L, L, H), device=device, dtype=torch.float32)
        grid_M = (N, T, L, L, H)
        _apply_mask_and_store_M[grid_M](
            G, L_tensor, M,
            N, T, L, H,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L_tensor.stride(0), L_tensor.stride(1), L_tensor.stride(2), L_tensor.stride(3), L_tensor.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=1, num_stages=1,
        )

        # 4) Diagonal matvec: Y[n, t, i, h, d] = sum_j M[n, t, i, j, h] * HS[n, t, j, h, d]
        HS_f32 = hidden_states.to(torch.float32)
        Y = torch.empty((N, T, L, H, D), device=device, dtype=torch.float32)
        grid_Y = (N, T, L, H, D)
        _diag_matvec_sum_M_and_HS[grid_Y](
            M, HS_f32, Y,
            N, T, L, H, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            HS_f32.stride(0), HS_f32.stride(1), HS_f32.stride(2), HS_f32.stride(3), HS_f32.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=1, num_stages=1,
        )

        # Return in bfloat16 to match original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
