import torch
import triton
import triton.language as tl


@triton.jit
def _contract_bc_to_g(B_ptr, C_ptr, G_ptr,
                      N, T, L, G, K,
                      stride_B_n, stride_B_t, stride_B_l, stride_B_g, stride_B_k,
                      stride_C_n, stride_C_t, stride_C_l, stride_C_g, stride_C_k,
                      stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h,
                      H,
                      num_warps: tl.constexpr, num_stages: tl.constexpr):
    # Grid over (N, T, L, L, H)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    acc = 0.0
    # Sum over groups g and K
    g = 0
    while g < G:
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
def _apply_mask(G_ptr, L_ptr, M_ptr,
                N, T, L, H,
                stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h,
                stride_L_n, stride_L_h, stride_L_t, stride_L_i, stride_L_j,
                stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j, stride_M_h,
                num_warps: tl.constexpr, num_stages: tl.constexpr):
    # Grid over (N, T, L, L, H)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    g_val = tl.load(G_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_i * stride_G_l_i + pid_j * stride_G_l_j + pid_h * stride_G_h)
    l_val = tl.load(L_ptr + pid_n * stride_L_n + pid_t * stride_L_t + pid_i * stride_L_i + pid_j * stride_L_j + pid_h * stride_L_h)
    tl.store(M_ptr + pid_n * stride_M_n + pid_t * stride_M_t + pid_i * stride_M_l_i + pid_j * stride_M_l_j + pid_h * stride_M_h, g_val * l_val)


@triton.jit
def _diag_matvec_sum(G_masked_ptr, HS_ptr, Y_ptr,
                     N, T, L, H, D,
                     stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h,
                     stride_HS_n, stride_HS_t, stride_HS_l, stride_HS_h, stride_HS_d,
                     stride_Y_n, stride_Y_t, stride_Y_l, stride_Y_h, stride_Y_d,
                     num_warps: tl.constexpr, num_stages: tl.constexpr):
    # Grid over (N, T, H, D)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_d = tl.program_id(3)

    acc = 0.0
    j = 0
    while j < L:
        g_val = tl.load(G_masked_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_j * stride_G_l_i + j * stride_G_l_j + pid_h * stride_G_h)
        hs_val = tl.load(HS_ptr + pid_n * stride_HS_n + pid_t * stride_HS_t + j * stride_HS_l + pid_h * stride_HS_h + pid_d * stride_HS_d)
        acc += g_val * hs_val
        j += 1

    tl.store(Y_ptr + pid_n * stride_Y_n + pid_t * stride_Y_t + pid_j * stride_Y_l + pid_h * stride_Y_h + pid_d * stride_Y_d, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_diag = sum over j of (M * hidden_states) where M = (G * L) and
        G = B @ C^T over groups, L is lower-triangular exp(A_cumsum).
        We use Triton for contraction and reduction; host computes L using torch.
        """
        assert hidden_states.is_cuda and B.is_cuda and C.is_cuda, "Tensors must be on CUDA."
        N, T, L_hs, H, D = hidden_states.shape

        # Compute L using torch: lower-triangular exp of cumulative sum over chunk dimension
        # A_cumsum expected shape: [N, H, T, L_hs] (matches original). If your environment provides different shapes,
        # adjust the slicing accordingly. Here we assume A_cumsum has L_hs last dim per (n, h, t).
        # We need segment_sum[i, j] = sum_{m=0..j} A_cumsum[n, h, t, i]; then L = exp(segment_sum) for i<=j.
        # Build L as zeros, then fill lower triangle.
        L = torch.zeros((N, H, T, L_hs, L_hs), device=hidden_states.device, dtype=torch.float32)
        for n in range(N):
            for h in range(H):
                for t in range(T):
                    # segment_sum per row i
                    segment_sum = torch.zeros((L_hs, L_hs), device=hidden_states.device, dtype=torch.float32)
                    # sum over m from 0 to j (we iterate j to compute segment_sum[i,j])
                    for j in range(L_hs):
                        # A_cumsum[n, h, t, i] for i in [0..L_hs-1]
                        a_row = A_cumsum[n, h, t, :].to(torch.float32)  # shape [L_hs]
                        # For each i, add A_cumsum[n, h, t, i] to all positions (i, j) where i <= j
                        i_vals = torch.arange(L_hs, device=hidden_states.device)
                        # only add for i <= j
                        mask_i_le_j = i_vals <= j
                        segment_sum[j, :] = torch.where(mask_i_le_j, segment_sum[j, :] + a_row, segment_sum[j, :])
                    # L[n, h, t, i, j] = exp(segment_sum[i, j]) for i <= j
                    # populate lower triangle
                    for i in range(L_hs):
                        L[n, h, t, i, :].copy_(
                            torch.exp(segment_sum[i, :]).to(torch.float32)
                        )
        # Ensure L is contiguous in memory (not strictly necessary for Triton load)
        L = L.contiguous()

        # Prepare B, C in float32
        B_f32 = B.contiguous().to(torch.float32)
        C_f32 = C.contiguous().to(torch.float32)

        # Output G, M, and final Y (float32 for stability, then cast to bfloat16)
        G = torch.empty((N, T, L_hs, L_hs, H), device=hidden_states.device, dtype=torch.float32)
        grid_contr = (N, T, L_hs, L_hs, H)
        _contract_bc_to_g[grid_contr](
            B_f32, C_f32, G,
            N, T, L_hs, 8, 32,
            B_f32.stride(0), B_f32.stride(1), B_f32.stride(2), B_f32.stride(3), B_f32.stride(4),
            C_f32.stride(0), C_f32.stride(1), C_f32.stride(2), C_f32.stride(3), C_f32.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            H,
            num_warps=1, num_stages=1,
        )

        # Apply mask L to G -> M
        M = torch.empty((N, T, L_hs, L_hs, H), device=hidden_states.device, dtype=torch.float32)
        grid_mask = grid_contr
        _apply_mask[grid_mask](
            G, L, M,
            N, T, L_hs, H,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=1, num_stages=1,
        )

        # hidden_states is already [N, T, L, H, D]; ensure float32 and contiguous
        HS = hidden_states.contiguous().to(torch.float32)

        # Compute Y_diag via diagonal matvec over j, i.e., sum_j M[..., j] * HS[..., j]
        Y = torch.empty((N, T, L_hs, H, D), device=hidden_states.device, dtype=torch.float32)
        grid_diag = (N, T, H, D)
        _diag_matvec_sum[grid_diag](
            M, HS, Y,
            N, T, L_hs, H, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            HS.stride(0), HS.stride(1), HS.stride(2), HS.stride(3), HS.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=1, num_stages=1,
        )

        # Return in bfloat16 to match original run function
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
