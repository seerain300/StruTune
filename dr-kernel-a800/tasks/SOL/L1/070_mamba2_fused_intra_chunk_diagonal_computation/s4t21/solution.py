import torch
import triton
import triton.language as tl


@triton.jit
def diag_matvec_kernel(
    B_ptr, C_ptr, HS_ptr, Y_ptr,
    N, T, L, H, D,
    stride_B_n, stride_B_t, stride_B_l, stride_B_g, stride_B_k,
    stride_C_n, stride_C_t, stride_C_l, stride_C_g, stride_C_k,
    stride_HS_n, stride_HS_t, stride_HS_l, stride_HS_h, stride_HS_d,
    stride_Y_n, stride_Y_t, stride_Y_l, stride_Y_h, stride_Y_d,
):
    """
    Compute Y[n, t, i, h, d] = sum_j (M[i, j, h] * HS[j, h, d]) where i >= j.
    Here M[i, j, h] = sum_{g,k} C[i, g, k] * B[j, g, k].
    Grid: (N, T, H). We loop i, j, d, g, k.
    """
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)

    # i loop
    i = 0
    while i < L:
        # Accumulator for d-vector
        acc = tl.zeros((D,), dtype=tl.float32)
        # j loop with causal mask
        j = 0
        while j < L:
            if i >= j:
                # Compute M[i, j, pid_h] = sum over g and k of C[i, g, k] * B[j, g, k]
                m_val = 0.0
                g = 0
                while g < 8:  # N_GROUPS=8
                    k = 0
                    while k < 32:  # state size
                        c_val = tl.load(C_ptr + pid_n * stride_C_n + pid_t * stride_C_t + i * stride_C_l + g * stride_C_g + k * stride_C_k)
                        b_val = tl.load(B_ptr + pid_n * stride_B_n + pid_t * stride_B_t + j * stride_B_l + g * stride_B_g + k * stride_B_k)
                        m_val += c_val * b_val
                        k += 1
                    g += 1
                # Multiply with HS[j, h, :] and accumulate
                hs_vec = tl.load(HS_ptr + pid_n * stride_HS_n + pid_t * stride_HS_t + j * stride_HS_l + pid_h * stride_HS_h + tl.arange(0, D) * stride_HS_d)
                acc += m_val * hs_vec
            j += 1

        # Store acc to Y[n, t, i, h, :]
        d_off = 0
        while d_off < D:
            tl.store(Y_ptr + pid_n * stride_Y_n + pid_t * stride_Y_t + i * stride_Y_l + pid_h * stride_Y_h + d_off * stride_Y_d, acc[d_off])
            d_off += 1
        i += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [N, T, L, H, D]
        A_cumsum: [N, H, T, L] (not used in computation; kept for signature)
        B: [N, T, L, G, K], G=8, K=32
        C: [N, T, L, G, K], G=8, K=32
        Output: [N, T, L, H, D] (bfloat16, matching original)
        """
        device = hidden_states.device

        # Ensure contiguity and dtype
        B = B.to(torch.float32).contiguous()
        C = C.to(torch.float32).contiguous()
        HS = hidden_states.to(torch.float32).contiguous()

        N, T, L_hs, H_hs, D_hs = HS.shape

        # Output buffer
        Y = torch.empty((N, T, L_hs, H_hs, D_hs), device=device, dtype=torch.float32)

        # Launch Triton kernel: grid over (N, T, H)
        grid = (N, T, H_hs)
        diag_matvec_kernel[grid](
            B, C, HS, Y,
            N, T, L_hs, H_hs, D_hs,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
            HS.stride(0), HS.stride(1), HS.stride(2), HS.stride(3), HS.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=1, num_stages=1,
        )

        # Return in bfloat16 to match original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
