import torch
import triton
import triton.language as tl

# Triton kernel: compute G = B @ C^T over groups and K
# Input: B [N, T, L, G, K], C [N, T, L, G, K]
# Output: Gout [N, T, L, L, H]
@triton.jit
def _contract_bc_to_g(B_ptr, C_ptr, G_ptr,
                      N, T, L, G_GROUPS, K,
                      stride_B_n, stride_B_t, stride_B_l, stride_B_g, stride_B_k,
                      stride_C_n, stride_C_t, stride_C_l, stride_C_g, stride_C_k,
                      stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_l_i = tl.program_id(2)  # i index
    pid_l_j = tl.program_id(3)  # j index
    pid_h = tl.program_id(4)

    # Accumulate G[i, j, h] = sum over groups g and k
    acc = 0.0
    g = 0
    while g < G_GROUPS:
        k = 0
        while k < K:
            # Load B[n, t, j, g, k] and C[n, t, i, g, k]
            b_val = tl.load(B_ptr + pid_n * stride_B_n + pid_t * stride_B_t + pid_l_j * stride_B_l + g * stride_B_g + k * stride_B_k)
            c_val = tl.load(C_ptr + pid_n * stride_C_n + pid_t * stride_C_t + pid_l_i * stride_C_l + g * stride_C_g + k * stride_C_k)
            acc += b_val * c_val
            k += 1
        g += 1
    tl.store(G_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_l_i * stride_G_l_i + pid_l_j * stride_G_l_j + pid_h * stride_G_h, acc)


# Triton kernel: diagonal matvec with lower-triangular mask (i >= j)
# Input: Gout [N, T, L, L, H], HS [N, T, L, H, D]
# Output: Y [N, T, L, H, D]
@triton.jit
def _diag_matvec_sum_ltri(G_ptr, HS_ptr, Y_ptr,
                          N, T, L, H, D,
                          stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_h,
                          stride_HS_n, stride_HS_t, stride_HS_l, stride_HS_h, stride_HS_d,
                          stride_Y_n, stride_Y_t, stride_Y_l, stride_Y_h, stride_Y_d):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_d = tl.program_id(3)  # d index

    # Accumulate over j for fixed i (row). We'll do i in a loop.
    i = 0
    while i < L:
        acc = 0.0
        j = 0
        while j < L:
            # Apply lower-triangular mask: only sum for i >= j
            if i >= j:
                g_val = tl.load(G_ptr + pid_n * stride_G_n + pid_t * stride_G_t + i * stride_G_l_i + j * stride_G_l_j + pid_h * stride_G_h)
                hs_val = tl.load(HS_ptr + pid_n * stride_HS_n + pid_t * stride_HS_t + j * stride_HS_l + pid_h * stride_HS_h + pid_d * stride_HS_d)
                acc += g_val * hs_val
            j += 1
        # Store result into Y[n, t, i, h, d]
        tl.store(Y_ptr + pid_n * stride_Y_n + pid_t * stride_Y_t + i * stride_Y_l + pid_h * stride_Y_h + pid_d * stride_Y_d, acc)
        i += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants implied by original code: chunk_size=128, num_heads=32, n_groups=8
        self.CHUNK_SIZE = 128
        self.NUM_HEADS = 32
        self.N_GROUPS = 8
        # Group and state size (K) — original code uses head_dim=64, K is half (32)
        self.K = 32

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous and float32 for compute stability
        device = hidden_states.device
        N, T, L,


def run(*args):
    return ModelNew()(*args)
