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

    i = 0
    while i < L:
        # We'll compute segment_sum[i, j] for all j in [0..L-1]
        j = 0
        while j < L:
            segment_sum = 0.0
            # Sum over m from 0 to j; if i > j, we won't use it (masked later)
            m = 0
            while m <= j:
                a_val = tl.load(
                    A_ptr + pid_n * stride_A_n + pid_h * stride_A_h + pid_t * stride_A_t + i * stride_A_l,
                    mask=(i < L) & (m < L),
                    other=0.0,
                )
                segment_sum += a_val
                m += 1
            l_val = segment_sum
            if i <= j:
                l_val = tl.exp(l_val)
            tl.store(
                L_ptr + pid_n * stride_L_n + pid_h * stride_L_h + pid_t * stride_L_t + i * stride_L_i + j * stride_L_j,
                l_val,
                mask=(i < L) & (j < L),
            )
        j += 1
    i += 1

# Kernel 2: Contract B @ C^T to form G: G[n,t,i,j,g] = sum_k C[n,t,i,g,k]*B[n,t,j,g,k]
# Inputs: B [N,T,L,G,K], C [N,T,L,G,K], Output Gout [N,T,L,L,G]
@triton.jit
def _contract_to_G_out(
    B_ptr, C_ptr, Gout_ptr,
    N, T, L, G, K,
    stride_B_n, stride_B_t, stride_B_l, stride_B_g, stride_B_k,
    stride_C_n, stride_C_t, stride_C_l, stride_C_g, stride_C_k,
    stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_g,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_li = tl.program_id(2)
    pid_lj = tl.program_id(3)
    pid_g = tl.program_id(4)

    acc = 0.0
    k = 0
    while k < K:
        b_val = tl.load(
            B_ptr + pid_n * stride_B_n + pid_t * stride_B_t + pid_lj * stride_B_l + pid_g * stride_B_g + k * stride_B_k
        )
        c_val = tl.load(
            C_ptr + pid_n * stride_C_n + pid_t * stride_C_t + pid_li * stride_C_l + pid_g * stride_C_g + k * stride_C_k
        )
        acc += b_val * c_val
        k += 1
    tl.store(
        Gout_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_li * stride_G_l_i + pid_lj * stride_G_l_j + pid_g * stride_G_g,
        acc
    )

# Kernel 3: Apply lower-triangular mask to G: M = G * L where i >= j
# Inputs: G [N,T,L,L,G], L [N,H=T,T,L,L] (we pass as H=T), Output M [N,T,L,L,G]
@triton.jit
def _apply_mask_to_G(
    G_ptr, L_ptr, M_ptr,
    N, T, L, G,
    stride_G_n, stride_G_t, stride_G_l_i, stride_G_l_j, stride_G_g,
    stride_L_n, stride_L_h, stride_L_t, stride_L_i, stride_L_j,
    stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j, stride_M_g,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_li = tl.program_id(2)
    pid_lj = tl.program_id(3)
    pid_g = tl.program_id(4)

    # Read G and L
    g_val = tl.load(G_ptr + pid_n * stride_G_n + pid_t * stride_G_t + pid_li * stride_G_l_i + pid_lj * stride_G_l_j + pid_g * stride_G_g)
    l_val = tl.load(L_ptr + pid_n * stride_L_n + pid_t * stride_L_h + pid_t * stride_L_t + pid_li * stride_L_i + pid_lj * stride_L_j)  # L is per (n,h,t)
    # Apply lower-triangular condition i >= j
    if pid_li >= pid_lj:
        m_val = g_val * l_val
    else:
        m_val = 0.0
    tl.store(M_ptr + pid_n * stride_M_n + pid_t * stride_M_t + pid_li * stride_M_l_i + pid_lj * stride_M_l_j + pid_g * stride_M_g, m_val)

# Kernel 4: Diagonal matvec: Y[n,t,i,g,d] = sum_j M[n,t,i,j,g] * HS[n,t,j,g,d]
# Inputs: M [N,T,L,L,G], HS [N,T,L,G,D], Output Y [N,T,L,G,D]
@triton.jit
def _diag_matvec_sum(
    M_ptr, HS_ptr, Y_ptr,
    N, T, L, G, D,
    stride_M_n, stride_M_t, stride_M_l_i, stride_M_l_j, stride_M_g,
    stride_HS_n, stride_HS_t, stride_HS_l, stride_HS_g, stride_HS_d,
    stride_Y_n, stride_Y_t, stride_Y_l, stride_Y_g, stride_Y_d,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_li = tl.program_id(2)
    pid_g = tl.program_id(3)
    pid_d = tl.program_id(4)

    acc = 0.0
    j = 0
    while j < L:
        m_val = tl.load(M_ptr + pid_n * stride_M_n + pid_t * stride_M_t + pid_li * stride_M_l_i + j * stride_M_l_j + pid_g * stride_M_g)
        hs_val = tl.load(HS_ptr + pid_n * stride_HS_n + pid_t * stride_HS_t + j * stride_HS_l + pid_g * stride_HS_g + pid_d * stride_HS_d)
        acc += m_val * hs_val
        j += 1
    tl.store(Y_ptr + pid_n * stride_Y_n + pid_t * stride_Y_t + pid_li * stride_Y_l + pid_g * stride_Y_g + pid_d * stride_Y_d, acc)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        # Shapes from original:
        # hidden_states: [N, T, L, H, D]
        # A_cumsum: [N, H, T, L]
        # B: [N, T, L, G, K] (G=8, K=32)
        # C: [N, T, L, G, K]
        assert hidden_states.ndim == 5, "hidden_states must be [N, T, L, H, D]"
        assert A_cumsum.ndim == 4, "A_cumsum must be [N, H, T, L]"
        assert B.ndim == 5 and C.ndim == 5, "B and C must be [N, T, L, G, K]"
        assert B.shape == C.shape, "B and C must have the same shape"
        N, T, L, H, D = hidden_states.shape
        # NUM_HEADS = H, N_GROUPS = 8 in original
        assert H % 8 == 0, "H must be divisible by N_GROUPS=8"
        G = 8
        K = B.shape[-1]  # K must equal C.shape[-1]

        device = hidden_states.device
        # Triton computations: all tensors in float32
        # 1) Build L: [N, H, T, L, L]
        L_out = torch.empty((N, H, T, L, L), device=device, dtype=torch.float32)
        grid_L = (N, H, T)
        _build_lower_tri_exp_kernel[grid_L](
            A_cumsum, L_out,
            N, H, T, L,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L_out.stride(0), L_out.stride(1), L_out.stride(2), L_out.stride(3), L_out.stride(4),
            num_warps=1, num_stages=1,
        )

        # 2) Contract B @ C^T to G: [N, T, L, L, G]
        Gout = torch.empty((N, T, L, L, G), device=device, dtype=torch.float32)
        grid_G = (N, T, L, L, G)
        _contract_to_G_out[grid_G](
            B, C, Gout,
            N, T, L, G, K,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            num_warps=1, num_stages=1,
        )

        # 3) Apply mask: M = G * L with lower-triangular condition
        M = torch.empty((N, T, L, L, G), device=device, dtype=torch.float32)
        grid_M = (N, T, L, L, G)
        _apply_mask_to_G[grid_M](
            Gout, L_out, M,
            N, T, L, G,
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
            L_out.stride(0), L_out.stride(1), L_out.stride(2), L_out.stride(3), L_out.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=1, num_stages=1,
        )

        # 4) Diagonal matvec: Y[n,t,i,g,d] = sum_j M[n,t,i,j,g] * HS[n,t,j,g,d]
        # hidden_states shape [N, T, L, H, D], we use H as input heads, mapping to G groups:
        # g = h // (H//G) = h // 4 since H=32, G=8
        HS_in = hidden_states.to(torch.float32)  # [N, T, L, H, D]
        Y = torch.empty((N, T, L, G, D), device=device, dtype=torch.float32)
        grid_Y = (N, T, L, G, D)
        _diag_matvec_sum[grid_Y](
            M, HS_in, Y,
            N, T, L, G, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            HS_in.stride(0), HS_in.stride(1), HS_in.stride(2), HS_in.stride(3), HS_in.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=1, num_stages=1,
        )

        # Return in bfloat16 to match original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
