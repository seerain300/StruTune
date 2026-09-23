import torch
import triton
import triton.language as tl

# Kernel 1: Compute L = exp(masked_cumsum(A, dim source j)) with lower-triangular mask (j <= i)
# A: [B, C, S, N_groups], L: [B, C, S, S, N_groups]
@triton.jit
def masked_cumsum_exp_kernel(
    A_ptr, L_ptr,
    B: tl.constexpr, C: tl.constexpr, S: tl.constexpr, N_groups: tl.constexpr,
    A_stride_b, A_stride_c, A_stride_s, A_stride_ng,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_ng,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    n_group = tl.program_id(2)  # n_group in [0, N_groups)
    i = tl.program_id(3)        # source position i in [0, S)

    # Running sum for cumsum along j for fixed (b, c, n_group, i)
    running_sum = 0.0
    for j in range(S):
        # Load A[b, c, j, n_group]
        a_ptr = A_ptr + b * A_stride_b + c * A_stride_c + j * A_stride_s + n_group * A_stride_ng
        a_val = tl.load(a_ptr)
        # Apply mask: include if j <= i, exclude if j > i
        include = j <= i
        contrib = a_val if include else 0.0
        running_sum += contrib
        # Store exp(running_sum) to L[b, c, i, j, n_group]
        l_ptr = L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n_group * L_stride_ng
        tl.store(l_ptr, tl.exp(running_sum))


# Kernel 2: Contract B_expanded and C_expanded to form G: G[b, c, i, j, n] = sum_k C_expanded[b,c,i,n,k] * B_expanded[b,c,j,n,k]
# B_expanded: [B, C, S, N, K], C_expanded: [B, C, S, N, K], G: [B, C, S, S, N]
@triton.jit
def contract_BC_to_G_kernel(
    B_exp_ptr, C_exp_ptr, G_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    B_exp_stride_b, B_exp_stride_c, B_exp_stride_s, B_exp_stride_n, B_exp_stride_k,
    C_exp_stride_b, C_exp_stride_c, C_exp_stride_s, C_exp_stride_n, C_exp_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    acc = 0.0
    for k in range(K):
        b_ptr = B_exp_ptr + b * B_exp_stride_b + c * B_exp_stride_c + j * B_exp_stride_s + n * B_exp_stride_n + k * B_exp_stride_k
        c_ptr = C_exp_ptr + b * C_exp_stride_b + c * C_exp_stride_c + i * C_exp_stride_s + n * C_exp_stride_n + k * C_exp_stride_k
        b_val = tl.load(b_ptr)
        c_val = tl.load(c_ptr)
        acc += b_val * c_val

    # Store acc to G[b, c, i, j, n]
    g_ptr = G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    tl.store(g_ptr, acc)


# Kernel 3: Diagonal contraction to produce Y: Y[b, c, i, n, d] = sum_j M[b,c,i,j,n] * hidden_states[b,c,j,n,d]
# M: [B, C, S, S, N], hidden_states: [B, C, S, N, D], Y: [B, C, S, N, D]
@triton.jit
def diag_contract_Y_kernel(
    M_ptr, HS_ptr, Y_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in range(S):
        m_ptr = M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n
        hs_ptr = HS_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d
        m_val = tl.load(m_ptr)
        hs_val = tl.load(hs_ptr)
        acc += m_val * hs_val

    y_ptr = Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d
    tl.store(y_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on CUDA and float32 for computation
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Tensors must be on CUDA."
        device = hidden_states.device

        # Extract dynamic axes: hidden_states [B, C, S, N, D]
        Bsz, Csz, S, N, D = hidden_states.shape
        N_groups = A_cumsum.shape[3]  # 8
        K = D  # state_size equals head_dim here

        # Cast to float32 for computation
        A = A_cumsum.to(torch.float32)
        B_hs = B.to(torch.float32)
        C_hs = C.to(torch.float32)

        # Expand B and C to num_heads (N=32) via repeat_interleave(EXPAND_FACTOR=4)
        B_expanded = B_hs.repeat_interleave(4, dim=3)  # [Bsz, Csz, S, 32, K]
        C_expanded = C_hs.repeat_interleave(4, dim=3)  # [Bsz, Csz, S, 32, K]

        # 1) Compute L = exp(masked_cumsum(A, dim source j)) with lower-triangular mask (j <= i), shape [B, C, S, S, N_groups]
        L = torch.empty((Bsz, Csz, S, S, N_groups), device=device, dtype=torch.float32)
        grid_L = (Bsz, Csz, N_groups, S)
        masked_cumsum_exp_kernel[grid_L](
            A, L,
            Bsz, Csz, S, N_groups,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=1, num_stages=1,
        )

        # 2) Compute G = contract(B_expanded, C_expanded): G[b, c, i, j, n] in Triton
        G = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)
        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G_kernel[grid_G](
            B_expanded, C_expanded, G,
            Bsz, Csz, S, N, K,
            B_expanded.stride(0), B_expanded.stride(1), B_expanded.stride(2), B_expanded.stride(3), B_expanded.stride(4),
            C_expanded.stride(0), C_expanded.stride(1), C_expanded.stride(2), C_expanded.stride(3), C_expanded.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            num_warps=1, num_stages=1,
        )

        # 3) Elementwise M = G * L (PyTorch elementwise multiply; not reduction-heavy)
        M = G * L  # float32

        # 4) Final diagonal contraction to produce Y[b, c, i, n, d]
        Y = torch.empty((Bsz, Csz, S, N, D), device=device, dtype=torch.float32)
        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y_kernel[grid_Y](
            M, hidden_states.to(torch.float32), Y,
            Bsz, Csz, S, N, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_states.to(torch.float32).stride(0), hidden_states.to(torch.float32).stride(1), hidden_states.to(torch.float32).stride(2), hidden_states.to(torch.float32).stride(3), hidden_states.to(torch.float32).stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=1, num_stages=1,
        )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
