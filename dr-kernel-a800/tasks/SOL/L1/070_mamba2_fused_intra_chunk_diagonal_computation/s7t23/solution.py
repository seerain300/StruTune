import torch
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_exp_kernel(
    A_ptr, L_ptr,
    Bsz, Csz, S, N_groups,
    A_stride_b, A_stride_c, A_stride_i, A_stride_n,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    # Grid: (b, c, n_group, i)
    b = tl.program_id(0)
    c = tl.program_id(1)
    n_group = tl.program_id(2)
    i = tl.program_id(3)

    # Running sum over j (source index). Include j <= i (diagonal=0).
    running = 0.0
    for j in range(0, S):
        if j <= i:
            a_ptr = A_ptr + b * A_stride_b + c * A_stride_c + j * A_stride_i + n_group * A_stride_n
            a_val = tl.load(a_ptr)
            running += a_val
        # else: exclude j > i
        # store exp(running) to L[b, c, i, j, n_group]
        l_ptr = L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n_group * L_stride_n
        tl.store(l_ptr, tl.exp(running))


@triton.jit
def contract_BC_to_G(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz, S, N, D,
    B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
    C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    # Grid: (b, c, i, j, n)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    # Accumulate G[i, j, n] = sum_k B[j, n, k] * C[i, n, k]
    acc = 0.0
    for k in range(0, D):
        b_ptr = B_ptr + b * B_stride_b + c * B_stride_c + j * B_stride_j + n * B_stride_n + k * B_stride_k
        c_ptr = C_ptr + b * C_stride_b + c * C_stride_c + i * C_stride_i + n * C_stride_n + k * C_stride_k
        b_val = tl.load(b_ptr)
        c_val = tl.load(c_ptr)
        acc += b_val * c_val

    g_ptr = G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    tl.store(g_ptr, acc)


@triton.jit
def diag_contract_Y(
    M_ptr, HS_ptr, Y_ptr,
    Bsz, Csz, S, N, D,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    # Grid: (b, c, i, n, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    # Accumulate Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * HS[b, c, j, n, d]
    acc = 0.0
    for j in range(0, S):
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
        """
        Compute Y_diag as per the original function:
        hidden_states: [B, C, S, N, D]
        A_cumsum: [B, C, S, N_groups] (N_groups=8)
        B, C: [B, C, S, N_groups, D]
        Output: [B, C, S, N, D], cast to bfloat16.
        """
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Tensors must be on CUDA."

        # Dynamic axes from hidden_states: [B, C, S, N, D]
        Bsz, Csz, S, N, D = hidden_states.shape
        N_groups = A_cumsum.shape[3]  # 8

        # Cast to float32 for computation
        A = A_cumsum.to(torch.float32)
        B_hs = B.to(torch.float32)
        C_hs = C.to(torch.float32)

        # Expand B and C to num_heads=32 via repeat_interleave(EXPAND_FACTOR=4)
        EXPAND_FACTOR = N // N_groups  # 4
        B_expanded = B_hs.repeat_interleave(EXPAND_FACTOR, dim=3)  # [B, C, S, 32, D]
        C_expanded = C_hs.repeat_interleave(EXPAND_FACTOR, dim=3)  # [B, C, S, 32, D]

        # 1) Compute L = exp(masked_cumsum(A, dim=-2)) with lower-triangular mask (diagonal=0) in Triton:
        #    L[b, c, i, j, n_group] = exp(cumsum(A[b, c, :, n_group]) masked by j <= i)
        L = torch.empty((Bsz, Csz, S, S, N_groups), device=hidden_states.device, dtype=torch.float32)
        grid_L = (Bsz, Csz, N_groups, S)
        masked_cumsum_exp_kernel[grid_L](
            A, L,
            Bsz, Csz, S, N_groups,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=1, num_stages=1,
        )

        # 2) Compute G = contract(B_expanded, C_expanded): G[b, c, i, j, n] in Triton
        G = torch.empty((Bsz, Csz, S, S, N), device=hidden_states.device, dtype=torch.float32)
        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G[grid_G](
            B_expanded, C_expanded, G,
            Bsz, Csz, S, N, D,
            B_expanded.stride(0), B_expanded.stride(1), B_expanded.stride(2), B_expanded.stride(3), B_expanded.stride(4),
            C_expanded.stride(0), C_expanded.stride(1), C_expanded.stride(2), C_expanded.stride(3), C_expanded.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            num_warps=1, num_stages=1,
        )

        # 3) Elementwise M = G * L (PyTorch for simplicity; Triton is not needed here)
        M = G * L  # [B, C, S, S, N], float32

        # 4) Final diagonal contraction to produce Y: [B, C, S, N, D]
        Y = torch.empty((Bsz, Csz, S, N, D), device=hidden_states.device, dtype=torch.float32)
        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y[grid_Y](
            M, hidden_states.to(torch.float32), Y,
            Bsz, Csz, S, N, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=1, num_stages=1,
        )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
