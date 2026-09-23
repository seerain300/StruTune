import torch
import triton
import triton.language as tl


# Triton kernel: compute L[b, h, n, i, j] = exp(sum_{ii<=i} A[b, h, n, ii]) for j <= i; else 0.
# We expand A_cumsum to [B, H, N, S, S] and mask j>i -> 0, then cumsum along i (S-dim), then exp.
@triton.jit
def cumsum_exp_tril_kernel(
    A_ptr, L_ptr,
    Bsz, H, N, S,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid over (b, h, n)
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)

    # For each (b,h,n), compute cumsum along i then exp, j<=i
    for i in range(S):
        prefix = 0.0
        for j in range(S):
            # Load A[b, h, n, i]
            a_val = tl.load(A_ptr + b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_s)
            if j <= i:
                prefix += a_val
            else:
                prefix += 0.0
            # Store exp(prefix) to L[b, h, n, i, j]
            tl.store(L_ptr + b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_s1 + j * L_stride_s2, tl.exp(prefix))


# Triton kernel: compute G[i, j, h] = sum_d C[b, n, i, h, d] * B[b, n, j, h, d]
# Grid over (b, n, i, j, h). We loop over d in tiles of 64 (as in provided inputs).
@triton.jit
def g_contract_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, N, S, H, D,
    B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
    C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    g_val = 0.0
    # sum over d from 0 to D-1
    for d in range(0, D):
        b_val = tl.load(B_ptr + b * B_stride_b + n * B_stride_n + j * B_stride_s + h * B_stride_h + d * B_stride_d)
        c_val = tl.load(C_ptr + b * C_stride_b + n * C_stride_n + i * C_stride_s + h * C_stride_h + d * C_stride_d)
        g_val += b_val * c_val
    tl.store(G_ptr + b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h, g_val)


# Triton kernel: elementwise multiply M = G * L with L permuted to [B, N, S, S, H]
@triton.jit
def m_mul_kernel(
    G_ptr, L_perm_ptr, M_ptr,
    Bsz, N, S, H,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    g_val = tl.load(G_ptr + b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h)
    l_val = tl.load(L_perm_ptr + b * L_stride_b + L_stride_h + n * L_stride_n + i * L_stride_s1 + j * L_stride_s2)
    m_val = g_val * l_val
    tl.store(M_ptr + b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h, m_val)


# Triton kernel: compute Y[b, n, i, h] = sum_j M[b, n, i, j, h] * hidden[b, n, j, h]
# We loop over j from 0 to S-1 and over D from 0 to D-1, accumulate dot.
@triton.jit
def y_diag_reduce_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    Bsz, N, S, H, D,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
    Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    acc = 0.0
    # Accumulate over j
    for j in range(S):
        # dot over D
        dot = 0.0
        for d in range(D):
            m_val = tl.load(M_ptr + b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h)
            hid_val = tl.load(hidden_ptr + b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_s + h * hidden_stride_h + d * hidden_stride_d)
            dot += m_val * hid_val
        acc += dot
    tl.store(Y_ptr + b * Y_stride_b + n * Y_stride_n + i * Y_stride_s1 + h * Y_stride_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure contiguity and dtypes
        Bsz, N, S, H, D = hidden_states.shape
        assert A_cumsum.shape == (Bsz, H, N, S), "A_cumsum shape must be [B, H, N, S]"
        # Expand A to [B, H, N, S, S]
        A = A_cumsum  # keep as is; we will expand in the kernel implicitly by striding
        # Prepare output buffers
        device = hidden_states.device

        # 1) Triton: L = exp(cumsum(masked A)) for j <= i
        L = torch.empty((Bsz, H, N, S, S), dtype=torch.float32, device=device)

        A_stride_b, A_stride_h, A_stride_n, A_stride_s = A.stride()
        L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2 = L.stride()

        cumsum_exp_tril_kernel[(Bsz, H, N)](
            A, L,
            Bsz, H, N, S,
            A_stride_b, A_stride_h, A_stride_n, A_stride_s,
            L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
            num_warps=1, num_stages=1
        )

        # 2) Triton: G contraction
        B_exp = B  # already expanded to H
        C_exp = C  # already expanded to H
        G = torch.empty((Bsz, N, S, S, H), dtype=torch.float32, device=device)

        B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d = B_exp.stride()
        C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d = C_exp.stride()
        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()

        g_contract_kernel[(Bsz, N, S, S, H)](
            B_exp, C_exp, G,
            Bsz, N, S, H, D,
            B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
            C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            num_warps=1, num_stages=1
        )

        # 3) Triton: M = G * L
        L_perm = L.permute(0, 2, 3, 4, 1).contiguous()  # [B, N, S, S, H]
        M = torch.empty_like(G)  # [B, N, S, S, H], float32

        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()
        L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2 = L_perm.stride()
        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()

        m_mul_kernel[(Bsz, N, S, S, H)](
            G, L_perm, M,
            Bsz, N, S, H,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            num_warps=1, num_stages=1
        )

        # 4) Triton: Y_diag reduction
        hidden = hidden_states.contiguous()  # [B, N, S, H, D]
        Y = torch.empty((Bsz, N, S, H), dtype=torch.float32, device=device)

        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()
        hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d = hidden.stride()
        Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h = Y.stride()

        y_diag_reduce_kernel[(Bsz, N, S, H)](
            M, hidden, Y,
            Bsz, N, S, H, D,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
            Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
            num_warps=1, num_stages=1
        )

        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
