import torch
import triton
import triton.language as tl


# Triton: build a lower-triangular mask M_lower[i, j] = 1 if j <= i else 0
@triton.jit
def tril_mask_kernel(M_ptr, S: tl.constexpr):
    i = tl.program_id(0)
    j = tl.program_id(1)
    if j <= i:
        tl.store(M_ptr + i * S + j, 1)
    else:
        tl.store(M_ptr + i * S + j, 0)


# Triton: compute L[b, h, n, i, j] = exp(cumsum(A_masked[b, h, n, i])) for j <= i; else 0
# A_cumsum is [B, H, N, S], we expand to [B, H, N, S, S], mask j>i -> 0, then cumsum along i (S-dim), then exp.
@triton.jit
def cumsum_exp_tril_kernel(
    A_ptr, L_ptr,
    Bsz, H, N, S,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid is (B, H, N)
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)

    # Loop over i (source positions) and j (target positions)
    # i in [0, S), j in [0, S)
    for i in range(S):
        prefix = 0.0
        for j in range(S):
            # Load A[b, h, n, i]
            val = tl.load(A_ptr + b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_s)
            if j <= i:
                prefix += val
            # Store exp(prefix) to L[b, h, n, i, j]
            tl.store(L_ptr + b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_s1 + j * L_stride_s2, tl.exp(prefix))


# Triton: compute G[b, n, i, j, h] = sum over d of C[b, n, i, h, d] * B[b, n, j, h, d]
# We assume D=64; loops over d. Grid is (B, N, i, j, h).
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

    acc = 0.0
    for d in range(D):
        b_val = tl.load(B_ptr + b * B_stride_b + n * B_stride_n + j * B_stride_s + h * B_stride_h + d * B_stride_d)
        c_val = tl.load(C_ptr + b * C_stride_b + n * C_stride_n + i * C_stride_s + h * C_stride_h + d * C_stride_d)
        acc += b_val * c_val

    tl.store(G_ptr + b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h, acc)


# Triton: M = G * L (elementwise). We need to permute L to [B, N, S, S, H] and multiply.
@triton.jit
def m_mul_kernel(
    G_ptr, L_ptr, M_ptr,
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
    l_val = tl.load(L_ptr + b * L_stride_b + L_stride_h + n * L_stride_n + i * L_stride_s1 + j * L_stride_s2)  # L is [B, N, S, S, H]
    tl.store(M_ptr + b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h, g_val * l_val)


# Triton: Y_diag[b, n, i, h] = sum_j M[b, n, i, j, h] * hidden[b, n, j, h], where hidden is [B, N, S, H, D] and D=64
# Grid over (B, N, i, h), loop over j and D.
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
    for j in range(S):
        # M[b, n, i, j, h]
        m_val = tl.load(M_ptr + b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h)
        # hidden[b, n, j, h, :]
        # We need to multiply m_val with the dot over D: sum_d hidden[b, n, j, h, d]
        # hidden is [B, N, S, H, D]
        for d in range(D):
            h_val = tl.load(hidden_ptr + b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_s + h * hidden_stride_h + d * hidden_stride_d)
            acc += m_val * h_val

    tl.store(Y_ptr + b * Y_stride_b + n * Y_stride_n + i * Y_stride_s1 + h * Y_stride_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_diag as in the original model but using Triton kernels for all numerical operations.
        Returns: [B, N, S, H] in bfloat16.
        Assumes:
          - hidden_states: [B, N, S, H, D], contiguous, D=64
          - A_cumsum: [B, H, N, S], contiguous
          - B, C: [B, N, S, G, D], contiguous; G can be any; we contract to H via h loop
        """
        assert hidden_states.dim() == 5, "hidden_states must be [B, N, S, H, D]"
        assert A_cumsum.dim() == 4, "A_cumsum must be [B, H, N, S]"
        assert B.dim() == 5 and C.dim() == 5, "B and C must be [B, N, S, G, D]"

        Bsz, N, S, H, D = hidden_states.shape
        # For safety and correctness, we enforce D=64 to match provided get_inputs.
        assert D == 64, "This Triton implementation currently assumes D=64."

        device = hidden_states.device

        # 1) Triton: build lower-triangular mask M_lower[S, S]
        M_lower = torch.empty((S, S), dtype=torch.int8, device=device)
        tril_mask_kernel[(S, S)](M_lower, S=S, num_warps=1, num_stages=1)

        # 2) Triton: compute L = exp(cumsum(masked A)) with diagonal=-1
        # A_cumsum: [B,H,N,S] -> expand to [B,H,N,S,S]
        A_expanded = A_cumsum.unsqueeze(-1).expand(Bsz, H, N, S, S).contiguous()  # [B,H,N,S,S]
        L = torch.empty((Bsz, H, N, S, S), dtype=torch.float32, device=device)

        A_stride_b, A_stride_h, A_stride_n, A_stride_s, _ = A_expanded.stride()
        L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2 = L.stride()

        cumsum_exp_tril_kernel[(Bsz, H, N)](
            A_expanded, L,
            Bsz, H, N, S,
            A_stride_b, A_stride_h, A_stride_n, A_stride_s,
            L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
            num_warps=1, num_stages=1
        )

        # 3) Triton: compute G[b, n, i, j, h] = sum_d C[b, n, i, h, d] * B[b, n, j, h, d]
        B_exp = B  # [B, N, S, G, D]
        C_exp = C  # [B, N, S, G, D]
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

        # 4) Triton: M = G * L (permute L to [B,N,S,S,H])
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

        # 5) Triton: Y_diag[b, n, i, h] = sum_j M[b, n, i, j, h] * hidden[b, n, j, h]
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

        # Return in bfloat16 to match original behavior
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
