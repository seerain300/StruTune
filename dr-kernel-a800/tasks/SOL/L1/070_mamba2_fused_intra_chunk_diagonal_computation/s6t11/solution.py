import torch
import triton
import triton.language as tl


# Triton kernel: compute L[b, h, n, i, j] = exp(sum_{ii<=i} A[b, h, n, ii]) for j <= i; else 0.
# We launch over (b, h, n). Internal loops over i and j.
@triton.jit
def cumsum_exp_tril_kernel(
    A_ptr, L_ptr,
    Bsz, H, N, S,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)

    # Accumulator for cumsum along source i
    acc = 0.0

    # Loop over target j (lower triangle with diagonal=-1 => j <= i)
    for j in range(0, S):
        # Reset accumulator at each new j; we are computing prefix sum along i, per j
        # But since we need j<=i, we keep updating acc as we iterate i.
        # Instead, we perform a sequential scan:
        # For each i, add A[b,h,n,i] and write exp(acc) into L[b,h,n,i,j]
        for i in range(0, S):
            # Load A[b, h, n, i]
            a_ptr = A_ptr + b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_s
            a_val = tl.load(a_ptr)
            acc = acc + a_val
            # Write exp(acc) to L[b,h,n,i,j] if j <= i; else 0
            # Only store when j <= i
            if j <= i:
                l_ptr = L_ptr + b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_s1 + j * L_stride_s2
                tl.store(l_ptr, tl.exp(acc))


# Triton kernel: compute G[i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
# Grid over (b, n, i, j), and we loop over h and d.
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

    # Accumulator for G[i,j,:]
    g_vec = tl.zeros((H,), dtype=tl.float32)

    # Loop over state dimension D
    for d in range(0, D):
        # Compute dot for each h
        # B[b,n,j,h,d]
        b_ptr = B_ptr + b * B_stride_b + n * B_stride_n + j * B_stride_s + 0 * B_stride_h + d * B_stride_d  # h is varying below
        # We need B[b,n,j, h, d], so we loop h explicitly
        for h in range(0, H):
            bh_ptr = B_ptr + b * B_stride_b + n * B_stride_n + j * B_stride_s + h * B_stride_h + d * B_stride_d
            ch_ptr = C_ptr + b * C_stride_b + n * C_stride_n + i * C_stride_s + h * C_stride_h + d * C_stride_d
            b_val = tl.load(bh_ptr)
            c_val = tl.load(ch_ptr)
            g_vec[h] = g_vec[h] + c_val * b_val

    # Store g_vec to G[b, n, i, j, :]
    for h in range(0, H):
        g_ptr = G_ptr + b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h
        tl.store(g_ptr, g_vec[h])


# Triton kernel: elementwise M = G * L where L is permuted to [B,N,S,S,H]
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

    g_ptr = G_ptr + b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h
    l_ptr = L_perm_ptr + b * L_stride_b + L_stride_h * h + n * L_stride_n + i * L_stride_s1 + j * L_stride_s2
    m_ptr = M_ptr + b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h

    g_val = tl.load(g_ptr)
    l_val = tl.load(l_ptr)
    tl.store(m_ptr, g_val * l_val)


# Triton kernel: compute Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden[b,n,j,h]
# Grid over (b, n, i) and loop over h and j.
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

    y_vec = tl.zeros((H,), dtype=tl.float32)

    # Loop over j
    for j in range(0, S):
        # Loop over h
        for h in range(0, H):
            m_ptr = M_ptr + b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h
            m_val = tl.load(m_ptr)
            h_ptr = hidden_ptr + b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_s + h * hidden_stride_h + 0 * hidden_stride_d  # D is not used here since we only need scalar multiply
            # Note: hidden has shape [B, N, S, H, D]; we multiply M[b,n,i,j,h] with hidden[b,n,j,h,0]
            h_val = tl.load(h_ptr)
            y_vec[h] = y_vec[h] + m_val * h_val

    # Store y_vec to Y[b, n, i, :]
    for h in range(0, H):
        y_ptr = Y_ptr + b * Y_stride_b + n * Y_stride_n + i * Y_stride_s1 + h * Y_stride_h
        tl.store(y_ptr, y_vec[h])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants assumed from the original code
        self.S = 128  # chunk_size
        self.H = 32   # num_heads (assumed)
        self.D = 64   # state_size (assumed)

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure dtypes
        device = hidden_states.device

        # Prepare inputs
        hidden = hidden_states.contiguous()   # [B, N, S, H, D]
        A = A_cumsum.contiguous()             # [B, H, N, S]
        Bexp = B.contiguous()                 # [B, N, S, H, D]
        Cexp = C.contiguous()                 # [B, N, S, H, D]

        Bsz, N, S, H, D = hidden.shape
        assert H == self.H, "num_heads must be 32"
        assert D == self.D, "state_size must be 64"
        assert S == self.S, "chunk_size must be 128"
        assert A.shape == (Bsz, self.H, N, self.S), "A_cumsum must have shape [B, H, N, S]"

        # Allocate L: [B, H, N, S, S] float32
        L = torch.empty((Bsz, self.H, N, self.S, self.S), dtype=torch.float32, device=device)

        # Launch cumsum + exp Triton kernel
        A_stride_b, A_stride_h, A_stride_n, A_stride_s = A.stride()
        L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2 = L.stride()

        cumsum_exp_tril_kernel[(Bsz, self.H, N)](
            A, L,
            Bsz, self.H, N, self.S,
            A_stride_b, A_stride_h, A_stride_n, A_stride_s,
            L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
            num_warps=1, num_stages=1
        )

        # Compute G: [B, N, S, S, H] float32
        G = torch.empty((Bsz, N, self.S, self.S, self.H), dtype=torch.float32, device=device)

        B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d = Bexp.stride()
        C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d = Cexp.stride()
        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()

        g_contract_kernel[(Bsz, N, self.S, self.S)](
            Bexp, Cexp, G,
            Bsz, N, self.S, self.H, self.D,
            B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
            C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            num_warps=1, num_stages=1
        )

        # Compute M = G * L (permute L to [B, N, S, S, H])
        L_perm = L.permute(0, 2, 3, 4, 1).contiguous()  # [B, N, S, S, H]
        M = torch.empty_like(G)  # [B, N, S, S, H], float32

        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()
        L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2 = L_perm.stride()
        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()

        m_mul_kernel[(Bsz, N, self.S, self.S, self.H)](
            G, L_perm, M,
            Bsz, N, self.S, self.H,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            num_warps=1, num_stages=1
        )

        # Compute Y_diag: [B, N, S, H]
        Y = torch.empty((Bsz, N, self.S, self.H), dtype=torch.float32, device=device)

        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()
        hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d = hidden.stride()
        Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h = Y.stride()

        y_diag_reduce_kernel[(Bsz, N, self.S)](
            M, hidden, Y,
            Bsz, N, self.S, self.H, self.D,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
            Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
            num_warps=1, num_stages=1
        )

        # Return bfloat16 to match original behavior
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
