import torch
import triton
import triton.language as tl


# Kernel 1: Masked cumsum along source dimension with lower-triangular mask (diagonal=-1).
# Input: A [B, N, C, S] (float32)
# Output: A_masked [B, N, C, S] (float32) where A_masked[b, n, c, i] = cumsum over s < i of A[b, n, c, s]
@triton.jit
def masked_cumsum_lower(A_ptr, Aout_ptr,
                        Bsz: tl.constexpr, N: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr,
                        A_stride_b, A_stride_n, A_stride_c, A_stride_s,
                        Aout_stride_b, Aout_stride_n, Aout_stride_c, Aout_stride_s,
                        num_warps: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_c = tl.program_id(2)

    running = tl.zeros((S,), dtype=tl.float32)

    for i in range(0, S):
        include = i > 0  # for i=0, exclude diagonal; for i>=1, include s < i
        sum_val = 0.0
        for s in range(0, S):
            m = s < i & include
            a_ptr = A_ptr + pid_b * A_stride_b + pid_n * A_stride_n + pid_c * A_stride_c + s * A_stride_s
            val = tl.load(a_ptr, mask=m, other=0.0)
            sum_val += val.to(tl.float32)
        running[i] = sum_val
        aout_ptr = Aout_ptr + pid_b * Aout_stride_b + pid_n * Aout_stride_n + pid_c * Aout_stride_c + i * Aout_stride_s
        tl.store(aout_ptr, running[i])

# Kernel 2: Exponential to produce causal mask L: lower-triangular (diagonal=0)
# Input: A_masked [B, N, C, S] (float32)
# Output: L [B, C, S, S, N] (float32) with L[b, c, i, j, n] = exp(A_masked[b, n, c, i]) if j <= i else 0
@triton.jit
def exp_causal_mask(A_ptr, L_ptr,
                    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr,
                    A_stride_b, A_stride_n, A_stride_c, A_stride_s,
                    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
                    num_warps: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(4)

    for i in range(0, S):
        row_ptr = A_ptr + pid_b * A_stride_b + pid_n * A_stride_n + pid_c * A_stride_c + i * A_stride_s
        a_val = tl.load(row_ptr)
        for j in range(0, S):
            include = j <= i
            val = tl.exp(a_val) if include else 0.0
            l_ptr = L_ptr + pid_b * L_stride_b + pid_c * L_stride_c + i * L_stride_i + j * L_stride_j + pid_n * L_stride_n
            tl.store(l_ptr, val)

# Kernel 3: Contraction to form G: G[b, c, i, j, n] = sum_k C[b, c, i, n, k] * B[b, c, j, n, k]
# B_expanded: [B, C, S, N, K], C_expanded: [B, C, S, N, K]
# Output G: [B, C, S, S, N] (float32)
@triton.jit
def contract_BC_to_G(B_ptr, C_ptr, G_ptr,
                     Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     B_stride_b, B_stride_c, B_stride_i, B_stride_n, B_stride_k,
                     C_stride_b, C_stride_c, C_stride_j, C_stride_n, C_stride_k,
                     G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
                     num_warps: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_n = tl.program_id(4)

    acc = 0.0
    for k in range(0, K):
        b_ptr = B_ptr + pid_b * B_stride_b + pid_c * B_stride_c + pid_j * B_stride_i + pid_n * B_stride_n + k * B_stride_k
        c_ptr = C_ptr + pid_b * C_stride_b + pid_c * C_stride_c + pid_i * C_stride_j + pid_n * C_stride_n + k * C_stride_k
        b_val = tl.load(b_ptr)
        c_val = tl.load(c_ptr)
        acc += b_val * c_val
    g_ptr = G_ptr + pid_b * G_stride_b + pid_c * G_stride_c + pid_i * G_stride_i + pid_j * G_stride_j + pid_n * G_stride_n
    tl.store(g_ptr, acc)

# Kernel 4: Diagonal contraction to compute Y_diag:
# Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
# M = G * L elementwise; we read M and hidden_states and write vector Y for each d
@triton.jit
def diag_contract_Y(M_ptr, HS_ptr, Y_ptr,
                    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
                    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
                    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
                    num_warps: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_n = tl.program_id(3)

    # Vector accumulate over D
    acc_vec = tl.zeros((D,), dtype=tl.float32)

    for j in range(0, S):
        m_ptr = M_ptr + pid_b * M_stride_b + pid_c * M_stride_c + pid_i * M_stride_i + j * M_stride_j + pid_n * M_stride_n
        m_val = tl.load(m_ptr)  # scalar
        # base pointer for HS row j, head n
        hs_base = HS_ptr + pid_b * HS_stride_b + pid_c * HS_stride_c + j * HS_stride_j + pid_n * HS_stride_n
        # accumulate over d
        for d in range(0, D):
            hs_d_ptr = hs_base + d * HS_stride_d
            y_val = tl.load(hs_d_ptr)
            acc_vec[d] += m_val * y_val

    y_base_ptr = Y_ptr + pid_b * Y_stride_b + pid_c * Y_stride_c + pid_i * Y_stride_i + pid_n * Y_stride_n
    for d in range(0, D):
        tl.store(y_base_ptr + d * Y_stride_d, acc_vec[d])

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants as in original: NUM_HEADS=32, N_GROUPS=8
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
          hidden_states: [B, C, S, N, D] (as in original)
          A_cumsum:      [B, N, C, S]
          B:             [B, C, S, G, K]
          C:             [B, C, S, G, K]
        Output:
          Y_diag:        [B, C, S, N, D] in bfloat16
        """
        device = hidden_states.device

        # Ensure contiguous and dtype
        Bsz, Csz, S, N, D = hidden_states.shape
        A_cumsum = A_cumsum.to(device=device, dtype=torch.float32).contiguous()
        B = B.to(device=device, dtype=torch.float32).contiguous()
        C = C.to(device=device, dtype=torch.float32).contiguous()
        hidden_states = hidden_states.contiguous()

        # 1) Masked cumsum along source dim (S) with lower-triangular (diagonal=-1) -> A_masked [B, N, C, S]
        A_masked = torch.empty((Bsz, N, Csz, S), device=device, dtype=torch.float32)
        grid_mask = (Bsz, N, Csz)
        masked_cumsum_lower[grid_mask](
            A_cumsum, A_masked,
            Bsz=Bsz, N=N, Csz=Csz, S=S,
            A_stride_b=A_cumsum.stride(0), A_stride_n=A_cumsum.stride(1), A_stride_c=A_cumsum.stride(2), A_stride_s=A_cumsum.stride(3),
            Aout_stride_b=A_masked.stride(0), Aout_stride_n=A_masked.stride(1), Aout_stride_c=A_masked.stride(2), Aout_stride_s=A_masked.stride(3),
            num_warps=1, num_stages=1
        )

        # 2) Exponential to produce causal mask L: [B, C, S, S, N]
        L = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)
        grid_L = (Bsz, Csz, S, S, N)
        exp_causal_mask[grid_L](
            A_masked, L,
            Bsz=Bsz, Csz=Csz, S=S, N=N,
            A_stride_b=A_masked.stride(0), A_stride_n=A_masked.stride(1), A_stride_c=A_masked.stride(2), A_stride_s=A_masked.stride(3),
            L_stride_b=L.stride(0), L_stride_c=L.stride(1), L_stride_i=L.stride(2), L_stride_j=L.stride(3), L_stride_n=L.stride(4),
            num_warps=1, num_stages=1
        )

        # 3) Expand B and C to include num_heads (repeat_interleave by NUM_HEADS // N_GROUPS = 4)
        repeat_factor = self.NUM_HEADS // self.N_GROUPS
        B_expanded = B.repeat_interleave(repeat_factor, dim=3)  # [B, C, S, N, K]
        C_expanded = C.repeat_interleave(repeat_factor, dim=3)  # [B, C, S, N, K]
        K = B_expanded.shape[4]

        # 4) Contraction to form G: [B, C, S, S, N]
        G = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)
        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G[grid_G](
            B_expanded, C_expanded, G,
            Bsz=Bsz, Csz=Csz, S=S, N=N, K=K,
            B_stride_b=B_expanded.stride(0), B_stride_c=B_expanded.stride(1), B_stride_i=B_expanded.stride(2), B_stride_n=B_expanded.stride(3), B_stride_k=B_expanded.stride(4),
            C_stride_b=C_expanded.stride(0), C_stride_c=C_expanded.stride(1), C_stride_j=C_expanded.stride(2), C_stride_n=C_expanded.stride(3), C_stride_k=C_expanded.stride(4),
            G_stride_b=G.stride(0), G_stride_c=G.stride(1), G_stride_i=G.stride(2), G_stride_j=G.stride(3), G_stride_n=G.stride(4),
            num_warps=1, num_stages=1
        )

        # 5) Elementwise multiply: M = G * L
        M = G * L  # elementwise in PyTorch; acceptable as not part of masked cumsum/exp

        # 6) Diagonal contraction to compute Y_diag: Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
        Y = torch.empty((Bsz, Csz, S, N, D), device=device, dtype=torch.bfloat16)
        grid_Y = (Bsz, Csz, S, N)
        diag_contract_Y[grid_Y](
            M, hidden_states.to(torch.float32), Y,
            Bsz=Bsz, Csz=Csz, S=S, N=N, D=D,
            M_stride_b=M.stride(0), M_stride_c=M.stride(1), M_stride_i=M.stride(2), M_stride_j=M.stride(3), M_stride_n=M.stride(4),
            HS_stride_b=hidden_states.stride(0), HS_stride_c=hidden_states.stride(1), HS_stride_j=hidden_states.stride(2), HS_stride_n=hidden_states.stride(3), HS_stride_d=hidden_states.stride(4),
            Y_stride_b=Y.stride(0), Y_stride_c=Y.stride(1), Y_stride_i=Y.stride(2), Y_stride_n=Y.stride(3), Y_stride_d=Y.stride(4),
            num_warps=1, num_stages=1
        )

        return Y


def run(*args):
    return ModelNew()(*args)
