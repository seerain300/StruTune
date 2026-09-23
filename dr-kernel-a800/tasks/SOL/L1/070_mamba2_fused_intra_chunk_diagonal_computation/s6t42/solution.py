import torch
import triton
import triton.language as tl


# Kernel 1: Generate 1D lower-triangular mask (diagonal=-1) of length S*S: lower[i*S + j] = 1 if j <= i else 0.
@triton.jit
def tril_mask_kernel(mask_ptr, S: tl.constexpr):
    idx = tl.program_id(0)  # 0..S*S-1
    i = idx // S
    j = idx % S
    if j <= i:
        tl.store(mask_ptr + idx, 1)
    else:
        tl.store(mask_ptr + idx, 0)


# Kernel 2: Compute L[b,h,n,i,j] = exp(sum_{k=0..i} A[b,h,n,k]) if j <= i else 0.0
# A: [B,H,N,S]; L: [B,H,N,S,S] float32.
@triton.jit
def a_segment_sum_exp_kernel(
    A_ptr, L_ptr,
    B_size, H_size, N_size, S_size,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
):
    pid_b = tl.program_id(0)  # b
    pid_h = tl.program_id(1)  # h
    pid_n = tl.program_id(2)  # n

    # For each i, compute cumulative sum over k <= i, then for each j <= i, write exp(sum) to L[i,j]
    for i in range(0, S_size):
        acc = 0.0
        for k in range(0, i + 1):
            a_ptr = A_ptr + pid_b * A_stride_b + pid_h * A_stride_h + pid_n * A_stride_n + k * A_stride_s
            a_val = tl.load(a_ptr)
            acc += a_val
        for j in range(0, S_size):
            if j <= i:
                l_ptr = L_ptr + pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + i * L_stride_s1 + j * L_stride_s2
                tl.store(l_ptr, tl.exp(acc))
            else:
                l_ptr = L_ptr + pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + i * L_stride_s1 + j * L_stride_s2
                tl.store(l_ptr, 0.0)


# Kernel 3: Compute G[i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
# B_expanded/C_expanded: [B,N,S,H,D]; G: [B,N,S,S,H] float32.
@triton.jit
def g_contract_kernel(
    B_ptr, C_ptr, G_ptr,
    B_size, N_size, S_size, H_size, D_size,
    B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
    C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_si = tl.program_id(2)  # i
    pid_sj = tl.program_id(3)  # j
    pid_h = tl.program_id(4)

    acc = 0.0
    for d_start in range(0, D_size, BLOCK_D):
        d_range = d_start + tl.arange(0, BLOCK_D)
        mask_d = d_range < D_size
        b_ptr = B_ptr + pid_b * B_stride_b + pid_n * B_stride_n + pid_sj * B_stride_s + pid_h * B_stride_h + d_range * B_stride_d
        c_ptr = C_ptr + pid_b * C_stride_b + pid_n * C_stride_n + pid_si * C_stride_s + pid_h * C_stride_h + d_range * C_stride_d
        b_vec = tl.load(b_ptr, mask=mask_d, other=0.0)
        c_vec = tl.load(c_ptr, mask=mask_d, other=0.0)
        acc += tl.sum(b_vec * c_vec, axis=0)
    g_ptr = G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + pid_si * G_stride_s1 + pid_sj * G_stride_s2 + pid_h * G_stride_h
    tl.store(g_ptr, acc)


# Kernel 4: Elementwise multiply M = G * L
# G: [B,N,S,S,H], L: [B,N,S,S,H], M: [B,N,S,S,H]
@triton.jit
def m_mul_kernel(
    G_ptr, L_ptr, M_ptr,
    B_size, N_size, S_size, H_size,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    L_stride_b, L_stride_n, L_stride_s1, L_stride_s2, L_stride_h,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_si = tl.program_id(2)  # i
    pid_sj = tl.program_id(3)  # j
    pid_h = tl.program_id(4)

    g_ptr = G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + pid_si * G_stride_s1 + pid_sj * G_stride_s2 + pid_h * G_stride_h
    l_ptr = L_ptr + pid_b * L_stride_b + pid_n * L_stride_n + pid_si * L_stride_s1 + pid_sj * L_stride_s2 + pid_h * L_stride_h
    m_ptr = M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_si * M_stride_s1 + pid_sj * M_stride_s2 + pid_h * M_stride_h
    g = tl.load(g_ptr)
    l = tl.load(l_ptr)
    tl.store(m_ptr, g * l)


# Kernel 5: Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h,:]
# M: [B,N,S,S,H], hidden: [B,N,S,H,D], Y: [B,N,S,H]
@triton.jit
def y_diag_reduce_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    B_size, N_size, S_size, H_size, D_size,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
    Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_si = tl.program_id(2)  # i
    pid_h = tl.program_id(3)

    acc = 0.0
    for j in range(0, S_size):
        # Load M[b,n,i,j,h]
        m_ptr = M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_si * M_stride_s1 + j * M_stride_s2 + pid_h * M_stride_h
        m = tl.load(m_ptr)
        # Reduce hidden[b,n,j,h,:] over D in tiles
        h_acc = 0.0
        for d_start in range(0, D_size, BLOCK_D):
            d_range = d_start + tl.arange(0, BLOCK_D)
            mask_d = d_range < D_size
            h_ptr = hidden_ptr + pid_b * hidden_stride_b + pid_n * hidden_stride_n + j * hidden_stride_s + pid_h * hidden_stride_h + d_range * hidden_stride_d
            h_vec = tl.load(h_ptr, mask=mask_d, other=0.0)
            h_acc += tl.sum(h_vec * m, axis=0)
        acc += h_acc
    y_ptr = Y_ptr + pid_b * Y_stride_b + pid_n * Y_stride_n + pid_si * Y_stride_s1 + pid_h * Y_stride_h
    tl.store(y_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, NUM_HEADS: int = 32, N_GROUPS: int = 8, D_STATE: int = 32):
        super().__init__()
        self.NUM_HEADS = NUM_HEADS
        self.N_GROUPS = N_GROUPS
        self.D_STATE = D_STATE

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
        """
        hidden_states: [B, N, S, H, D]
        A_cumsum: [B, H, N, S]
        B: [B, N, S, G, D] (G=N_GROUPS)
        C: [B, N, S, G, D]
        Returns: Y_diag: [B, N, S, H] bfloat16
        """
        device = hidden_states.device
        B_size, N_size, S_size, H_size, D_size = hidden_states.shape

        # Ensure inputs are contiguous and float32 for computation
        A = A_cumsum.to(torch.float32).contiguous()       # [B,H,N,S]
        Bc = B.to(torch.float32).contiguous()             # [B,N,S,G,D]
        Cc = C.to(torch.float32).contiguous()             # [B,N,S,G,D]

        # Expand B/C from G to H (NUM_HEADS=32, G=N_GROUPS=8 => repeat_interleave by 4)
        if Bc.shape[3] != self.NUM_HEADS:
            Bc = Bc.repeat_interleave(self.NUM_HEADS // Bc.shape[3], dim=3)
        if Cc.shape[3] != self.NUM_HEADS:
            Cc = Cc.repeat_interleave(self.NUM_HEADS // Cc.shape[3], dim=3)

        # 1) Compute lower-triangular mask (diagonal=-1) using Triton
        mask = torch.empty(S_size * S_size, dtype=torch.int8, device=device)
        tril_mask_kernel[(S_size * S_size,)](
            mask,
            S=S_size,
            num_warps=1, num_stages=1
        )

        # 2) Compute L = exp(segment_sum(A)) with tril(diagonal=-1) via Triton
        L = torch.empty((B_size, H_size, N_size, S_size, S_size), dtype=torch.float32, device=device)
        a_segment_sum_exp_kernel[(B_size, H_size, N_size)](
            A, L,
            B_size, H_size, N_size, S_size,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=1, num_stages=1
        )

        # 3) Compute G[i,j,h] via Triton contraction over D
        G = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)
        g_contract_kernel[(B_size, N_size, S_size, S_size, H_size)](
            Bc, Cc, G,
            B_size, N_size, S_size, H_size, D_size,
            Bc.stride(0), Bc.stride(1), Bc.stride(2), Bc.stride(3), Bc.stride(4),
            Cc.stride(0), Cc.stride(1), Cc.stride(2), Cc.stride(3), Cc.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            BLOCK_D=64,  # D_size=32; masked loads handle tail
            num_warps=4, num_stages=2
        )

        # 4) Elementwise multiply M = G * L (launch kernel)
        M = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)
        m_mul_kernel[(B_size, N_size, S_size, S_size, H_size)](
            G, L, M,
            B_size, N_size, S_size, H_size,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=1, num_stages=1
        )

        # 5) Reduce to Y[b,n,i,h] via Triton kernel (launch)
        Y = torch.empty((B_size, N_size, S_size, H_size), dtype=torch.float32, device=device)
        y_diag_reduce_kernel[(B_size, N_size, S_size, H_size)](
            M, hidden_states.to(torch.float32), Y,
            B_size, N_size, S_size, H_size, D_size,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            BLOCK_D=64,
            num_warps=4, num_stages=2
        )

        # Cast to bfloat16 to match original output dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
