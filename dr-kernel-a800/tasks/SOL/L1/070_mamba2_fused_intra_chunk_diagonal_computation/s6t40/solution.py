import torch
import triton
import triton.language as tl


# Kernel 1: Compute L = exp(segment_sum(A)) with tril(diagonal=-1).
# A: [B,H,N,S] float32, Output L: [B,H,N,S,S] float32
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

    # For each i in [0..S-1], compute segment_sum = sum_{k=0..i} A[b,h,n,k]
    # Then for each j in [0..S-1], if j <= i store exp(segment_sum), else 0.0
    for i in range(0, S_size):
        seg_sum = 0.0
        # Compute cumulative sum up to i
        for k in range(0, i + 1):
            a_val = tl.load(A_ptr + pid_b * A_stride_b + pid_h * A_stride_h + pid_n * A_stride_n + k * A_stride_s)
            seg_sum += a_val
        # Fill L[b,h,n,i,j] for all j
        for j in range(0, S_size):
            if j <= i:
                l_val = tl.exp(seg_sum)
            else:
                l_val = 0.0
            tl.store(
                L_ptr + pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + i * L_stride_s1 + j * L_stride_s2,
                l_val
            )


# Kernel 2: Compute G[i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
# Inputs: B [B,N,S,H,D], C [B,N,S,H,D], Output G [B,N,S,S,H], float32
@triton.jit
def g_contract_kernel(
    B_ptr, C_ptr, G_ptr,
    B_size, N_size, S_size, H_size, D_size,
    B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
    C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)  # b
    pid_n = tl.program_id(1)  # n
    pid_si = tl.program_id(2)  # i
    pid_sj = tl.program_id(3)  # j
    pid_h = tl.program_id(4)   # h

    acc = 0.0
    for d_start in range(0, D_size, BLOCK_D):
        d_idx = d_start + tl.arange(0, BLOCK_D)
        mask = d_idx < D_size
        b_vec = tl.load(B_ptr + pid_b * B_stride_b + pid_n * B_stride_n + pid_sj * B_stride_s + pid_h * B_stride_h + d_idx * B_stride_d, mask=mask, other=0.0)
        c_vec = tl.load(C_ptr + pid_b * C_stride_b + pid_n * C_stride_n + pid_si * C_stride_s + pid_h * C_stride_h + d_idx * C_stride_d, mask=mask, other=0.0)
        acc += tl.sum(b_vec * c_vec, axis=0)
    tl.store(G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + pid_si * G_stride_s1 + pid_sj * G_stride_s2 + pid_h * G_stride_h, acc)


# Kernel 3: Elementwise M = G * L
# G: [B,N,S,S,H], L: [B,N,S,S,H], Output M: [B,N,S,S,H]
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
    pid_si = tl.program_id(2)
    pid_sj = tl.program_id(3)
    pid_h = tl.program_id(4)

    g_val = tl.load(G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + pid_si * G_stride_s1 + pid_sj * G_stride_s2 + pid_h * G_stride_h)
    l_val = tl.load(L_ptr + pid_b * L_stride_b + pid_n * L_stride_n + pid_si * L_stride_s1 + pid_sj * L_stride_s2 + pid_h * L_stride_h)
    m_val = g_val * l_val
    tl.store(M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_si * M_stride_s1 + pid_sj * M_stride_s2 + pid_h * M_stride_h, m_val)


# Kernel 4: Reduce Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h,:]
# M: [B,N,S,S,H], hidden: [B,N,S,H,D], Output Y: [B,N,S,H] float32
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
    pid_h = tl.program_id(3)   # h

    acc = 0.0
    for j in range(0, S_size):
        m_val = tl.load(M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_si * M_stride_s1 + j * M_stride_s2 + pid_h * M_stride_h)
        d_idx = tl.arange(0, BLOCK_D)
        mask = d_idx < D_size
        hidden_vec = tl.load(
            hidden_ptr + pid_b * hidden_stride_b + pid_n * hidden_stride_n + j * hidden_stride_s + pid_h * hidden_stride_h + d_idx * hidden_stride_d,
            mask=mask, other=0.0
        )
        acc += tl.sum(hidden_vec * m_val, axis=0)
    tl.store(Y_ptr + pid_b * Y_stride_b + pid_n * Y_stride_n + pid_si * Y_stride_s1 + pid_h * Y_stride_h, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, CHUNK_SIZE=128, NUM_HEADS=32, N_GROUPS=8, D_STATE=32):
        super().__init__()
        self.CHUNK_SIZE = CHUNK_SIZE
        self.NUM_HEADS = NUM_HEADS
        self.N_GROUPS = N_GROUPS
        self.D_STATE = D_STATE

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, N, S, H, D]
        A_cumsum:      [B, H, N, S]
        B:             [B, N, S, G, D] where G=N_GROUPS=8
        C:             [B, N, S, G, D]
        Output:        [B, N, S, H] in bfloat16
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

        # 1) Compute L = exp(segment_sum(A)) with tril(diagonal=-1) via Triton
        # L: [B,H,N,S,S] float32
        L = torch.empty((B_size, H_size, N_size, S_size, S_size), dtype=torch.float32, device=device)
        a_segment_sum_exp_kernel[(B_size, H_size, N_size)](
            A, L,
            B_size, H_size, N_size, S_size,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=1, num_stages=1
        )

        # 2) Compute G[i,j,h] via Triton contraction over D
        G = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)
        g_contract_kernel[(B_size, N_size, S_size, S_size, H_size)](
            Bc, Cc, G,
            B_size, N_size, S_size, H_size, self.D_STATE,
            Bc.stride(0), Bc.stride(1), Bc.stride(2), Bc.stride(3), Bc.stride(4),
            Cc.stride(0), Cc.stride(1), Cc.stride(2), Cc.stride(3), Cc.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            BLOCK_D=64,  # D_STATE=32, masked loads handle tail
            num_warps=4, num_stages=2
        )

        # 3) M = G * L (elementwise


def run(*args):
    return ModelNew()(*args)
