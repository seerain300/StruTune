import torch
import triton
import triton.language as tl


# Kernel 1: Generate 1D lower-triangular mask (diagonal=-1) of length S*S into mask[0:S*S].
# mask[i*S + j] = 1 if j <= i, else 0. Output as int8 (1/0). Always invoked in forward.
@triton.jit
def tril_mask_kernel(mask_ptr, S: tl.constexpr):
    idx = tl.program_id(0)  # linear index over S*S
    i = idx // S
    j = idx % S
    lower = j <= i
    if lower:
        tl.store(mask_ptr + idx, 1)
    else:
        tl.store(mask_ptr + idx, 0)


# Kernel 2: Compute L = exp(segment_sum(A)) with tril(diagonal=-1).
# Input A: [B,H,N,S], Output L: [B,H,N,S,S] float32
# For each (b,h,n), L[i,j] = exp(sum_{k=0..i} A[b,h,n,k]) if j <= i else 0.0.
@triton.jit
def a_segment_sum_exp_kernel(A_ptr, L_ptr,
                             B_size, H_size, N_size, S: tl.constexpr,
                             A_stride_b, A_stride_h, A_stride_n, A_stride_s,
                             L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2):
    pid_b = tl.program_id(0)  # b
    pid_h = tl.program_id(1)  # h
    pid_n = tl.program_id(2)  # n

    for i in range(S):
        seg_sum = 0.0
        for k in range(i + 1):  # sum from k=0..i
            a_ptr = A_ptr + pid_b * A_stride_b + pid_h * A_stride_h + pid_n * A_stride_n + k * A_stride_s
            val = tl.load(a_ptr)
            seg_sum += val
        for j in range(S):
            if j <= i:
                l_ptr = L_ptr + pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + i * L_stride_s1 + j * L_stride_s2
                tl.store(l_ptr, tl.exp(seg_sum))
            else:
                l_ptr = L_ptr + pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + i * L_stride_s1 + j * L_stride_s2
                tl.store(l_ptr, 0.0)


# Kernel 3: G[i,j,h] = sum over D of C[b,n,i,h,d] * B[b,n,j,h,d].
# B_expanded and C_expanded are already expanded to H on host.
# Grid: (B,N,S,S,H). Inner loop accumulates over D in tiles.
@triton.jit
def g_contract_kernel(B_ptr, C_ptr, G_ptr,
                      B_size, N_size, S_size, H_size, D_size,
                      B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
                      C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
                      G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
                      BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_i = tl.program_id(2)  # i in [0..S-1]
    pid_j = tl.program_id(3)  # j in [0..S-1]
    pid_h = tl.program_id(4)  # h in [0..H-1]

    acc = 0.0
    for d0 in range(0, D_size, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D_size
        c_ptr = C_ptr + pid_b * C_stride_b + pid_n * C_stride_n + pid_i * C_stride_s + pid_h * C_stride_h + offs_d * C_stride_d
        c_vals = tl.load(c_ptr, mask=mask_d, other=0.0)
        b_ptr = B_ptr + pid_b * B_stride_b + pid_n * B_stride_n + pid_j * B_stride_s + pid_h * B_stride_h + offs_d * B_stride_d
        b_vals = tl.load(b_ptr, mask=mask_d, other=0.0)
        acc += tl.sum(c_vals * b_vals, axis=0)
    g_ptr = G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + pid_i * G_stride_s1 + pid_j * G_stride_s2 + pid_h * G_stride_h
    tl.store(g_ptr, acc)


# Kernel 4: M = G * L elementwise. L is assumed to be [B,H,N,S,S]; G is [B,N,S,S,H].
# Launch grid (B,N,S,S,H). No permuting needed: G has H as last dim; L is indexed by h for that row.
@triton.jit
def m_mul_kernel(G_ptr, L_ptr, M_ptr,
                 B_size, N_size, S_size, H_size,
                 G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
                 L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
                 M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_i = tl.program_id(2)  # i
    pid_j = tl.program_id(3)  # j
    pid_h = tl.program_id(4)  # h

    g_ptr = G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + pid_i * G_stride_s1 + pid_j * G_stride_s2 + pid_h * G_stride_h
    g_val = tl.load(g_ptr)

    l_ptr = L_ptr + pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + pid_i * L_stride_s1 + pid_j * L_stride_s2
    l_val = tl.load(l_ptr)

    m_ptr = M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_i * M_stride_s1 + pid_j * M_stride_s2 + pid_h * M_stride_h
    tl.store(m_ptr, g_val * l_val)


# Kernel 5: Reduce to Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h,:].
# Inputs: M [B,N,S,S,H], hidden_states [B,N,S,H,D], Output Y [B,N,S,H] float32
@triton.jit
def y_diag_reduce_kernel(M_ptr, hidden_ptr, Y_ptr,
                         B_size, N_size, S_size, H_size, D_size,
                         M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
                         hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
                         Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
                         BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_i = tl.program_id(2)  # i
    pid_h = tl.program_id(3)  # h

    acc = 0.0
    for j in range(0, S_size):
        m_ptr = M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_i * M_stride_s1 + j * M_stride_s2 + pid_h * M_stride_h
        m_val = tl.load(m_ptr)

        # Reduce over D for hidden_states[b,n,j,h,:]
        for d0 in range(0, D_size, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D_size
            hs_ptr = hidden_ptr + pid_b * hidden_stride_b + pid_n * hidden_stride_n + j * hidden_stride_s + pid_h * hidden_stride_h + offs_d * hidden_stride_d
            hs_vals = tl.load(hs_ptr, mask=mask_d, other=0.0)
            acc += tl.sum(hs_vals * m_val, axis=0)

    y_ptr = Y_ptr + pid_b * Y_stride_b + pid_n * Y_stride_n + pid_i * Y_stride_s1 + pid_h * Y_stride_h
    tl.store(y_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.NUM_HEADS = 32
        self.D_STATE = 32  # state size, as implied by original usage
        self.CHUNK_SIZE = 128
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, N, S, H, D]
        A_cumsum: [B, H, N, S]
        B, C: [B, N, S, G, D], with G=N_GROUPS=8. We expand to H in forward.
        Output: Y: [B, N, S, H] in bfloat16.
        """
        # Ensure contiguity and dtypes
        hidden = hidden_states.contiguous()
        A = A_cumsum.contiguous()
        Bc = B.contiguous()
        Cc = C.contiguous()

        # Expand B/C from G to H (NUM_HEADS=32, N_GROUPS=8 => repeat_interleave by 4)
        if Bc.shape[3] != self.NUM_HEADS:
            Bc = Bc.repeat_interleave(self.NUM_HEADS // Bc.shape[3], dim=3)
        if Cc.shape[3] != self.NUM_HEADS:
            Cc = Cc.repeat_interleave(self.NUM_HEADS // Cc.shape[3], dim=3)

        B_exp = Bc  # already expanded
        C_exp = Cc  # already expanded

        device = hidden.device
        B_size, N_size, S_size, H_size, D_size = hidden.shape

        # 1) Launch tril_mask_kernel (always invoked) to produce mask of length S*S
        mask = torch.empty(S_size * S_size, dtype=torch.int8, device=device)
        tril_mask_kernel[(S_size * S_size,)](
            mask, S_size,
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
            B_exp, C_exp, G,
            B_size, N_size, S_size, H_size, D_size,
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            BLOCK_D=64,  # D_size is 32 in the original; masked loads handle tail
            num_warps=4, num_stages=2
        )

        # 4) Elementwise multiply M = G * L
        M = torch.empty_like(G, dtype=torch.float32, device=device)
        m_mul_kernel[(B_size, N_size, S_size, S_size, H_size)](
            G, L, M,
            B_size, N_size, S_size, H_size,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=1, num_stages=1
        )

        # 5) Reduce to Y[b,n,i,h] using y_diag_reduce_kernel
        Y = torch.empty((B_size, N_size, S_size, H_size), dtype=torch.float32, device=device)
        y_diag_reduce_kernel[(B_size, N_size, S_size, H_size)](
            M, hidden.to(torch.float32), Y,
            B_size, N_size, S_size, H_size, D_size,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            BLOCK_D=64,
            num_warps=4, num_stages=2
        )

        # Cast to bfloat16 to match original output dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
