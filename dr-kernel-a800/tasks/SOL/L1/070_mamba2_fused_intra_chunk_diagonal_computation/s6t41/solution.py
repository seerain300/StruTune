import torch
import triton
import triton.language as tl


# Kernel 1: Generate 1D lower-triangular mask (diagonal=-1) for SxS.
# mask[i*S + j] = 1 if j <= i else 0. Stored as int8.
@triton.jit
def tril_mask_kernel(mask_ptr, S: tl.constexpr):
    idx = tl.program_id(0)
    S_int = S
    i = idx // S
    j = idx % S
    lower = j <= i
    if lower:
        tl.store(mask_ptr + idx, 1)
    else:
        tl.store(mask_ptr + idx, 0)


# Kernel 2: Compute L[b,h,n,i,j] = exp(cumsum(A_cumsum with tril(diagonal=-1))).
# For each (b,h,n): initialize segment_sum = 0; for i in 0..S-1:
#   for j in 0..S-1: if j <= i: segment_sum += A[b,h,n,j]; else: segment_sum = 0
#   store L[b,h,n,i,j] = exp(segment_sum) (float32)
@triton.jit
def a_segment_sum_exp_kernel(
    A_ptr, L_ptr,
    B_size, H_size, N_size, S_size,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
    S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    # segment_sum scalar accumulator
    segment_sum = 0.0  # float32

    # Iterate i from 0 to S-1
    for i in range(S):
        # Reinitialize segment_sum if i > 0? No, we will overwrite per j; segment_sum is local.
        # For each j in 0..S-1
        for j in range(S):
            # Read A[b,h,n,j] if j <= i
            lower = j <= i
            # Pointer for A[b,h,n,j]
            a_ptr = A_ptr + pid_b * A_stride_b + pid_h * A_stride_h + pid_n * A_stride_n + j * A_stride_s
            if lower:
                a_val = tl.load(a_ptr)
                segment_sum += a_val
            else:
                segment_sum = 0.0  # j > i: contribution is 0

            # Store exp(segment_sum) to L[b,h,n,i,j]
            l_ptr = L_ptr + pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + i * L_stride_s1 + j * L_stride_s2
            tl.store(l_ptr, tl.exp(segment_sum))


# Kernel 3: Compute G[i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d], reduce over D_STATE.
# Grid: (B,N,S,S,H). We loop d in tiles of BLOCK_D, use masked loads for tail.
@triton.jit
def g_contract_kernel(
    B_ptr, C_ptr, G_ptr,
    B_size, N_size, S_size, H_size, D_STATE,
    B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
    C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
    G_stride_i, G_stride_j, G_stride_h,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    # Accumulator scalar
    acc = 0.0

    # Loop over D in tiles
    for d0 in range(0, D_STATE, BLOCK_D):
        d_offsets = d0 + tl.arange(0, BLOCK_D)
        mask_d = d_offsets < D_STATE

        # Load B[b,n,j,h,d_offsets]
        b_ptr = B_ptr + pid_b * B_stride_b + pid_n * B_stride_n + pid_j * B_stride_s + pid_h * B_stride_h + d_offsets * B_stride_d
        b_vec = tl.load(b_ptr, mask=mask_d, other=0.0)

        # Load C[b,n,i,h,d_offsets]
        c_ptr = C_ptr + pid_b * C_stride_b + pid_n * C_stride_n + pid_i * C_stride_s + pid_h * C_stride_h + d_offsets * C_stride_d
        c_vec = tl.load(c_ptr, mask=mask_d, other=0.0)

        # FMA
        acc += tl.sum(b_vec * c_vec, axis=0)

    # Store G[i,j,h] = acc
    g_ptr = G_ptr + pid_i * G_stride_i + pid_j * G_stride_j + pid_h * G_stride_h
    tl.store(g_ptr, acc)


# Kernel 4: Elementwise multiply M = G * L. We assume:
# G: [B,N,S,S,H], L: [B,N,S,S,H] (after permute from A_segment_sum_exp_kernel).
@triton.jit
def m_mul_kernel(G_ptr, L_ptr, M_ptr,
                 B_size, N_size, S_size, H_size,
                 G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
                 L_stride_b, L_stride_n, L_stride_s1, L_stride_s2, L_stride_h,
                 M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
                 BLOCK_S: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s1 = tl.program_id(2)
    pid_s2 = tl.program_id(3)
    pid_h = tl.program_id(4)

    # Vectorize over s2 (columns) in tiles of BLOCK_S
    for start in range(0, S_size, BLOCK_S):
        s2_offsets = start + tl.arange(0, BLOCK_S)
        mask = s2_offsets < S_size

        g_ptr_vec = G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + pid_s1 * G_stride_s1 + s2_offsets * G_stride_s2 + pid_h * G_stride_h
        l_ptr_vec = L_ptr + pid_b * L_stride_b + pid_n * L_stride_n + pid_s1 * L_stride_s1 + s2_offsets * L_stride_s2 + pid_h * L_stride_h
        g_vec = tl.load(g_ptr_vec, mask=mask, other=0.0)
        l_vec = tl.load(l_ptr_vec, mask=mask, other=1.0)  # l_vec may contain zeros for s2>=S, but we should not load for masked; better keep mask and ensure L is correct.

        m_vec = g_vec * l_vec
        m_ptr_vec = M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_s1 * M_stride_s1 + s2_offsets * M_stride_s2 + pid_h * M_stride_h
        tl.store(m_ptr_vec, m_vec, mask=mask)


# Kernel 5: Compute Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h,:].
# Grid: (B,N,S,H). We loop j and D in tiles.
@triton.jit
def y_diag_reduce_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    B_size, N_size, S_size, H_size, D_STATE,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
    Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s1 = tl.program_id(2)  # i
    pid_h = tl.program_id(3)

    acc = 0.0

    # Loop over j in tiles
    for j0 in range(0, S_size, 1):  # i is fixed; we iterate j
        j = j0
        # Inner accumulation over D
        for d0 in range(0, D_STATE, BLOCK_D):
            d_offsets = d0 + tl.arange(0, BLOCK_D)
            mask_d = d_offsets < D_STATE

            # Load M[b,n,i,j,h]
            m_ptr = M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_s1 * M_stride_s1 + j * M_stride_s2 + pid_h * M_stride_h
            m_val = tl.load(m_ptr)
            # Load hidden[b,n,j,h,d_offsets]
            hidden_ptr_vec = hidden_ptr + pid_b * hidden_stride_b + pid_n * hidden_stride_n + j * hidden_stride_s + pid_h * hidden_stride_h + d_offsets * hidden_stride_d
            hidden_vec = tl.load(hidden_ptr_vec, mask=mask_d, other=0.0)

            acc += m_val * tl.sum(hidden_vec, axis=0)

    # Store Y[b,n,i,h]
    y_ptr = Y_ptr + pid_b * Y_stride_b + pid_n * Y_stride_n + pid_s1 * Y_stride_s1 + pid_h * Y_stride_h
    tl.store(y_ptr, acc)


class ModelNew(torch.nn.Module):
    NUM_HEADS = 32  # constant from original
    N_GROUPS = 8    # constant from original
    D_STATE = 32    # constant from original

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, N, S, H, D]
        A_cumsum:      [B, H, N, S]
        B:             [B, N, S, G, D]
        C:             [B, N, S, G, D]
        Returns:       [B, N, S, H] in bfloat16
        """
        device = hidden_states.device
        B_size, N_size, S_size, H_size, D_size = hidden_states.shape

        # Ensure inputs are contiguous and float32 for computation
        A = A_cumsum.to(torch.float32).contiguous()          # [B,H,N,S]
        Bc = B.to(torch.float32).contiguous()                # [B,N,S,G,D]
        Cc = C.to(torch.float32).contiguous()                # [B,N,S,G,D]

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
            S=S_size,
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
            BLOCK_D=64,
            num_warps=4, num_stages=2
        )

        # 3) M = G * L (elementwise multiply) using Triton
        # Note: original L has shape [B,H,N,S,S], G has [B,N,S,S,H], M must be [B,N,S,S,H]
        # Permute L to [B,N,S,S,H] for multiplication
        L_perm = L.permute(0, 2, 3, 1, 4).contiguous()  # [B,N,S,S,H]
        M = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)
        m_mul_kernel[(B_size, N_size, S_size, S_size, H_size)](
            G, L_perm, M,
            B_size, N_size, S_size, H_size,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L_perm.stride(0), L_perm.stride(1), L_perm.stride(2), L_perm.stride(3), L_perm.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            BLOCK_S=64,
            num_warps=4, num_stages=2
        )

        # 4) Y reduction via Triton
        Y = torch.empty((B_size, N_size, S_size, H_size), dtype=torch.float32, device=device)
        y_diag_reduce_kernel[(B_size, N_size, S_size, H_size)](
            M, hidden_states.to(torch.float32).contiguous(), Y,
            B_size, N_size, S_size, H_size, self.D_STATE,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_states.contiguous().stride(0), hidden_states.contiguous().stride(1), hidden_states.contiguous().stride(2),
            hidden_states.contiguous().stride(3), hidden_states.contiguous().stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            BLOCK_D=64,
            num_warps=4, num_stages=2
        )

        # Cast to bfloat16 to match original output dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
