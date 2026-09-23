import torch
import triton
import triton.language as tl


# Kernel 1: Generate 1D lower-triangular mask (diagonal=-1) of length S*S
@triton.jit
def tril_mask_1d_kernel(mask_ptr, S: tl.constexpr):
    idx = tl.program_id(0)  # linear index over S*S
    S_int = S
    i = idx // S
    j = idx % S
    lower = j <= i
    if lower:
        tl.store(mask_ptr + idx, 1)
    else:
        tl.store(mask_ptr + idx, 0)


# Kernel 2: Compute L[b,h,n,i,j] = exp(sum_{k=0..i} A[b,h,n,k]) if j <= i else 0.0
# Grid: (B, H, N). Inside kernel loops over i and j.
@triton.jit
def a_segment_sum_exp_kernel(
    A_ptr, L_ptr,
    B_size, H_size, N_size, S_size,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2
):
    pid_b = tl.program_id(0)  # b
    pid_h = tl.program_id(1)  # h
    pid_n = tl.program_id(2)  # n

    # Loop over i and j to fill the lower-triangular part
    for i in range(S_size):
        segment_sum = 0.0
        for j in range(S_size):
            if j <= i:
                # A[b,h,n,j] (note: A is [B,H,N,S], last dim index j)
                val = tl.load(A_ptr + pid_b * A_stride_b + pid_h * A_stride_h + pid_n * A_stride_n + j * A_stride_s)
                segment_sum += val
            # For j > i, we explicitly set 0.0 (no contribution)
            exp_val = tl.exp(segment_sum) if j <= i else 0.0
            # Store to L[b,h,n,i,j]
            tl.store(L_ptr + pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n +
                     i * L_stride_s1 + j * L_stride_s2, exp_val)


# Kernel 3: Compute G[b,n,i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
# Grid: (B, N). Inside kernel loops over i, j, h and tiles over d with BLOCK_D.
@triton.jit
def g_contract_kernel(
    B_ptr, C_ptr, G_ptr,
    B_size, N_size, S_size, H_size, D_size,
    B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
    C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Iterate over i, j, h with outer loops; reduce over d in tiles
    for i in range(S_size):
        for j in range(S_size):
            for h in range(H_size):
                acc = 0.0
                # Tile over D
                for d_start in range(0, D_size, BLOCK_D):
                    d_offsets = d_start + tl.arange(0, BLOCK_D)
                    mask = d_offsets < D_size
                    # Load B[b,n,j,h,d] and C[b,n,i,h,d]
                    b_vals = tl.load(B_ptr + pid_b * B_stride_b + pid_n * B_stride_n + j * B_stride_s + h * B_stride_h +
                                     d_offsets * B_stride_d, mask=mask, other=0.0)
                    c_vals = tl.load(C_ptr + pid_b * C_stride_b + pid_n * C_stride_n + i * C_stride_s + h * C_stride_h +
                                     d_offsets * C_stride_d, mask=mask, other=0.0)
                    acc += tl.sum(b_vals * c_vals, axis=0)
                # Store G[b,n,i,j,h] = acc
                tl.store(G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h, acc)


# Kernel 4: Permute G to G_perm with shape [B,N,S,S,H]
# Input: G [B,N,S,S,H], Output: G_perm [B,N,S,S,H] (same data, here just a dummy to ensure we define it; we will compute M directly).
# Note: We'll instead use a separate kernel to produce M = G * L_perm without explicit G_perm: see m_mul_kernel using G and L_perm.
# Placeholder: Not used directly, but we ensure L_perm is produced by m_permute_kernel below.


# Kernel 5: Elementwise multiply M = G_perm * L_perm, where both are [B,N,S,S,H]
# Grid: (B,N,S,S,H). We implement this multiply directly.
@triton.jit
def m_mul_kernel(
    G_ptr, L_ptr, M_ptr,
    B_size, N_size, S_size, H_size,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    L_stride_b, L_stride_n, L_stride_s1, L_stride_s2, L_stride_h,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h
):
    # Grid over (b, n, s1, s2, h)
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s1 = tl.program_id(2)
    pid_s2 = tl.program_id(3)
    pid_h = tl.program_id(4)

    # Load G[b,n,s1,s2,h] and L[b,n,s1,s2,h], multiply, store
    g_val = tl.load(G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + pid_s1 * G_stride_s1 + pid_s2 * G_stride_s2 + pid_h * G_stride_h)
    l_val = tl.load(L_ptr + pid_b * L_stride_b + pid_n * L_stride_n + pid_s1 * L_stride_s1 + pid_s2 * L_stride_s2 + pid_h * L_stride_h)
    m_val = g_val * l_val
    tl.store(M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_s1 * M_stride_s1 + pid_s2 * M_stride_s2 + pid_h * M_stride_h, m_val)


# Kernel 6: Reduce Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h,:]
# Grid: (B,N,S,H). Inside kernel loops over j (BLOCK_J tiling).
@triton.jit
def y_diag_reduce_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    B_size, N_size, S_size, H_size, D_size,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
    Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
    BLOCK_J: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s1 = tl.program_id(2)  # i
    pid_h = tl.program_id(3)   # h

    acc = 0.0
    for j_start in range(0, S_size, BLOCK_J):
        j_offsets = j_start + tl.arange(0, BLOCK_J)
        mask = j_offsets < S_size
        # Load M[b,n,i,j,h] for the vector of j
        m_ptrs = M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_s1 * M_stride_s1 + j_offsets * M_stride_s2 + pid_h * M_stride_h
        m_vals = tl.load(m_ptrs, mask=mask, other=0.0)
        # Load hidden_states[b,n,j,h,:] vector over j and D
        h_ptrs = hidden_ptr + pid_b * hidden_stride_b + pid_n * hidden_stride_n + j_offsets * hidden_stride_s + pid_h * hidden_stride_h + tl.arange(0, D_size) * hidden_stride_d
        # Broadcast j_offsets to match pointer arithmetic over j dimension; Triton expects per-dimension loops here, so we do it explicitly:
        # We will reduce per j element separately
        for jj in range(BLOCK_J):
            if mask[jj]:
                h_single_ptr = hidden_ptr + pid_b * hidden_stride_b + pid_n * hidden_stride_n + (j_start + jj) * hidden_stride_s + pid_h * hidden_stride_h
                # Load hidden vector for that j
                h_vec = tl.load(h_single_ptr + tl.arange(0, D_size) * hidden_stride_d)
                acc += m_vals[jj] * tl.sum(h_vec, axis=0)  # sum over D
    # Store result
    tl.store(Y_ptr + pid_b * Y_stride_b + pid_n * Y_stride_n + pid_s1 * Y_stride_s1 + pid_h * Y_stride_h, acc)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        A_cumsum: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Y_diag using Triton kernels:
        - L via segment_sum with tril(diagonal=-1) and exp
        - G contraction over state dim
        - M = G * L
        - Y = sum_j M * hidden_states along j
        Returns: [B, N, S, H] bfloat16
        """
        device = hidden_states.device
        # Input shapes as in original: hidden_states [B, N, S, H, D], A_cumsum [B, H, N, S], B, C [B, N, S, H, D]
        B_size, N_size, S_size, H_size, D_size = hidden_states.shape

        # Ensure dtype float32 for compute
        A = A_cumsum.to(torch.float32)       # [B,H,N,S]
        B_f = B.to(torch.float32)            # [B,N,S,H,D]
        C_f = C.to(torch.float32)            # [B,N,S,H,D]
        hidden = hidden_states.to(torch.float32)  # [B,N,S,H,D]

        # 1) Generate tril mask 1D (diagonal=-1), length S*S
        mask_1d = torch.empty(S_size * S_size, dtype=torch.int8, device=device)
        tril_mask_1d_kernel[(S_size * S_size,)](
            mask_1d, S_size, num_warps=1, num_stages=1
        )
        # Note: We do not need mask_1d further; the logic in a_segment_sum_exp_kernel implements the tril mask via condition j <= i.

        # 2) Compute L[b,h,n,i,j] = exp(cumulative sum(A)) for j <= i
        L = torch.empty((B_size, H_size, N_size, S_size, S_size), dtype=torch.float32, device=device)
        a_segment_sum_exp_kernel[(B_size, H_size, N_size)](
            A, L,
            B_size, H_size, N_size, S_size,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=4, num_stages=1
        )

        # 3) Compute G[b,n,i,j,h] = sum_d B[b,n,j,h,d] * C[b,n,i,h,d]
        G = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)
        g_contract_kernel[(B_size, N_size)](
            B_f, C_f, G,
            B_size, N_size, S_size, H_size, D_size,
            B_f.stride(0), B_f.stride(1), B_f.stride(2), B_f.stride(3), B_f.stride(4),
            C_f.stride(0), C_f.stride(1), C_f.stride(2), C_f.stride(3), C_f.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            BLOCK_D=32, num_warps=4, num_stages=1
        )

        # 4) Elementwise M = G * L, with L permuted to [B,N,S,S,H]
        # L is [B,H,N,S,S]; permute to [B,N,S,S,H]
        L_perm = L.permute(0, 2, 3, 4, 1).contiguous()  # [B,N,S,S,H]
        M = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)
        m_mul_kernel[(B_size, N_size, S_size, S_size, H_size)](
            G, L_perm, M,
            B_size, N_size, S_size, H_size,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L_perm.stride(0), L_perm.stride(1), L_perm.stride(2), L_perm.stride(3), L_perm.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=4, num_stages=1
        )

        # 5) Reduce to Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h,:]
        Y = torch.empty((B_size, N_size, S_size, H_size), dtype=torch.float32, device=device)
        y_diag_reduce_kernel[(B_size, N_size, S_size, H_size)](
            M, hidden, Y,
            B_size, N_size, S_size, H_size, D_size,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            BLOCK_J=64, num_warps=4, num_stages=1
        )

        # Return in bfloat16 as original run() returns
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
