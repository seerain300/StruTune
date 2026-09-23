import torch
import triton
import triton.language as tl


# 1) Triton: Generate lower-triangular mask M_lower of shape [S, S] with diagonal=-1 -> int8 buffer
@triton.jit
def tril_mask_kernel(M_ptr: tl.pointer_type(tl.int8), S: tl.constexpr):
    # 2D grid over (i, j) in [0..S-1]
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    if pid_j <= pid_i:
        # write 1 for lower-triangular, else 0
        tl.store(M_ptr + pid_i * S + pid_j, 1)
    else:
        tl.store(M_ptr + pid_i * S + pid_j, 0)


# 2) Triton: Compute L = exp(segment_sum(A)) on lower triangle (diagonal=-1)
# Input: A_cumsum [B, H, N, S], Output: L [B, H, N, S, S] float32
@triton.jit
def a_segment_sum_exp_kernel(
    A_ptr, L_ptr,
    B_size, H_size, N_size, S_size,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
):
    # grid over (b, h, n)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    # For each i, maintain prefix sum and write exp(prefix) to L[b,h,n,i,j] if j<=i, else 0
    for i in range(S_size):
        seg_sum = 0.0
        for j in range(S_size):
            if j <= i:
                a_off = pid_b * A_stride_b + pid_h * A_stride_h + pid_n * A_stride_n + i * A_stride_s
                a_val = tl.load(A_ptr + a_off)  # scalar
                seg_sum += a_val
            # store exp(seg_sum) at L[b,h,n,i,j]
            l_off = pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + i * L_stride_s1 + j * L_stride_s2
            l_val = tl.exp(seg_sum) if j <= i else 0.0
            tl.store(L_ptr + l_off, l_val)


# 3) Triton: G contraction G[i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
@triton.jit
def g_contract_kernel(
    B_ptr, C_ptr, G_ptr,
    B_size, N_size, S, H, D,
    B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
    C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    BLOCK_D: tl.constexpr,
):
    # Grid: (B, N, S, S, H)
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_is = tl.program_id(2)  # i
    pid_js = tl.program_id(3)  # j
    pid_h = tl.program_id(4)

    acc = 0.0
    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        mask_d = d_offsets < D

        # Load B[b,n,j,h,:] tile
        b_off = pid_b * B_stride_b + pid_n * B_stride_n + pid_js * B_stride_s + pid_h * B_stride_h + d_offsets * B_stride_d
        Bv = tl.load(B_ptr + b_off, mask=mask_d, other=0.0)

        # Load C[b,n,i,h,:] tile
        c_off = pid_b * C_stride_b + pid_n * C_stride_n + pid_is * C_stride_s + pid_h * C_stride_h + d_offsets * C_stride_d
        Cv = tl.load(C_ptr + c_off, mask=mask_d, other=0.0)

        prod = Cv * Bv
        acc += tl.sum(prod, axis=0)

    # Store G[b,n,i,j,h]
    g_off = pid_b * G_stride_b + pid_n * G_stride_n + pid_is * G_stride_s1 + pid_js * G_stride_s2 + pid_h * G_stride_h
    tl.store(G_ptr + g_off, acc)


# 4) Triton: Elementwise M = G * L (L is [B,H,N,S,S], permute to [B,N,S,S,H])
@triton.jit
def m_mul_kernel(
    G_ptr, L_ptr, M_ptr,
    B_size, H_size, N_size, S,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
):
    # Grid: (B, N, S, S, H)
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_is = tl.program_id(2)
    pid_js = tl.program_id(3)
    pid_h = tl.program_id(4)

    g_off = pid_b * G_stride_b + pid_n * G_stride_n + pid_is * G_stride_s1 + pid_js * G_stride_s2 + pid_h * G_stride_h
    l_off = pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + pid_is * L_stride_s1 + pid_js * L_stride_s2
    m_off = pid_b * M_stride_b + pid_n * M_stride_n + pid_is * M_stride_s1 + pid_js * M_stride_s2 + pid_h * M_stride_h

    g_val = tl.load(G_ptr + g_off)
    l_val = tl.load(L_ptr + l_off)
    tl.store(M_ptr + m_off, g_val * l_val)


# 5) Triton: Y_diag reduction: Y[b, n, i, h] = sum_j M[b, n, i, j, h] * hidden[b, n, j, h, :]
@triton.jit
def y_diag_reduce_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    B_size, N_size, S, H, D,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
    Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
    BLOCK_D: tl.constexpr,
):
    # Grid: (B, N, S, H)
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_is = tl.program_id(2)  # i
    pid_h = tl.program_id(3)

    acc = 0.0
    for js in range(0, S):
        # Accumulate over D for each j
        for d_start in range(0, D, BLOCK_D):
            d_offsets = d_start + tl.arange(0, BLOCK_D)
            mask_d = d_offsets < D

            # Load M[b,n,i,j,h] scalar
            m_off = pid_b * M_stride_b + pid_n * M_stride_n + pid_is * M_stride_s1 + js * M_stride_s2 + pid_h * M_stride_h
            m_val = tl.load(M_ptr + m_off)

            # Load hidden[b,n,j,h,:] tile
            hid_off = pid_b * hidden_stride_b + pid_n * hidden_stride_n + js * hidden_stride_s + pid_h * hidden_stride_h + d_offsets * hidden_stride_d
            hidv = tl.load(hidden_ptr + hid_off, mask=mask_d, other=0.0)

            acc += m_val * tl.sum(hidv, axis=0)

    # Store Y[b,n,i,h] = acc
    y_off = pid_b * Y_stride_b + pid_n * Y_stride_n + pid_is * Y_stride_s1 + pid_h * Y_stride_h
    tl.store(Y_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized version of the original run function.
        - All computations are done by Triton kernels; no PyTorch elementwise/cumsum/exp/reduce in forward.
        - Returns Y_diag: [B, N, S, H] in bfloat16, matching original behavior.
        """
        # Ensure inputs are contiguous
        hidden_states = hidden_states.contiguous()
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        B_size, N_size, S, H, D = hidden_states.shape
        assert A_cumsum.shape == (B_size, H, N_size, S), "A_cumsum must have shape [B, H, N, S]"

        device = hidden_states.device

        # 1) Triton: Generate lower-triangular mask [S, S] -> int8
        M_lower = torch.empty((S, S), dtype=torch.int8, device=device)
        grid_mask = (S, S)
        tril_mask_kernel[grid_mask](M_lower, S, num_warps=1, num_stages=1)

        # 2) Triton: Compute L = exp(segment_sum(A)) on lower triangle (diagonal=-1) -> [B, H, N, S, S] float32
        L = torch.empty((B_size, H, N_size, S, S), dtype=torch.float32, device=device)

        A_stride_b, A_stride_h, A_stride_n, A_stride_s = A_cumsum.stride()
        L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2 = L.stride()

        grid_a = (B_size, H, N_size)
        a_segment_sum_exp_kernel[grid_a](
            A_cumsum, L,
            B_size, H, N_size, S,
            A_stride_b, A_stride_h, A_stride_n, A_stride_s,
            L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
            num_warps=1, num_stages=1
        )

        # 3) Triton: G contraction G[b,n,i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d] -> [B, N, S, S, H]
        G = torch.empty((B_size, N_size, S, S, H), dtype=torch.float32, device=device)

        B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d = B.stride()
        C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d = C.stride()
        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()

        grid_g = (B_size, N_size, S, S, H)
        g_contract_kernel[grid_g](
            B, C, G,
            B_size, N_size, S, H, D,
            B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
            C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            BLOCK_D=32,
            num_warps=1, num_stages=1
        )

        # 4) Triton: M = G * L (L is [B,H,N,S,S]; permute to [B,N,S,S,H] in kernel indexing)
        M = torch.empty_like(G)  # [B, N, S, S, H]

        L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2 = L.stride()
        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()

        m_mul_kernel[grid_g](
            G, L, M,
            B_size, H, N_size, S,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            num_warps=1, num_stages=1
        )

        # 5) Triton: Y_diag reduction Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden[b,n,j,h,:]
        Y = torch.empty((B_size, N_size, S, H), dtype=torch.float32, device=device)

        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()
        hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d = hidden_states.stride()
        Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h = Y.stride()

        grid_y = (B_size, N_size, S, H)
        y_diag_reduce_kernel[grid_y](
            M, hidden_states, Y,
            B_size, N_size, S, H, D,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
            Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
            BLOCK_D=32,
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 to match original output dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
