import torch
import triton
import triton.language as tl


# Triton kernel: compute L[b,h,n,i,j] = exp(sum_{k=0..i} A[b,h,n,k]) if j <= i else 0.0
# A shape: [B, H, N, S]; L shape: [B, H, N, S, S] (float32).
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

    # For each i, accumulate segment sum; for each j<=i, write exp(segment_sum), else 0.0
    for i in range(0, S_size):
        seg_sum = 0.0
        # sum over k in [0..i]
        for k in range(0, i + 1):
            a_off = pid_b * A_stride_b + pid_h * A_stride_h + pid_n * A_stride_n + k * A_stride_s
            a_val = tl.load(A_ptr + a_off)
            seg_sum += a_val
        # set L[b,h,n,i,j] for all j<=i
        for j in range(0, S_size):
            if j <= i:
                l_val = tl.exp(seg_sum)
            else:
                l_val = 0.0
            l_off = pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + i * L_stride_s1 + j * L_stride_s2
            tl.store(L_ptr + l_off, l_val)


# Triton kernel: G contraction G[b,n,i,j,h] = sum_d C_exp[b,n,i,h,d] * B_exp[b,n,j,h,d]
# Inputs: B_exp and C_exp [B,N,S,H,D] float32; Output G [B,N,S,S,H] float32.
@triton.jit
def g_contract_kernel(
    B_exp, C_exp, G_ptr,
    B_size, N_size, S_size, H_size, D_size,
    B_exp_stride_b, B_exp_stride_n, B_exp_stride_s, B_exp_stride_h, B_exp_stride_d,
    C_exp_stride_b, C_exp_stride_n, C_exp_stride_s, C_exp_stride_h, C_exp_stride_d,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s1 = tl.program_id(2)  # i
    pid_s2 = tl.program_id(3)  # j
    pid_h = tl.program_id(4)   # h

    acc = 0.0
    # Tile over D dimension
    for d0 in range(0, D_size, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        mask_d = d_idx < D_size
        b_vals = tl.load(B_exp + pid_b * B_exp_stride_b + pid_n * B_exp_stride_n + pid_s2 * B_exp_stride_s + pid_h * B_exp_stride_h + d_idx * B_exp_stride_d, mask=mask_d, other=0.0)
        c_vals = tl.load(C_exp + pid_b * C_exp_stride_b + pid_n * C_exp_stride_n + pid_s1 * C_exp_stride_s + pid_h * C_exp_stride_h + d_idx * C_exp_stride_d, mask=mask_d, other=0.0)
        acc += tl.sum(b_vals * c_vals, axis=0)
    # Store G[b,n,i,j,h] = acc
    g_off = pid_b * G_stride_b + pid_n * G_stride_n + pid_s1 * G_stride_s1 + pid_s2 * G_stride_s2 + pid_h * G_stride_h
    tl.store(G_ptr + g_off, acc)


# Triton kernel: M = G * L after permuting L to [B,N,S,S,H]
@triton.jit
def m_mul_kernel(
    G_ptr, Lp_ptr, M_ptr,
    B_size, N_size, S_size, H_size,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    Lp_stride_b, Lp_stride_n, Lp_stride_s1, Lp_stride_s2, Lp_stride_h,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s1 = tl.program_id(2)  # i
    pid_s2 = tl.program_id(3)  # j
    pid_h = tl.program_id(4)   # h

    g_off = pid_b * G_stride_b + pid_n * G_stride_n + pid_s1 * G_stride_s1 + pid_s2 * G_stride_s2 + pid_h * G_stride_h
    lp_off = pid_b * Lp_stride_b + pid_n * Lp_stride_n + pid_s1 * Lp_stride_s1 + pid_s2 * Lp_stride_s2 + pid_h * Lp_stride_h
    m_off = pid_b * M_stride_b + pid_n * M_stride_n + pid_s1 * M_stride_s1 + pid_s2 * M_stride_s2 + pid_h * M_stride_h

    g_val = tl.load(G_ptr + g_off)
    lp_val = tl.load(Lp_ptr + lp_off)
    tl.store(M_ptr + m_off, g_val * lp_val)


# Triton kernel: Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h]
# Inputs: M [B,N,S,S,H], hidden_states [B,N,S,H,D], Output Y [B,N,S,H] float32.
@triton.jit
def y_diag_reduce_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    B_size, N_size, S_size, H_size, D_size,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
    Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
    BLOCK_J: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s1 = tl.program_id(2)  # i
    pid_h = tl.program_id(3)   # h

    acc = 0.0
    # Loop over j in tiles
    for j0 in range(0, S_size, BLOCK_J):
        # For each j in tile, accumulate M[b,n,i,j,h] * hidden[b,n,j,h,:] over D
        for jj in range(0, BLOCK_J):
            j_idx = j0 + jj
            if j_idx < S_size:
                m_off = pid_b * M_stride_b + pid_n * M_stride_n + pid_s1 * M_stride_s1 + j_idx * M_stride_s2 + pid_h * M_stride_h
                m_val = tl.load(M_ptr + m_off)
                sum_hs = 0.0
                # Reduce over D dimension
                for d0 in range(0, D_size, BLOCK_D):
                    d_idx = d0 + tl.arange(0, BLOCK_D)
                    mask_d = d_idx < D_size
                    hs_off = pid_b * hidden_stride_b + pid_n * hidden_stride_n + j_idx * hidden_stride_s + pid_h * hidden_stride_h + d_idx * hidden_stride_d
                    hs_vals = tl.load(hidden_ptr + hs_off, mask=mask_d, other=0.0)
                    sum_hs += tl.sum(hs_vals, axis=0)
                acc += m_val * sum_hs
    # Store Y[b,n,i,h]
    y_off = pid_b * Y_stride_b + pid_n * Y_stride_n + pid_s1 * Y_stride_s1 + pid_h * Y_stride_h
    tl.store(Y_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, N, S, H, D]; A_cumsum: [B, H, N, S]
        B_size, N_size, S_size, H_size, D_size = hidden_states.shape

        # Expand B and C from N_GROUPS=8 to NUM_HEADS=32
        B_exp = B.repeat_interleave(H_size // 8, dim=3)  # [B, N, S, H, D]
        C_exp = C.repeat_interleave(H_size // 8, dim=3)  # [B, N, S, H, D]

        # Allocate L [B, H, N, S, S] float32
        L = torch.empty((B_size, H_size, N_size, S_size, S_size), dtype=torch.float32, device=hidden_states.device)

        # Launch L kernel: grid over (B,H,N)
        grid_L = (B_size, H_size, N_size)
        a_segment_sum_exp_kernel[grid_L](
            A_cumsum, L,
            B_size, H_size, N_size, S_size,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=4, num_stages=2,
        )

        # Permute L to [B, N, S, S, H] for multiply
        Lp = L.permute(0, 2, 3, 4, 1).contiguous()  # [B, N, S, S, H]

        # Compute G [B, N, S, S, H] float32
        G = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=hidden_states.device)

        grid_G = (B_size, N_size, S_size, S_size, H_size)
        g_contract_kernel[grid_G](
            B_exp, C_exp, G,
            B_size, N_size, S_size, H_size, D_size,
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            BLOCK_D=32,
            num_warps=4, num_stages=2,
        )

        # Elementwise M = G * Lp
        M = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=hidden_states.device)

        grid_M = (B_size, N_size, S_size, S_size, H_size)
        m_mul_kernel[grid_M](
            G, Lp, M,
            B_size, N_size, S_size, H_size,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            Lp.stride(0), Lp.stride(1), Lp.stride(2), Lp.stride(3), Lp.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=4, num_stages=2,
        )

        # Final reduction to Y [B, N, S, H]
        Y = torch.empty((B_size, N_size, S_size, H_size), dtype=torch.float32, device=hidden_states.device)

        grid_Y = (B_size, N_size, S_size, H_size)
        y_diag_reduce_kernel[grid_Y](
            M, hidden_states.to(torch.float32).contiguous(), Y,
            B_size, N_size, S_size, H_size, D_size,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            BLOCK_J=32, BLOCK_D=32,
            num_warps=4, num_stages=2,
        )

        # Cast to bfloat16 to match original output dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
