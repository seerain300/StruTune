import torch
import triton
import triton.language as tl


# Kernel 1: Compute L = exp(segment_sum(A)) on lower triangle (diagonal=-1).
# Input: A_cumsum [B, H, N, S], Output: L [B, H, N, S, S] float32
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

    # Compute in float32 for accuracy
    # segment_sum over i for each i, then exp and store for j<=i
    for i in range(0, S_size):
        seg_sum = 0.0
        # sum over k in [0..i]
        for k in range(0, i + 1):
            a_offset = pid_b * A_stride_b + pid_h * A_stride_h + pid_n * A_stride_n + k * A_stride_s
            val = tl.load(A_ptr + a_offset)  # assume A is float32
            seg_sum += val
        exp_val = tl.exp(seg_sum)
        for j in range(0, S_size):
            if j <= i:
                l_offset = pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + i * L_stride_s1 + j * L_stride_s2
                # store exp_val as float32
                tl.store(L_ptr + l_offset, exp_val)


# Kernel 2: G contraction: G[b,n,i,j,h] = sum_d C_expanded[b,n,i,h,d] * B_expanded[b,n,j,h,d]
@triton.jit
def g_contract_kernel(
    B_exp_ptr, C_exp_ptr, G_ptr,
    B_size, N_size, S_size, H_size, D_size,
    B_exp_stride0, B_exp_stride1, B_exp_stride2, B_exp_stride3, B_exp_stride4,
    C_exp_stride0, C_exp_stride1, C_exp_stride2, C_exp_stride3, C_exp_stride4,
    G_stride0, G_stride1, G_stride2, G_stride3, G_stride4,
    BLOCK_D: tl.constexpr = 32,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    # tile over D
    for d_start in range(0, D_size, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D_size
        # load B_exp[b,n,j,h,d] and C_exp[b,n,i,h,d]
        b_off = pid_b * B_exp_stride0 + pid_n * B_exp_stride1 + pid_j * B_exp_stride2 + pid_h * B_exp_stride3 + (offs_d * B_exp_stride4)
        c_off = pid_b * C_exp_stride0 + pid_n * C_exp_stride1 + pid_i * C_exp_stride2 + pid_h * C_exp_stride3 + (offs_d * C_exp_stride4)
        b_vals = tl.load(B_exp_ptr + b_off, mask=mask_d, other=0.0)
        c_vals = tl.load(C_exp_ptr + c_off, mask=mask_d, other=0.0)
        acc += tl.sum(b_vals * c_vals, axis=0)

    g_off = pid_b * G_stride0 + pid_n * G_stride1 + pid_i * G_stride2 + pid_j * G_stride3 + pid_h * G_stride4
    tl.store(G_ptr + g_off, acc)


# Kernel 3: Elementwise M = G * Lp (L permuted to [B,N,S,S,H])
@triton.jit
def m_mul_kernel(
    G_ptr, Lp_ptr, M_ptr,
    B_size, N_size, S_size, H_size,
    G_stride0, G_stride1, G_stride2, G_stride3, G_stride4,
    Lp_stride0, Lp_stride1, Lp_stride2, Lp_stride3, Lp_stride4,
    M_stride0, M_stride1, M_stride2, M_stride3, M_stride4,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    g_off = pid_b * G_stride0 + pid_n * G_stride1 + pid_i * G_stride2 + pid_j * G_stride3 + pid_h * G_stride4
    lp_off = pid_b * Lp_stride0 + pid_n * Lp_stride1 + pid_i * Lp_stride2 + pid_j * Lp_stride3 + pid_h * Lp_stride4
    m_off = pid_b * M_stride0 + pid_n * M_stride1 + pid_i * M_stride2 + pid_j * M_stride3 + pid_h * M_stride4

    g_val = tl.load(G_ptr + g_off)
    lp_val = tl.load(Lp_ptr + lp_off)
    prod = g_val * lp_val
    tl.store(M_ptr + m_off, prod)


# Kernel 4: Y reduction: Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h]
@triton.jit
def y_diag_reduce_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    B_size, N_size, S_size, H_size, D_size,
    M_stride0, M_stride1, M_stride2, M_stride3, M_stride4,
    hidden_stride0, hidden_stride1, hidden_stride2, hidden_stride3, hidden_stride4,
    Y_stride0, Y_stride1, Y_stride2, Y_stride3,
    BLOCK_J: tl.constexpr = 32,
    BLOCK_D: tl.constexpr = 32,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_h = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)
    for j_start in range(0, S_size, BLOCK_J):
        offs_j = j_start + tl.arange(0, BLOCK_J)
        mask_j = offs_j < S_size
        # dot over d tiles
        for d_start in range(0, D_size, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D_size
            # M[b,n,i,j,h]
            m_off = pid_b * M_stride0 + pid_n * M_stride1 + pid_i * M_stride2 + (offs_j[:, None] * M_stride3) + pid_h * M_stride4
            m_vals = tl.load(M_ptr + m_off, mask=mask_j[:, None], other=0.0)  # shape [BLOCK_J, 1]
            # hidden[b,n,j,h]
            h_off = pid_b * hidden_stride0 + pid_n * hidden_stride1 + (offs_j * hidden_stride2) + pid_h * hidden_stride3 + (offs_d[None, :] * hidden_stride4)
            h_vals = tl.load(hidden_ptr + h_off, mask=mask_j[:, None] & mask_d[None, :], other=0.0)  # shape [BLOCK_J, BLOCK_D]
            prod = m_vals * h_vals  # [BLOCK_J, BLOCK_D]
            # reduce over d tile
            acc += tl.sum(prod, axis=1)  # sum over BLOCK_J rows, each row reduces over BLOCK_D

    y_off = pid_b * Y_stride0 + pid_n * Y_stride1 + pid_i * Y_stride2 + pid_h * Y_stride3
    tl.store(Y_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, N, S, H, D]
        A_cumsum: [B, H, N, S] (float32 recommended)
        B: [B, N, S, G, D]
        C: [B, N, S, G, D]
        Output: [B, N, S, H] in bfloat16
        """
        B_size, N_size, S_size, H_size, D_size = hidden_states.shape
        device = hidden_states.device

        # 1) Compute L in Triton
        L = torch.empty((B_size, H_size, N_size, S_size, S_size), dtype=torch.float32, device=device)
        grid_L = (B_size, H_size, N_size)
        a_segment_sum_exp_kernel[grid_L](
            A_cumsum, L,
            B_size, H_size, N_size, S_size,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=1, num_stages=1,
        )

        # 2) Expand B and C to H (repeat_interleave(4) from G=8 -> H=32)
        B_expanded = B.repeat_interleave(4, dim=3)  # [B, N, S, H, D]
        C_expanded = C.repeat_interleave(4, dim=3)

        # 3) Compute G in Triton: [B, N, S, S, H]
        G = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)
        grid_G = (B_size, N_size, S_size, S_size, H_size)
        g_contract_kernel[grid_G](
            B_expanded, C_expanded, G,
            B_size, N_size, S_size, H_size, D_size,
            B_expanded.stride(0), B_expanded.stride(1), B_expanded.stride(2), B_expanded.stride(3), B_expanded.stride(4),
            C_expanded.stride(0), C_expanded.stride(1), C_expanded.stride(2), C_expanded.stride(3), C_expanded.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            BLOCK_D=32,  # D_size should be 32 in provided setup
            num_warps=4, num_stages=2,
        )

        # 4) Permute L to [B, N, S, S, H] and multiply elementwise
        Lp = L.permute(0, 2, 3, 4, 1).contiguous()  # [B, N, S, S, H]
        M = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)
        grid_M = (B_size, N_size, S_size, S_size, H_size)
        m_mul_kernel[grid_M](
            G, Lp, M,
            B_size, N_size, S_size, H_size,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            Lp.stride(0), Lp.stride(1), Lp.stride(2), Lp.stride(3), Lp.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=4, num_stages=2,
        )

        # 5) Reduce to Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h]
        Y = torch.empty((B_size, N_size, S_size, H_size), dtype=torch.float32, device=device)
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
