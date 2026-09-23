import torch
import triton
import triton.language as tl


# Kernel 1: Compute L = exp(cumsum(masked A)) on lower triangle (diagonal=-1).
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

    # Initialize segment sum (prefix) to 0.0
    prefix = 0.0  # float32

    # Loop over i from 0 to S-1
    for i in range(0, S_size):
        # Load A[b, h, n, i]
        a_val = tl.load(A_ptr + pid_b * A_stride_b + pid_h * A_stride_h + pid_n * A_stride_n + i * A_stride_s)
        prefix += a_val
        # Loop over j from 0 to S-1: for j<=i, L[b,h,n,i,j] = exp(prefix); else 0.0
        for j in range(0, S_size):
            # Compute L address: L[b, h, n, i, j]
            addr_L = pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + i * L_stride_s1 + j * L_stride_s2
            # If j <= i, store exp(prefix); else 0.0
            if j <= i:
                val = tl.exp(prefix)
            else:
                val = 0.0
            tl.store(L_ptr + addr_L, val)


# Kernel 2: Compute G[b,n,i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
# Inputs: B [B, N, S, H, D], C [B, N, S, H, D], Output G [B, N, S, S, H]
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
    pid_i = tl.program_id(2)  # i
    pid_j = tl.program_id(3)  # j
    pid_h = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    # Loop over D in tiles
    for d0 in range(0, D_size, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        mask_d = d_idx < D_size
        # Load B[b, n, j, h, d] and C[b, n, i, h, d]
        # For B: s=pid_j
        B_offsets = pid_b * B_stride_b + pid_n * B_stride_n + pid_j * B_stride_s + pid_h * B_stride_h + d_idx * B_stride_d
        B_vals = tl.load(B_ptr + B_offsets, mask=mask_d, other=0.0)
        # For C: s=pid_i
        C_offsets = pid_b * C_stride_b + pid_n * C_stride_n + pid_i * C_stride_s + pid_h * C_stride_h + d_idx * C_stride_d
        C_vals = tl.load(C_ptr + C_offsets, mask=mask_d, other=0.0)
        acc += tl.sum(C_vals * B_vals, axis=0)
    # Store G[b, n, i, j, h] = acc
    G_addr = pid_b * G_stride_b + pid_n * G_stride_n + pid_i * G_stride_s1 + pid_j * G_stride_s2 + pid_h * G_stride_h
    tl.store(G_ptr + G_addr, acc)


# Kernel 3: M = G * L
@triton.jit
def m_mul_kernel(
    G_ptr, L_ptr, M_ptr,
    B_size, N_size, S_size, H_size,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    g_addr = pid_b * G_stride_b + pid_n * G_stride_n + pid_i * G_stride_s1 + pid_j * G_stride_s2 + pid_h * G_stride_h
    l_addr = pid_b * L_stride_b + pid_n * L_stride_h + pid_n * L_stride_n + pid_i * L_stride_s1 + pid_j * L_stride_s2
    m_addr = pid_b * M_stride_b + pid_n * M_stride_n + pid_i * M_stride_s1 + pid_j * M_stride_s2 + pid_h * M_stride_h

    g_val = tl.load(G_ptr + g_addr)
    l_val = tl.load(L_ptr + l_addr)
    tl.store(M_ptr + m_addr, g_val * l_val)


# Kernel 4: Y[b, n, i, h] = sum_j M[b, n, i, j, h] * hidden[b, n, j, h]
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
    pid_i = tl.program_id(2)
    pid_h = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)
    # Loop over j (chunk dimension)
    for j in range(0, S_size):
        # Accumulate sum over D for hidden
        tmp_acc = tl.zeros((), dtype=tl.float32)
        for d0 in range(0, D_size, BLOCK_D):
            d_idx = d0 + tl.arange(0, BLOCK_D)
            mask_d = d_idx < D_size
            # hidden[b, n, j, h, d]
            hidden_offsets = pid_b * hidden_stride_b + pid_n * hidden_stride_n + j * hidden_stride_s + pid_h * hidden_stride_h + d_idx * hidden_stride_d
            h_vals = tl.load(hidden_ptr + hidden_offsets, mask=mask_d, other=0.0)
            # M[b, n, i, j, h]
            M_addr = pid_b * M_stride_b + pid_n * M_stride_n + pid_i * M_stride_s1 + j * M_stride_s2 + pid_h * M_stride_h
            m_val = tl.load(M_ptr + M_addr)
            tmp_acc += tl.sum(h_vals * m_val, axis=0)
        acc += tmp_acc
    # Store Y[b, n, i, h]
    Y_addr = pid_b * Y_stride_b + pid_n * Y_stride_n + pid_i * Y_stride_s1 + pid_h * Y_stride_h
    tl.store(Y_ptr + Y_addr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on CUDA and contiguous
        device = hidden_states.device
        B = B.contiguous()
        C = C.contiguous()
        hidden = hidden_states.contiguous()
        A = A_cumsum.contiguous()

        # Shapes
        B_size, N_size, S, H, D = hidden.shape
        assert A.shape == (B_size, H, N_size, S), "A_cumsum must have shape [B, H, N, S]"
        # Assert B/C expanded to H
        assert B.shape[3] == H and C.shape[3] == H, "B and C must be expanded to num_heads (H)"

        # 1) Triton: Compute L = exp(cumsum(masked A)) with lower-triangular (diagonal=-1)
        L = torch.empty((B_size, H, N_size, S, S), dtype=torch.float32, device=device)
        a_segment_sum_exp_kernel[(B_size, H_size, N_size)](
            A, L,
            B_size, H, N_size, S,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=1, num_stages=1
        )

        # 2) Triton: Compute G[b,n,i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
        G = torch.empty((B_size, N_size, S, S, H), dtype=torch.float32, device=device)

        B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d = B.stride()
        C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d = C.stride()
        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()

        g_contract_kernel[(B_size, N_size, S, S, H)](
            B, C, G,
            B_size, N_size, S, H, D,
            B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
            C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            BLOCK_D=32,
            num_warps=1, num_stages=1
        )

        # 3) Triton: M = G * L
        M = torch.empty_like(G, dtype=torch.float32, device=device)

        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()
        L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2 = L.stride()
        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()

        m_mul_kernel[(B_size, N_size, S, S, H)](
            G, L, M,
            B_size, N_size, S, H,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            num_warps=1, num_stages=1
        )

        # 4) Triton: Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden[b,n,j,h]
        Y = torch.empty((B_size, N_size, S, H), dtype=torch.float32, device=device)

        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()
        hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d = hidden.stride()
        Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h = Y.stride()

        y_diag_reduce_kernel[(B_size, N_size, S, H)](
            M, hidden, Y,
            B_size, N_size, S, H, D,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
            Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
            BLOCK_D=32,
            num_warps=1, num_stages=1
        )

        # Return in bfloat16 to match original behavior
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
