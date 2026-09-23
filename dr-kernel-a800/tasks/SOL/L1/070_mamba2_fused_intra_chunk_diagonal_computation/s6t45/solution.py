import torch
import triton
import triton.language as tl


# Kernel 1: L = exp(segment_sum(A)) for lower triangle (j <= i)
# Inputs:
#   A: [B, H, N, S], float32
#   L: [B, H, N, S, S], float32
@triton.jit
def a_segment_sum_exp_kernel(
    A_ptr, L_ptr,
    B_size, H_size, N_size, S_size,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
):
    # program ids for batch, head, chunk
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    # precompute strides
    A_b = pid_b * A_stride_b
    A_h = pid_h * A_stride_h
    A_n = pid_n * A_stride_n

    # Loop over i from 0 to S-1
    for i in range(0, S_size):
        i_off = i * A_stride_s
        # compute segment_sum = sum_{k=0..i} A[b,h,n,k]
        segment_sum = 0.0
        for k in range(0, i + 1):
            ptr = A_ptr + A_b + A_h + A_n + k * A_stride_s
            val = tl.load(ptr)  # A[b,h,n,k]
            segment_sum += val
        # For each j <= i, set L[b,h,n,i,j] = exp(segment_sum)
        for j in range(0, S_size):
            if j <= i:
                exp_val = tl.exp(segment_sum)
                ptr_L = L_ptr + pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + i * L_stride_s1 + j * L_stride_s2
                tl.store(ptr_L, exp_val)
            else:
                # j > i -> 0 (mask by tril with diagonal=-1)
                ptr_L = L_ptr + pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + i * L_stride_s1 + j * L_stride_s2
                tl.store(ptr_L, 0.0)


# Kernel 2: G[b,n,i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
# Inputs:
#   B: [B, N, S, H, D], float32
#   C: [B, N, S, H, D], float32
#   G: [B, N, S, S, H], float32
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

    i = pid_si
    j = pid_sj

    # Accumulate over D
    acc = 0.0
    for d_off in range(0, D_size, BLOCK_D):
        offs = d_off + tl.arange(0, BLOCK_D)
        mask = offs < D_size
        # Load B[b,n,j,h, offs]
        B_ptr_j = B_ptr + pid_b * B_stride_b + pid_n * B_stride_n + j * B_stride_s + pid_h * B_stride_h
        B_vals = tl.load(B_ptr_j + offs * B_stride_d, mask=mask, other=0.0)
        # Load C[b,n,i,h, offs]
        C_ptr_i = C_ptr + pid_b * C_stride_b + pid_n * C_stride_n + i * C_stride_s + pid_h * C_stride_h
        C_vals = tl.load(C_ptr_i + offs * C_stride_d, mask=mask, other=0.0)
        # Accumulate dot: sum_{k} C_vals[k] * B_vals[k]
        # Note: Triton does not have a built-in dot over vector; we do elementwise multiply and reduce via tl.sum
        acc += tl.sum(C_vals * B_vals, axis=0)

    # Store G[b,n,i,j,h] = acc
    G_ptr_out = G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + pid_h * G_stride_h
    tl.store(G_ptr_out, acc)


# Kernel 3: M = G * L (elementwise)
# Inputs:
#   G: [B, N, S, S, H], float32
#   L_perm: [B, N, S, S, H], float32  (permuted L from [B,H,N,S,S] -> [B,N,S,S,H])
#   M: [B, N, S, S, H], float32
@triton.jit
def m_mul_kernel(
    G_ptr, Lp_ptr, M_ptr,
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

    i = pid_si
    j = pid_sj

    G_val = tl.load(G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + pid_h * G_stride_h)
    Lp_val = tl.load(Lp_ptr + pid_b * L_stride_b + pid_n * L_stride_n + i * L_stride_s1 + j * L_stride_s2 + pid_h * L_stride_h)
    out = G_val * Lp_val
    tl.store(M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + pid_h * M_stride_h, out)


# Kernel 4: Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h]
# Inputs:
#   M: [B, N, S, S, H], float32
#   hidden_states: [B, N, S, H, D], float32
#   Y: [B, N, S, H], float32
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
    pid_h = tl.program_id(3)  # h

    i = pid_si
    acc = 0.0
    # Loop j across S with tiles
    for j in range(0, S_size):
        # load M[b,n,i,j,h]
        M_val = tl.load(M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + pid_h * M_stride_h)
        # load hidden[b,n,j,h,:D] then sum over D
        # We iterate D in tiles
        for d_off in range(0, D_size, BLOCK_D):
            offs = d_off + tl.arange(0, BLOCK_D)
            mask = offs < D_size
            hidden_ptr_jh = hidden_ptr + pid_b * hidden_stride_b + pid_n * hidden_stride_n + j * hidden_stride_s + pid_h * hidden_stride_h
            hidden_vals = tl.load(hidden_ptr_jh + offs * hidden_stride_d, mask=mask, other=0.0)
            acc += M_val * tl.sum(hidden_vals, axis=0)
    # store Y[b,n,i,h] = acc
    tl.store(Y_ptr + pid_b * Y_stride_b + pid_n * Y_stride_n + i * Y_stride_s1 + pid_h * Y_stride_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, N, S, H, D]
        A_cumsum:      [B, H, N, S]
        B:             [B, N, S, H, D]  (note: original B is [B,N,S,G,D], after expand to H)
        C:             [B, N, S, H, D]
        Output:        [B, N, S, H] in bfloat16, same as original
        """
        # Ensure dtype/device and contiguous
        B_size, N_size, S_size, H_size, D_size = hidden_states.shape
        device = hidden_states.device

        # A_cumsum: float32 contiguous
        A = A_cumsum.to(torch.float32).contiguous()

        # Allocate L: [B, H, N, S, S] float32
        L = torch.empty((B_size, H_size, N_size, S_size, S_size), dtype=torch.float32, device=device)

        # Launch kernel to compute L = exp(segment_sum(A)) for lower triangle
        grid_L = (B_size, H_size, N_size)
        a_segment_sum_exp_kernel[grid_L](
            A, L,
            B_size, H_size, N_size, S_size,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=1, num_stages=1,
        )

        # G: [B, N, S, S, H] float32
        G = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)

        # Launch G contraction kernel
        grid_G = (B_size, N_size, S_size, S_size, H_size)
        g_contract_kernel[grid_G](
            B, C, G,
            B_size, N_size, S_size, H_size, D_size,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            BLOCK_D=32,  # H=D=32 in given model; tuneable
            num_warps=4, num_stages=2,
        )

        # Permute L to [B,N,S,S,H]
        Lp = L.permute(0, 2, 3, 4, 1).contiguous()

        # M: [B, N, S, S, H] float32
        M = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)

        # Launch elementwise multiply
        grid_M = (B_size, N_size, S_size, S_size, H_size)
        m_mul_kernel[grid_M](
            G, Lp, M,
            B_size, N_size, S_size, H_size,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            Lp.stride(0), Lp.stride(1), Lp.stride(2), Lp.stride(3), Lp.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=4, num_stages=2,
        )

        # Y: [B, N, S, H] float32
        Y = torch.empty((B_size, N_size, S_size, H_size), dtype=torch.float32, device=device)

        # Launch reduction kernel
        grid_Y = (B_size, N_size, S_size, H_size)
        y_diag_reduce_kernel[grid_Y](
            M, hidden_states.to(torch.float32).contiguous(), Y,
            B_size, N_size, S_size, H_size, D_size,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_states.to(torch.float32).contiguous().stride(0), hidden_states.to(torch.float32).contiguous().stride(1),
            hidden_states.to(torch.float32).contiguous().stride(2), hidden_states.to(torch.float32).contiguous().stride(3),
            hidden_states.to(torch.float32).contiguous().stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            BLOCK_D=32,
            num_warps=4, num_stages=2,
        )

        # Cast to bfloat16 to match original output dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
