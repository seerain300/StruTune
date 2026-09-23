import torch
import triton
import triton.language as tl


# Kernel 1: Create 1D lower-triangular mask (diagonal=-1) of length S*S. Output: mask[i*S + j] = 1 if j <= i else 0 (int8).
@triton.jit
def tril_mask_kernel(mask_ptr, S: tl.int32):
    pid = tl.program_id(0)
    # Derive i and j for 1D index
    i = pid // S
    j = pid % S
    lower = j <= i
    # Store 1 for True, 0 for False
    if lower:
        tl.store(mask_ptr + pid, 1)
    else:
        tl.store(mask_ptr + pid, 0)


# Kernel 2: Compute L[b,h,n,i,j] = exp(sum_{k=0..i} A[b,h,n,k]) if j <= i else 0.0
# Inputs: A_cumsum [B, H, N, S], mask [S*S] (int8), Output: L [B, H, N, S, S] (float32)
@triton.jit
def a_segment_sum_exp_kernel(
    A_ptr, mask_ptr, L_ptr,
    B_size, H_size, N_size, S_size,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
):
    b = tl.program_id(0)  # 0..B_size-1
    h = tl.program_id(1)  # 0..H_size-1
    n = tl.program_id(2)  # 0..N_size-1

    # Initialize segment_sum
    segment_sum = 0.0
    # Loop over i from 0 to S-1
    i = 0
    while i < S_size:
        # Load A[b,h,n,i]
        a_val = tl.load(
            A_ptr + b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_s
        )
        segment_sum += a_val

        # For each j <= i, store exp(segment_sum) into L[b,h,n,i,j]
        j = 0
        while j <= i:
            # Use mask to decide if lower-triangular position: j <= i -> mask[j*S + i] == 1
            # We only need lower-triangular condition j <= i; for j > i we do nothing (0.0 is written in Py code).
            lower = (j <= i)
            if lower:
                # store exp(segment_sum) at L[b,h,n,i,j]
                tl.store(
                    L_ptr + b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_s1 + j * L_stride_s2,
                    tl.exp(segment_sum)
                )
            j += 1
        i += 1


# Kernel 3: G contraction: G[b,n,i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
# Inputs: B_exp [B,N,S,H,D], C_exp [B,N,S,H,D], Output: G [B,N,S,S,H] float32
@triton.jit
def g_contract_kernel(
    B_ptr, C_ptr, G_ptr,
    B_size, N_size, S_size, H_size, G_SIZE, D_size,
    B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
    C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)  # source index
    j = tl.program_id(3)  # target index
    h = tl.program_id(4)  # head index

    g_val = 0.0
    d0 = 0
    while d0 < D_size:
        d_off = d0 + tl.arange(0, BLOCK_D)
        mask = d_off < D_size
        # Load B[b,n,j,h,d] and C[b,n,i,h,d] vectors
        B_vec = tl.load(
            B_ptr + b * B_stride_b + n * B_stride_n + j * B_stride_s + h * B_stride_h + d_off * B_stride_d,
            mask=mask, other=0.0
        )
        C_vec = tl.load(
            C_ptr + b * C_stride_b + n * C_stride_n + i * C_stride_s + h * C_stride_h + d_off * C_stride_d,
            mask=mask, other=0.0
        )
        # Accumulate sum
        g_val += tl.sum(B_vec * C_vec, axis=0)
        d0 += BLOCK_D

    # Store G[b,n,i,j,h] = g_val
    tl.store(
        G_ptr + b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h,
        g_val
    )


# Kernel 4: M = G * L elementwise, where G: [B,N,S,S,H], L: [B,H,N,S,S] permuted to [B,N,S,S,H]
@triton.jit
def m_mul_kernel(
    G_ptr, L_ptr, M_ptr,
    B_size, N_size, S_size, H_size,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    L_stride_b, L_stride_n, L_stride_s1, L_stride_s2, L_stride_h,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    g_val = tl.load(
        G_ptr + b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h
    )
    l_val = tl.load(
        L_ptr + b * L_stride_b + n * L_stride_n + i * L_stride_s1 + j * L_stride_s2 + h * L_stride_h
    )
    prod = g_val * l_val
    tl.store(
        M_ptr + b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h,
        prod
    )


# Kernel 5: Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h,:]
# Inputs: M [B,N,S,S,H], hidden [B,N,S,H,D], Output: Y [B,N,S,H] float32
@triton.jit
def y_diag_reduce_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    B_size, N_size, S_size, H_size, D_size,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
    Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)  # source index
    h = tl.program_id(3)  # head index

    acc = 0.0
    d0 = 0
    while d0 < D_size:
        d_off = d0 + tl.arange(0, BLOCK_D)
        mask = d_off < D_size
        # Load M[b,n,i,j,h] for all j as a vector by looping over j? We need vector over j, but Triton grid only has one j per program? Instead, we loop j and accumulate.
        j = 0
        total = 0.0
        while j < S_size:
            m_val = tl.load(
                M_ptr + b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h
            )
            hidden_j = tl.load(
                hidden_ptr + b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_s + h * hidden_stride_h + d_off * hidden_stride_d,
                mask=mask, other=0.0
            )
            total += m_val * hidden_j
            j += 1
        acc += tl.sum(total, axis=0)
        d0 += BLOCK_D

    tl.store(
        Y_ptr + b * Y_stride_b + n * Y_stride_n + i * Y_stride_s1 + h * Y_stride_h,
        acc
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants from the original code
        self.NUM_HEADS = 32
        self.N_GROUPS = 8
        self.CHUNK_SIZE = 128

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, N, S, H, D]
        A_cumsum:      [B, H, N, S]
        B:             [B, N, S, G, D] (assumed expanded to H on host before call)
        C:             [B, N, S, G, D] (assumed expanded to H on host before call)
        Output:        [B, N, S, H] in bfloat16
        """
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All inputs must be CUDA tensors."

        # Shapes
        B_size, N_size, S_size, H_size, D_size = hidden_states.shape

        # Ensure contiguous and float32 for computation
        A = A_cumsum.to(torch.float32).contiguous()            # [B, H, N, S]
        B_exp = B.to(torch.float32).contiguous()              # [B, N, S, H, D] (note: H is passed in as H_size)
        C_exp = C.to(torch.float32).contiguous()              # [B, N, S, H, D]
        hidden = hidden_states.to(torch.float32).contiguous() # [B, N, S, H, D]

        # 1) Lower-triangular mask (diagonal=-1) as int8: length S_size * S_size
        mask = torch.empty(S_size * S_size, dtype=torch.int8, device=hidden.device)
        tril_mask_kernel[(S_size * S_size,)](
            mask,
            S_size,
            num_warps=1, num_stages=1
        )

        # 2) Compute L in Triton: L[b,h,n,i,j] = exp(sum_{k=0..i} A[b,h,n,k]) if j <= i else 0.0
        L = torch.empty((B_size, H_size, N_size, S_size, S_size), dtype=torch.float32, device=hidden.device)
        a_segment_sum_exp_kernel[(B_size, H_size, N_size)](
            A, mask, L,
            B_size, H_size, N_size, S_size,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=1, num_stages=1
        )

        # 3) G contraction: G[b,n,i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
        G = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=hidden.device)
        g_contract_kernel[(B_size, N_size, S_size, S_size, H_size)](
            B_exp, C_exp, G,
            B_size, N_size, S_size, H_size, self.N_GROUPS, D_size,
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            BLOCK_D=32,
            num_warps=1, num_stages=1
        )

        # 4) M = G * L (elementwise multiply). Permute L to [B,N,S,S,H] for multiply.
        # Note: Triton kernel expects L permuted as [B,N,S,S,H]. We permute here:
        L_perm = L.permute(0, 2, 3, 4, 1)  # [B, N, S, S, H]
        M = torch.empty_like(G)            # [B,N,S,S,H]
        m_mul_kernel[(B_size, N_size, S_size, S_size, H_size)](
            G, L_perm, M,
            B_size, N_size, S_size, H_size,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L_perm.stride(0), L_perm.stride(1), L_perm.stride(2), L_perm.stride(3), L_perm.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=1, num_stages=1
        )

        # 5) Y_diag reduction: Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h,:]
        Y = torch.empty((B_size, N_size, S_size, H_size), dtype=torch.float32, device=hidden.device)
        y_diag_reduce_kernel[(B_size, N_size, S_size, H_size)](
            M, hidden, Y,
            B_size, N_size, S_size, H_size, D_size,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            BLOCK_D=32,
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 to match original output dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
