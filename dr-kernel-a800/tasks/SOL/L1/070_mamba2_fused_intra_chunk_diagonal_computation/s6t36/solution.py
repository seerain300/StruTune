import torch
import triton
import triton.language as tl


# Kernel 1: Create 1D lower-triangular mask (diagonal=-1) of length S*S. Output: mask[i*S + j] = 1 if j <= i else 0 (int8).
@triton.jit
def tril_mask_kernel(mask_ptr, S: tl.constexpr):
    pid = tl.program_id(0)
    S_int = S
    i = pid // S
    j = pid % S
    lower = j <= i
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

    # We fill L row by row for fixed (b,h,n). For each i in [0..S_size-1], segment_sum = sum_{k=0..i} A[b,h,n,k]
    # Then for j in [0..S_size-1], if j <= i: L[b,h,n,i,j] = exp(segment_sum), else 0.0
    for i in range(S_size):
        segment_sum = 0.0
        # accumulate A for this i
        k = 0
        while k <= i:
            a_off = b * A_stride_b + h * A_stride_h + n * A_stride_n + k * A_stride_s
            a_val = tl.load(A_ptr + a_off)
            segment_sum += a_val
            k += 1

        # write L[b,h,n,i,j] for all j
        for j in range(S_size):
            # lower triangular check using mask or j<=i always holds because loop goes j from 0..S_size-1 and i>=j per loop order,
            # but to be robust we can re-check via mask: L_idx = b*L_stride_b + h*L_stride_h + n*L_stride_n + i*L_stride_s1 + j*L_stride_s2
            L_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_s1 + j * L_stride_s2
            if j <= i:
                tl.store(L_ptr + L_off, tl.exp(segment_sum))
            else:
                tl.store(L_ptr + L_off, 0.0)


# Kernel 3: G contraction: G[b,n,i,j,h] = sum over d of C[b,n,i,h,d] * B[b,n,j,h,d]
# Inputs: B_exp [B, N, S, H, D], C_exp [B, N, S, H, D], Output: G [B, N, S, S, H] (float32)
@triton.jit
def g_contract_kernel(
    B_ptr, C_ptr, G_ptr,
    B_size, N_size, S_size, H_size, D_size,
    B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
    C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    BLOCK_D: tl.constexpr,
):
    # Grid over (b, n, i, j, h)
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    d0 = 0
    while d0 < D_size:
        offs_d = d0 + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D_size

        # B[b,n,j,h,d]
        b_off = b * B_stride_b + n * B_stride_n + j * B_stride_s + h * B_stride_h + offs_d * B_stride_d
        B_vals = tl.load(B_ptr + b_off, mask=d_mask, other=0.0)

        # C[b,n,i,h,d]
        c_off = b * C_stride_b + n * C_stride_n + i * C_stride_s + h * C_stride_h + offs_d * C_stride_d
        C_vals = tl.load(C_ptr + c_off, mask=d_mask, other=0.0)

        acc += tl.sum(B_vals * C_vals, axis=0)
        d0 += BLOCK_D

    G_off = b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h
    tl.store(G_ptr + G_off, acc)


# Kernel 4: M = G * L (elementwise)
# Inputs: G [B, N, S, S, H], L [B, H, N, S, S], Output: M [B, N, S, S, H]
@triton.jit
def m_mul_kernel(G_ptr, L_ptr, M_ptr,
                 B_size, N_size, S_size, H_size,
                 G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
                 L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
                 M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    G_off = b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h
    L_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_s1 + j * L_stride_s2
    M_off = b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h

    m_val = tl.load(G_ptr + G_off) * tl.load(L_ptr + L_off)
    tl.store(M_ptr + M_off, m_val)


# Kernel 5: Y_diag reduction: Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h,:]
# Inputs: M [B, N, S, S, H], hidden_states [B, N, S, H, D], Output: Y [B, N, S, H] (float32)
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
    i = tl.program_id(2)
    h = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over j in tiles
    j0 = 0
    while j0 < S_size:
        j_vec = j0 + tl.arange(0, S_size)  # vector for j positions
        mask_j = j_vec < S_size

        # For each j, accumulate M[b,n,i,j,h] * hidden[b,n,j,h,:]
        # We'll use D tiling to handle D_size
        for jj in range(S_size):  # S_size is constexpr here, loop over j
            # skip if jj >= S_size (mask handles)
            # Compute pointers
            M_off = b * M_stride_b + n * M_stride_n + i * M_stride_s1 + jj * M_stride_s2 + h * M_stride_h
            m_val = tl.load(M_ptr + M_off)  # scalar

            hidden_off = b * hidden_stride_b + n * hidden_stride_n + jj * hidden_stride_s + h * hidden_stride_h
            acc_j = tl.zeros((), dtype=tl.float32)

            d0 = 0
            while d0 < D_size:
                offs_d = d0 + tl.arange(0, BLOCK_D)
                d_mask = offs_d < D_size
                hidden_ptr_d = hidden_ptr + hidden_off + offs_d * hidden_stride_d
                hs_d = tl.load(hidden_ptr_d, mask=d_mask, other=0.0)
                # m_val is scalar; broadcast multiply
                acc_j += tl.sum(m_val * hs_d, axis=0)
                d0 += BLOCK_D

            # add to total
            acc += acc_j

        j0 += 1

    Y_off = b * Y_stride_b + n * Y_stride_n + i * Y_stride_s1 + h * Y_stride_h
    tl.store(Y_ptr + Y_off, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.CHUNK_SIZE = 128
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_diag as in the original, but all computations in Triton.
        Inputs:
          hidden_states: [B, N, S, H, D]
          A_cumsum:      [B, H, N, S]
          B:             [B, N, S, G, D] (already expanded to H on host)
          C:             [B, N, S, G, D] (already expanded to H on host)
        Output:
          Y_diag: [B, N, S, H] (cast to bfloat16)
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All inputs must be CUDA tensors for Triton kernels."

        B_size, N_size, S_size, H_size, D_size = hidden_states.shape

        # Ensure float32 and contiguous for kernels
        A = A_cumsum.to(torch.float32).contiguous()  # [B, H, N, S]
        B_exp = B.to(torch.float32).contiguous()     # [B, N, S, H, D]
        C_exp = C.to(torch.float32).contiguous()     # [B, N, S, H, D]
        hidden = hidden_states.to(torch.float32).contiguous()  # [B, N, S, H, D]

        # 1) Create lower-triangular mask (diagonal=-1) in Triton and write to device memory
        mask = torch.empty(S_size * S_size, dtype=torch.int8, device=hidden.device)
        tril_mask_kernel[(S_size * S_size,)](
            mask,
            S_size,
            num_warps=1, num_stages=1
        )

        # 2) Compute L in Triton
        L = torch.empty((B_size, H_size, N_size, S_size, S_size), dtype=torch.float32, device=hidden.device)
        a_segment_sum_exp_kernel[(B_size, H_size, N_size)](
            A, mask, L,
            B_size, H_size, N_size, S_size,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=1, num_stages=1
        )

        # 3) Compute G in Triton
        G = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=hidden.device)
        g_contract_kernel[(B_size, N_size, S_size, S_size, H_size)](
            B_exp, C_exp, G,
            B_size, N_size, S_size, H_size, D_size,
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            BLOCK_D=32,
            num_warps=4, num_stages=2
        )

        # 4) Multiply M = G * L (elementwise), permute L to [B, N, S, S, H]
        M = torch.empty_like(G)  # [B, N, S, S, H]
        # We need L in [B, N, S, S, H] for multiply; permute in Triton by re-ordering strides via tensor permute,
        # but Triton kernels operate on raw strides. We can compute L_perm by passing L with strides (b,h,n,i,j) and
        # treat M_ptr as G. Actually, we will compute M directly: M[b,n,i,j,h] = G[b,n,i,j,h] * L[b,h,n,i,j].
        # Here, since L is [B,H,N,S,S], we index it with strides (b,h,n,i,j) to get L[b,h,n,i,j]. We pass L's strides to kernel.
        # In Python, to use L as [B,N,S,S,H], we permute by creating new strides mapping. However, Triton expects raw pointers,
        # and we cannot pass permuted strides directly. So we compute M using original L strides by indexing L as (b,h,n,i,j)
        # and write into M at (b,n,i,j,h). Below we implement pointer arithmetic accordingly in kernel.
        # Note: For elementwise multiply, we keep G and L original layouts and write M in G's layout [B,N,S,S,H], using L strides.
        # The kernel below already accepts L as [B,H,N,S,S]; we will index L with (b,h,n,i,j) and M with (b,n,i,j,h).
        m_mul_kernel[(B_size, N_size, S_size, S_size, H_size)](
            G, L, M,
            B_size, N_size, S_size, H_size,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),      # G strides: (b,n,i,j,h)
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),      # L strides: (b,h,n,i,j)
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),      # M strides: (b,n,i,j,h)
            num_warps=1, num_stages=1
        )

        # 5) Reduce to Y_diag in Triton
        Y = torch.empty((B_size, N_size, S_size, H_size), dtype=torch.float32, device=hidden.device)
        y_diag_reduce_kernel[(B_size, N_size, S_size, H_size)](
            M, hidden, Y,
            B_size, N_size, S_size, H_size, D_size,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),         # M strides: (b,n,i,j,h)
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            BLOCK_D=32,
            num_warps=4, num_stages=2
        )

        # Return in bfloat16, matching the original code's output dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
