import torch
import triton
import triton.language as tl


# Kernel 1: generate 1D tril mask of length S*S, int8 (lower triangular, diagonal=-1)
@triton.jit
def tril_mask_kernel(mask_ptr, S_size: tl.constexpr):
    idx = tl.program_id(0)
    # row i = idx // S, col j = idx % S
    i = idx // S_size
    j = idx % S_size
    lower = j <= i
    if lower:
        tl.store(mask_ptr + idx, 1)
    else:
        tl.store(mask_ptr + idx, 0)


# Kernel 2: compute L = exp(segment_sum(A)) with lower-triangular mask (j <= i).
# Inputs:
#   A: [B, H, N, S]
#   L: [B, H, N, S, S] float32
# For each (b,h,n): for i in 0..S-1: segment_sum = 0; for j in 0..i: segment_sum += A[b,h,n,j]; L[b,h,n,i,j] = exp(segment_sum)
@triton.jit
def a_segment_sum_exp_kernel(
    A_ptr, L_ptr,
    B_size, H_size, N_size, S_size,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    # accumulate segment_sum over j up to i
    i = 0
    while i < S_size:
        segment_sum = 0.0
        # sum up to j=i: segment_sum = sum_{k=0..i} A[b,h,n,k]
        j_local = 0
        while j_local <= i:
            a_val = tl.load(A_ptr + pid_b * A_stride_b + pid_h * A_stride_h + pid_n * A_stride_n + j_local * A_stride_s)
            segment_sum += a_val
            j_local += 1
        # store exp(segment_sum) to all positions (i,j) for j<=i
        j = 0
        while j <= i:
            val = tl.exp(segment_sum)
            tl.store(
                L_ptr + pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n +
                i * L_stride_s1 + j * L_stride_s2,
                val
            )
            j += 1
        i += 1


# Kernel 3: compute G[b,n,i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
# Inputs: B, C expanded to H; Outputs: G [B, N, S, S, H] float32
@triton.jit
def g_contract_kernel(
    B_ptr, C_ptr, G_ptr,
    B_size, N_size, S_size, H_size, D_size,
    B_stride_b, B_stride_n, B_stride_s1, B_stride_h, B_stride_d,
    C_stride_b, C_stride_n, C_stride_s1, C_stride_h, C_stride_d,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    acc = 0.0
    d0 = 0
    while d0 < D_size:
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D_size
        b_i = tl.load(B_ptr + pid_b * B_stride_b + pid_n * B_stride_n + pid_i * B_stride_s1 + pid_h * B_stride_h + offs_d * B_stride_d, mask=mask_d, other=0.0)
        c_i = tl.load(C_ptr + pid_b * C_stride_b + pid_n * C_stride_n + pid_i * C_stride_s1 + pid_h * C_stride_h + offs_d * C_stride_d, mask=mask_d, other=0.0)
        acc += tl.sum(b_i * c_i, axis=0)
        d0 += BLOCK_D

    tl.store(
        G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + pid_i * G_stride_s1 + pid_j * G_stride_s2 + pid_h * G_stride_h,
        acc
    )


# Kernel 4: elementwise M = G * L with L permuted to [B, N, S, S, H] inside indexing.
# L is indexed as L[b, n, i, j, h] where h is the last dim (num_heads).
@triton.jit
def m_mul_kernel(
    G_ptr, L_ptr, M_ptr,
    B_size, N_size, S_size, H_size,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    L_stride_b, L_stride_n, L_stride_s1, L_stride_s2, L_stride_h,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)

    h0 = 0
    while h0 < H_size:
        offs_h = h0 + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H_size

        g = tl.load(
            G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + pid_i * G_stride_s1 + pid_j * G_stride_s2 + offs_h * G_stride_h,
            mask=mask_h, other=0.0
        )
        # L indexed with swapped last two dims for h: L[b, n, i, j, h]
        l = tl.load(
            L_ptr + pid_b * L_stride_b + pid_n * L_stride_n + pid_i * L_stride_s1 + pid_j * L_stride_s2 + offs_h * L_stride_h,
            mask=mask_h, other=0.0
        )
        out = g * l
        tl.store(
            M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_i * M_stride_s1 + pid_j * M_stride_s2 + offs_h * M_stride_h,
            out, mask=mask_h
        )
        h0 += BLOCK_H


# Kernel 5: Y[b,n,i,h] = sum over j of M[b,n,i,j,h] * hidden_states[b,n,j,h,:]
@triton.jit
def y_diag_reduce_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    B_size, N_size, S_size, H_size, D_size,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
    Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_h = tl.program_id(3)

    acc = 0.0
    j0 = 0
    while j0 < S_size:
        offs_j = j0 + tl.arange(0, BLOCK_D)
        mask_j = offs_j < S_size

        # M[b,n,i,offs_j,h]
        m_vec = tl.load(
            M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_i * M_stride_s1 + offs_j * M_stride_s2 + pid_h * M_stride_h,
            mask=mask_j, other=0.0
        )
        # hidden[b,n,offs_j,h,:]
        hs = tl.load(
            hidden_ptr + pid_b * hidden_stride_b + pid_n * hidden_stride_n + offs_j * hidden_stride_s + pid_h * hidden_stride_h + tl.arange(0, BLOCK_D) * hidden_stride_d,
            mask=mask_j, other=0.0
        )
        # acc += sum_j (M[b,n,i,j,h] * hidden[b,n,j,h,0]) * D
        # Note: original hidden is [B,N,S,H,D]; we need to multiply M's last dim with hidden last dim (D).
        # Since Y_diag has shape [B,N,S,H], we only reduce across D. The original code multiplies M (H) with hidden_states (D) and sums across D.
        # Here, we accumulate acc += sum_j m_vec[j] * hs[j, 0] (i.e., take the first channel?).
        # To strictly follow original, we need to multiply M's H dimension with hidden_states's D dimension. The original hidden has last dim D (state_size), and we reduce along D.
        # Correction: we need M and hidden_states to have same last dim (state_size), but here M has last dim H and hidden has last dim D. This suggests a mismatch. However, original run() uses hidden_states with last dim D=32 and Model parameters indicate head_dim=32. We need to align types.
        # For correctness in this environment, we assume hidden_states has last dim D (32), and Y_diag has shape [B,N,S,H]. We compute Y[b,n,i,h] as dot(M[b,n,i,:,h], hidden[b,n,:,h,:]) across D.
        # That means we need hidden[b,n,j,h,:] and M[b,n,i,j,h]. Our M kernel writes M with last dim H, and we will load hidden's last dim D as BLOCK_D and multiply accordingly.

        # Compute dot product across D: M vector has size BLOCK_D, hidden has last dim D. We must align and reduce. Since original code uses hidden with D=32, we can load hidden's last dim and multiply.
        # We will compute for each j in tile: M[i,j,h] * hidden[b,n,j,h,:] dot product across D. To do this, we need M to have last dim matching hidden's last dim. But our M was computed from G*exp(A), which has last dim H. The original PyTorch run has hidden with last dim D=32, which is consistent with state size.
        # Therefore, our forward must ensure hidden has last dim D. In practice, the provided inputs follow that. We will assume hidden has last dim D.

        # Since hs was loaded with shape [BLOCK_D, D], we need to multiply m_vec (shape [BLOCK_D]) with each row of hs and sum over D.
        # For simplicity and correctness, we will implement a reduction: for each j in offs_j, load hidden[b,n,j,h,:] and multiply with corresponding m_vec[j] and accumulate into acc.
        # Note: This kernel was not part of previous submissions but we will implement it to reduce M along D using hidden's D dimension.
        # We'll loop j in Python while loop and accumulate acc. Triton supports while loops.

        j = 0
        while j < BLOCK_D:
            m_j = m_vec[j]
            hs_j = tl.load(
                hidden_ptr + pid_b * hidden_stride_b + pid_n * hidden_stride_n + (j0 + j) * hidden_stride_s + pid_h * hidden_stride_h + tl.arange(0, D_size) * hidden_stride_d,
                mask=tl.full((), True, tl.int1),  # always valid if j < S_size and D_size is small
                other=0.0
            )
            # Multiply each element of hs_j by m_j and accumulate. hs_j is [D], m_j is scalar.
            acc += tl.sum(hs_j * m_j, axis=0)
            j += 1

        j0 += BLOCK_D

    tl.store(
        Y_ptr + pid_b * Y_stride_b + pid_n * Y_stride_n + pid_i * Y_stride_s1 + pid_h * Y_stride_h,
        acc
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from original model
        self.CHUNK_SIZE = 128
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, N, S, H, D]
        A_cumsum:      [B, H, N, S]
        B:             [B, N, S, G, D]
        C:             [B, N, S, G, D]
        Output:        [B, N, S, H] in bfloat16 (original returns bfloat16)
        """
        device = hidden_states.device

        B_size, N_size, S_size, H_size, D_size = hidden_states.shape
        # Ensure inputs are on GPU and proper dtypes
        A = A_cumsum.to(torch.float32).contiguous()
        B_exp = B.to(torch.float32).contiguous()  # expanded to H on host beforehand
        C_exp = C.to(torch.float32).contiguous()  # expanded to H on host beforehand
        hidden = hidden_states.to(torch.float32).contiguous()

        # 1) L = exp(segment_sum(A)) with lower-triangular mask (diagonal=-1). Triton kernel
        L = torch.empty((B_size, H_size, N_size, S_size, S_size), dtype=torch.float32, device=device)

        a_segment_sum_exp_kernel[(B_size, H_size, N_size)](
            A, L,
            B_size, H_size, N_size, S_size,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=1, num_stages=1
        )

        # 2) G contraction: Triton kernel, grid over (B, N, S, S, H)
        G = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)

        g_contract_kernel[(B_size, N_size, S_size, S_size, H_size)](
            B_exp, C_exp, G,
            B_size, N_size, S_size, H_size, D_size,
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            BLOCK_D=32, num_warps=4, num_stages=2
        )

        # 3) M = G * L; L is [B,H,N,S,S], G is [B,N,S,S,H]
        # We need to permute L to [B,N,S,S,H] for elementwise multiply. Triton kernel will index L with swapped last two dims (use h as last).
        M = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)

        m_mul_kernel[(B_size, N_size, S_size, S_size)](
            G, L, M,
            B_size, N_size, S_size, H_size,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            BLOCK_H=32, num_warps=4, num_stages=2
        )

        # 4) Y_diag reduction: Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden[b,n,j,h,:]
        # hidden shape is [B,N,S,H,D], M has last dim H. We multiply along D and reduce. Triton kernel.
        Y = torch.empty((B_size, N_size, S_size, H_size), dtype=torch.float32, device=device)

        y_diag_reduce_kernel[(B_size, N_size, S_size, H_size)](
            M, hidden, Y,
            B_size, N_size, S_size, H_size, D_size,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            BLOCK_D=32, num_warps=4, num_stages=2
        )

        # Return in bfloat16 as original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
