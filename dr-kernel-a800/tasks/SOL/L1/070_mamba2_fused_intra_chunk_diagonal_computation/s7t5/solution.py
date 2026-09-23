import torch
import triton
import triton.language as tl


# Triton Kernel 1: Compute masked cumsum along source dimension to produce segment_sum_A with lower-triangular mask (diagonal=-1).
# A: [B, N, C, S], where S = hidden_states.shape[2]
# Output A_masked: [B, N, C, S, S] (we store 2D matrices per (b,n,c))
@triton.jit
def segment_cumsum_lower_masked(A_ptr, Out_ptr,
                                 Bsz: tl.constexpr, N: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr,
                                 A_stride_b, A_stride_n, A_stride_c, A_stride_s,
                                 Out_stride_b, Out_stride_n, Out_stride_c, Out_stride_i, Out_stride_j,
                                 num_warps=1, num_stages=1):
    # One program per (b, n, c)
    b = tl.program_id(0)
    n = tl.program_id(1)
    c = tl.program_id(2)

    # Initialize running sum for each i (row index)
    # We compute cumsum for each i (exclude diagonal j==i per mask). Lower-triangular with diagonal=-1 means j < i.
    for i in tl.static_range(S):
        running = 0.0
        # Iterate over source j and apply mask (j < i)
        for j in tl.static_range(S):
            if j < i:
                val = tl.load(A_ptr + b * A_stride_b + n * A_stride_n + c * A_stride_c + j * A_stride_s, mask=True, other=0.0)
                running += val
        # Store segment_sum for row i (all j < i included; diagonal excluded)
        # Out[b, n, c, i, j] = segment_sum up to j for row i
        for j in tl.static_range(S):
            if j < i:
                # store running at Out[b, n, c, i, j]
                tl.store(Out_ptr + b * Out_stride_b + n * Out_stride_n + c * Out_stride_c + i * Out_stride_i + j * Out_stride_j, running)


# Triton Kernel 2: Apply exp to segment_sum result to produce L (causal mask), also with diagonal=-1 mask.
# In: A_masked [B, N, C, S, S]
# Out: L_exp [B, N, C, S, S] in float32
@triton.jit
def exp_causal_mask(In_ptr, Out_ptr,
                    Bsz: tl.constexpr, N: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr,
                    In_stride_b, In_stride_n, In_stride_c, In_stride_i, In_stride_j,
                    Out_stride_b, Out_stride_c, Out_stride_i, Out_stride_j, Out_stride_n,
                    num_warps=1, num_stages=1):
    b = tl.program_id(0)
    n = tl.program_id(1)
    c = tl.program_id(2)
    # Only lower-triangular (j < i), diagonal=-1
    for i in tl.static_range(S):
        for j in tl.static_range(S):
            if j < i:
                val = tl.load(In_ptr + b * In_stride_b + n * In_stride_n + c * In_stride_c + i * In_stride_i + j * In_stride_j)
                # exclude diagonal: not used; keep running unchanged when j >= i
                tl.store(Out_ptr + b * Out_stride_b + n * Out_stride_n + c * Out_stride_c + i * Out_stride_i + j * Out_stride_j,
                         tl.exp(val))
            else:
                # j >= i: masked, store 0 (or leave as 0). We ensure multiplication by M will zero out upper part anyway.
                tl.store(Out_ptr + b * Out_stride_b + n * Out_stride_n + c * Out_stride_c + i * Out_stride_i + j * Out_stride_j, 0.0)


# Triton Kernel 3: Contraction BC -> G: G[b, c, i, j, n] = sum_k C[b, c, i, n, k] * B[b, c, j, n, k]
# B_expanded: [B, C, S, N, K], C_expanded: [B, C, S, N, K]
# Output G: [B, C, S, S, N] in float32
@triton.jit
def contract_BC_to_G(B_ptr, C_ptr, G_ptr,
                     Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     B_stride_b, B_stride_c, B_stride_i, B_stride_n, B_stride_k,
                     C_stride_b, C_stride_c, C_stride_j, C_stride_n, C_stride_k,
                     G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
                     num_warps=1, num_stages=1):
    b = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(4)  # n dimension is the last grid axis
    # Loop over i and j (compile-time static to avoid runtime loops)
    for i in tl.static_range(S):
        for j in tl.static_range(S):
            acc = 0.0
            for k in tl.static_range(K):
                b_i = tl.load(B_ptr + b * B_stride_b + c * B_stride_c + i * B_stride_i + n * B_stride_n + k * B_stride_k)
                c_j = tl.load(C_ptr + b * C_stride_b + c * C_stride_c + j * C_stride_j + n * C_stride_n + k * C_stride_k)
                acc += b_i * c_j
            tl.store(G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n, acc)


# Triton Kernel 4: Diagonal contraction to compute Y_diag:
# Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
# M: [B, C, S, S, N] (float32), hidden_states: [B, C, S, N, D] (float32 in compute), output Y: [B, C, S, N, D] (bf16)
@triton.jit
def diag_contract_Y(M_ptr, HS_ptr, Y_ptr,
                    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
                    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
                    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
                    num_warps=1, num_stages=1):
    b = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(3)  # n is the 4th grid axis
    # Accumulator across d
    for d in tl.static_range(D):
        acc = tl.zeros((), dtype=tl.float32)
        for j in tl.static_range(S):
            m_val = tl.load(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n)
            hs_val = tl.load(HS_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d)
            acc += m_val * hs_val
        tl.store(Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-strict implementation mirroring the original run function.
        Computes: Y_diag = sum_j (M[b,c,i,j,n] * hidden_states[b,c,j,n,d]) with M = G * L, where:
        - L is causal mask derived from A_cumsum via lower-triangular masked cumsum (diagonal=-1) and exp.
        - G is contraction of expanded B and C over state_size.
        Output shape: [batch, num_chunks, chunk_size, num_heads, head_dim], dtype bfloat16.
        """
        device = hidden_states.device
        Bsz, Csz, S, N, D = hidden_states.shape  # batch, num_chunks, chunk_size, num_heads, head_dim

        # Ensure inputs are contiguous and in float32 for computation
        A = A_cumsum.contiguous().to(torch.float32)  # [B, N, C, S]
        Bf32 = B.contiguous().to(torch.float32)      # [B, C, S, N, K]
        Cf32 = C.contiguous().to(torch.float32)      # [B, C, S, N, K]

        # 1) Compute masked cumsum along source dimension to produce segment_sum_A (diagonal=-1)
        segment_sum_A = torch.empty((Bsz, N, Csz, S, S), device=device, dtype=torch.float32)
        A_stride_b, A_stride_n, A_stride_c, A_stride_s = A.stride()
        Out_stride_b, Out_stride_n, Out_stride_c, Out_stride_i, Out_stride_j = segment_sum_A.stride()
        grid = (Bsz, N, Csz)
        segment_cumsum_lower_masked[grid](
            A, segment_sum_A,
            Bsz=Bsz, N=N, Csz=Csz, S=S,
            A_stride_b=A_stride_b, A_stride_n=A_stride_n, A_stride_c=A_stride_c, A_stride_s=A_stride_s,
            Out_stride_b=Out_stride_b, Out_stride_n=Out_stride_n, Out_stride_c=Out_stride_c, Out_stride_i=Out_stride_i, Out_stride_j=Out_stride_j,
            num_warps=1, num_stages=1
        )

        # 2) Apply exp to segment_sum result to produce L (causal mask) with diagonal=-1
        L = torch.empty_like(segment_sum_A)  # [B, N, C, S, S] in float32
        In_stride_b, In_stride_n, In_stride_c, In_stride_i, In_stride_j = segment_sum_A.stride()
        Out_stride_b_L, Out_stride_c_L, Out_stride_i_L, Out_stride_j_L, Out_stride_n_L = L.stride()
        grid_L = (Bsz, N, Csz)
        exp_causal_mask[grid_L](
            segment_sum_A, L,
            Bsz=Bsz, N=N, Csz=Csz, S=S,
            In_stride_b=In_stride_b, In_stride_n=In_stride_n, In_stride_c=In_stride_c, In_stride_i=In_stride_i, In_stride_j=In_stride_j,
            Out_stride_b=Out_stride_b_L, Out_stride_c=Out_stride_c_L, Out_stride_i=Out_stride_i_L, Out_stride_j=Out_stride_j_L, Out_stride_n=Out_stride_n_L,
            num_warps=1, num_stages=1
        )

        # 3) Expand B and C from n_groups to num_heads (NUM_HEADS // N_GROUPS = 4)
        # Original NUM_HEADS=32, N_GROUPS=8, so we repeat_interleave by 4. We'll do it per tensor:
        B_expanded = Bf32.repeat_interleave(4, dim=3)  # [B, C, S, N, K]
        C_expanded = Cf32.repeat_interleave(4, dim=3)  # [B, C, S, N, K]

        # 4) Compute G = contraction of B_expanded and C_expanded over K
        G = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)
        B_stride_b, B_stride_c, B_stride_i, B_stride_n, B_stride_k = B_expanded.stride()
        C_stride_b, C_stride_c, C_stride_j, C_stride_n, C_stride_k = C_expanded.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()
        K = B_expanded.shape[4]  # state_size; must match B and C last dim
        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G[grid_G](
            B_expanded, C_expanded, G,
            Bsz=Bsz, Csz=Csz, S=S, N=N, K=K,
            B_stride_b=B_stride_b, B_stride_c=B_stride_c, B_stride_i=B_stride_i, B_stride_n=B_stride_n, B_stride_k=B_stride_k,
            C_stride_b=C_stride_b, C_stride_c=C_stride_c, C_stride_j=C_stride_j, C_stride_n=C_stride_n, C_stride_k=C_stride_k,
            G_stride_b=G_stride_b, G_stride_c=G_stride_c, G_stride_i=G_stride_i, G_stride_j=G_stride_j, G_stride_n=G_stride_n,
            num_warps=1, num_stages=1
        )

        # 5) Compute M = G * L (elementwise) using PyTorch (OK for this step)
        M = G * L  # [B, C, S, S, N], float32

        # 6) Diagonal contraction to compute Y_diag: Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
        Y = torch.empty((Bsz, Csz, S, N, D), device=device, dtype=torch.bfloat16)
        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = hidden_states.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()
        grid_Y = (Bsz, Csz, S, N)
        diag_contract_Y[grid_Y](
            M, hidden_states.to(torch.float32), Y,
            Bsz=Bsz, Csz=Csz, S=S, N=N, D=D,
            M_stride_b=M_stride_b, M_stride_c=M_stride_c, M_stride_i=M_stride_i, M_stride_j=M_stride_j, M_stride_n=M_stride_n,
            HS_stride_b=HS_stride_b, HS_stride_c=HS_stride_c, HS_stride_j=HS_stride_j, HS_stride_n=HS_stride_n, HS_stride_d=HS_stride_d,
            Y_stride_b=Y_stride_b, Y_stride_c=Y_stride_c, Y_stride_i=Y_stride_i, Y_stride_n=Y_stride_n, Y_stride_d=Y_stride_d,
            num_warps=1, num_stages=1
        )

        return Y


def run(*args):
    return ModelNew()(*args)
