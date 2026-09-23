import torch
import triton
import triton.language as tl


@triton.jit
def make_lower_mask(mask_out_ptr, S: tl.constexpr):
    # Build a lower-triangular mask (diagonal=-1) of shape [S, S].
    # mask[i, j] = True if j <= i - 1 (exclude diagonal), else False.
    # We launch a 1D grid over S*S elements and compute 2D indices.
    total = S * S
    idx = tl.program_id(0)
    if idx >= total:
        return
    i = idx // S
    j = idx % S
    include = j <= (i - 1)  # diagonal=-1, exclude j == i
    # Store as 0/1 int
    val = tl.where(include, 1, 0)
    tl.store(mask_out_ptr + idx, val)


@triton.jit
def masked_cumsum_lower(A_ptr, A_seg_ptr,
                        B_size, Csz, S, N,
                        A_stride_b, A_stride_c, A_stride_i, A_stride_n,
                        Aseg_stride_b, Aseg_stride_c, Aseg_stride_i, Aseg_stride_j, Aseg_stride_n):
    # Grid: (B, C, N, S) where S is looped over inside
    b = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2)
    i_block = tl.program_id(3)  # unused for now, but included in case we extend later

    # We loop over j (source index) from 0 to S-1, compute cumsum along i, and write to A_seg[b, c, i, j, n].
    # Load mask vector for this (n). Make_lower_mask produces mask of shape [S, S]; we can reuse it by loading mask[j, i].
    # However, since mask depends on (i,j), we will recompute include condition in kernel using indices.
    # To avoid ambiguity, we assume we already have mask as a tensor; for simplicity, we recompute include via indices:
    # include[i,j] = (j <= i - 1). We'll implement mask by checking (j <= i - 1) inside kernel.
    for j in range(S):
        # Compute running sum only for positions where include is True (j < i)
        # But since j is fixed, we need to update cumsum for all i; we'll handle mask per i via if.
        # Alternative approach: load A[b,c,:,n] into a vector, apply mask, and compute cumsum. Triton doesn't support
        # vectorized dynamic loads like numpy, but we can emulate by loading scalar A[b,c,i,n] and updating a scalar cumsum.
        cum = 0.0
        for i in range(S):
            include = j <= (i - 1)  # diagonal=-1, exclude j == i
            a_val = tl.load(A_ptr + b * A_stride_b + c * A_stride_c + i * A_stride_i + n * A_stride_n)
            a_val = tl.where(include, a_val, 0.0)
            cum += a_val
            tl.store(A_seg_ptr + b * Aseg_stride_b + c * Aseg_stride_c + i * Aseg_stride_i + j * Aseg_stride_j + n * Aseg_stride_n, cum)


@triton.jit
def exp_lower_masked_cumsum(A_seg_ptr, L_ptr,
                            B_size, Csz, S, N,
                            Aseg_stride_b, Aseg_stride_c, Aseg_stride_i, Aseg_stride_j, Aseg_stride_n,
                            L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n):
    # Compute L = exp(A_seg) with lower-tri mask (diagonal=-1) already applied in A_seg
    # Grid: (B, C, N, S, S)
    b = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2)
    i = tl.program_id(3)
    j = tl.program_id(4)

    # A_seg[b, c, i, j, n] has mask include if j < i (diagonal=-1). Since A_seg already applied mask,
    # all values where j >= i are zero. We can simply load and exp.
    val = tl.load(A_seg_ptr + b * Aseg_stride_b + c * Aseg_stride_c + i * Aseg_stride_i + j * Aseg_stride_j + n * Aseg_stride_n)
    # Apply exp
    val = tl.exp(val)
    tl.store(L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n, val)


@triton.jit
def contract_BC_to_G(B_expanded_ptr, C_expanded_ptr, G_ptr,
                     B_size, C_size, S, N, K,
                     B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
                     C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
                     G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n):
    # Compute G[b, c, i, j, n] = sum_k C[b, c, i, n, k] * B[b, c, j, n, k]
    # Grid: (B, C, S, S, N)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    acc = 0.0
    for k in range(K):
        b_elem = tl.load(B_expanded_ptr + b * B_stride_b + c * B_stride_c + j * B_stride_j + n * B_stride_n + k * B_stride_k)
        c_elem = tl.load(C_expanded_ptr + b * C_stride_b + c * C_stride_c + i * C_stride_i + n * C_stride_n + k * C_stride_k)
        acc += b_elem * c_elem
    tl.store(G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n, acc)


@triton.jit
def G_mul_L(G_ptr, L_ptr, M_ptr,
            B_size, C_size, S, N,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
            M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n):
    # Elementwise M = G * L
    # Grid: (B, C, S, S, N)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    g_val = tl.load(G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n)
    l_val = tl.load(L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n)
    m_val = g_val * l_val
    tl.store(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n, m_val)


@triton.jit
def diag_contract_Y(M_ptr, hidden_states_ptr, Y_ptr,
                    B_size, C_size, S, N, D,
                    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
                    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
                    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d):
    # Compute Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
    # Grid: (B, C, S, N, D)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in range(S):
        m_val = tl.load(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n)
        hs_val = tl.load(hidden_states_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d)
        acc += m_val * hs_val
    tl.store(Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, num_chunks: int, chunk_size: int, num_heads: int = 32, n_groups: int = 8, state_size: int = 128):
        super().__init__()
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.n_groups = n_groups
        self.state_size = state_size  # K

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, C, S, N, D]
        A_cumsum:      [B, C, S, N]  (from original code)
        B:             [B, C, S, N_GROUPS, K]
        C:             [B, C, S, N_GROUPS, K]
        Output:        [B, C, S, N, D], bfloat16
        """
        # Ensure inputs are on GPU and contiguous
        device = hidden_states.device
        Bsz, Csz, S, N, D = hidden_states.shape
        # Original signature uses NUM_HEADS=32, N_GROUPS=8
        assert N == 32, "num_heads must be 32"
        assert self.n_groups == 8, "n_groups must be 8"
        # Expand B and C from N_GROUPS=8 to NUM_HEADS=32
        B_expanded = B.repeat_interleave(4, dim=3).contiguous()  # [B, C, S, N, K]
        C_expanded = C.repeat_interleave(4, dim=3).contiguous()  # [B, C, S, N, K]

        # 1) Build lower-tri mask [S, S] with diagonal=-1
        mask = torch.empty((S, S), dtype=torch.int32, device=device)
        mask_kernel = make_lower_mask
        grid_mask = (S * S,)
        mask_kernel[grid_mask](mask)

        # 2) masked cumsum of A_cumsum along source dimension (i) with lower-tri mask (exclude diagonal), result A_seg: [B, C, S, S, N], float32
        A_seg = torch.empty((Bsz, Csz, S, S, N), dtype=torch.float32, device=device)
        grid1 = (Bsz, Csz, N, 1)  # we loop over S inside kernel
        masked_cumsum_lower[grid1](A_cumsum, A_seg,
                                   Bsz, Csz, S, N,
                                   A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
                                   A_seg.stride(0), A_seg.stride(1), A_seg.stride(2), A_seg.stride(3), A_seg.stride(4))

        # 3) exp of masked cumsum to form L: [B, C, S, S, N], float32
        L = torch.empty((Bsz, Csz, S, S, N), dtype=torch.float32, device=device)
        grid2 = (Bsz, Csz, S, S, N)
        exp_lower_masked_cumsum[grid2](A_seg, L,
                                       Bsz, Csz, S, N,
                                       A_seg.stride(0), A_seg.stride(1), A_seg.stride(2), A_seg.stride(3), A_seg.stride(4),
                                       L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4))

        # 4) Contract B_expanded and C_expanded to G: [B, C, S, S, N], float32
        G = torch.empty((Bsz, Csz, S, S, N), dtype=torch.float32, device=device)
        grid3 = (Bsz, Csz, S, S, N)
        # Strides for B_expanded and C_expanded: both [B, C, S, N, K]
        B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k = B_expanded.stride()
        C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k = C_expanded.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()
        contract_BC_to_G[grid3](
            B_expanded, C_expanded, G,
            Bsz, Csz, S, N, self.state_size,
            B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
            C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            num_warps=1, num_stages=1
        )

        # 5) Elementwise M = G * L
        M = torch.empty((Bsz, Csz, S, S, N), dtype=torch.float32, device=device)
        grid4 = (Bsz, Csz, S, S, N)
        G_mul_L[grid4](
            G, L, M,
            Bsz, Csz, S, N,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=1, num_stages=1
        )

        # 6) Diagonal contraction: Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
        Y = torch.empty((Bsz, Csz, S, N, D), dtype=torch.float32, device=device)
        grid_Y = (Bsz, Csz, S, N, D)
        # Strides for hidden_states: [B, C, S, N, D]
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = hidden_states.stride()
        # Y strides
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()
        diag_contract_Y[grid_Y](
            M, hidden_states, Y,
            Bsz, Csz, S, N, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
            Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
