import torch
import triton
import triton.language as tl


# 1) Triton: generate lower-triangular mask [S, S] with diagonal=-1 as 1D int8 array
# mask[i*S + j] = 1 if j <= i else 0
@triton.jit
def tril_mask_kernel(mask: tl.pointer_type(tl.int8), S: tl.constexpr):
    pid = tl.program_id(axis=0)
    total = S * S
    if pid < total:
        i = pid // S
        j = pid % S
        if j <= i:
            tl.store(mask + pid, 1)
        else:
            tl.store(mask + pid, 0)


# 2) Triton: compute L = exp(cumsum(masked A)) along source chunk dimension S for each (b,h,n)
# A is [B, H, N, S], expanded to [B, H, N, S, S]. We loop i (rows) and j (cols) to compute prefix sum for j<=i then exp.
@triton.jit
def masked_cumsum_exp_kernel(A_expanded: tl.pointer_type(tl.float32),
                             L: tl.pointer_type(tl.float32),
                             Bsz: tl.constexpr, Hsz: tl.constexpr, Nsz: tl.constexpr, S: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)

    # For each i (row), compute prefix sum along j (col) up to i, then exp and store.
    for i in range(0, S):
        prefix = 0.0
        for j in range(0, S):
            a_val = tl.load(A_expanded + b * (Hsz * Nsz * S * S) + h * (Nsz * S * S) + n * (S * S) + i * S + j)
            prefix += a_val
            l_val = tl.exp(prefix)
            tl.store(L + b * (Hsz * Nsz * S * S) + h * (Nsz * S * S) + n * (S * S) + i * S + j, l_val)


# 3) Triton: compute G[i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d] with tiling over i and j.
@triton.jit
def g_contract_kernel(B: tl.pointer_type(tl.float32),
                      C: tl.pointer_type(tl.float32),
                      G: tl.pointer_type(tl.float32),
                      Bsz: tl.constexpr, Nsz: tl.constexpr, S: tl.constexpr, Hsz: tl.constexpr, D: tl.constexpr,
                      BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BK: tl.constexpr):
    # grid over (b, n, h) + tiles over i and j
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    tile_i = tl.program_id(3)
    tile_j = tl.program_id(4)

    i_start = tile_i * BLOCK_I
    j_start = tile_j * BLOCK_J

    i_offsets = i_start + tl.arange(0, BLOCK_I)
    j_offsets = j_start + tl.arange(0, BLOCK_J)

    # accumulator for G[i,j,h]
    G_acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)

    # loop over D dimension in blocks
    for d_start in range(0, D, BK):
        d_offsets = d_start + tl.arange(0, BK)

        # For each (i,j) in tile, accumulate over D
        for ii in range(BLOCK_I):
            i_idx = i_start + ii
            valid_i = i_idx < S
            per_j = tl.zeros((BLOCK_J,), dtype=tl.float32)
            for jj in range(BLOCK_J):
                j_idx = j_start + jj
                valid_j = j_idx < S
                acc_ij = tl.zeros((), dtype=tl.float32)
                for dd in range(0, BK):
                    d_idx = d_start + dd
                    valid_d = d_idx < D
                    # B[b,n,j,h,d]
                    b_ptr = B + b * (Nsz * S * Hsz * D) + n * (S * Hsz * D) + j_idx * (Hsz * D) + h * D + d_idx
                    # C[b,n,i,h,d]
                    c_ptr = C + b * (Nsz * S * Hsz * D) + n * (S * Hsz * D) + i_idx * (Hsz * D) + h * D + d_idx
                    # load with validity checks
                    b_val = tl.load(b_ptr, mask=valid_i & valid_j & valid_d, other=0.0)
                    c_val = tl.load(c_ptr, mask=valid_i & valid_j & valid_d, other=0.0)
                    acc_ij += c_val * b_val
                per_j[jj] = acc_ij
            G_acc[ii, :] = per_j

    # Store G_acc to output
    G_base = G + b * (Nsz * S * S * Hsz) + n * (S * S * Hsz) + h * (S * Hsz)
    # write tiles
    for ii in range(BLOCK_I):
        i_idx = i_start + ii
        if i_idx < S:
            for jj in range(BLOCK_J):
                j_idx = j_start + jj
                if j_idx < S:
                    tl.store(G_base + i_idx * (S * Hsz) + j_idx * Hsz, G_acc[ii, jj])


# 4) Triton: elementwise M = G * L_expanded (L is [B,H,N,S,S], permute to [B,N,S,S,H] before calling)
@triton.jit
def m_mul_kernel(G: tl.pointer_type(tl.float32),
                 L_perm: tl.pointer_type(tl.float32),
                 M: tl.pointer_type(tl.float32),
                 Bsz: tl.constexpr, Nsz: tl.constexpr, S: tl.constexpr, Hsz: tl.constexpr):
    # grid over (b,n,i,j,h)
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    g_ptr = G + b * (Nsz * S * S * Hsz) + n * (S * S * Hsz) + i * (S * Hsz) + j * Hsz
    l_ptr = L_perm + b * (Nsz * S * S * Hsz) + n * (S * S * Hsz) + i * (S * Hsz) + j * Hsz
    m_ptr = M + b * (Nsz * S * S * Hsz) + n * (S * S * Hsz) + i * (S * Hsz) + j * Hsz

    g_val = tl.load(g_ptr)
    l_val = tl.load(l_ptr)
    tl.store(m_ptr, g_val * l_val)


# 5) Triton: Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden[b,n,j,h,:]
@triton.jit
def y_diag_reduce_kernel(M: tl.pointer_type(tl.float32),
                         hidden: tl.pointer_type(tl.float32),
                         Y: tl.pointer_type(tl.float32),
                         Bsz: tl.constexpr, Nsz: tl.constexpr, S: tl.constexpr, Hsz: tl.constexpr, D: tl.constexpr):
    # grid over (b,n,i,h)
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)
    for j in range(0, S):
        m_ptr = M + b * (Nsz * S * S * Hsz) + n * (S * S * Hsz) + i * (S * Hsz) + j * Hsz
        m_val = tl.load(m_ptr)

        hidden_base = hidden + b * (Nsz * S * Hsz * D) + n * (S * Hsz * D) + j * (Hsz * D) + h * D
        dot = 0.0
        for d in range(0, D):
            h_ptr = hidden_base + d
            h_val = tl.load(h_ptr)
            dot += m_val * h_val
        acc += dot

    Y_ptr = Y + b * (Nsz * S * Hsz) + n * (S * Hsz) + i * Hsz + h
    tl.store(Y_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, N, S, H, D]
        A_cumsum: [B, H, N, S]
        B, C: [B, N, S, H, D] (assumed expanded from G to H)
        Returns Y: [B, N, S, H] in bfloat16.
        """
        # Extract shapes
        Bsz, Nsz, S, Hsz, D = hidden_states.shape
        assert A_cumsum.shape == (Bsz, Hsz, Nsz, S), "A_cumsum must be [B, H, N, S]"

        # Create expanded A for cumsum along source dimension
        device = hidden_states.device
        A_expanded = A_cumsum.unsqueeze(-1).expand(Bsz, Hsz, Nsz, S, S).to(torch.float32).contiguous()

        # 1) Triton: generate lower-triangular mask [S,S] as 1D int8
        mask = torch.empty(S * S, dtype=torch.int8, device=device)
        tril_mask_kernel[(S * S,)](mask, S)

        # 2) Triton: compute L = exp(cumsum(masked A)) along S for each (b,h,n)
        L = torch.empty((Bsz, Hsz, Nsz, S, S), dtype=torch.float32, device=device)
        masked_cumsum_exp_kernel[(Bsz, Hsz, Nsz)](A_expanded, L, Bsz, Hsz, Nsz, S)

        # 3) Triton: compute G[i,j,h] contraction
        G = torch.empty((Bsz, Nsz, S, S, Hsz), dtype=torch.float32, device=device)
        # Launch: grid over (b,n,h) with tiles over i,j
        BLOCK_I = 64
        BLOCK_J = 64
        grid_g = (Bsz, Nsz, Hsz, triton.cdiv(S, BLOCK_I), triton.cdiv(S, BLOCK_J))
        g_contract_kernel[grid_g](
            B.contiguous(), C.contiguous(), G,
            Bsz, Nsz, S, Hsz, D,
            BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, BK=32,
            num_warps=4, num_stages=2
        )

        # 4) Triton: M = G * L (permute L to [B,N,S,S,H])
        L_perm = L.permute(0, 2, 3, 4, 1).contiguous()  # [B, N, S, S, H]
        M = torch.empty_like(G, dtype=torch.float32, device=device)
        m_mul_kernel[(Bsz, Nsz, S, S, Hsz)](G, L_perm, M, Bsz, Nsz, S, Hsz)

        # 5) Triton: Y_diag reduction: Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden[b,n,j,h,:]
        hidden = hidden_states.contiguous()  # [B, N, S, H, D]
        Y = torch.empty((Bsz, Nsz, S, Hsz), dtype=torch.float32, device=device)
        y_diag_reduce_kernel[(Bsz, Nsz, S, Hsz)](
            M, hidden, Y,
            Bsz, Nsz, S, Hsz, D,
            num_warps=1, num_stages=1
        )

        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
