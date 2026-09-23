import torch
import triton
import triton.language as tl


# =============================================================================
# L1/070 Mamba2 fused intra-chunk diagonal computation.
#
# Per (batch b, chunk c, head h) tile, with group g = h // (H // Gp):
#   C_h = C[b, c, :, g, :]              # [L, N]  indexed [i, n]
#   B_h = B[b, c, :, g, :]              # [L, N]  indexed [j, n]
#   X_h = hidden_states[b, c, :, h, :]  # [L, P]  indexed [j, d]
#   a   = A_cumsum[b, h, c, :]          # [L]
#
#   G[i,j]    = sum_n C_h[i,n] * B_h[j,n]          (= C_h @ B_h^T)
#   S[i]      = inclusive cumsum(a)[i]
#   Ldec[i,j] = exp(S[i] - S[j]) if i >= j else 0  (diagonal included)
#   M[i,j]    = G[i,j] * Ldec[i,j]
#   Y[i,d]    = sum_j M[i,j] * X_h[j,d]            (= M @ X_h)
#   store Y as bfloat16.
#
# c002: query-row split. Grid = (b*nc, H, num_m_blocks) with BLOCK_M rows of i
# per program -> more programs to fill the SMs on small b*nc workloads. A
# dynamic j-loop `range(m_block + 1)` genuinely skips the strict-upper causal
# triangle of j-blocks. Single fused kernel; no [L,L,*] HBM temporaries.
# bf16 inputs, fp32 tensor-core accumulation, bf16 store.
# =============================================================================


@triton.jit
def _mamba2_diag_kernel(
    X_ptr, A_ptr, B_ptr, C_ptr, Y_ptr,
    NC,
    sX_b, sX_c, sX_l, sX_h, sX_p,
    sA_b, sA_h, sA_c, sA_l,
    sB_b, sB_c, sB_l, sB_g, sB_n,
    sC_b, sC_c, sC_l, sC_g, sC_n,
    sY_b, sY_c, sY_l, sY_h, sY_p,
    HEADS_PER_GROUP: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    P: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    bc = tl.program_id(0)
    b = bc // NC
    c = bc % NC
    h = tl.program_id(1)
    m_block = tl.program_id(2)
    g = h // HEADS_PER_GROUP

    m0 = m_block * BLOCK_M
    offs_L = tl.arange(0, L)                       # [L]
    i_idx = m0 + tl.arange(0, BLOCK_M)             # [BLOCK_M]

    # --- full inclusive prefix sum S[L] (same numerics as c001) ---
    a_full = tl.load(A_ptr + b * sA_b + h * sA_h + c * sA_c + offs_L * sA_l).to(tl.float32)
    S_full = tl.cumsum(a_full, axis=0)             # [L]
    # S restricted to the i-block via one-hot selection.
    sel_i = offs_L[None, :] == i_idx[:, None]      # [BLOCK_M, L]
    S_i = tl.sum(tl.where(sel_i, S_full[None, :], 0.0), axis=1)  # [BLOCK_M]

    # --- C rows for this i-block: [BLOCK_M, N] ---
    C_bp = tl.make_block_ptr(
        base=C_ptr + b * sC_b + c * sC_c + g * sC_g,
        shape=(L, N), strides=(sC_l, sC_n),
        offsets=(m0, 0), block_shape=(BLOCK_M, N), order=(1, 0),
    )
    c_tile = tl.load(C_bp)                          # [BLOCK_M, N] bf16

    acc = tl.zeros((BLOCK_M, P), dtype=tl.float32)

    # --- causal j-loop: only j-blocks <= m_block contribute (i >= j) ---
    for jb in range(m_block + 1):
        j0 = jb * BLOCK_M
        j_idx = j0 + tl.arange(0, BLOCK_M)          # [BLOCK_M]

        B_bp = tl.make_block_ptr(
            base=B_ptr + b * sB_b + c * sB_c + g * sB_g,
            shape=(L, N), strides=(sB_l, sB_n),
            offsets=(j0, 0), block_shape=(BLOCK_M, N), order=(1, 0),
        )
        X_bp = tl.make_block_ptr(
            base=X_ptr + b * sX_b + c * sX_c + h * sX_h,
            shape=(L, P), strides=(sX_l, sX_p),
            offsets=(j0, 0), block_shape=(BLOCK_M, P), order=(1, 0),
        )
        b_tile = tl.load(B_bp)                      # [BLOCK_N=BLOCK_M, N] bf16
        x_tile = tl.load(X_bp)                      # [BLOCK_N, P] bf16

        # G_block[i,j] = C_rows @ B_block^T
        g_block = tl.dot(c_tile, tl.trans(b_tile), out_dtype=tl.float32)  # [BM, BN]

        sel_j = offs_L[None, :] == j_idx[:, None]   # [BLOCK_M, L]
        S_j = tl.sum(tl.where(sel_j, S_full[None, :], 0.0), axis=1)  # [BLOCK_M]

        diff = S_i[:, None] - S_j[None, :]          # [BM, BN]
        # Causal mask (correct for every block: for jb<m_block all i>j anyway).
        causal = i_idx[:, None] >= j_idx[None, :]
        diff = tl.where(causal, diff, float("-inf"))
        Ldec = tl.exp(diff)

        m_block_w = g_block * Ldec                  # [BM, BN] fp32
        acc += tl.dot(m_block_w.to(x_tile.dtype), x_tile, out_dtype=tl.float32)

    Y_bp = tl.make_block_ptr(
        base=Y_ptr + b * sY_b + c * sY_c + h * sY_h,
        shape=(L, P), strides=(sY_l, sY_p),
        offsets=(m0, 0), block_shape=(BLOCK_M, P), order=(1, 0),
    )
    tl.store(Y_bp, acc.to(tl.bfloat16))


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    A_cumsum: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """Fused intra-chunk diagonal output Y_diag for Mamba2 SSD (Triton)."""
    hs = hidden_states.contiguous()
    A = A_cumsum.contiguous()
    Bt = B.contiguous()
    Ct = C.contiguous()

    b, nc, L, H, P = hs.shape
    Gp = Bt.shape[3]
    N = Bt.shape[4]
    heads_per_group = H // Gp

    BLOCK_M = 64
    num_m_blocks = L // BLOCK_M

    Y = torch.empty_like(hs)

    grid = (b * nc, H, num_m_blocks)
    _mamba2_diag_kernel[grid](
        hs, A, Bt, Ct, Y,
        nc,
        hs.stride(0), hs.stride(1), hs.stride(2), hs.stride(3), hs.stride(4),
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        Bt.stride(0), Bt.stride(1), Bt.stride(2), Bt.stride(3), Bt.stride(4),
        Ct.stride(0), Ct.stride(1), Ct.stride(2), Ct.stride(3), Ct.stride(4),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
        HEADS_PER_GROUP=heads_per_group,
        L=L, N=N, P=P,
        BLOCK_M=BLOCK_M,
        num_warps=4,
        num_stages=2,
    )
    return Y
