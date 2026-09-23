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
# c004: GROUP-LEVEL G REUSE (hypothesis H5). G depends on the head only through
# its group g = h // (H//Gp); the 4 heads of a group share the same C_g, B_g,
# hence the same G. Grid = (b, nc, Gp): compute G = C_g @ B_g^T ONCE per group
# and inner-loop the HEADS_PER_GROUP heads (each with its own a, X, Y, Ldec).
# This cuts the first matmul and the B/C HBM reads 4x while keeping the 128^3
# tiles. Trade-off: 4x fewer programs (occupancy risk on small b*nc).
#
# Single fused kernel; no [L,L]/[L,L,*] HBM temporaries. bf16 inputs, fp32
# tensor-core accumulation, bf16 store. (Parent c001; c002/c003 rejected.)
# =============================================================================


@triton.jit
def _mamba2_diag_group_kernel(
    X_ptr, A_ptr, B_ptr, C_ptr, Y_ptr,
    sX_b, sX_c, sX_l, sX_h, sX_p,
    sA_b, sA_h, sA_c, sA_l,
    sB_b, sB_c, sB_l, sB_g, sB_n,
    sC_b, sC_c, sC_l, sC_g, sC_n,
    sY_b, sY_c, sY_l, sY_h, sY_p,
    HEADS_PER_GROUP: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    P: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    grp = tl.program_id(2)

    # --- load C_g [i, n] and B_g [j, n] once for the group (bf16) ---
    C_bp = tl.make_block_ptr(
        base=C_ptr + b * sC_b + c * sC_c + grp * sC_g,
        shape=(L, N), strides=(sC_l, sC_n),
        offsets=(0, 0), block_shape=(L, N), order=(1, 0),
    )
    B_bp = tl.make_block_ptr(
        base=B_ptr + b * sB_b + c * sB_c + grp * sB_g,
        shape=(L, N), strides=(sB_l, sB_n),
        offsets=(0, 0), block_shape=(L, N), order=(1, 0),
    )
    c_tile = tl.load(C_bp)   # [i, n] bf16
    b_tile = tl.load(B_bp)   # [j, n] bf16

    # --- G[i,j] = C_g @ B_g^T (fp32 accumulate), shared by all heads ---
    g_tile = tl.dot(c_tile, tl.trans(b_tile), out_dtype=tl.float32)  # [i, j] fp32

    offs = tl.arange(0, L)
    causal = offs[:, None] >= offs[None, :]

    # --- inner-loop the heads of this group ---
    for hpg in tl.static_range(HEADS_PER_GROUP):
        h = grp * HEADS_PER_GROUP + hpg

        a = tl.load(A_ptr + b * sA_b + h * sA_h + c * sA_c + offs * sA_l).to(tl.float32)
        S = tl.cumsum(a, axis=0)                   # inclusive prefix sum [L]
        diff = S[:, None] - S[None, :]             # [i, j]
        diff = tl.where(causal, diff, float("-inf"))
        Ldec = tl.exp(diff)                        # [i, j] fp32

        m_tile = g_tile * Ldec                     # [i, j] fp32

        X_bp = tl.make_block_ptr(
            base=X_ptr + b * sX_b + c * sX_c + h * sX_h,
            shape=(L, P), strides=(sX_l, sX_p),
            offsets=(0, 0), block_shape=(L, P), order=(1, 0),
        )
        x_tile = tl.load(X_bp)                     # [j, d] bf16

        y_tile = tl.dot(m_tile.to(x_tile.dtype), x_tile, out_dtype=tl.float32)  # [i, d]

        Y_bp = tl.make_block_ptr(
            base=Y_ptr + b * sY_b + c * sY_c + h * sY_h,
            shape=(L, P), strides=(sY_l, sY_p),
            offsets=(0, 0), block_shape=(L, P), order=(1, 0),
        )
        tl.store(Y_bp, y_tile.to(tl.bfloat16))


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

    Y = torch.empty_like(hs)

    grid = (b, nc, Gp)
    _mamba2_diag_group_kernel[grid](
        hs, A, Bt, Ct, Y,
        hs.stride(0), hs.stride(1), hs.stride(2), hs.stride(3), hs.stride(4),
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        Bt.stride(0), Bt.stride(1), Bt.stride(2), Bt.stride(3), Bt.stride(4),
        Ct.stride(0), Ct.stride(1), Ct.stride(2), Ct.stride(3), Ct.stride(4),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
        HEADS_PER_GROUP=heads_per_group,
        L=L, N=N, P=P,
        num_warps=4,
        num_stages=2,
    )
    return Y
