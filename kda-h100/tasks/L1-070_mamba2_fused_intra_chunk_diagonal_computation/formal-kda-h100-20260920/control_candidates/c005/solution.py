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
# Single fused kernel: G and M live in SRAM/registers; no [L,L]/[L,L,*] HBM
# temporaries are materialized (the whole point of the fusion). bf16 inputs,
# fp32 tensor-core accumulation, bf16 store.
#
# Best-known-good parent (c001). Rejected explorations:
#   c002 - BLOCK_M=64 query-row split + causal j-loop -> regressed every workload.
#   c003 - num_warps=8                                -> regressed.
#   c004 - group-level G reuse, grid (b,nc,Gp)        -> large regression (occupancy).
# c005: c001 kernel+grid unchanged, only num_stages 2 -> 3. Deeper software
# pipelining of the two dots may overlap loads better on the large BW-bound cases.
# =============================================================================


@triton.jit
def _mamba2_diag_kernel(
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
    h = tl.program_id(2)
    g = h // HEADS_PER_GROUP

    # --- load C_h [i, n] and B_h [j, n] (bf16) ---
    C_bp = tl.make_block_ptr(
        base=C_ptr + b * sC_b + c * sC_c + g * sC_g,
        shape=(L, N), strides=(sC_l, sC_n),
        offsets=(0, 0), block_shape=(L, N), order=(1, 0),
    )
    B_bp = tl.make_block_ptr(
        base=B_ptr + b * sB_b + c * sB_c + g * sB_g,
        shape=(L, N), strides=(sB_l, sB_n),
        offsets=(0, 0), block_shape=(L, N), order=(1, 0),
    )
    X_bp = tl.make_block_ptr(
        base=X_ptr + b * sX_b + c * sX_c + h * sX_h,
        shape=(L, P), strides=(sX_l, sX_p),
        offsets=(0, 0), block_shape=(L, P), order=(1, 0),
    )

    c_tile = tl.load(C_bp)   # [i, n] bf16
    b_tile = tl.load(B_bp)   # [j, n] bf16
    x_tile = tl.load(X_bp)   # [j, d] bf16

    # --- G[i,j] = C_h @ B_h^T (fp32 accumulate) ---
    g_tile = tl.dot(c_tile, tl.trans(b_tile), out_dtype=tl.float32)  # [i, j] fp32

    # --- decay mask Ldec[i,j] = exp(S[i] - S[j]), i >= j else 0 ---
    offs = tl.arange(0, L)
    a = tl.load(A_ptr + b * sA_b + h * sA_h + c * sA_c + offs * sA_l).to(tl.float32)  # [L]
    S = tl.cumsum(a, axis=0)                       # inclusive prefix sum [L]
    diff = S[:, None] - S[None, :]                 # [i, j]
    causal = offs[:, None] >= offs[None, :]
    diff = tl.where(causal, diff, float("-inf"))
    Ldec = tl.exp(diff)                            # [i, j] fp32

    m_tile = g_tile * Ldec                         # [i, j] fp32

    # --- Y[i,d] = M @ X_h (fp32 accumulate) ---
    y_tile = tl.dot(m_tile.to(x_tile.dtype), x_tile, out_dtype=tl.float32)  # [i, d] fp32

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

    grid = (b, nc, H)
    _mamba2_diag_kernel[grid](
        hs, A, Bt, Ct, Y,
        hs.stride(0), hs.stride(1), hs.stride(2), hs.stride(3), hs.stride(4),
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        Bt.stride(0), Bt.stride(1), Bt.stride(2), Bt.stride(3), Bt.stride(4),
        Ct.stride(0), Ct.stride(1), Ct.stride(2), Ct.stride(3), Ct.stride(4),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
        HEADS_PER_GROUP=heads_per_group,
        L=L, N=N, P=P,
        num_warps=4,
        num_stages=3,
    )
    return Y
