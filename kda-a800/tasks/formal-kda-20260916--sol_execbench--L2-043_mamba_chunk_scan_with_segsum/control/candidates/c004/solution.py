"""L2/043 Mamba-2 chunk scan with segment sum — Triton solution.

Candidate c004: fuse the K1 cumsum kernel away (recompute in K2/K3/K4).

Evidence driving this change: on c002 (5/5 PASS, geomean 10.64x) the per-workload
timing vector shows the three 64-tile workloads (WL1/3/4) all sit at ~0.158 ms regardless
of NC in {1,2,4} or B in {1,2,4}, while 192-tile=0.210 ms and 256-tile=0.254 ms. A linear
fit ``time ≈ floor + slope·tiles`` gives floor ≈ 0.126 ms (~80% of the 64-tile runtime) and
slope ≈ 0.5 us/tile — i.e. a large fixed overhead (4 kernel launches + the acum/alast DRAM
round-trips) dominates the small cases. c003 already showed c002 is NOT bound by redundant
G = C·Bᵀ matmul, so the right lever is cutting fixed overhead, not matmul FLOPs.

This candidate removes kernel K1 entirely: the per-chunk cumsum a[t] = cumsum_t A and its
total a_last are recomputed inside K2, K3 and K4 (a[t] is a cheap 256-element scan). That
deletes one launch and the ``acum:[B,H,NC,Q]`` + ``alast:[B,H,NC]`` DRAM tensors (read
twice, written once). Algebra is unchanged and numerically identical to c002 (cumsum in
fp32; padded rows load A=0 so the scan stays flat, matching the reference). In K4 the query
/ key cumsum sub-blocks a_i / a_j are extracted from the full fp32 cumsum via an exact
one-hot gather (a [BLOCK_M, Q] mask-and-sum, negligible next to the matmuls).

All decays / cumsum / exp / accumulators / D residual stay in fp32; bf16 only for
tensor-core operands and the final output / final_state casts.

Stages (all working intermediates in fp32; only output / final_state cast to bf16):
  K2 chunk_state    : a=cumsum(A); states[p,n] = sum_t exp(a_last - a[t]) * X[t,p] * B[t,n].
  K3 state_passing  : a_last(c)=sum_t A; recurrence seeded by initial_states,
                      new_states[c+1] = exp(a_last(c)) * new_states[c] + states[c];
                      states_out[c] = new_states[c], final_state = new_states[NC].
  K4 chunk_scan     : a=cumsum(A); Y = (G∘L)·X + exp(a[t])·(C·states_out) + D[h]·X,
                      G[i,j]=C_i·B_j, L[i,j]=exp(a[i]-a[j]) lower-tri; no [Q,Q] to DRAM.

Constants for this task: H=16, P=64, N=256, n_groups=1, Q(chunk)=256.
No Torch/CPU/NumPy computational fallback anywhere: Triton is the only compute path.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# K2: per-(b,chunk,head) state = sum_t decay_state[t] * X[t,:] outer B[t,:].
#     grid = (B*NC*H, N // BLOCK_N).  Recomputes a = cumsum(A) inline.
# ---------------------------------------------------------------------------
@triton.jit
def _chunk_state_kernel(
    X_ptr, B_ptr, A_ptr, states_ptr,
    sX_b, sX_l, sX_h, sX_p,
    sB_b, sB_l, sB_n,
    sA_b, sA_h, sA_l,
    sst_b, sst_c, sst_h, sst_p, sst_n,
    L, NC, H,
    Q: tl.constexpr, P: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr,
):
    bch = tl.program_id(0)
    nblk = tl.program_id(1)
    b = bch // (NC * H)
    rem = bch % (NC * H)
    c = rem // H
    h = rem % H

    t = tl.arange(0, Q)
    seq = c * Q + t
    tmask = seq < L
    p = tl.arange(0, P)
    n = nblk * BLOCK_N + tl.arange(0, BLOCK_N)

    Arow = tl.load(A_ptr + b * sA_b + h * sA_h + seq * sA_l, mask=tmask, other=0.0).to(tl.float32)
    acum = tl.cumsum(Arow, axis=0)          # [Q]
    alast = tl.sum(Arow, axis=0)            # scalar
    decay = tl.exp(alast - acum)            # [Q]

    X = tl.load(
        X_ptr + b * sX_b + seq[:, None] * sX_l + h * sX_h + p[None, :] * sX_p,
        mask=tmask[:, None], other=0.0,
    ).to(tl.float32)  # [Q, P]
    Xd = (X * decay[:, None]).to(tl.bfloat16)  # [Q, P] bf16 (decay applied in fp32)

    Bt = tl.load(
        B_ptr + b * sB_b + seq[:, None] * sB_l + n[None, :] * sB_n,
        mask=tmask[:, None], other=0.0,
    )  # [Q, BLOCK_N] bf16 (native)

    states = tl.dot(tl.trans(Xd), Bt)  # [P, BLOCK_N] fp32 accumulate
    tl.store(
        states_ptr + b * sst_b + c * sst_c + h * sst_h + p[:, None] * sst_p + n[None, :] * sst_n,
        states,
    )


# ---------------------------------------------------------------------------
# K3: inter-chunk sequential recurrence. grid = (B, H, N // BLOCK_N).
#     Recomputes a_last(c) = sum_t A inline (no alast tensor).
# ---------------------------------------------------------------------------
@triton.jit
def _state_passing_kernel(
    states_ptr, init_ptr, states_out_ptr, final_ptr, A_ptr,
    sst_b, sst_c, sst_h, sst_p, sst_n,
    si_b, si_h, si_p, si_n,
    so_b, so_c, so_h, so_p, so_n,
    sf_b, sf_h, sf_p, sf_n,
    sA_b, sA_h, sA_l,
    L, NC, H,
    Q: tl.constexpr, P: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    nblk = tl.program_id(2)

    p = tl.arange(0, P)
    n = nblk * BLOCK_N + tl.arange(0, BLOCK_N)
    t = tl.arange(0, Q)

    state = tl.load(
        init_ptr + b * si_b + h * si_h + p[:, None] * si_p + n[None, :] * si_n
    ).to(tl.float32)  # [P, BLOCK_N]

    for c in range(0, NC):
        tl.store(
            states_out_ptr + b * so_b + c * so_c + h * so_h + p[:, None] * so_p + n[None, :] * so_n,
            state,
        )
        seq = c * Q + t
        smask = seq < L
        Arow = tl.load(A_ptr + b * sA_b + h * sA_h + seq * sA_l, mask=smask, other=0.0).to(tl.float32)
        g = tl.exp(tl.sum(Arow, axis=0))
        st = tl.load(
            states_ptr + b * sst_b + c * sst_c + h * sst_h + p[:, None] * sst_p + n[None, :] * sst_n
        ).to(tl.float32)
        state = g * state + st

    tl.store(
        final_ptr + b * sf_b + h * sf_h + p[:, None] * sf_p + n[None, :] * sf_n,
        state.to(tl.bfloat16),
    )


# ---------------------------------------------------------------------------
# K4: chunk scan. grid = (B*NC*H, Q // BLOCK_M).  Recomputes a = cumsum(A) inline.
# ---------------------------------------------------------------------------
@triton.jit
def _chunk_scan_kernel(
    C_ptr, B_ptr, X_ptr, A_ptr, states_out_ptr, D_ptr, out_ptr,
    sC_b, sC_l, sC_n,
    sB_b, sB_l, sB_n,
    sX_b, sX_l, sX_h, sX_p,
    sA_b, sA_h, sA_l,
    so_b, so_c, so_h, so_p, so_n,
    sD_h,
    sout_b, sout_l, sout_hp,
    L, NC, H,
    Q: tl.constexpr, P: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    bch = tl.program_id(0)
    iblk = tl.program_id(1)
    b = bch // (NC * H)
    rem = bch % (NC * H)
    c = rem // H
    h = rem % H

    qidx = tl.arange(0, Q)
    seq_full = c * Q + qidx
    fmask = seq_full < L
    Arow = tl.load(A_ptr + b * sA_b + h * sA_h + seq_full * sA_l, mask=fmask, other=0.0).to(tl.float32)
    a_full = tl.cumsum(Arow, axis=0)  # [Q]

    im = iblk * BLOCK_M + tl.arange(0, BLOCK_M)  # chunk-local query rows
    seq_i = c * Q + im
    imask = seq_i < L
    nidx = tl.arange(0, N)
    pidx = tl.arange(0, P)

    # exact gather a_i = a_full[im] via one-hot mask-sum
    a_i = tl.sum(tl.where(qidx[None, :] == im[:, None], a_full[None, :], 0.0), axis=1)  # [BLOCK_M]

    C_i = tl.load(
        C_ptr + b * sC_b + seq_i[:, None] * sC_l + nidx[None, :] * sC_n,
        mask=imask[:, None], other=0.0,
    )  # [BLOCK_M, N] bf16

    acc = tl.zeros((BLOCK_M, P), dtype=tl.float32)

    for jblk in range(0, iblk + 1):
        jm = jblk * BLOCK_M + tl.arange(0, BLOCK_M)
        seq_j = c * Q + jm
        jmask = seq_j < L
        a_j = tl.sum(tl.where(qidx[None, :] == jm[:, None], a_full[None, :], 0.0), axis=1)  # [BLOCK_M]
        B_j = tl.load(
            B_ptr + b * sB_b + seq_j[:, None] * sB_l + nidx[None, :] * sB_n,
            mask=jmask[:, None], other=0.0,
        )  # [BLOCK_M, N] bf16
        X_j = tl.load(
            X_ptr + b * sX_b + seq_j[:, None] * sX_l + h * sX_h + pidx[None, :] * sX_p,
            mask=jmask[:, None], other=0.0,
        )  # [BLOCK_M, P] bf16

        G = tl.dot(C_i, tl.trans(B_j))  # [BLOCK_M, BLOCK_M] fp32
        diff = a_i[:, None] - a_j[None, :]
        cmask = im[:, None] >= jm[None, :]
        Lm = tl.where(cmask, tl.exp(diff), 0.0)
        M = (G * Lm).to(tl.bfloat16)
        acc += tl.dot(M, X_j)

    states_o = tl.load(
        states_out_ptr + b * so_b + c * so_c + h * so_h + pidx[:, None] * so_p + nidx[None, :] * so_n
    ).to(tl.bfloat16)  # [P, N]
    Cstate = tl.dot(C_i, tl.trans(states_o))  # [BLOCK_M, P] fp32
    a_i_exp = tl.exp(a_i)
    Y = acc + a_i_exp[:, None] * Cstate

    X_i = tl.load(
        X_ptr + b * sX_b + seq_i[:, None] * sX_l + h * sX_h + pidx[None, :] * sX_p,
        mask=imask[:, None], other=0.0,
    ).to(tl.float32)
    Dh = tl.load(D_ptr + h * sD_h).to(tl.float32)
    Y = Y + Dh * X_i

    out_off = out_ptr + b * sout_b + seq_i[:, None] * sout_l + (h * P + pidx[None, :]) * sout_hp
    tl.store(out_off, Y.to(tl.bfloat16), mask=imask[:, None])


# ---------------------------------------------------------------------------
# Host entry point.
# ---------------------------------------------------------------------------
@torch.no_grad()
def run(hidden_states, A, B, C, D, initial_states):
    Bb, L, H, P = hidden_states.shape
    N = 256
    Q = 256
    device = hidden_states.device

    pad = (Q - L % Q) % Q
    L_pad = L + pad
    NC = L_pad // Q

    BLOCK_M = 64
    BLOCK_N = 64

    states = torch.empty((Bb, NC, H, P, N), dtype=torch.float32, device=device)
    states_out = torch.empty((Bb, NC, H, P, N), dtype=torch.float32, device=device)
    final_state = torch.empty((Bb, H, P, N), dtype=torch.bfloat16, device=device)
    output = torch.empty((Bb, L, H * P), dtype=torch.bfloat16, device=device)

    # strides
    sA_b, sA_h, sA_l = A.stride(0), A.stride(1), A.stride(2)
    sX_b, sX_l, sX_h, sX_p = hidden_states.stride()
    sB_b, sB_l, sB_n = B.stride(0), B.stride(1), B.stride(3)
    sC_b, sC_l, sC_n = C.stride(0), C.stride(1), C.stride(3)
    sD_h = D.stride(0)
    si_b, si_h, si_p, si_n = initial_states.stride()

    sst_b, sst_c, sst_h, sst_p, sst_n = states.stride()
    so_b, so_c, so_h, so_p, so_n = states_out.stride()
    sf_b, sf_h, sf_p, sf_n = final_state.stride()
    sout_b, sout_l, sout_hp = output.stride()

    # K2: chunk_state (recomputes cumsum inline)
    _chunk_state_kernel[(Bb * NC * H, N // BLOCK_N)](
        hidden_states, B, A, states,
        sX_b, sX_l, sX_h, sX_p,
        sB_b, sB_l, sB_n,
        sA_b, sA_h, sA_l,
        sst_b, sst_c, sst_h, sst_p, sst_n,
        L, NC, H,
        Q=Q, P=P, N=N, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2,
    )

    # K3: state_passing (recomputes a_last inline)
    _state_passing_kernel[(Bb, H, N // BLOCK_N)](
        states, initial_states, states_out, final_state, A,
        sst_b, sst_c, sst_h, sst_p, sst_n,
        si_b, si_h, si_p, si_n,
        so_b, so_c, so_h, so_p, so_n,
        sf_b, sf_h, sf_p, sf_n,
        sA_b, sA_h, sA_l,
        L, NC, H,
        Q=Q, P=P, N=N, BLOCK_N=BLOCK_N, num_warps=4,
    )

    # K4: chunk_scan (recomputes cumsum inline)
    _chunk_scan_kernel[(Bb * NC * H, Q // BLOCK_M)](
        C, B, hidden_states, A, states_out, D, output,
        sC_b, sC_l, sC_n,
        sB_b, sB_l, sB_n,
        sX_b, sX_l, sX_h, sX_p,
        sA_b, sA_h, sA_l,
        so_b, so_c, so_h, so_p, so_n,
        sD_h,
        sout_b, sout_l, sout_hp,
        L, NC, H,
        Q=Q, P=P, N=N, BLOCK_M=BLOCK_M, num_warps=4, num_stages=2,
    )

    return output, final_state
