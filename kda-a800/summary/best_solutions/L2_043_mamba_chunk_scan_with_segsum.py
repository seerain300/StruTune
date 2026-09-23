# KDA A800 best solution: L2/043_mamba_chunk_scan_with_segsum
# candidate: c007  |  feedback: 19.23x  |  final (authoritative): 19.85x
# campaign formal-kda-20260916 (A800, g0056)  |  evaluations: 10
# source: tasks/formal-kda-20260916--sol_execbench--L2-043_mamba_chunk_scan_with_segsum/control/candidates/c007/solution.py (sha256-frozen snapshot)

"""L2/043 Mamba-2 chunk scan with segment sum — Triton solution.

Candidate c007: fuse K2 (chunk_state) and K3 (state_passing) into one kernel.

Parent c006 (5/5 PASS, geomean 14.45x, 3 launches K2/K3/K4) is the current best. The two
accepted wins so far were (c005) storing the big state tensors in bf16 and (c006) removing
a launch by folding K1 into K2 — i.e. both DRAM traffic on the state tensors AND launch
count are first-order on these tiny (NC<=4) shapes.

This candidate removes the whole ``states`` intermediate tensor and one more launch by
fusing the per-chunk state computation and the inter-chunk recurrence into a single kernel
``_state_pass_kernel`` (grid (B, H, N/BLOCK_N)). Each program carries the [P, BLOCK_N] SSM
state in fp32 registers and loops chunks c=0..NC-1:
    store states_out[c] = carried-in state (bf16)
    a = cumsum(A[c]);  a_last = sum(A[c]);  decay = exp(a_last - a)
    contrib[p,n] = sum_t (X[t,p]*decay[t]) * B[t,n]     (bf16 dot, fp32 accumulate)
    state = exp(a_last) * state + contrib               (fp32 recurrence)
final_state = state (bf16). The cumsum ``acum`` is still persisted (only the nblk==0
program stores it) because K4 needs a_i / a_j. The ``states`` and ``alast`` tensors are
gone entirely, and the launch count drops 3 -> 2.

Numerics vs c006: identical algebra, and the per-chunk ``contrib`` now stays fp32 in a
register instead of being rounded to bf16 in the ``states`` tensor and reloaded — i.e.
marginally MORE accurate. The state recurrence, decays and exp args are all fp32 as before.
Trade-off: the state-compute stage loses the NC factor of grid parallelism (chunks are now
serialized within a block). NC<=4 so this only bites the B=1 workload (WL4); the removed
launch + states round-trip is expected to more than compensate, and the single-chunk WL3
is a pure win (NC=1, no serialization).

Stages (compute fp32; states_out/output/final_state stored bf16):
  KS state+pass     : fused K2+K3 as above (stores acum for K4, states_out, final_state).
  K4 chunk_scan     : Y = (G∘L)·X + exp(a[t])·(C·states_out) + D[h]·X (reads acum).

Constants for this task: H=16, P=64, N=256, n_groups=1, Q(chunk)=256.
No Torch/CPU/NumPy computational fallback anywhere: Triton is the only compute path.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# KS: fused chunk_state + state_passing.  grid = (B, H, N // BLOCK_N).
#     Carries the SSM state [P, BLOCK_N] in fp32 registers across chunks.
#     nblk==0 program persists acum:[B,H,NC,Q] for K4.
# ---------------------------------------------------------------------------
@triton.jit
def _state_pass_kernel(
    X_ptr, B_ptr, A_ptr, init_ptr, acum_ptr, states_out_ptr, final_ptr,
    sX_b, sX_l, sX_h, sX_p,
    sB_b, sB_l, sB_n,
    sA_b, sA_h, sA_l,
    si_b, si_h, si_p, si_n,
    sac_b, sac_h, sac_c, sac_q,
    so_b, so_c, so_h, so_p, so_n,
    sf_b, sf_h, sf_p, sf_n,
    L, NC, H,
    Q: tl.constexpr, P: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    nblk = tl.program_id(2)

    t = tl.arange(0, Q)
    p = tl.arange(0, P)
    n = nblk * BLOCK_N + tl.arange(0, BLOCK_N)

    state = tl.load(
        init_ptr + b * si_b + h * si_h + p[:, None] * si_p + n[None, :] * si_n
    ).to(tl.float32)  # [P, BLOCK_N]

    for c in range(0, NC):
        # states_out[c] = state carried in from chunks < c (before this chunk's update)
        tl.store(
            states_out_ptr + b * so_b + c * so_c + h * so_h + p[:, None] * so_p + n[None, :] * so_n,
            state.to(tl.bfloat16),
        )
        seq = c * Q + t
        tmask = seq < L
        Arow = tl.load(A_ptr + b * sA_b + h * sA_h + seq * sA_l, mask=tmask, other=0.0).to(tl.float32)
        acum = tl.cumsum(Arow, axis=0)   # [Q]
        alast = tl.sum(Arow, axis=0)     # scalar
        if nblk == 0:
            tl.store(acum_ptr + b * sac_b + h * sac_h + c * sac_c + t * sac_q, acum)
        decay = tl.exp(alast - acum)     # [Q]

        X = tl.load(
            X_ptr + b * sX_b + seq[:, None] * sX_l + h * sX_h + p[None, :] * sX_p,
            mask=tmask[:, None], other=0.0,
        ).to(tl.float32)  # [Q, P]
        Xd = (X * decay[:, None]).to(tl.bfloat16)  # [Q, P] bf16 (decay applied in fp32)
        Bt = tl.load(
            B_ptr + b * sB_b + seq[:, None] * sB_l + n[None, :] * sB_n,
            mask=tmask[:, None], other=0.0,
        )  # [Q, BLOCK_N] bf16 (native)

        contrib = tl.dot(tl.trans(Xd), Bt)   # [P, BLOCK_N] fp32
        state = tl.exp(alast) * state + contrib

    tl.store(
        final_ptr + b * sf_b + h * sf_h + p[:, None] * sf_p + n[None, :] * sf_n,
        state.to(tl.bfloat16),
    )


# ---------------------------------------------------------------------------
# K4: chunk scan. grid = (B*NC*H, Q // BLOCK_M).  states_out loaded bf16.
# ---------------------------------------------------------------------------
@triton.jit
def _chunk_scan_kernel(
    C_ptr, B_ptr, X_ptr, acum_ptr, states_out_ptr, D_ptr, out_ptr,
    sC_b, sC_l, sC_n,
    sB_b, sB_l, sB_n,
    sX_b, sX_l, sX_h, sX_p,
    sac_b, sac_h, sac_c, sac_q,
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

    im = iblk * BLOCK_M + tl.arange(0, BLOCK_M)  # chunk-local query rows
    seq_i = c * Q + im
    imask = seq_i < L
    nidx = tl.arange(0, N)
    pidx = tl.arange(0, P)

    a_i = tl.load(acum_ptr + b * sac_b + h * sac_h + c * sac_c + im * sac_q).to(tl.float32)  # [BLOCK_M]
    C_i = tl.load(
        C_ptr + b * sC_b + seq_i[:, None] * sC_l + nidx[None, :] * sC_n,
        mask=imask[:, None], other=0.0,
    )  # [BLOCK_M, N] bf16

    acc = tl.zeros((BLOCK_M, P), dtype=tl.float32)

    for jblk in range(0, iblk + 1):
        jm = jblk * BLOCK_M + tl.arange(0, BLOCK_M)
        seq_j = c * Q + jm
        jmask = seq_j < L
        a_j = tl.load(acum_ptr + b * sac_b + h * sac_h + c * sac_c + jm * sac_q).to(tl.float32)
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
    )  # [P, N] bf16 (stored bf16 by KS)
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

    acum = torch.empty((Bb, H, NC, Q), dtype=torch.float32, device=device)
    states_out = torch.empty((Bb, NC, H, P, N), dtype=torch.bfloat16, device=device)
    final_state = torch.empty((Bb, H, P, N), dtype=torch.bfloat16, device=device)
    output = torch.empty((Bb, L, H * P), dtype=torch.bfloat16, device=device)

    # strides
    sA_b, sA_h, sA_l = A.stride(0), A.stride(1), A.stride(2)
    sX_b, sX_l, sX_h, sX_p = hidden_states.stride()
    sB_b, sB_l, sB_n = B.stride(0), B.stride(1), B.stride(3)
    sC_b, sC_l, sC_n = C.stride(0), C.stride(1), C.stride(3)
    sD_h = D.stride(0)
    si_b, si_h, si_p, si_n = initial_states.stride()

    sac_b, sac_h, sac_c, sac_q = acum.stride()
    so_b, so_c, so_h, so_p, so_n = states_out.stride()
    sf_b, sf_h, sf_p, sf_n = final_state.stride()
    sout_b, sout_l, sout_hp = output.stride()

    # KS: fused chunk_state + state_passing
    _state_pass_kernel[(Bb, H, N // BLOCK_N)](
        hidden_states, B, A, initial_states, acum, states_out, final_state,
        sX_b, sX_l, sX_h, sX_p,
        sB_b, sB_l, sB_n,
        sA_b, sA_h, sA_l,
        si_b, si_h, si_p, si_n,
        sac_b, sac_h, sac_c, sac_q,
        so_b, so_c, so_h, so_p, so_n,
        sf_b, sf_h, sf_p, sf_n,
        L, NC, H,
        Q=Q, P=P, N=N, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2,
    )

    # K4: chunk_scan
    _chunk_scan_kernel[(Bb * NC * H, Q // BLOCK_M)](
        C, B, hidden_states, acum, states_out, D, output,
        sC_b, sC_l, sC_n,
        sB_b, sB_l, sB_n,
        sX_b, sX_l, sX_h, sX_p,
        sac_b, sac_h, sac_c, sac_q,
        so_b, so_c, so_h, so_p, so_n,
        sD_h,
        sout_b, sout_l, sout_hp,
        L, NC, H,
        Q=Q, P=P, N=N, BLOCK_M=BLOCK_M, num_warps=4, num_stages=2,
    )

    return output, final_state
