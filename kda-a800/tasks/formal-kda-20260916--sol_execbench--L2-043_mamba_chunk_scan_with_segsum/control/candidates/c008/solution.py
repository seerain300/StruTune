"""L2/043 Mamba-2 chunk scan with segment sum — Triton solution.

Candidate c008: K4 query-tile size BLOCK_M 64 -> 128 (single knob vs c007).

Parent c007 (5/5 PASS, geomean 19.23x, 2 launches KS/K4) is the current best. Structural
fusion is largely exhausted (only acum + states_out cross DRAM between KS and K4). This is
a single-knob micro-tune: the K4 chunk-scan query-block size ``BLOCK_M`` goes 64 -> 128.

Rationale: K4's matmuls (G = C_i·B_jᵀ, M·X, C·states) are all small [64,·] tiles with a
short lower-triangular j-loop (BLOCK_M=64 → up to 4 iterations, 10 sub-block dots). Doubling
to BLOCK_M=128 gives larger, more tensor-core-efficient [128,·] tiles and a shorter j-loop
(≤2 iters, 3 sub-blocks) at the cost of some grid parallelism (grid 2nd dim Q/BLOCK_M: 4→2).
Occupancy check: even the 64-tile workloads keep B·NC·H·(Q/BLOCK_M) = 64·2 = 128 programs
(> 108 SMs), and the diagonal lower-tri mask (im>=jm) and partial-chunk row masks are
unchanged and still correct at 128 rows. Everything else is byte-identical to c007, so any
delta is attributable purely to the K4 tile size. KS is untouched.

Stages (compute fp32; states_out/output/final_state stored bf16):
  KS state+pass     : fused chunk_state + inter-chunk recurrence, state carried in fp32
                      registers; stores acum for K4, states_out, final_state.
  K4 chunk_scan     : Y = (G∘L)·X + exp(a[t])·(C·states_out) + D[h]·X (reads acum),
                      BLOCK_M=128.

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

    BLOCK_M = 128
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
