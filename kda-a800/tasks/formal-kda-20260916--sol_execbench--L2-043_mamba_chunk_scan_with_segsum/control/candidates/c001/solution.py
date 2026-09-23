"""L2/043 Mamba-2 chunk scan with segment sum — Triton solution.

Candidate c001: faithful Option-A 4-stage SSD decomposition.

Stages (all working intermediates in fp32; only output / final_state cast to bf16):
  K1 chunk_cumsum   : a[t] = cumsum_t A over each chunk, plus a_last = a[Q-1].
  K2 chunk_state    : states[p,n] = sum_t exp(a_last - a[t]) * X[t,p] * B[t,n].
  K3 state_passing  : sequential inter-chunk recurrence seeded by initial_states,
                      new_states[c+1] = exp(a_last(c)) * new_states[c] + states[c];
                      states_out[c] = new_states[c], final_state = new_states[NC].
  K4 chunk_scan     : Y = (G∘L)·X  +  exp(a[t])·(C·states_out)  +  D[h]·X
                      with G[i,j] = C_i·B_j (head-independent), L[i,j]=exp(a[i]-a[j]),
                      lower-triangular; never materializes [Q,Q] to DRAM.

Constants for this task: H=16, P=64, N=256, n_groups=1, Q(chunk)=256.
B/C are shared across all heads (n_groups=1).

All matmuls use input_precision="ieee" (true fp32) in this correctness-first baseline.
No Torch/CPU/NumPy computational fallback anywhere: Triton is the only compute path.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# K1: per-chunk inclusive cumsum of A, and per-chunk total (a_last).
# ---------------------------------------------------------------------------
@triton.jit
def _cumsum_kernel(
    A_ptr, acum_ptr, alast_ptr,
    sA_b, sA_h, sA_l,
    sac_b, sac_h, sac_c, sac_q,
    sal_b, sal_h, sal_c,
    L,
    Q: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    t = tl.arange(0, Q)
    seq = c * Q + t
    m = seq < L
    a = tl.load(A_ptr + b * sA_b + h * sA_h + seq * sA_l, mask=m, other=0.0).to(tl.float32)
    acum = tl.cumsum(a, axis=0)
    tl.store(acum_ptr + b * sac_b + h * sac_h + c * sac_c + t * sac_q, acum)
    last = tl.sum(a, axis=0)
    tl.store(alast_ptr + b * sal_b + h * sal_h + c * sal_c, last)


# ---------------------------------------------------------------------------
# K2: per-(b,chunk,head) state = sum_t decay_state[t] * X[t,:] outer B[t,:].
#     grid = (B*NC*H, N // BLOCK_N)
# ---------------------------------------------------------------------------
@triton.jit
def _chunk_state_kernel(
    X_ptr, B_ptr, acum_ptr, alast_ptr, states_ptr,
    sX_b, sX_l, sX_h, sX_p,
    sB_b, sB_l, sB_n,
    sac_b, sac_h, sac_c, sac_q,
    sal_b, sal_h, sal_c,
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

    acum = tl.load(acum_ptr + b * sac_b + h * sac_h + c * sac_c + t * sac_q).to(tl.float32)
    alast = tl.load(alast_ptr + b * sal_b + h * sal_h + c * sal_c).to(tl.float32)
    decay = tl.exp(alast - acum)  # [Q]

    X = tl.load(
        X_ptr + b * sX_b + seq[:, None] * sX_l + h * sX_h + p[None, :] * sX_p,
        mask=tmask[:, None], other=0.0,
    ).to(tl.float32)  # [Q, P]
    Xd = X * decay[:, None]

    Bt = tl.load(
        B_ptr + b * sB_b + seq[:, None] * sB_l + n[None, :] * sB_n,
        mask=tmask[:, None], other=0.0,
    ).to(tl.float32)  # [Q, BLOCK_N]

    states = tl.dot(tl.trans(Xd), Bt, input_precision="ieee")  # [P, BLOCK_N]
    tl.store(
        states_ptr + b * sst_b + c * sst_c + h * sst_h + p[:, None] * sst_p + n[None, :] * sst_n,
        states,
    )


# ---------------------------------------------------------------------------
# K3: inter-chunk sequential recurrence. grid = (B, H, N // BLOCK_N).
#     Elementwise in (p, n), so tiling over n is safe.
# ---------------------------------------------------------------------------
@triton.jit
def _state_passing_kernel(
    states_ptr, init_ptr, states_out_ptr, final_ptr, alast_ptr,
    sst_b, sst_c, sst_h, sst_p, sst_n,
    si_b, si_h, si_p, si_n,
    so_b, so_c, so_h, so_p, so_n,
    sf_b, sf_h, sf_p, sf_n,
    sal_b, sal_h, sal_c,
    NC, H,
    P: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    nblk = tl.program_id(2)

    p = tl.arange(0, P)
    n = nblk * BLOCK_N + tl.arange(0, BLOCK_N)

    state = tl.load(
        init_ptr + b * si_b + h * si_h + p[:, None] * si_p + n[None, :] * si_n
    ).to(tl.float32)  # [P, BLOCK_N]

    for c in range(0, NC):
        tl.store(
            states_out_ptr + b * so_b + c * so_c + h * so_h + p[:, None] * so_p + n[None, :] * so_n,
            state,
        )
        alast = tl.load(alast_ptr + b * sal_b + h * sal_h + c * sal_c).to(tl.float32)
        g = tl.exp(alast)
        st = tl.load(
            states_ptr + b * sst_b + c * sst_c + h * sst_h + p[:, None] * sst_p + n[None, :] * sst_n
        ).to(tl.float32)
        state = g * state + st

    tl.store(
        final_ptr + b * sf_b + h * sf_h + p[:, None] * sf_p + n[None, :] * sf_n,
        state.to(tl.bfloat16),
    )


# ---------------------------------------------------------------------------
# K4: chunk scan. grid = (B*NC*H, Q // BLOCK_M).
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
    ).to(tl.float32)  # [BLOCK_M, N]

    acc = tl.zeros((BLOCK_M, P), dtype=tl.float32)

    for jblk in range(0, iblk + 1):
        jm = jblk * BLOCK_M + tl.arange(0, BLOCK_M)
        seq_j = c * Q + jm
        jmask = seq_j < L
        a_j = tl.load(acum_ptr + b * sac_b + h * sac_h + c * sac_c + jm * sac_q).to(tl.float32)
        B_j = tl.load(
            B_ptr + b * sB_b + seq_j[:, None] * sB_l + nidx[None, :] * sB_n,
            mask=jmask[:, None], other=0.0,
        ).to(tl.float32)  # [BLOCK_M, N]
        X_j = tl.load(
            X_ptr + b * sX_b + seq_j[:, None] * sX_l + h * sX_h + pidx[None, :] * sX_p,
            mask=jmask[:, None], other=0.0,
        ).to(tl.float32)  # [BLOCK_M, P]

        G = tl.dot(C_i, tl.trans(B_j), input_precision="ieee")  # [BLOCK_M, BLOCK_M]
        diff = a_i[:, None] - a_j[None, :]
        cmask = im[:, None] >= jm[None, :]
        Lm = tl.where(cmask, tl.exp(diff), 0.0)
        M = G * Lm
        acc += tl.dot(M, X_j, input_precision="ieee")

    states_o = tl.load(
        states_out_ptr + b * so_b + c * so_c + h * so_h + pidx[:, None] * so_p + nidx[None, :] * so_n
    ).to(tl.float32)  # [P, N]
    Cstate = tl.dot(C_i, tl.trans(states_o), input_precision="ieee")  # [BLOCK_M, P]
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
    alast = torch.empty((Bb, H, NC), dtype=torch.float32, device=device)
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

    sac_b, sac_h, sac_c, sac_q = acum.stride()
    sal_b, sal_h, sal_c = alast.stride()
    sst_b, sst_c, sst_h, sst_p, sst_n = states.stride()
    so_b, so_c, so_h, so_p, so_n = states_out.stride()
    sf_b, sf_h, sf_p, sf_n = final_state.stride()
    sout_b, sout_l, sout_hp = output.stride()

    # K1: cumsum
    _cumsum_kernel[(Bb, NC, H)](
        A, acum, alast,
        sA_b, sA_h, sA_l,
        sac_b, sac_h, sac_c, sac_q,
        sal_b, sal_h, sal_c,
        L,
        Q=Q, num_warps=4,
    )

    # K2: chunk_state
    _chunk_state_kernel[(Bb * NC * H, N // BLOCK_N)](
        hidden_states, B, acum, alast, states,
        sX_b, sX_l, sX_h, sX_p,
        sB_b, sB_l, sB_n,
        sac_b, sac_h, sac_c, sac_q,
        sal_b, sal_h, sal_c,
        sst_b, sst_c, sst_h, sst_p, sst_n,
        L, NC, H,
        Q=Q, P=P, N=N, BLOCK_N=BLOCK_N, num_warps=8,
    )

    # K3: state_passing
    _state_passing_kernel[(Bb, H, N // BLOCK_N)](
        states, initial_states, states_out, final_state, alast,
        sst_b, sst_c, sst_h, sst_p, sst_n,
        si_b, si_h, si_p, si_n,
        so_b, so_c, so_h, so_p, so_n,
        sf_b, sf_h, sf_p, sf_n,
        sal_b, sal_h, sal_c,
        NC, H,
        P=P, N=N, BLOCK_N=BLOCK_N, num_warps=4,
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
        Q=Q, P=P, N=N, BLOCK_M=BLOCK_M, num_warps=8,
    )

    return output, final_state
