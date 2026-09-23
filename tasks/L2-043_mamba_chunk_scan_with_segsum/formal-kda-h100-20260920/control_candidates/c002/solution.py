import torch
import triton
import triton.language as tl

# =============================================================================
# L2/043 Mamba-2 chunk scan with segment sum  (H100 / sm_90)
#
# Candidate c002 -- Phase-0 bugfix of c001.
#
# c001 failed with a uniform RUNTIME_ERROR on every shape (clean run, not code-3).
# Diagnosis: register/resource pressure -- _chunk_scan_kernel held several full
# state-dim N=256 fp32 tiles live at once (C_i[64,256], R_c[64,256], B_j[64,256]),
# ~272 regs/thread @ 8 warps, exceeding the 255-reg limit -> PTXAS launch failure
# identical across all shapes. Also used the fp32 'ieee' dot path.
#
# c002 changes (single theme = shrink live footprint + supported dot path):
#   * Tile the N=256 contraction with an explicit BLOCK_K loop in chunk_scan
#     (both the Y_off C.R contraction and the CB = C.B contraction) so every
#     live matmul tile is at most [64,64].
#   * Tile the chunk_state output over N via a grid dimension (acc is [P,BLOCK_S]).
#   * Use input_precision="tf32" (the default, always-supported Hopper path)
#     instead of "ieee".
#   * Keep cumsum, exp, decays, and the D residual in fp32 (algorithm unchanged;
#     it was verified correct in docs/draft.md section 2).
#
# Fixed constants: H=16, P=64, N=256, G=1, Q(chunk)=256. Variable: bsz, S.
# Triton-only compute; torch used only for metadata / allocation / launch.
# =============================================================================

_H = 16
_P = 64
_N = 256
_Q = 256

_PREC = "tf32"


@triton.jit
def _chunk_cumsum_kernel(
    A_ptr, ACS_ptr, DCHUNK_ptr,
    S, NC,
    H: tl.constexpr, Q: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid % H
    tmp = pid // H
    c = tmp % NC
    b = tmp // NC

    t = tl.arange(0, Q)
    seq = c * Q + t
    mask = seq < S
    a = tl.load(A_ptr + (b * H + h) * S + seq, mask=mask, other=0.0).to(tl.float32)
    acs = tl.cumsum(a, axis=0)

    acs_base = ((b * H + h) * NC + c) * Q
    tl.store(ACS_ptr + acs_base + t, acs)
    # dchunk = acs[Q-1] = total sum over the (zero-padded) chunk
    tl.store(DCHUNK_ptr + (b * H + h) * NC + c, tl.sum(a, axis=0))


@triton.jit
def _chunk_state_kernel(
    X_ptr, B_ptr, ACS_ptr, DCHUNK_ptr, CS_ptr,
    S, NC,
    H: tl.constexpr, P: tl.constexpr, N: tl.constexpr, Q: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    sblock = tl.program_id(1)

    h = pid % H
    tmp = pid // H
    c = tmp % NC
    b = tmp // NC

    d = tl.arange(0, P)
    s = sblock * BLOCK_S + tl.arange(0, BLOCK_S)
    acs_base = ((b * H + h) * NC + c) * Q
    dchunk = tl.load(DCHUNK_ptr + (b * H + h) * NC + c)  # scalar

    acc = tl.zeros((P, BLOCK_S), dtype=tl.float32)
    for t0 in range(0, Q, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        seq = c * Q + t
        seq_mask = seq < S
        acs = tl.load(ACS_ptr + acs_base + t)  # [BLOCK_T]
        decay = tl.exp(dchunk - acs)           # [BLOCK_T]

        x_off = ((b * S + seq)[:, None] * H + h) * P + d[None, :]
        x = tl.load(X_ptr + x_off, mask=seq_mask[:, None], other=0.0).to(tl.float32)  # [BT,P]
        b_off = (b * S + seq)[:, None] * N + s[None, :]
        Bt = tl.load(B_ptr + b_off, mask=seq_mask[:, None], other=0.0).to(tl.float32)  # [BT,BS]

        Bd = Bt * decay[:, None]
        acc += tl.dot(tl.trans(x), Bd, input_precision=_PREC)  # [P,BS]

    cs_base = ((b * NC + c) * H + h) * P * N
    cs_off = cs_base + d[:, None] * N + s[None, :]
    tl.store(CS_ptr + cs_off, acc)


@triton.jit
def _state_passing_kernel(
    INIT_ptr, CS_ptr, DCHUNK_ptr, STATES_IN_ptr, FINAL_ptr,
    S, NC,
    H: tl.constexpr, P: tl.constexpr, N: tl.constexpr, BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    npb = P // BLOCK_P
    pblock = pid % npb
    tmp = pid // npb
    h = tmp % H
    b = tmp // H

    d = pblock * BLOCK_P + tl.arange(0, BLOCK_P)
    s = tl.arange(0, N)

    init_off = ((b * H + h) * P + d[:, None]) * N + s[None, :]
    R = tl.load(INIT_ptr + init_off).to(tl.float32)  # [BLOCK_P, N] = R_0

    for c in range(0, NC):
        si_base = ((b * NC + c) * H + h) * P * N
        si_off = si_base + d[:, None] * N + s[None, :]
        tl.store(STATES_IN_ptr + si_off, R)  # states_in[c] = R_c

        cs = tl.load(CS_ptr + si_off).to(tl.float32)  # chunk_state_c
        gamma = tl.exp(tl.load(DCHUNK_ptr + (b * H + h) * NC + c))
        R = cs + gamma * R  # R_{c+1}

    fin_off = ((b * H + h) * P + d[:, None]) * N + s[None, :]
    tl.store(FINAL_ptr + fin_off, R.to(tl.bfloat16))  # R_NC


@triton.jit
def _chunk_scan_kernel(
    X_ptr, B_ptr, C_ptr, D_ptr, ACS_ptr, STATES_IN_ptr, OUT_ptr,
    S, NC,
    H: tl.constexpr, P: tl.constexpr, N: tl.constexpr, Q: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid0 = tl.program_id(0)
    mblock = tl.program_id(1)

    h = pid0 % H
    tmp = pid0 // H
    c = tmp % NC
    b = tmp // NC

    i0 = mblock * BLOCK_M
    rows = i0 + tl.arange(0, BLOCK_M)  # local chunk row indices [BM]
    d = tl.arange(0, P)

    seq_i = c * Q + rows
    seq_i_mask = seq_i < S

    acs_base = ((b * H + h) * NC + c) * Q
    acs_i = tl.load(ACS_ptr + acs_base + rows)  # [BM]

    # x_i [BM, P]
    xi_off = (seq_i[:, None] * H + h) * P + d[None, :]
    x_i = tl.load(X_ptr + (b * S) * H * P + xi_off, mask=seq_i_mask[:, None], other=0.0).to(tl.float32)

    # --- Y_off = (C_i . R_c^T) * exp(acs_i), contract over N in BLOCK_K tiles ---
    r_base = ((b * NC + c) * H + h) * P * N
    yoff = tl.zeros((BLOCK_M, P), dtype=tl.float32)
    for k0 in range(0, N, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        ci_off = (b * S + seq_i)[:, None] * N + ks[None, :]
        C_ik = tl.load(C_ptr + ci_off, mask=seq_i_mask[:, None], other=0.0).to(tl.float32)  # [BM,BK]
        r_off = r_base + d[:, None] * N + ks[None, :]
        R_ck = tl.load(STATES_IN_ptr + r_off).to(tl.float32)  # [P,BK]
        yoff += tl.dot(C_ik, tl.trans(R_ck), input_precision=_PREC)  # [BM,P]
    y = yoff * tl.exp(acs_i)[:, None]

    # --- Y_diag = sum_{jblock<=mblock} (C_i . B_j^T) * L(i,j) * x_j ---
    for jblock in range(0, mblock + 1):
        j0 = jblock * BLOCK_N
        cols = j0 + tl.arange(0, BLOCK_N)  # [BN]
        seq_j = c * Q + cols
        seq_j_mask = seq_j < S
        acs_j = tl.load(ACS_ptr + acs_base + cols)  # [BN]

        CB = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, N, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            ci_off = (b * S + seq_i)[:, None] * N + ks[None, :]
            C_ik = tl.load(C_ptr + ci_off, mask=seq_i_mask[:, None], other=0.0).to(tl.float32)  # [BM,BK]
            bj_off = (b * S + seq_j)[:, None] * N + ks[None, :]
            B_jk = tl.load(B_ptr + bj_off, mask=seq_j_mask[:, None], other=0.0).to(tl.float32)  # [BN,BK]
            CB += tl.dot(C_ik, tl.trans(B_jk), input_precision=_PREC)  # [BM,BN]

        L = tl.exp(acs_i[:, None] - acs_j[None, :])  # [BM,BN]
        causal = rows[:, None] >= cols[None, :]
        M = tl.where(causal, CB * L, 0.0)

        xj_off = (seq_j[:, None] * H + h) * P + d[None, :]
        x_j = tl.load(X_ptr + (b * S) * H * P + xj_off, mask=seq_j_mask[:, None], other=0.0).to(tl.float32)  # [BN,P]
        y += tl.dot(M, x_j, input_precision=_PREC)  # [BM,P]

    # D residual (fp32) then cast
    Dh = tl.load(D_ptr + h).to(tl.float32)
    y += Dh * x_i

    HP = H * P
    out_off = (b * S + seq_i)[:, None] * HP + (h * P + d[None, :])
    tl.store(OUT_ptr + out_off, y.to(tl.bfloat16), mask=seq_i_mask[:, None])


@torch.no_grad()
def run(hidden_states, A, B, C, D, initial_states):
    x = hidden_states.contiguous()
    A_c = A.contiguous()
    Bm = B.contiguous()
    Cm = C.contiguous()
    Dv = D.contiguous()
    init = initial_states.contiguous()

    bsz, S, H, P = x.shape
    N = Bm.shape[-1]
    Q = _Q
    assert H == _H and P == _P and N == _N

    pad = (Q - S % Q) % Q
    S_pad = S + pad
    NC = S_pad // Q

    dev = x.device
    f32 = torch.float32
    acs = torch.empty((bsz, H, NC, Q), device=dev, dtype=f32)
    dchunk = torch.empty((bsz, H, NC), device=dev, dtype=f32)
    chunk_states = torch.empty((bsz, NC, H, P, N), device=dev, dtype=f32)
    states_in = torch.empty((bsz, NC, H, P, N), device=dev, dtype=f32)
    output = torch.empty((bsz, S, H * P), device=dev, dtype=torch.bfloat16)
    final_state = torch.empty((bsz, H, P, N), device=dev, dtype=torch.bfloat16)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    BLOCK_S = 64
    BLOCK_P = 16
    BLOCK_T = 64

    # Stage 0: chunk cumsum
    _chunk_cumsum_kernel[(bsz * H * NC,)](
        A_c, acs, dchunk, S, NC, H=H, Q=Q, num_warps=4,
    )
    # Stage 1: chunk state (grid tiled over N)
    _chunk_state_kernel[(bsz * NC * H, N // BLOCK_S)](
        x, Bm, acs, dchunk, chunk_states, S, NC,
        H=H, P=P, N=N, Q=Q, BLOCK_T=BLOCK_T, BLOCK_S=BLOCK_S, num_warps=4,
    )
    # Stage 2: state passing (sequential over chunks)
    _state_passing_kernel[(bsz * H * (P // BLOCK_P),)](
        init, chunk_states, dchunk, states_in, final_state, S, NC,
        H=H, P=P, N=N, BLOCK_P=BLOCK_P, num_warps=4,
    )
    # Stage 3: chunk scan (parallel over b,c,h,row-block; N contraction tiled)
    _chunk_scan_kernel[(bsz * NC * H, Q // BLOCK_M)](
        x, Bm, Cm, Dv, acs, states_in, output, S, NC,
        H=H, P=P, N=N, Q=Q, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8,
    )

    return output, final_state
