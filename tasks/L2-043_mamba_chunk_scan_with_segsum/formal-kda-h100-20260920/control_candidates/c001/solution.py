import torch
import triton
import triton.language as tl

# =============================================================================
# L2/043 Mamba-2 chunk scan with segment sum  (H100 / sm_90)
#
# Candidate c001 -- correctness-first fused Triton baseline (Phase 0 of plan).
#
# Fixed problem constants (definition.json):
#   H (num_heads) = 16, P (head_dim) = 64, N (state_size) = 256,
#   G (n_groups) = 1, Q (chunk_size) = 256.
# Variable: batch (bsz), seq_len (S).
#
# 4-stage decomposition (all math/accumulation in fp32, matching reference):
#   0. chunk_cumsum : acs = cumsum_t(A) per chunk (pad->0); dchunk = acs[Q-1]
#   1. chunk_state  : cs[d,s] = sum_t x[t,d]*B[t,s]*exp(dchunk-acs[t])
#   2. state_passing: R_0=init; R_{c+1}=cs_c+exp(dchunk_c)*R_c; states_in[c]=R_c;
#                     final_state = R_NC
#   3. chunk_scan   : Y_diag = sum_{j<=i} (C_i.B_j)*exp(acs_i-acs_j) * x_j
#                     Y_off  = (C_i . R_c) * exp(acs_i)
#                     y = Y_diag + Y_off + D[h]*x_i   -> [B,S,H*P] bf16
#
# Triton-only compute; torch used only for metadata / buffer allocation / launch.
# =============================================================================

_H = 16
_P = 64
_N = 256
_Q = 256

# Full-precision fp32 matmul for the correctness-first baseline.
_PREC = "ieee"


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
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid % H
    tmp = pid // H
    c = tmp % NC
    b = tmp // NC

    d = tl.arange(0, P)
    s = tl.arange(0, N)
    acs_base = ((b * H + h) * NC + c) * Q
    dchunk = tl.load(DCHUNK_ptr + (b * H + h) * NC + c)  # scalar

    acc = tl.zeros((P, N), dtype=tl.float32)
    for t0 in range(0, Q, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        seq = c * Q + t
        seq_mask = seq < S
        acs = tl.load(ACS_ptr + acs_base + t)  # [BLOCK_T]
        decay = tl.exp(dchunk - acs)           # [BLOCK_T]

        x_off = ((b * S + seq)[:, None] * H + h) * P + d[None, :]
        x = tl.load(X_ptr + x_off, mask=seq_mask[:, None], other=0.0).to(tl.float32)  # [BT,P]
        b_off = (b * S + seq)[:, None] * N + s[None, :]
        Bt = tl.load(B_ptr + b_off, mask=seq_mask[:, None], other=0.0).to(tl.float32)  # [BT,N]

        Bd = Bt * decay[:, None]
        acc += tl.dot(tl.trans(x), Bd, input_precision=_PREC)  # [P,N]

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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
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
    s = tl.arange(0, N)

    seq_i = c * Q + rows
    seq_i_mask = seq_i < S

    acs_base = ((b * H + h) * NC + c) * Q
    acs_i = tl.load(ACS_ptr + acs_base + rows)  # [BM]

    # C_i [BM, N]
    ci_off = (b * S + seq_i)[:, None] * N + s[None, :]
    C_i = tl.load(C_ptr + ci_off, mask=seq_i_mask[:, None], other=0.0).to(tl.float32)
    # x_i [BM, P]
    xi_off = ((b * S + seq_i)[:, None] * H + h) * P + d[None, :]
    x_i = tl.load(X_ptr + xi_off, mask=seq_i_mask[:, None], other=0.0).to(tl.float32)

    # R_c [P, N] entering-state
    r_base = ((b * NC + c) * H + h) * P * N
    r_off = r_base + d[:, None] * N + s[None, :]
    R_c = tl.load(STATES_IN_ptr + r_off).to(tl.float32)

    # Y_off = (C_i . R_c^T) * exp(acs_i)
    y = tl.dot(C_i, tl.trans(R_c), input_precision=_PREC)  # [BM, P]
    y = y * tl.exp(acs_i)[:, None]

    # Y_diag = sum_{jblock<=mblock} (C_i . B_j^T) * L(i,j) * x_j
    for jblock in range(0, mblock + 1):
        j0 = jblock * BLOCK_N
        cols = j0 + tl.arange(0, BLOCK_N)  # [BN]
        seq_j = c * Q + cols
        seq_j_mask = seq_j < S

        acs_j = tl.load(ACS_ptr + acs_base + cols)  # [BN]
        bj_off = (b * S + seq_j)[:, None] * N + s[None, :]
        B_j = tl.load(B_ptr + bj_off, mask=seq_j_mask[:, None], other=0.0).to(tl.float32)
        xj_off = ((b * S + seq_j)[:, None] * H + h) * P + d[None, :]
        x_j = tl.load(X_ptr + xj_off, mask=seq_j_mask[:, None], other=0.0).to(tl.float32)

        CB = tl.dot(C_i, tl.trans(B_j), input_precision=_PREC)  # [BM, BN]
        L = tl.exp(acs_i[:, None] - acs_j[None, :])             # [BM, BN]
        causal = rows[:, None] >= cols[None, :]
        M = tl.where(causal, CB * L, 0.0)
        y += tl.dot(M, x_j, input_precision=_PREC)             # [BM, P]

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
    BLOCK_P = 16
    BLOCK_T = 64

    # Stage 0: chunk cumsum
    _chunk_cumsum_kernel[(bsz * H * NC,)](
        A_c, acs, dchunk, S, NC, H=H, Q=Q, num_warps=4,
    )
    # Stage 1: chunk state
    _chunk_state_kernel[(bsz * NC * H,)](
        x, Bm, acs, dchunk, chunk_states, S, NC,
        H=H, P=P, N=N, Q=Q, BLOCK_T=BLOCK_T, num_warps=8,
    )
    # Stage 2: state passing (sequential over chunks)
    _state_passing_kernel[(bsz * H * (P // BLOCK_P),)](
        init, chunk_states, dchunk, states_in, final_state, S, NC,
        H=H, P=P, N=N, BLOCK_P=BLOCK_P, num_warps=4,
    )
    # Stage 3: chunk scan (parallel over b,c,h,row-block)
    _chunk_scan_kernel[(bsz * NC * H, Q // BLOCK_M)](
        x, Bm, Cm, Dv, acs, states_in, output, S, NC,
        H=H, P=P, N=N, Q=Q, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=8,
    )

    return output, final_state
