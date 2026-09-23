import torch
import triton
import triton.language as tl


@triton.jit
def _mamba2_diag_group_kernel(
    X_ptr, A_ptr, B_ptr, C_ptr, Y_ptr,
    # hidden_states / Y strides: [batch, nchunks, chunk, nheads, headdim]
    sx_b, sx_c, sx_s, sx_h, sx_d,
    # A_cumsum strides: [batch, nheads, nchunks, chunk]
    sa_b, sa_h, sa_c, sa_s,
    # B strides: [batch, nchunks, chunk, ngroups, dstate]
    sb_b, sb_c, sb_s, sb_g, sb_n,
    # C strides: [batch, nchunks, chunk, ngroups, dstate]
    sc_b, sc_c, sc_s, sc_g, sc_n,
    # Y strides: [batch, nchunks, chunk, nheads, headdim]
    sy_b, sy_c, sy_s, sy_h, sy_d,
    num_chunks,
    HEADS_PER_GROUP: tl.constexpr,
    CHUNK: tl.constexpr,     # 128
    DSTATE: tl.constexpr,    # 128
    HEADDIM: tl.constexpr,   # 128
    NUM_GROUPS: tl.constexpr,
):
    pid = tl.program_id(0)
    g = pid % NUM_GROUPS
    tmp = pid // NUM_GROUPS
    c = tmp % num_chunks
    b = tmp // num_chunks

    offs_i = tl.arange(0, CHUNK)     # query rows i (rows of C / of L / output rows)
    offs_j = tl.arange(0, CHUNK)     # key/value rows j (rows of B and of X)
    offs_n = tl.arange(0, DSTATE)    # state contraction dim n
    offs_d = tl.arange(0, HEADDIM)   # feature dim d

    causal = offs_i[:, None] >= offs_j[None, :]

    # ---- Load Cmat [i, n] and Bmat [j, n] for group g (bf16); G shared by 4 heads ----
    c_base = C_ptr + b * sc_b + c * sc_c + g * sc_g
    Cmat = tl.load(c_base + offs_i[:, None] * sc_s + offs_n[None, :] * sc_n)
    b_base = B_ptr + b * sb_b + c * sb_c + g * sb_g
    Bmat = tl.load(b_base + offs_j[:, None] * sb_s + offs_n[None, :] * sb_n)

    # ---- G = Cmat @ Bmat^T -> [i, j] (bf16 MMA, fp32 accumulate); computed ONCE ----
    G = tl.dot(Cmat, tl.trans(Bmat))  # fp32 [i, j]

    # ---- Loop the HEADS_PER_GROUP heads of this group (unrolled) ----
    for hh in tl.static_range(HEADS_PER_GROUP):
        h = g * HEADS_PER_GROUP + hh

        # decay mask L[i,j] = exp(cA[i]-cA[j]) for i>=j else 0  (per-head)
        a_base = A_ptr + b * sa_b + h * sa_h + c * sa_c
        a = tl.load(a_base + offs_j * sa_s).to(tl.float32)  # [128]
        cA = tl.cumsum(a, axis=0)
        segsum = cA[:, None] - cA[None, :]
        L = tl.where(causal, tl.exp(segsum), 0.0)           # fp32 [i, j]

        M = G * L                                           # fp32 [i, j]

        x_base = X_ptr + b * sx_b + c * sx_c + h * sx_h
        X = tl.load(x_base + offs_j[:, None] * sx_s + offs_d[None, :] * sx_d)  # bf16 [j, d]
        Y = tl.dot(M.to(tl.bfloat16), X)                    # fp32 [i, d]

        y_base = Y_ptr + b * sy_b + c * sy_c + h * sy_h
        tl.store(y_base + offs_i[:, None] * sy_s + offs_d[None, :] * sy_d, Y.to(tl.bfloat16))


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    A_cumsum: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """Fused intra-chunk diagonal computation for Mamba2 SSD (Triton).

    Per-group fusion: one program per (batch b, chunk c, group g) computes the
    head-independent scores G = C[b,c,:,g,:] @ B[b,c,:,g,:]^T once, then loops the
    heads_per_group heads h of the group, each with its own decay mask L_h, mask
    M_h = G * L_h, and output Y_h = M_h @ hidden_states[b,c,:,h,:].
    """
    batch, nchunks, chunk, nheads, headdim = hidden_states.shape
    ngroups = B.shape[3]
    dstate = B.shape[4]
    heads_per_group = nheads // ngroups

    Y = torch.empty_like(hidden_states)

    grid = (batch * nchunks * ngroups,)
    _mamba2_diag_group_kernel[grid](
        hidden_states, A_cumsum, B, C, Y,
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        hidden_states.stride(3), hidden_states.stride(4),
        A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
        nchunks,
        HEADS_PER_GROUP=heads_per_group,
        CHUNK=chunk, DSTATE=dstate, HEADDIM=headdim, NUM_GROUPS=ngroups,
        num_warps=8, num_stages=2,
    )
    return Y
