import torch
import triton
import triton.language as tl


@triton.jit
def expand_repeat_interleave(B_ptr, B_exp_ptr, C_ptr, C_exp_ptr,
                             Bsz, Csz, S, N_GROUPS, NUM_HEADS, K,
                             B_stride_b, B_stride_c, B_stride_s, B_stride_ng, B_stride_k,
                             C_stride_b, C_stride_c, C_stride_s, C_stride_ng, C_stride_k,
                             B_exp_stride_b, B_exp_stride_c, B_exp_stride_s, B_exp_stride_n, B_exp_stride_k,
                             C_exp_stride_b, C_exp_stride_c, C_exp_stride_s, C_exp_stride_n, C_exp_stride_k,
                             REPEAT: tl.constexpr):
    # grid = (Bsz, Csz, S, N_GROUPS, REPEAT)
    b = tl.program_id(0)
    c = tl.program_id(1)
    s = tl.program_id(2)
    ng = tl.program_id(3)
    r = tl.program_id(4)

    n = ng * REPEAT + r

    # Load B[ng] and store to B[n]
    B_off = b * B_stride_b + c * B_stride_c + s * B_stride_s + ng * B_stride_ng
    val = tl.load(B_ptr + B_off)
    B_exp_off = b * B_exp_stride_b + c * B_exp_stride_c + s * B_exp_stride_s + n * B_exp_stride_n
    tl.store(B_exp_ptr + B_exp_off, val)

    # Load C[ng] and store to C[n]
    C_off = c * C_stride_c + s * C_stride_s + ng * C_stride_ng
    C_off = b * C_stride_b + c * C_stride_c + s * C_stride_s + ng * C_stride_ng
    C_val = tl.load(C_ptr + C_off)
    C_exp_off = b * C_exp_stride_b + c * C_exp_stride_c + s * C_exp_stride_s + n * C_exp_stride_n
    tl.store(C_exp_ptr + C_exp_off, C_val)


@triton.jit
def masked_cumsum_lower_exp_L(L_ptr, A_ptr,
                              Bsz, N, Csz, S,
                              L_stride_b, L_stride_n, L_stride_c, L_stride_i, L_stride_j,
                              A_stride_b, A_stride_n, A_stride_c, A_stride_s):
    # grid = (Bsz, N, Csz, S)
    b = tl.program_id(0)
    n = tl.program_id(1)
    c = tl.program_id(2)
    i = tl.program_id(3)

    # running sum along j (source), include j < i (diagonal=-1), exclude j == i
    running = 0.0
    for j in range(0, S):
        # load A[b, n, c, j]
        A_off = b * A_stride_b + n * A_stride_n + c * A_stride_c + j * A_stride_s
        a_j = tl.load(A_ptr + A_off)
        if j < i:
            running += a_j
        # L[b, n, c, i, j] = exp(running) if j < i else 0
        L_off = b * L_stride_b + n * L_stride_n + c * L_stride_c + i * L_stride_i + j * L_stride_j
        if j < i:
            tl.store(L_ptr + L_off, tl.exp(running))
        else:
            tl.store(L_ptr + L_off, 0.0)


@triton.jit
def contract_BC_to_G(B_exp_ptr, C_exp_ptr, G_ptr,
                     Bsz, Csz, S, N, K,
                     B_exp_stride_b, B_exp_stride_c, B_exp_stride_s, B_exp_stride_n, B_exp_stride_k,
                     C_exp_stride_b, C_exp_stride_c, C_exp_stride_s, C_exp_stride_n, C_exp_stride_k,
                     G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n):
    # grid = (Bsz, Csz, S, S, N)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    acc = 0.0
    for k in range(0, K):  # K is chunk_size, equals S in this setup
        B_off = b * B_exp_stride_b + c * B_exp_stride_c + j * B_exp_stride_s + n * B_exp_stride_n + k * B_exp_stride_k
        C_off = b * C_exp_stride_b + c * C_exp_stride_c + i * C_exp_stride_s + n * C_exp_stride_n + k * C_exp_stride_k
        b_val = tl.load(B_exp_ptr + B_off)
        c_val = tl.load(C_exp_ptr + C_off)
        acc += b_val * c_val

    G_off = b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    tl.store(G_ptr + G_off, acc)


@triton.jit
def elementwise_mul_M(G_ptr, L_ptr, M_ptr,
                      Bsz, Csz, S, N,
                      G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
                      L_stride_b, L_stride_n, L_stride_c, L_stride_i, L_stride_j,
                      M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n):
    # grid = (Bsz, Csz, S, S, N)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    G_off = b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    L_off = b * L_stride_b + n * L_stride_n + c * L_stride_c + i * L_stride_i + j * L_stride_j
    G_val = tl.load(G_ptr + G_off)
    L_val = tl.load(L_ptr + L_off)
    M_val = G_val * L_val
    M_off = b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n
    tl.store(M_ptr + M_off, M_val)


@triton.jit
def diag_contract_Y(M_ptr, HS_ptr, Y_ptr,
                    Bsz, Csz, S, N, D,
                    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
                    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
                    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d):
    # grid = (Bsz, Csz, S, N, D)
    # For each (b, c, i, n, d), accumulate over j
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in range(0, S):
        M_off = b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n
        HS_off = b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d
        M_val = tl.load(M_ptr + M_off)
        HS_val = tl.load(HS_ptr + HS_off)
        acc += M_val * HS_val

    Y_off = b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d
    tl.store(Y_ptr + Y_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on the same device and dtype float32 for compute
        device = hidden_states.device
        Bsz, Csz, S, N, D = hidden_states.shape
        N_GROUPS = 8
        NUM_HEADS = 32
        REPEAT = NUM_HEADS // N_GROUPS  # 4

        # Expand B and C from N_GROUPS to NUM_HEADS using Triton kernel
        B_exp = torch.empty((Bsz, Csz, S, NUM_HEADS, hidden_states.shape[4]), device=device, dtype=torch.float32)
        C_exp = torch.empty((Bsz, Csz, S, NUM_HEADS, hidden_states.shape[4]), device=device, dtype=torch.float32)

        # Strides
        B_stride_b, B_stride_c, B_stride_s, B_stride_ng, B_stride_k = B.stride()
        C_stride_b, C_stride_c, C_stride_s, C_stride_ng, C_stride_k = C.stride()
        B_exp_stride_b, B_exp_stride_c, B_exp_stride_s, B_exp_stride_n, B_exp_stride_k = B_exp.stride()
        C_exp_stride_b, C_exp_stride_c, C_exp_stride_s, C_exp_stride_n, C_exp_stride_k = C_exp.stride()

        # Launch expand_repeat_interleave with grid = (Bsz, Csz, S, N_GROUPS, REPEAT)
        grid_expand = (Bsz, Csz, S, N_GROUPS, REPEAT)
        expand_repeat_interleave[grid_expand](
            B, C, B_exp, C_exp,
            Bsz, Csz, S, N_GROUPS, NUM_HEADS, hidden_states.shape[4],
            B_stride_b, B_stride_c, B_stride_s, B_stride_ng, B_stride_k,
            C_stride_b, C_stride_c, C_stride_s, C_stride_ng, C_stride_k,
            B_exp_stride_b, B_exp_stride_c, B_exp_stride_s, B_exp_stride_n, B_exp_stride_k,
            C_exp_stride_b, C_exp_stride_c, C_exp_stride_s, C_exp_stride_n, C_exp_stride_k,
            REPEAT=REPEAT, num_warps=1, num_stages=1
        )

        # Compute L[b, n, c, i, j] via masked cumsum (exclude diagonal), then exp
        L = torch.empty((Bsz, NUM_HEADS, Csz, S, S), device=device, dtype=torch.float32)
        L_stride_b, L_stride_n, L_stride_c, L_stride_i, L_stride_j = L.stride()
        A_stride_b, A_stride_n, A_stride_c, A_stride_s = A_cumsum.stride()

        grid_L = (Bsz, NUM_HEADS, Csz, S)
        masked_cumsum_lower_exp_L[grid_L](
            L, A_cumsum,
            Bsz, NUM_HEADS, Csz, S,
            L_stride_b, L_stride_n, L_stride_c, L_stride_i, L_stride_j,
            A_stride_b, A_stride_n, A_stride_c, A_stride_s,
            num_warps=1, num_stages=1
        )

        # Compute G[b, c, i, j, n] = sum over k of C_exp[b, c, i, n, k] * B_exp[b, c, j, n, k]
        G = torch.empty((Bsz, Csz, S, S, NUM_HEADS), device=device, dtype=torch.float32)
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()

        grid_G = (Bsz, Csz, S, S, NUM_HEADS)
        contract_BC_to_G[grid_G](
            B_exp, C_exp, G,
            Bsz, Csz, S, NUM_HEADS, hidden_states.shape[4],  # K equals S
            B_exp_stride_b, B_exp_stride_c, B_exp_stride_s, B_exp_stride_n, B_exp_stride_k,
            C_exp_stride_b, C_exp_stride_c, C_exp_stride_s, C_exp_stride_n, C_exp_stride_k,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            num_warps=1, num_stages=1
        )

        # Compute M = G * L (elementwise) via Triton
        M = torch.empty_like(G)
        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        L2 = L  # reuse L

        grid_M = (Bsz, Csz, S, S, NUM_HEADS)
        elementwise_mul_M[grid_M](
            G, L2, M,
            Bsz, Csz, S, NUM_HEADS,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            L_stride_b, L_stride_n, L_stride_c, L_stride_i, L_stride_j,
            M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
            num_warps=1, num_stages=1
        )

        # Compute Y_diag[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
        Y = torch.empty((Bsz, Csz, S, NUM_HEADS, D), device=device, dtype=torch.float32)

        HS = hidden_states
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = HS.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        grid_Y = (Bsz, Csz, S, NUM_HEADS, D)
        diag_contract_Y[grid_Y](
            M, HS, Y,
            Bsz, Csz, S, NUM_HEADS, D,
            M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
            HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
            Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 as original returns
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
