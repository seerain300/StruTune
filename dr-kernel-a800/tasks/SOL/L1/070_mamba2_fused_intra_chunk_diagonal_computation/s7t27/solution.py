import torch
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_lower_exp(
    A: tl.pointer_type(tl.float32, ()),  # [B, C, S, N]
    L: tl.pointer_type(tl.float32, ()),  # [B, C, S, S, N]
    B_size: tl.int32, C_size: tl.int32, S: tl.int32, N: tl.int32,
    A_stride_b: tl.int32, A_stride_c: tl.int32, A_stride_j: tl.int32, A_stride_n: tl.int32,
    L_stride_b: tl.int32, L_stride_c: tl.int32, L_stride_i: tl.int32, L_stride_j: tl.int32, L_stride_n: tl.int32,
):
    # Grid: (B, C, S, S, N) -> each program computes L[b, c, i, j, n] for fixed (b,c,n,i), loops over j
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    # Running sum for j < i only (diagonal=-1). For j >= i, value is 0. Then exp(0) = 1 for excluded positions.
    running_sum = tl.zeros((), dtype=tl.float32)

    # Loop over source j from 0 to S-1
    for j_src in range(S):
        # Only include j_src < i
        include = j_src < i
        # Load A[b, c, j_src, n, i] if include else 0
        a_ptr = A + b * A_stride_b + c * A_stride_c + j_src * A_stride_j + n * A_stride_n
        a_val = tl.load(a_ptr, mask=include, other=0.0)
        running_sum += a_val

        # Store exp(running_sum) to L[b, c, i, j, n] if include, else 1.0
        # If j_src != j, we just skip storing; only one writer per (i,j) across loop.
        if include:
            l_ptr = L + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n
            tl.store(l_ptr, tl.exp(running_sum))


@triton.jit
def contract_BC_to_G(
    B_exp: tl.pointer_type(tl.float32, ()),  # [B, C, S, N, K]
    C_exp: tl.pointer_type(tl.float32, ()),  # [B, C, S, N, K]
    G: tl.pointer_type(tl.float32, ()),      # [B, C, S, S, N]
    B_size: tl.int32, C_size: tl.int32, S: tl.int32, N: tl.int32, K: tl.constexpr,
    B_stride_b: tl.int32, B_stride_c: tl.int32, B_stride_j: tl.int32, B_stride_n: tl.int32, B_stride_k: tl.int32,
    C_stride_b: tl.int32, C_stride_c: tl.int32, C_stride_i: tl.int32, C_stride_n: tl.int32, C_stride_k: tl.int32,
    G_stride_b: tl.int32, G_stride_c: tl.int32, G_stride_i: tl.int32, G_stride_j: tl.int32, G_stride_n: tl.int32,
):
    # Grid: (B, C, S, S, N)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    g_ptr = G + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    acc = tl.zeros((), dtype=tl.float32)

    # Sum over state dimension K
    for k in range(K):
        b_ptr = B_exp + b * B_stride_b + c * B_stride_c + j * B_stride_j + n * B_stride_n + k * B_stride_k
        c_ptr = C_exp + b * C_stride_b + c * C_stride_c + i * C_stride_i + n * C_stride_n + k * C_stride_k
        b_val = tl.load(b_ptr)
        c_val = tl.load(c_ptr)
        acc += b_val * c_val

    tl.store(g_ptr, acc)


@triton.jit
def elementwise_mul_M(
    G: tl.pointer_type(tl.float32, ()),  # [B, C, S, S, N]
    L: tl.pointer_type(tl.float32, ()),  # [B, C, S, S, N]
    M: tl.pointer_type(tl.float32, ()),  # [B, C, S, S, N]
    B_size: tl.int32, C_size: tl.int32, S: tl.int32, N: tl.int32,
    G_stride_b: tl.int32, G_stride_c: tl.int32, G_stride_i: tl.int32, G_stride_j: tl.int32, G_stride_n: tl.int32,
    L_stride_b: tl.int32, L_stride_c: tl.int32, L_stride_i: tl.int32, L_stride_j: tl.int32, L_stride_n: tl.int32,
    M_stride_b: tl.int32, M_stride_c: tl.int32, M_stride_i: tl.int32, M_stride_j: tl.int32, M_stride_n: tl.int32,
):
    # Grid: (B, C, S, S, N) elementwise M = G * L
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    g_ptr = G + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    l_ptr = L + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n
    m_ptr = M + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n

    g_val = tl.load(g_ptr)
    l_val = tl.load(l_ptr)
    m_val = g_val * l_val
    tl.store(m_ptr, m_val)


@triton.jit
def diag_contract_Y(
    M: tl.pointer_type(tl.float32, ()),          # [B, C, S, S, N]
    hidden: tl.pointer_type(tl.float32, ()),     # [B, C, S, N, D]
    Y: tl.pointer_type(tl.float32, ()),          # [B, C, S, N, D]
    B_size: tl.int32, C_size: tl.int32, S: tl.int32, N: tl.int32, D: tl.constexpr,
    M_stride_b: tl.int32, M_stride_c: tl.int32, M_stride_i: tl.int32, M_stride_j: tl.int32, M_stride_n: tl.int32,
    hidden_stride_b: tl.int32, hidden_stride_c: tl.int32, hidden_stride_j: tl.int32, hidden_stride_n: tl.int32, hidden_stride_d: tl.int32,
    Y_stride_b: tl.int32, Y_stride_c: tl.int32, Y_stride_i: tl.int32, Y_stride_n: tl.int32, Y_stride_d: tl.int32,
):
    # Grid: (B, C, S, N, D) each program handles one (b, c, i, n, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)

    # Sum over j (chunk positions)
    for j in range(S):
        m_ptr = M + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n
        h_ptr = hidden + b * hidden_stride_b + c * hidden_stride_c + j * hidden_stride_j + n * hidden_stride_n + d * hidden_stride_d
        m_val = tl.load(m_ptr)
        h_val = tl.load(h_ptr)
        acc += m_val * h_val

    y_ptr = Y + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d
    tl.store(y_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, C, S, N, D]
        device = hidden_states.device
        dtype = hidden_states.dtype  # typically float32

        Bsz, Csz, S, N, D = hidden_states.shape

        # Prepare expanded B and C to num_heads (N=32) by repeat_interleave(4)
        # Original: NUM_HEADS=32, N_GROUPS=8, so factor=4
        factor = 4
        B_exp = B.repeat_interleave(factor, dim=3).contiguous()  # [B, C, S, N, K]
        C_exp = C.repeat_interleave(factor, dim=3).contiguous()  # [B, C, S, N, K]

        # Allocate L: [B, C, S, S, N] as float32
        L = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        # Ensure A_cumsum is float32
        A = A_cumsum.to(torch.float32)

        # Launch masked cumsum + exp to produce L
        A_stride_b, A_stride_c, A_stride_j, A_stride_n = A.stride()
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = L.stride()

        grid_L = (Bsz, Csz, S, S, N)
        masked_cumsum_lower_exp[grid_L](
            A, L,
            B_size=Bsz, C_size=Csz, S=S, N=N,
            A_stride_b=A_stride_b, A_stride_c=A_stride_c, A_stride_j=A_stride_j, A_stride_n=A_stride_n,
            L_stride_b=L_stride_b, L_stride_c=L_stride_c, L_stride_i=L_stride_i, L_stride_j=L_stride_j, L_stride_n=L_stride_n,
            num_warps=1, num_stages=1
        )

        # Allocate G: [B, C, S, S, N] as float32
        G = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        # Launch contraction to compute G
        B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k = B_exp.stride()
        C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k = C_exp.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()

        grid_G = (Bsz, Csz, S, S, N)
        # K is head_dim (D) of hidden_states, but the original G contraction sums over the state_size dimension of B/C, which is not D.
        # In the original code, K is hidden_states.shape[4]; however, B/C have state_size different from head_dim. We must infer K from B.
        # To be safe, we take K from B_exp.shape[4].
        K = B_exp.shape[4]
        contract_BC_to_G[grid_G](
            B_exp, C_exp, G,
            B_size=Bsz, C_size=Csz, S=S, N=N, K=K,
            B_stride_b=B_stride_b, B_stride_c=B_stride_c, B_stride_j=B_stride_j, B_stride_n=B_stride_n, B_stride_k=B_stride_k,
            C_stride_b=C_stride_b, C_stride_c=C_stride_c, C_stride_i=C_stride_i, C_stride_n=C_stride_n, C_stride_k=C_stride_k,
            G_stride_b=G_stride_b, G_stride_c=G_stride_c, G_stride_i=G_stride_i, G_stride_j=G_stride_j, G_stride_n=G_stride_n,
            num_warps=1, num_stages=1
        )

        # Elementwise multiply M = G * L
        M = torch.empty_like(G)
        G_stride = G.stride()
        L_stride = L.stride()
        M_stride = M.stride()
        grid_M = (Bsz, Csz, S, S, N)
        elementwise_mul_M[grid_M](
            G, L, M,
            B_size=Bsz, C_size=Csz, S=S, N=N,
            G_stride_b=G_stride[0], G_stride_c=G_stride[1], G_stride_i=G_stride[2], G_stride_j=G_stride[3], G_stride_n=G_stride[4],
            L_stride_b=L_stride[0], L_stride_c=L_stride[1], L_stride_i=L_stride[2], L_stride_j=L_stride[3], L_stride_n=L_stride[4],
            M_stride_b=M_stride[0], M_stride_c=M_stride[1], M_stride_i=M_stride[2], M_stride_j=M_stride[3], M_stride_n=M_stride[4],
            num_warps=1, num_stages=1
        )

        # Final diagonal contraction to produce Y: [B, C, S, N, D]
        Y = torch.empty((Bsz, Csz, S, N, D), device=device, dtype=torch.float32)

        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        hidden_f32 = hidden_states.to(torch.float32)
        hidden_stride_b, hidden_stride_c, hidden_stride_j, hidden_stride_n, hidden_stride_d = hidden_f32.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y[grid_Y](
            M, hidden_f32, Y,
            B_size=Bsz, C_size=Csz, S=S, N=N, D=D,
            M_stride_b=M_stride_b, M_stride_c=M_stride_c, M_stride_i=M_stride_i, M_stride_j=M_stride_j, M_stride_n=M_stride_n,
            hidden_stride_b=hidden_stride_b, hidden_stride_c=hidden_stride_c, hidden_stride_j=hidden_stride_j, hidden_stride_n=hidden_stride_n, hidden_stride_d=hidden_stride_d,
            Y_stride_b=Y_stride_b, Y_stride_c=Y_stride_c, Y_stride_i=Y_stride_i, Y_stride_n=Y_stride_n, Y_stride_d=Y_stride_d,
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
