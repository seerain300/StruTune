import torch
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_lower_exp_kernel(
    A_ptr,  # [B, C, S, N]
    L_ptr,  # [B, C, S, S, N]
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr,
    A_stride_b, A_stride_c, A_stride_j, A_stride_n,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
):
    # Grid: (Bsz, Csz, N)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Vectorize over i positions
    i_vec = tl.arange(0, S)

    # Running sum per i: cumsum along j with lower-triangular mask j < i
    running = tl.zeros([S], dtype=tl.float32)

    for j in range(S):
        # Load A[b, c, j, n] (scalar)
        a = tl.load(
            A_ptr + pid_b * A_stride_b + pid_c * A_stride_c + j * A_stride_j + pid_n * A_stride_n,
            mask=True,
            other=0.0
        )
        # Mask: include if j < i
        include = j < i_vec
        a_vec = tl.where(include, a, 0.0)
        running += a_vec
        # Store exp(running) to L[b, c, i, j, n]
        tl.store(
            L_ptr + pid_b * L_stride_b + pid_c * L_stride_c + i_vec * L_stride_i + j * L_stride_j + pid_n * L_stride_n,
            tl.exp(running),
            mask=(i_vec < S)  # ensure i stays within S
        )


@triton.jit
def contract_BC_to_G_kernel(
    B_exp_ptr,  # [B, C, S, NUM_HEADS, K]
    C_exp_ptr,  # [B, C, S, NUM_HEADS, K]
    G_ptr,      # [B, C, S, S, NUM_HEADS]
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, NUM_HEADS: tl.constexpr, K_CONST: tl.constexpr,
    B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
    C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
):
    # Grid: (Bsz, Csz, S, S, NUM_HEADS)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_n = tl.program_id(4)

    # Accumulate over K
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(K_CONST):
        b_val = tl.load(
            B_exp_ptr + pid_b * B_stride_b + pid_c * B_stride_c + pid_j * B_stride_j + pid_n * B_stride_n + k * B_stride_k,
            mask=True,
            other=0.0
        )
        c_val = tl.load(
            C_exp_ptr + pid_b * C_stride_b + pid_c * C_stride_c + pid_i * C_stride_i + pid_n * C_stride_n + k * C_stride_k,
            mask=True,
            other=0.0
        )
        acc += b_val * c_val

    # Store G[b, c, i, j, n] = acc
    tl.store(
        G_ptr + pid_b * G_stride_b + pid_c * G_stride_c + pid_i * G_stride_i + pid_j * G_stride_j + pid_n * G_stride_n,
        acc,
        mask=True
    )


@triton.jit
def elementwise_mul_LG_kernel(
    G_ptr,  # [B, C, S, S, NUM_HEADS]
    L_ptr,  # [B, C, S, S, NUM_HEADS]
    M_ptr,  # [B, C, S, S, NUM_HEADS]
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, NUM_HEADS: tl.constexpr,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
):
    # Grid: (Bsz, Csz, S, S, NUM_HEADS)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_n = tl.program_id(4)

    g = tl.load(
        G_ptr + pid_b * G_stride_b + pid_c * G_stride_c + pid_i * G_stride_i + pid_j * G_stride_j + pid_n * G_stride_n,
        mask=True,
        other=0.0
    )
    l = tl.load(
        L_ptr + pid_b * L_stride_b + pid_c * L_stride_c + pid_i * L_stride_i + pid_j * L_stride_j + pid_n * L_stride_n,
        mask=True,
        other=0.0
    )
    m = g * l
    tl.store(
        M_ptr + pid_b * M_stride_b + pid_c * M_stride_c + pid_i * M_stride_i + pid_j * M_stride_j + pid_n * M_stride_n,
        m,
        mask=True
    )


@triton.jit
def diag_contract_Y_kernel(
    M_ptr,     # [B, C, S, S, NUM_HEADS]
    HS_ptr,    # [B, C, S, NUM_HEADS, D]
    Y_ptr,     # [B, C, S, NUM_HEADS, D]
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, NUM_HEADS: tl.constexpr, D_CONST: tl.constexpr,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
    d_const: tl.constexpr,
):
    # Grid: (Bsz, Csz, S, NUM_HEADS, 1) — specialize per d_const
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_n = tl.program_id(3)
    pid_d = tl.program_id(4)

    # Initialize accumulator for this (b, c, i, n, d)
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over j and accumulate sum_j M[b, c, i, j, n] * HS[b, c, j, n, d]
    for j in range(S):
        m_val = tl.load(
            M_ptr + pid_b * M_stride_b + pid_c * M_stride_c + pid_i * M_stride_i + j * M_stride_j + pid_n * M_stride_n,
            mask=True,
            other=0.0
        )
        hs_val = tl.load(
            HS_ptr + pid_b * HS_stride_b + pid_c * HS_stride_c + j * HS_stride_j + pid_n * HS_stride_n + d_const * HS_stride_d,
            mask=True,
            other=0.0
        )
        acc += m_val * hs_val

    # Store acc to Y[b, c, i, n, d_const]
    tl.store(
        Y_ptr + pid_b * Y_stride_b + pid_c * Y_stride_c + pid_i * Y_stride_i + pid_n * Y_stride_n + d_const * Y_stride_d,
        acc,
        mask=True
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Extract axes from hidden_states (batch, chunks, chunk_size, num_heads, head_dim)
        Bsz = hidden_states.shape[0]
        Csz = hidden_states.shape[1]
        S = hidden_states.shape[2]
        N = hidden_states.shape[3]  # num_heads used for G, Y
        D = hidden_states.shape[4]  # head_dim

        device = hidden_states.device
        # 1) Compute L from A_cumsum: [Bsz, Csz, S, N] -> [Bsz, Csz, S, S, N]
        L = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        # Strides for A_cumsum and L
        A_stride_b, A_stride_c, A_stride_j, A_stride_n = A_cumsum.stride()
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = L.stride()

        # Launch Triton kernel for masked cumsum + exp
        masked_cumsum_lower_exp_kernel[(Bsz, Csz, N)](
            A_cumsum, L,
            Bsz, Csz, S, N,
            A_stride_b, A_stride_c, A_stride_j, A_stride_n,
            L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
            num_warps=1, num_stages=1
        )

        # 2) Expand B and C to NUM_HEADS=32 via repeat_interleave(4) (original signature)
        NUM_HEADS = 32
        B_exp = B.repeat_interleave(NUM_HEADS // 8, dim=3)  # from n_groups=8 -> num_heads=32
        C_exp = C.repeat_interleave(NUM_HEADS // 8, dim=3)

        # Ensure float32 for computation
        B_exp = B_exp.to(torch.float32)
        C_exp = C_exp.to(torch.float32)

        # 3) Compute G: [Bsz, Csz, S, S, NUM_HEADS]
        G = torch.empty((Bsz, Csz, S, S, NUM_HEADS), device=device, dtype=torch.float32)

        # Strides for expanded B/C and G
        B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k = B_exp.stride()
        C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k = C_exp.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()

        # K_CONST = hidden_states.shape[4] (head_dim)
        K_CONST = D

        contract_BC_to_G_kernel[(Bsz, Csz, S, S, NUM_HEADS)](
            B_exp, C_exp, G,
            Bsz, Csz, S, NUM_HEADS, K_CONST,
            B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
            C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            K_CONST=K_CONST,
            num_warps=4, num_stages=2
        )

        # 4) Compute M = G * L in Triton
        M = torch.empty_like(G, dtype=torch.float32)

        G_stride = G.stride()
        L_stride = L.stride()
        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()

        elementwise_mul_LG_kernel[(Bsz, Csz, S, S, NUM_HEADS)](
            G, L, M,
            Bsz, Csz, S, NUM_HEADS,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
            M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
            num_warps=4, num_stages=2
        )

        # 5) Diagonal contraction to Y_diag: [Bsz, Csz, S, NUM_HEADS, D]
        Y = torch.empty((Bsz, Csz, S, NUM_HEADS, D), device=device, dtype=torch.float32)

        M_stride = M.stride()
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = hidden_states.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        # Launch Triton kernel specialized per d dimension
        for d in range(D):
            diag_contract_Y_kernel[(Bsz, Csz, S, NUM_HEADS, 1)](
                M, hidden_states.to(torch.float32), Y,
                Bsz, Csz, S, NUM_HEADS, D,
                M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
                HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
                Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
                d_const=d,
                num_warps=2, num_stages=2
            )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
