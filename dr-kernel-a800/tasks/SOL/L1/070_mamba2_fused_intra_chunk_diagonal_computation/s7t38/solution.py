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
    # Each program handles one (b, c, n) and vectorizes over i
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)
    i_vec = tl.arange(0, S)  # vector of i positions

    # Running sum per i: we compute cumsum along j (source) with mask j < i
    running = tl.zeros([S], dtype=tl.float32)

    for j in range(S):
        # Load A[b, c, j, n] for all i
        a = tl.load(
            A_ptr + pid_b * A_stride_b + pid_c * A_stride_c + j * A_stride_j + pid_n * A_stride_n,
            mask=True,
            other=0.0
        )  # scalar
        # Apply lower-triangular mask: include j < i
        include = j < i_vec  # vector
        a_vec = tl.where(include, a, 0.0)  # vectorized masked value
        running += a_vec  # cumsum along j
        # Store exp(running) to L[b, c, i, j, n]
        tl.store(
            L_ptr + pid_b * L_stride_b + pid_c * L_stride_c + i_vec * L_stride_i + j * L_stride_j + pid_n * L_stride_n,
            tl.exp(running),
            mask=(i_vec < S)
        )


@triton.jit
def contract_BC_to_G_kernel(
    B_ptr,  # [B, C, S, NUM_HEADS, K]
    C_ptr,  # [B, C, S, NUM_HEADS, K]
    G_ptr,  # [B, C, S, S, NUM_HEADS]
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, NUM_HEADS: tl.constexpr, K: tl.constexpr,
    B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
    C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
):
    # 5D grid: (b, c, i, j, n)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_n = tl.program_id(4)

    acc = tl.zeros([1], dtype=tl.float32)
    # Sum over k (state_size)
    for k in range(K):
        bj = tl.load(B_ptr + pid_b * B_stride_b + pid_c * B_stride_c + pid_j * B_stride_j + pid_n * B_stride_n + k * B_stride_k)
        ci = tl.load(C_ptr + pid_b * C_stride_b + pid_c * C_stride_c + pid_i * C_stride_i + pid_n * C_stride_n + k * C_stride_k)
        acc += bj * ci
    # Store G[b, c, i, j, n]
    tl.store(G_ptr + pid_b * G_stride_b + pid_c * G_stride_c + pid_i * G_stride_i + pid_j * G_stride_j + pid_n * G_stride_n, acc)


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
    # 5D grid: (b, c, i, j, n)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_n = tl.program_id(4)

    g = tl.load(G_ptr + pid_b * G_stride_b + pid_c * G_stride_c + pid_i * G_stride_i + pid_j * G_stride_j + pid_n * G_stride_n)
    l = tl.load(L_ptr + pid_b * L_stride_b + pid_c * L_stride_c + pid_i * L_stride_i + pid_j * L_stride_j + pid_n * L_stride_n)
    m = g * l
    tl.store(M_ptr + pid_b * M_stride_b + pid_c * M_stride_c + pid_i * M_stride_i + pid_j * M_stride_j + pid_n * M_stride_n, m)


@triton.jit
def diag_contract_Y_kernel(
    M_ptr,  # [B, C, S, S, NUM_HEADS]
    HS_ptr,  # [B, C, S, NUM_HEADS, D]
    Y_ptr,  # [B, C, S, NUM_HEADS, D]
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, NUM_HEADS: tl.constexpr, D: tl.constexpr,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
    d_const: tl.constexpr,
):
    # Grid: (B, C, S, NUM_HEADS, D); vectorize over j
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_n = tl.program_id(3)
    pid_d = tl.program_id(4)

    acc = tl.zeros([1], dtype=tl.float32)
    for j in range(S):
        m_val = tl.load(M_ptr + pid_b * M_stride_b + pid_c * M_stride_c + pid_i * M_stride_i + j * M_stride_j + pid_n * M_stride_n)
        hs_val = tl.load(HS_ptr + pid_b * HS_stride_b + pid_c * HS_stride_c + j * HS_stride_j + pid_n * HS_stride_n + d_const * HS_stride_d)
        acc += m_val * hs_val
    tl.store(Y_ptr + pid_b * Y_stride_b + pid_c * Y_stride_c + pid_i * Y_stride_i + pid_n * Y_stride_n + d_const * Y_stride_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Infer shapes from hidden_states
        Bsz = hidden_states.shape[0]  # batch_size (NUM_HEADS in original)
        Csz = hidden_states.shape[1]  # num_chunks
        S = hidden_states.shape[2]    # chunk_size
        N = hidden_states.shape[3]    # num_heads (used for A_cumsum and output)
        D = hidden_states.shape[4]    # head_dim

        # Ensure all tensors are on the same device and contiguous
        device = hidden_states.device
        B = B.to(device).contiguous()
        C = C.to(device).contiguous()
        A = A_cumsum.to(device).contiguous()
        HS = hidden_states.to(torch.float32).contiguous()

        # Expand A to [B, C, S, S, N] with lower-triangular mask and cumsum along j, then exp -> L
        L = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        A_stride_b, A_stride_c, A_stride_j, A_stride_n = A.stride()
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = L.stride()

        grid_A = (Bsz, Csz, N)
        masked_cumsum_lower_exp_kernel[grid_A](
            A, L,
            Bsz=Bsz, Csz=Csz, S=S, N=N,
            A_stride_b=A_stride_b, A_stride_c=A_stride_c, A_stride_j=A_stride_j, A_stride_n=A_stride_n,
            L_stride_b=L_stride_b, L_stride_c=L_stride_c, L_stride_i=L_stride_i, L_stride_j=L_stride_j, L_stride_n=L_stride_n,
            num_warps=1, num_stages=1
        )

        # Expand B and C from n_groups to NUM_HEADS=32 (original signature) via repeat_interleave factor = 4
        # Note: B and C are [B, C, S, n_groups, K]
        n_groups = B.shape[3]
        assert n_groups == C.shape[3], "B and C must have same n_groups"
        K = B.shape[4]  # state_size (equals hidden_states.shape[4] which is D)

        # Repeat_interleave along n_groups dimension to NUM_HEADS=32
        B_exp = B.repeat_interleave(4, dim=3).contiguous()  # [B, C, S, 32, K]
        C_exp = C.repeat_interleave(4, dim=3).contiguous()  # [B, C, S, 32, K]
        assert B_exp.shape[3] == 32 and C_exp.shape[3] == 32, "Expansion to NUM_HEADS=32 failed"

        # Allocate G: [B, C, S, S, NUM_HEADS] where NUM_HEADS=32 for B_exp/C_exp
        G = torch.empty((Bsz, Csz, S, S, 32), device=device, dtype=torch.float32)

        B_exp_stride_b, B_exp_stride_c, B_exp_stride_j, B_exp_stride_n, B_exp_stride_k = B_exp.stride()
        C_exp_stride_b, C_exp_stride_c, C_exp_stride_i, C_exp_stride_n, C_exp_stride_k = C_exp.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()

        grid_G = (Bsz, Csz, S, S, 32)
        contract_BC_to_G_kernel[grid_G](
            B_exp, C_exp, G,
            Bsz=Bsz, Csz=Csz, S=S, NUM_HEADS=32, K=K,
            B_stride_b=B_exp_stride_b, B_stride_c=B_exp_stride_c, B_stride_j=B_exp_stride_j, B_stride_n=B_exp_stride_n, B_stride_k=B_exp_stride_k,
            C_stride_b=C_exp_stride_b, C_stride_c=C_exp_stride_c, C_stride_i=C_exp_stride_i, C_stride_n=C_exp_stride_n, C_stride_k=C_exp_stride_k,
            G_stride_b=G_stride_b, G_stride_c=G_stride_c, G_stride_i=G_stride_i, G_stride_j=G_stride_j, G_stride_n=G_stride_n,
            num_warps=1, num_stages=1
        )

        # Elementwise M = G * L
        M = torch.empty_like(G)
        G_stride_bM, G_stride_cM, G_stride_iM, G_stride_jM, G_stride_nM = G.stride()
        L_stride_bM, L_stride_cM, L_stride_iM, L_stride_jM, L_stride_nM = L.stride()
        M_stride_bM, M_stride_cM, M_stride_iM, M_stride_jM, M_stride_nM = M.stride()

        grid_M = (Bsz, Csz, S, S, 32)
        elementwise_mul_LG_kernel[grid_M](
            G, L, M,
            Bsz=Bsz, Csz=Csz, S=S, NUM_HEADS=32,
            G_stride_b=G_stride_bM, G_stride_c=G_stride_cM, G_stride_i=G_stride_iM, G_stride_j=G_stride_jM, G_stride_n=G_stride_nM,
            L_stride_b=L_stride_bM, L_stride_c=L_stride_cM, L_stride_i=L_stride_iM, L_stride_j=L_stride_jM, L_stride_n=L_stride_nM,
            M_stride_b=M_stride_bM, M_stride_c=M_stride_cM, M_stride_i=M_stride_iM, M_stride_j=M_stride_jM, M_stride_n=M_stride_nM,
            num_warps=1, num_stages=1
        )

        # Final diagonal contraction: Y_diag [B, C, S, NUM_HEADS, D]
        Y = torch.empty((Bsz, Csz, S, 32, D), device=device, dtype=torch.float32)

        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = HS.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        grid_Y = (Bsz, Csz, S, 32, D)
        # Launch per d
        for d in range(D):
            diag_contract_Y_kernel[grid_Y](
                M, HS, Y,
                Bsz=Bsz, Csz=Csz, S=S, NUM_HEADS=32, D=D,
                M_stride_b=M_stride_b, M_stride_c=M_stride_c, M_stride_i=M_stride_i, M_stride_j=M_stride_j, M_stride_n=M_stride_n,
                HS_stride_b=HS_stride_b, HS_stride_c=HS_stride_c, HS_stride_j=HS_stride_j, HS_stride_n=HS_stride_n, HS_stride_d=HS_stride_d,
                Y_stride_b=Y_stride_b, Y_stride_c=Y_stride_c, Y_stride_i=Y_stride_i, Y_stride_n=Y_stride_n, Y_stride_d=Y_stride_d,
                d_const=d,
                num_warps=2, num_stages=2
            )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
