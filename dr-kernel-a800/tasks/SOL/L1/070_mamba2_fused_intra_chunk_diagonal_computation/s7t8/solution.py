import torch
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_lower_exp(
    A_ptr, L_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr,
    B_stride_b, B_stride_c, B_stride_i, B_stride_n, B_stride_j,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
    num_warps: tl.constexpr = 1, num_stages: tl.constexpr = 1,
):
    # Program ids for 5D grid: (b, c, i, n)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)

    # Running sum across j for masked cumsum
    running = 0.0
    for j in range(S):
        include = j < i  # diagonal=-1 semantics: include if j < i
        # A[b, c, i, j, n] load; if j>=i, mask out (include=False), then running += 0
        val = tl.load(A_ptr + b * B_stride_b + c * B_stride_c + i * B_stride_i + n * B_stride_n + j * B_stride_j)
        if include:
            running += val
        # store exp(running) into L[b, c, i, j, n]
        tl.store(L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n, tl.exp(running))


@triton.jit
def contract_BC_to_G(
    B_ptr, C_ptr, G_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    B_stride_b, B_stride_c, B_stride_i, B_stride_n, B_stride_k,
    C_stride_b, C_stride_c, C_stride_j, C_stride_n, C_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
    num_warps: tl.constexpr = 1, num_stages: tl.constexpr = 1,
):
    # 5D grid: (b, c, i, j, n)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    acc = 0.0
    for k in range(K):
        B_val = tl.load(B_ptr + b * B_stride_b + c * B_stride_c + i * B_stride_i + n * B_stride_n + k * B_stride_k)
        C_val = tl.load(C_ptr + b * C_stride_b + c * C_stride_c + j * C_stride_j + n * C_stride_n + k * C_stride_k)
        acc += B_val * C_val
    tl.store(G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n, acc)


@triton.jit
def diag_contract_Y(
    M_ptr, HS_ptr, Y_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
    num_warps: tl.constexpr = 1, num_stages: tl.constexpr = 1,
):
    # 5D grid: (b, c, i, n, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    # Accumulate over j: Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * HS[b, c, j, n, d]
    acc = 0.0
    for j in range(S):
        M_val = tl.load(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n)
        HS_val = tl.load(HS_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d)
        acc += M_val * HS_val
    tl.store(Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, NUM_HEADS: int = 32, N_GROUPS: int = 8):
        super().__init__()
        self.NUM_HEADS = NUM_HEADS
        self.N_GROUPS = N_GROUPS

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Infer shapes from hidden_states
        assert hidden_states.ndim == 5, "hidden_states must have shape [batch, num_chunks, chunk_size, num_heads, head_dim]"
        Bsz, Csz, S, N, D = hidden_states.shape

        # Prepare expanded B and C for num_heads
        # Original code uses repeat_interleave(NUM_HEADS // N_GROUPS), here NUM_HEADS=32, N_GROUPS=8 => repeat 4 times
        repeat_factor = self.NUM_HEADS // self.N_GROUPS
        assert self.NUM_HEADS % self.N_GROUPS == 0, "NUM_HEADS must be divisible by N_GROUPS"
        B_expanded = B.repeat_interleave(repeat_factor, dim=3)  # [B, C, S, N, D]
        C_expanded = C.repeat_interleave(repeat_factor, dim=3)  # [B, C, S, N, D]

        # 1) Compute L via Triton masked cumsum + exp
        # A_cumsum: [B, C, S, N]
        L = torch.empty((Bsz, Csz, S, S, N), device=hidden_states.device, dtype=torch.float32)

        B_stride_b, B_stride_c, B_stride_i, B_stride_n, B_stride_j = A_cumsum.stride()
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = L.stride()

        grid_L = (Bsz, Csz, S, N)
        masked_cumsum_lower_exp[grid_L](
            A_cumsum, L,
            Bsz=Bsz, Csz=Csz, S=S, N=N,
            B_stride_b=B_stride_b, B_stride_c=B_stride_c, B_stride_i=B_stride_i, B_stride_n=B_stride_n, B_stride_j=B_stride_j,
            L_stride_b=L_stride_b, L_stride_c=L_stride_c, L_stride_i=L_stride_i, L_stride_j=L_stride_j, L_stride_n=L_stride_n,
            num_warps=1, num_stages=1
        )

        # 2) Compute G via Triton contraction over K=D
        G = torch.empty((Bsz, Csz, S, S, N), device=hidden_states.device, dtype=torch.float32)

        B_exp_stride_b, B_exp_stride_c, B_exp_stride_i, B_exp_stride_n, B_exp_stride_k = B_expanded.stride()
        C_exp_stride_b, C_exp_stride_c, C_exp_stride_j, C_exp_stride_n, C_exp_stride_k = C_expanded.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()

        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G[grid_G](
            B_expanded, C_expanded, G,
            Bsz=Bsz, Csz=Csz, S=S, N=N, K=D,
            B_stride_b=B_exp_stride_b, B_stride_c=B_exp_stride_c, B_stride_i=B_exp_stride_i, B_stride_n=B_exp_stride_n, B_stride_k=B_exp_stride_k,
            C_stride_b=C_exp_stride_b, C_stride_c=C_exp_stride_c, C_stride_j=C_exp_stride_j, C_stride_n=C_exp_stride_n, C_stride_k=C_exp_stride_k,
            G_stride_b=G_stride_b, G_stride_c=G_stride_c, G_stride_i=G_stride_i, G_stride_j=G_stride_j, G_stride_n=G_stride_n,
            num_warps=1, num_stages=1
        )

        # 3) Diagonal contraction to compute Y_diag in Triton
        Y = torch.empty((Bsz, Csz, S, N, D), device=hidden_states.device, dtype=torch.float32)

        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = G.stride()
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = hidden_states.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y[grid_Y](
            G, hidden_states.to(torch.float32), Y,
            Bsz=Bsz, Csz=Csz, S=S, N=N, D=D,
            M_stride_b=M_stride_b, M_stride_c=M_stride_c, M_stride_i=M_stride_i, M_stride_j=M_stride_j, M_stride_n=M_stride_n,
            HS_stride_b=HS_stride_b, HS_stride_c=HS_stride_c, HS_stride_j=HS_stride_j, HS_stride_n=HS_stride_n, HS_stride_d=HS_stride_d,
            Y_stride_b=Y_stride_b, Y_stride_c=Y_stride_c, Y_stride_i=Y_stride_i, Y_stride_n=Y_stride_n, Y_stride_d=Y_stride_d,
            num_warps=1, num_stages=1
        )

        # Return in bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
