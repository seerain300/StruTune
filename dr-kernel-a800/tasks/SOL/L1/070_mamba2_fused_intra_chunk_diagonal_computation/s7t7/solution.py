import torch
import triton
import triton.language as tl


# Triton Kernel 1: masked_cumsum_exp_lower
# Compute L[b, c, i, j, n] = exp(masked cumsum along j for each i) with lower-triangular mask (diagonal=-1).
# A_cumsum: [B, C, S, N] where S = hidden_states.shape[2], N = hidden_states.shape[3]
# Output L: [B, C, S, S, N] in float32
@triton.jit
def masked_cumsum_exp_lower(
    A_ptr, L_ptr,
    B_size: tl.constexpr, C_size: tl.constexpr, S: tl.constexpr, N: tl.constexpr,
    A_stride_b, A_stride_c, A_stride_i, A_stride_n,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    # Program ids: 5D grid (b, c, i, j, n)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    # For each (b, c, i, n), compute masked cumsum over j from 0..S-1, exclude j == i
    # We'll accumulate acc and then write L[b, c, i, j, n] for each j < i
    acc = 0.0
    for jj in tl.static_range(S):
        if jj < i:
            # Load A[b, c, i, n] which is independent of j in original code
            a = tl.load(A_ptr + b * A_stride_b + c * A_stride_c + i * A_stride_i + n * A_stride_n)
            acc += a
    # Compute exp of the masked cumsum and store for all j < i
    l_val = tl.exp(acc)
    for jj in tl.static_range(S):
        if jj < i:
            tl.store(L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + jj * L_stride_j + n * L_stride_n, l_val)


# Triton Kernel 2: contract_BC_to_G
# Computes G[b, c, i, j, n] = sum over k of C[b, c, i, n, k] * B[b, c, j, n, k]
# B_expanded: [B, C, S, N, K], C_expanded: [B, C, S, N, K], G: [B, C, S, S, N]
@triton.jit
def contract_BC_to_G(
    B_ptr, C_ptr, G_ptr,
    B_size: tl.constexpr, C_size: tl.constexpr, S: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
    C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    acc = 0.0
    for k in tl.static_range(K):
        b_val = tl.load(B_ptr + b * B_stride_b + c * B_stride_c + j * B_stride_j + n * B_stride_n + k * B_stride_k)
        c_val = tl.load(C_ptr + b * C_stride_b + c * C_stride_c + i * C_stride_i + n * C_stride_n + k * C_stride_k)
        acc += b_val * c_val
    tl.store(G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n, acc)


# Triton Kernel 3: diagonal contraction for Y_diag
# Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
# M: [B, C, S, S, N] float32
# hidden_states: [B, C, S, N, D]
# Y: [B, C, S, N, D] float32
@triton.jit
def diag_contract_Y(
    M_ptr, HS_ptr, Y_ptr,
    B_size: tl.constexpr, C_size: tl.constexpr, S: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in tl.static_range(S):
        m = tl.load(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n)
        hs = tl.load(HS_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d)
        acc += m * hs
    tl.store(Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants consistent with original signature
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, C, S, N, D]
        device = hidden_states.device
        Bsz, Csz, S, N, D = hidden_states.shape

        # Prepare expanded B and C for num_heads = 32 (repeat_interleave by 4)
        B_expanded = B.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3)  # [B, C, S, N, K]
        C_expanded = C.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3)  # [B, C, S, N, K]
        # Ensure contiguous for Triton
        B_expanded = B_expanded.contiguous()
        C_expanded = C_expanded.contiguous()

        # 1) Compute L in Triton: masked cumsum + exp
        A_cumsum = A_cumsum.to(torch.float32).contiguous()  # [B, C, S, N]
        L = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        A_stride_b, A_stride_c, A_stride_i, A_stride_n = A_cumsum.stride()
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = L.stride()

        grid_L = (Bsz, Csz, S, S, N)
        masked_cumsum_exp_lower[grid_L](
            A_cumsum, L,
            B_size=Bsz, C_size=Csz, S=S, N=N,
            A_stride_b=A_stride_b, A_stride_c=A_stride_c, A_stride_i=A_stride_i, A_stride_n=A_stride_n,
            L_stride_b=L_stride_b, L_stride_c=L_stride_c, L_stride_i=L_stride_i, L_stride_j=L_stride_j, L_stride_n=L_stride_n,
            num_warps=1, num_stages=1
        )

        # 2) Compute G via Triton contraction over K (state_size)
        K = B_expanded.shape[4]  # same as hidden_states.shape[4] which is D in original, but here we take from expanded
        # If B_expanded has different K, we need to align with hidden_states last dim. Typically, B/C in original have state_size = hidden_states.shape[4].
        # To ensure correctness, we set K = hidden_states.shape[4] if available. In this code, B/C are expanded from original B/C where state_size matches hidden_states.last dim.
        # We infer K from B_expanded last dim; for robustness, we also require B_expanded.shape[4] == hidden_states.shape[4]. If not, we fallback to D.
        if B_expanded.shape[4] == hidden_states.shape[4]:
            K = B_expanded.shape[4]
        else:
            K = hidden_states.shape[4]

        G = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k = B_expanded.stride()
        C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k = C_expanded.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()

        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G[grid_G](
            B_expanded, C_expanded, G,
            B_size=Bsz, C_size=Csz, S=S, N=N, K=K,
            B_stride_b=B_stride_b, B_stride_c=B_stride_c, B_stride_j=B_stride_j, B_stride_n=B_stride_n, B_stride_k=B_stride_k,
            C_stride_b=C_stride_b, C_stride_c=C_stride_c, C_stride_i=C_stride_i, C_stride_n=C_stride_n, C_stride_k=C_stride_k,
            G_stride_b=G_stride_b, G_stride_c=G_stride_c, G_stride_i=G_stride_i, G_stride_j=G_stride_j, G_stride_n=G_stride_n,
            num_warps=1, num_stages=1
        )

        # 3) Elementwise M = G * L (PyTorch). Avoid torch.tril/cumsum/exp by computing L in Triton above.
        M = G * L  # float32

        # 4) Diagonal contraction to compute Y_diag via Triton
        Y = torch.empty((Bsz, Csz, S, N, D), device=device, dtype=torch.float32)

        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = hidden_states.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y[grid_Y](
            M, hidden_states.to(torch.float32), Y,
            B_size=Bsz, C_size=Csz, S=S, N=N, D=D,
            M_stride_b=M_stride_b, M_stride_c=M_stride_c, M_stride_i=M_stride_i, M_stride_j=M_stride_j, M_stride_n=M_stride_n,
            HS_stride_b=HS_stride_b, HS_stride_c=HS_stride_c, HS_stride_j=HS_stride_j, HS_stride_n=HS_stride_n, HS_stride_d=HS_stride_d,
            Y_stride_b=Y_stride_b, Y_stride_c=Y_stride_c, Y_stride_i=Y_stride_i, Y_stride_n=Y_stride_n, Y_stride_d=Y_stride_d,
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
