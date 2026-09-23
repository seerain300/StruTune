import torch
import triton
import triton.language as tl

# Kernel 1: Expand B and C along num_heads via repeat_interleave(NUM_HEADS // N_GROUPS) = 4
@triton.jit
def expand_repeat_interleave(
    B_in_ptr, C_in_ptr, B_out_ptr, C_out_ptr,
    B_size: tl.constexpr, C_size: tl.constexpr, S: tl.constexpr, N_GROUPS: tl.constexpr, K: tl.constexpr,
    B_in_stride_b, B_in_stride_c, B_in_stride_s, B_in_stride_ng, B_in_stride_k,
    C_in_stride_b, C_in_stride_c, C_in_stride_s, C_in_stride_ng, C_in_stride_k,
    B_out_stride_b, B_out_stride_c, B_out_stride_s, B_out_stride_n, B_out_stride_k,
    C_out_stride_b, C_out_stride_c, C_out_stride_s, C_out_stride_n, C_out_stride_k,
):
    # Each program handles one (b, c, s, k) and writes 4 times for n in [0..N_GROUPS*4)
    b = tl.program_id(0)
    c = tl.program_id(1)
    s = tl.program_id(2)
    k = tl.program_id(3)
    # base pointer for (b, c, s, k) in input
    base_in_B = B_in_ptr + b * B_in_stride_b + c * B_in_stride_c + s * B_in_stride_s + k * B_in_stride_k
    base_in_C = C_in_ptr + b * C_in_stride_b + c * C_in_stride_c + s * C_in_stride_s + k * C_in_stride_k
    # write to output for each expanded n
    for n in range(N_GROUPS * 4):
        B_out_ptr_addr = B_out_ptr + b * B_out_stride_b + c * B_out_stride_c + s * B_out_stride_s + n * B_out_stride_n + k * B_out_stride_k
        C_out_ptr_addr = C_out_ptr + b * C_out_stride_b + c * C_out_stride_c + s * C_out_stride_s + n * C_out_stride_n + k * C_out_stride_k
        # value is same for all n because repeat_interleave just replicates along n
        tl.store(B_out_ptr_addr, tl.load(base_in_B))
        tl.store(C_out_ptr_addr, tl.load(base_in_C))

# Kernel 2: Compute L via masked cumsum (lower-triangular, diagonal=-1) and exp
@triton.jit
def masked_cumsum_lower_exp_L(
    A_ptr, L_ptr,
    B_size: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr,
    A_stride_b, A_stride_n, A_stride_c, A_stride_s,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
):
    # grid: (B, N, Csz)
    b = tl.program_id(0)
    n = tl.program_id(1)
    c = tl.program_id(2)
    # running sums per i
    for i in range(S):
        sum_val = 0.0
        for j in range(S):
            # A[b, n, c, j] with pointer arithmetic
            A_addr = A_ptr + b * A_stride_b + n * A_stride_n + c * A_stride_c + j * A_stride_s
            a = tl.load(A_addr)
            # include j < i (diagonal=-1), exclude j == i
            if j < i:
                sum_val += a
            # L[i, j, n] = exp(sum_val)
            L_addr = L_ptr + b * L_stride_b + c * L_stride_i + i * L_stride_i + j * L_stride_j + n * L_stride_n
            tl.store(L_addr, tl.exp(sum_val))

# Kernel 3: Compute G[b, c, i, j, n] = sum_k C[b, c, i, n, k] * B[b, c, j, n, k]
@triton.jit
def contract_BC_to_G(
    B_expanded_ptr, C_expanded_ptr, G_ptr,
    B_size: tl.constexpr, C_size: tl.constexpr, S: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    B_stride_b, B_stride_c, B_stride_s, B_stride_n, B_stride_k,
    C_stride_b, C_stride_c, C_stride_s, C_stride_n, C_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)
    g_val = 0.0
    for k in range(K):
        B_addr = B_expanded_ptr + b * B_stride_b + c * B_stride_c + j * B_stride_s + n * B_stride_n + k * B_stride_k
        C_addr = C_expanded_ptr + b * C_stride_b + c * C_stride_c + i * C_stride_s + n * C_stride_n + k * C_stride_k
        b_elem = tl.load(B_addr)
        c_elem = tl.load(C_addr)
        g_val += b_elem * c_elem
    G_addr = G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    tl.store(G_addr, g_val)

# Kernel 4: Elementwise M = G * L
@triton.jit
def elementwise_mul_M(
    G_ptr, L_ptr, M_ptr,
    B_size: tl.constexpr, C_size: tl.constexpr, S: tl.constexpr, N: tl.constexpr,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)
    G_addr = G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    L_addr = L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n
    G_val = tl.load(G_addr)
    L_val = tl.load(L_addr)
    M_addr = M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n
    tl.store(M_addr, G_val * L_val)

# Kernel 5: Diagonal contraction Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
@triton.jit
def diag_contract_Y(
    M_ptr, hidden_ptr, Y_ptr,
    B_size: tl.constexpr, C_size: tl.constexpr, S: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    hidden_stride_b, hidden_stride_c, hidden_stride_j, hidden_stride_n, hidden_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)
    acc = 0.0
    for j in range(S):
        M_addr = M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n
        hidden_addr = hidden_ptr + b * hidden_stride_b + c * hidden_stride_c + j * hidden_stride_j + n * hidden_stride_n + d * hidden_stride_d
        m_val = tl.load(M_addr)
        hs_val = tl.load(hidden_addr)
        acc += m_val * hs_val
    Y_addr = Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d
    tl.store(Y_addr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, N_GROUPS: int = 8, NUM_HEADS: int = 32):
        super().__init__()
        self.N_GROUPS = N_GROUPS
        self.NUM_HEADS = NUM_HEADS

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Infer shapes
        Bsz, Csz, S, N, D = hidden_states.shape
        assert hidden_states.ndim == 5, "hidden_states must be [B, C, S, N, D]"
        assert A_cumsum.shape == (Bsz, N, Csz, S), "A_cumsum must be [B, N, C, S]"
        assert B.shape == (Bsz, Csz, S, self.N_GROUPS, D), "B must be [B, C, S, N_GROUPS, D]"
        assert C.shape == (Bsz, Csz, S, self.N_GROUPS, D), "C must be [B, C, S, N_GROUPS, D]"

        # Allocate expanded tensors
        B_expanded = torch.empty((Bsz, Csz, S, self.NUM_HEADS, D), device=hidden_states.device, dtype=torch.float32)
        C_expanded = torch.empty((Bsz, Csz, S, self.NUM_HEADS, D), device=hidden_states.device, dtype=torch.float32)

        # Launch expand_repeat_interleave (repeat_interleave by 4 to go from N_GROUPS to NUM_HEADS=32)
        B_in_stride_b, B_in_stride_c, B_in_stride_s, B_in_stride_ng, B_in_stride_k = B.stride()
        C_in_stride_b, C_in_stride_c, C_in_stride_s, C_in_stride_ng, C_in_stride_k = C.stride()
        B_out_stride_b, B_out_stride_c, B_out_stride_s, B_out_stride_n, B_out_stride_k = B_expanded.stride()
        C_out_stride_b, C_out_stride_c, C_out_stride_s, C_out_stride_n, C_out_stride_k = C_expanded.stride()
        grid_expand = (Bsz, Csz, S, D)  # k dimension is D (head_dim)
        expand_repeat_interleave[grid_expand](
            B, C, B_expanded, C_expanded,
            B_size=Bsz, C_size=Csz, S=S, N_GROUPS=self.N_GROUPS, K=D,
            B_in_stride_b=B_in_stride_b, B_in_stride_c=B_in_stride_c, B_in_stride_s=B_in_stride_s, B_in_stride_ng=B_in_stride_k,  # note: B_in_stride_ng is not used here; we pass k stride as B_in_stride_k
            C_in_stride_b=C_in_stride_b, C_in_stride_c=C_in_stride_c, C_in_stride_s=C_in_stride_s, C_in_stride_ng=C_in_stride_k,
            B_out_stride_b=B_out_stride_b, B_out_stride_c=B_out_stride_c, B_out_stride_s=B_out_stride_s, B_out_stride_n=B_out_stride_n, B_out_stride_k=B_out_stride_k,
            C_out_stride_b=C_out_stride_b, C_out_stride_c=C_out_stride_c, C_out_stride_s=C_out_stride_s, C_out_stride_n=C_out_stride_n, C_out_stride_k=C_out_stride_k,
            num_warps=1, num_stages=1
        )

        # Allocate L, G, M, Y
        # L: [B, N, C, S, S] float32
        L = torch.empty((Bsz, N, Csz, S, S), device=hidden_states.device, dtype=torch.float32)
        G = torch.empty((Bsz, Csz, S, S, N), device=hidden_states.device, dtype=torch.float32)
        M = torch.empty((Bsz, Csz, S, S, N), device=hidden_states.device, dtype=torch.float32)
        Y = torch.empty((Bsz, Csz, S, N, D), device=hidden_states.device, dtype=torch.float32)

        # Launch masked_cumsum_lower_exp_L
        A_stride_b, A_stride_n, A_stride_c, A_stride_s = A_cumsum.stride()
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = L.stride()
        grid_L = (Bsz, N, Csz)
        masked_cumsum_lower_exp_L[grid_L](
            A_cumsum, L,
            B_size=Bsz, Csz=Csz, S=S, N=N,
            A_stride_b=A_stride_b, A_stride_n=A_stride_n, A_stride_c=A_stride_c, A_stride_s=A_stride_s,
            L_stride_b=L_stride_b, L_stride_c=L_stride_c, L_stride_i=L_stride_i, L_stride_j=L_stride_j, L_stride_n=L_stride_n,
            num_warps=1, num_stages=1
        )

        # Launch contract_BC_to_G
        B_stride_b, B_stride_c, B_stride_s, B_stride_n, B_stride_k = B_expanded.stride()
        C_stride_b, C_stride_c, C_stride_s, C_stride_n, C_stride_k = C_expanded.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()
        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G[grid_G](
            B_expanded, C_expanded, G,
            B_size=Bsz, C_size=Csz, S=S, N=N, K=D,
            B_stride_b=B_stride_b, B_stride_c=B_stride_c, B_stride_s=B_stride_s, B_stride_n=B_stride_n, B_stride_k=B_stride_k,
            C_stride_b=C_stride_b, C_stride_c=C_stride_c, C_stride_s=C_stride_s, C_stride_n=C_stride_n, C_stride_k=C_stride_k,
            G_stride_b=G_stride_b, G_stride_c=G_stride_c, G_stride_i=G_stride_i, G_stride_j=G_stride_j, G_stride_n=G_stride_n,
            num_warps=1, num_stages=1
        )

        # Launch elementwise_mul_M
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = L.stride()
        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        grid_M = (Bsz, Csz, S, S, N)
        elementwise_mul_M[grid_M](
            G, L, M,
            B_size=Bsz, C_size=Csz, S=S, N=N,
            G_stride_b=G_stride_b, G_stride_c=G_stride_c, G_stride_i=G_stride_i, G_stride_j=G_stride_j, G_stride_n=G_stride_n,
            L_stride_b=L_stride_b, L_stride_c=L_stride_c, L_stride_i=L_stride_i, L_stride_j=L_stride_j, L_stride_n=L_stride_n,
            M_stride_b=M_stride_b, M_stride_c=M_stride_c, M_stride_i=M_stride_i, M_stride_j=M_stride_j, M_stride_n=M_stride_n,
            num_warps=1, num_stages=1
        )

        # Launch diag_contract_Y
        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        hidden_stride_b, hidden_stride_c, hidden_stride_j, hidden_stride_n, hidden_stride_d = hidden_states.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()
        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y[grid_Y](
            M, hidden_states.to(torch.float32), Y,
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
