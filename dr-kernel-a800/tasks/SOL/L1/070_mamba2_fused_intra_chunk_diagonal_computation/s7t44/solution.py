import torch
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_lower_exp(
    A_ptr,          # *float32, [B, C, S, N]
    L_ptr,          # *float32, [B, C, S, S, N]
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr,
    A_stride_b, A_stride_c, A_stride_j, A_stride_n,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
):
    # Each program handles one (b, c, i)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)

    # Running sum vector for each head n in [0, N)
    running = tl.zeros((N,), dtype=tl.float32)

    # Loop over source j from 0 to S-1; include only j < i (diagonal=-1)
    for j in range(0, S):
        include = j < i  # scalar predicate; broadcasts over N
        n_idx = tl.arange(0, N)
        a_offsets = b * A_stride_b + c * A_stride_c + j * A_stride_j + n_idx * A_stride_n
        a_vals = tl.load(A_ptr + a_offsets)
        # Zero out if j >= i
        a_vals = tl.where(include, a_vals, 0.0)
        running += a_vals

        # Store L[b, c, i, j, n] = exp(running)
        l_offsets = b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n_idx * L_stride_n
        tl.store(L_ptr + l_offsets, tl.exp(running))


@triton.jit
def contract_BC_to_G(
    B_ptr,          # *float32, [B, C, S, N, K]
    C_ptr,          # *float32, [B, C, S, N, K]
    G_ptr,          # *float32, [B, C, S, S, N]
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
    C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    # Sum over state dimension K
    for k in range(0, K):
        bval = tl.load(B_ptr + b * B_stride_b + c * B_stride_c + j * B_stride_j + n * B_stride_n + k * B_stride_k)
        cval = tl.load(C_ptr + b * C_stride_b + c * C_stride_c + i * C_stride_i + n * C_stride_n + k * C_stride_k)
        acc += bval * cval

    # Store G[b, c, i, j, n] = acc
    g_offset = b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    tl.store(G_ptr + g_offset, acc)


@triton.jit
def diag_contract_Y(
    M_ptr,          # *float32, [B, C, S, S, N]
    HS_ptr,         # *float32, [B, C, S, N, D]
    Y_ptr,          # *float32, [B, C, S, N, D]
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    # Sum over j dimension
    for j in range(0, S):
        mval = tl.load(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n)
        hsv = tl.load(HS_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d)
        acc += mval * hsv

    # Store Y[b, c, i, n, d] = acc
    y_offset = b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d
    tl.store(Y_ptr + y_offset, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from the original signature: NUM_HEADS=32, N_GROUPS=8
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Shapes inferred from input
        Bsz, Csz, S, N, D = hidden_states.shape

        # Ensure contiguity and dtype
        device = hidden_states.device
        A_cumsum = A_cumsum.to(torch.float32).contiguous()  # [B, C, S, N]
        B = B.to(torch.float32).contiguous()                # [B, C, S, N_GROUPS, K]
        C = C.to(torch.float32).contiguous()                # [B, C, S, N_GROUPS, K]

        # 1) Compute L via Triton masked cumsum with diagonal=-1, then exp.
        L = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        A_stride_b, A_stride_c, A_stride_j, A_stride_n = A_cumsum.stride()
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = L.stride()
        grid_L = (Bsz, Csz, S)
        masked_cumsum_lower_exp[grid_L](
            A_cumsum, L,
            Bsz, Csz, S, N,
            A_stride_b, A_stride_c, A_stride_j, A_stride_n,
            L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
            num_warps=1, num_stages=1
        )

        # 2) Compute G[b, c, i, j, n] via Triton contraction over K (state_size)
        # Expand B and C from n_groups to num_heads (32) by repeat_interleave
        B_expanded = B.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3).contiguous()  # [B, C, S, N, K]
        C_expanded = C.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3).contiguous()  # [B, C, S, N, K]

        K = D  # state_size equals hidden_states.last_dim (head_dim)
        G = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        B_expanded_str_b, B_expanded_str_c, B_expanded_str_j, B_expanded_str_n, B_expanded_str_k = B_expanded.stride()
        C_expanded_str_b, C_expanded_str_c, C_expanded_str_i, C_expanded_str_n, C_expanded_str_k = C_expanded.stride()
        G_str_b, G_str_c, G_str_i, G_str_j, G_str_n = G.stride()

        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G[grid_G](
            B_expanded, C_expanded, G,
            Bsz, Csz, S, N, K,
            B_expanded_str_b, B_expanded_str_c, B_expanded_str_j, B_expanded_str_n, B_expanded_str_k,
            C_expanded_str_b, C_expanded_str_c, C_expanded_str_i, C_expanded_str_n, C_expanded_str_k,
            G_str_b, G_str_c, G_str_i, G_str_j, G_str_n,
            num_warps=1, num_stages=1
        )

        # 3) Elementwise M = G * L (PyTorch; not a heavy op)
        M = G * L  # [B, C, S, S, N], float32

        # 4) Diagonal contraction to produce Y_diag: [B, C, S, N, D]
        HS = hidden_states.to(torch.float32).contiguous()  # [B, C, S, N, D]
        Y = torch.empty((Bsz, Csz, S, N, D), device=device, dtype=torch.float32)

        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = HS.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y[grid_Y](
            M, HS, Y,
            Bsz, Csz, S, N, D,
            M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
            HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
            Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
