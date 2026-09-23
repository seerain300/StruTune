import torch
import triton
import triton.language as tl


# Triton Kernel: Elementwise exp on a 5D tensor. We'll use it to compute L = exp(A_cumsum_seg).
# Input A_seg: [B, N, C, S, S] float32
# Output L_out: [B, N, C, S, S] float32
@triton.jit
def exp_5d_kernel(
    A_ptr, Out_ptr,
    B_size: tl.constexpr, N_size: tl.constexpr, C_size: tl.constexpr,
    S: tl.constexpr,
    A_stride_b, A_stride_n, A_stride_c, A_stride_i, A_stride_j,
    Out_stride_b, Out_stride_n, Out_stride_c, Out_stride_i, Out_stride_j,
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    c = tl.program_id(2)
    i = tl.program_id(3)
    j = tl.program_id(4)

    A_idx = b * A_stride_b + n * A_stride_n + c * A_stride_c + i * A_stride_i + j * A_stride_j
    val = tl.load(A_ptr + A_idx)
    out = tl.exp(val)
    Out_idx = b * Out_stride_b + n * Out_stride_n + c * Out_stride_c + i * Out_stride_i + j * Out_stride_j
    tl.store(Out_ptr + Out_idx, out)


# Triton Kernel: Contract B and C into G: G[b, c, i, j, n] = sum_k C[b, c, i, n, k] * B[b, c, j, n, k]
# B_expanded: [B, C, S, N, K]
# C_expanded: [B, C, S, N, K]
# G_out: [B, C, S, S, N] float32
@triton.jit
def contract_BC_G_dynamic(
    B_ptr, C_ptr, G_ptr,
    B_size: tl.constexpr, C_size: tl.constexpr, N_size: tl.constexpr,
    S: tl.constexpr, K: tl.constexpr,
    B_stride_b, B_stride_c, B_stride_i, B_stride_n, B_stride_k,
    C_stride_b, C_stride_c, C_stride_j, C_stride_n, C_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    acc = 0.0
    for k in range(K):
        bj = b * B_stride_b + c * B_stride_c + j * B_stride_i + n * B_stride_n + k * B_stride_k
        ci = c * C_stride_c + c * C_stride_c + i * C_stride_j + n * C_stride_n + k * C_stride_k
        bj_val = tl.load(B_ptr + bj)
        ci_val = tl.load(C_ptr + ci)
        acc += bj_val * ci_val
    G_idx = b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    tl.store(G_ptr + G_idx, acc)  # float32


# Triton Kernel: Multiply G by L to get M: M = G * L (elementwise)
# G: [B, C, S, S, N] float32
# L: [B, C, S, S, N] float32
# M_out: [B, C, S, S, N] float32
@triton.jit
def mul_GL_to_M(
    G_ptr, L_ptr, M_ptr,
    B_size: tl.constexpr, C_size: tl.constexpr, N_size: tl.constexpr,
    S: tl.constexpr,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    G_idx = b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    L_idx = b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n

    g_val = tl.load(G_ptr + G_idx)  # float32
    l_val = tl.load(L_ptr + L_idx)  # float32
    m_val = g_val * l_val
    M_idx = b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n
    tl.store(M_ptr + M_idx, m_val)


# Triton Kernel: Diagonal contraction Y_diag
# M: [B, C, S, S, N] float32
# hidden_states: [B, C, S, N, D]
# Y_out: [B, C, S, N, D] bfloat16
@triton.jit
def diag_contract_Y_dynamic(
    M_ptr, HS_ptr, Y_ptr,
    B_size: tl.constexpr, C_size: tl.constexpr, N_size: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)

    # Accumulator vector for each head dimension
    acc = tl.zeros((D,), dtype=tl.float32)
    for j in range(S):
        M_idx = b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n
        HS_base = b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n
        m_scalar = tl.load(M_ptr + M_idx)  # float32
        for d in range(D):
            HS_d = tl.load(HS_ptr + HS_base + d * HS_stride_d)  # hidden_states dtype, cast to float32 for compute
            acc[d] += m_scalar * tl.cast(HS_d, tl.float32)

    Y_base = b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n
    for d in range(D):
        Y_idx = Y_base + d * Y_stride_d
        tl.store(Y_ptr + Y_idx, tl.cast(acc[d], tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from the original code
        self.CHUNK_SIZE = 128
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors
        assert hidden_states.is_cuda, "Inputs must be CUDA tensors for Triton kernels."
        assert A_cumsum.is_cuda, "A_cumsum must be a CUDA tensor."
        assert B.is_cuda and C.is_cuda, "B and C must be CUDA tensors."

        # Extract shapes
        Bsz, Csz, S, N, D = hidden_states.shape
        # Original code uses fixed constants, but we must honor input shapes
        # A_cumsum shape: [B, N, C, S]
        assert A_cumsum.shape == (Bsz, N, Csz, S), f"A_cumsum shape must be [B, NUM_HEADS, C, S], got {A_cumsum.shape}"

        # Constants for expansion
        NUM_HEADS = self.NUM_HEADS  # 32
        N_GROUPS = self.N_GROUPS    # 8

        # Expand B and C to include NUM_HEADS (repeat_interleave by 4)
        B_expanded = B.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)  # [B, C, S, N, D]
        C_expanded = C.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)  # [B, C, S, N, D]

        # Compute L (causal mask) using original semantics:
        # 1) Expand A_cumsum to [B, N, C, S, S]
        # 2) Apply lower-triangular mask (diagonal = -1)
        # 3) Cumsum along dim=-2 (the source index), then exp
        A_expanded = A_cumsum.unsqueeze(-1).unsqueeze(-1).expand(Bsz, N, Csz, S, S)  # [B, N, C, S, S]
        # Lower-triangular mask with diagonal = -1
        # Create mask on the expanded shape
        device = A_expanded.device
        mask = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=-1)  # [S, S]
        # Broadcast mask to [N, C, S, S] then combine with B
        # For each (b, n, c), mask is the same
        A_masked = A_expanded.masked_fill(~mask, 0)  # apply mask across last two dims

        # Cumsum along the source index (dim=-2 of A_masked => last dim of A_masked is S, but cumsum is over C dimension)
        # Clarification: A_masked has shape [B, N, C, S, S]. Cumsum along dim=-2 means cumsum across C for fixed (b, n, c, i,j).
        # The original code does torch.cumsum(A_expanded, dim=-2). Since A_expanded is [B, N, C, S, S], cumsum along -2 (C).
        # So we need to cumsum across C.
        # Implement cumsum along C:
        A_cumsum_seg = torch.cumsum(A_masked, dim=2)  # [B, N, C, S, S]

        # Exponentiate to get L: L = exp(A_cumsum_seg)
        L = torch.exp(A_cumsum_seg)  # float32
        # Launch a simple Triton kernel to write L as float32 (we already have it via torch; Triton kernel can copy if needed)
        # For simplicity, use L as is; we only need to apply elementwise exp on L if we were to compute via Triton, but L is already computed.

        # Compute G = contract(B_expanded, C_expanded) over state_size K = D
        K = D  # state_size equals hidden_states last dim
        G = torch.empty((Bsz, Csz, S, S, NUM_HEADS), device=hidden_states.device, dtype=torch.float32)

        B_stride_b, B_stride_c, B_stride_i, B_stride_n, B_stride_k = B_expanded.stride()
        C_stride_b, C_stride_c, C_stride_j, C_stride_n, C_stride_k = C_expanded.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()

        grid_G = (Bsz, Csz, S, S, NUM_HEADS)
        contract_BC_G_dynamic[grid_G](
            B_expanded, C_expanded, G,
            Bsz, Csz, NUM_HEADS,
            S, K,
            B_stride_b, B_stride_c, B_stride_i, B_stride_n, B_stride_k,
            C_stride_b, C_stride_c, C_stride_j, C_stride_n, C_stride_k,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            num_warps=1, num_stages=1,
        )

        # Compute M = G * L (elementwise)
        M = torch.empty_like(G, dtype=torch.float32)

        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = L.stride()
        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()

        grid_M = (Bsz, Csz, S, S, NUM_HEADS)
        mul_GL_to_M[grid_M](
            G, L, M,
            Bsz, Csz, NUM_HEADS,
            S,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
            M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
            num_warps=1, num_stages=1,
        )

        # Compute Y_diag: Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
        Y = torch.empty((Bsz, Csz, S, NUM_HEADS, D), device=hidden_states.device, dtype=torch.bfloat16)

        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = hidden_states.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        grid_Y = (Bsz, Csz, S, NUM_HEADS)
        diag_contract_Y_dynamic[grid_Y](
            M, hidden_states, Y,
            Bsz, Csz, NUM_HEADS, S, D,
            M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
            HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
            Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
            num_warps=1, num_stages=1,
        )

        return Y


def run(*args):
    return ModelNew()(*args)
