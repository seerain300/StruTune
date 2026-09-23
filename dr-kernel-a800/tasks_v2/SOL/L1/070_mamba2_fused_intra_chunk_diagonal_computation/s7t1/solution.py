import torch
import triton
import triton.language as tl


# Kernel 1: Compute L from A_cumsum with lower-triangular (diagonal=0) cumsum + exp.
# A_cumsum: [B, N, C, S] where S=CHUNK_SIZE=128
# Output L: [B, C, S, S, N] in bfloat16
@triton.jit
def segment_sum_exp_mask(
    A_ptr, L_ptr,
    B_size: tl.constexpr, C_size: tl.constexpr, N_size: tl.constexpr,
    S: tl.constexpr,
    A_stride_b, A_stride_n, A_stride_c, A_stride_s,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2)
    # Compute triangular mask with diagonal=0: j <= i
    for i in range(S):
        # We need cumsum along source dimension (S). A_cumsum has shape [B, N, C, S]
        # For each i, we compute sum over k <= i for each j. Lower-triangular means j <= i.
        # We'll compute cumsum for each j from 0..S-1.
        total = tl.zeros((), dtype=tl.float32)
        for j in range(S):
            if j <= i:
                a_idx = b * A_stride_b + n * A_stride_n + c * A_stride_c + j * A_stride_s
                a_val = tl.load(A_ptr + a_idx)
                total += a_val
            else:
                total += 0.0
        exp_val = tl.exp(total)
        # Store to L[b, c, i, j, n] as bfloat16
        L_idx = b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n
        tl.store(L_ptr + L_idx, tl.cast(exp_val, tl.bfloat16))


# Kernel 2: Contract B and C into G. Compute G[b, c, i, j, n] = sum_k C[b, c, i, n, k] * B[b, c, j, n, k]
# B_expanded: [B, C, S, N, K]  (we pass K as meta-constexpr)
# C_expanded: [B, C, S, N, K]
# G_out: [B, C, S, S, N] float32
@triton.jit
def contract_BC_G(
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
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(K):
        bj = b * B_stride_b + c * B_stride_c + j * B_stride_i + n * B_stride_n + k * B_stride_k
        ci = c * C_stride_c + c * C_stride_c + i * C_stride_j + n * C_stride_n + k * C_stride_k
        bj_val = tl.load(B_ptr + bj)
        ci_val = tl.load(C_ptr + ci)
        acc += bj_val * ci_val
    G_idx = b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    tl.store(G_ptr + G_idx, acc)


# Kernel 3: Diagonal contraction Y_diag[b, c, i, n, :] = sum over j of M[b, c, i, j, n] * hidden_states[b, c, j, n, :]
# M: [B, C, S, S, N] in bfloat16 (we'll load as float32)
# hidden_states: [B, C, S, N, D]
# Y_out: [B, C, S, N, D] bfloat16
@triton.jit
def diag_contract_Y(
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
    # Accumulate over j
    for j in range(S):
        M_idx = b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n
        HS_idx = b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n
        M_val = tl.load(M_ptr + M_idx, eviction_policy='evict_last')  # bfloat16; load as float32 by cast
        HS_val = tl.load(HS_ptr + HS_idx, eviction_policy='evict_last')
        M_f32 = tl.cast(M_val, tl.float32)
        HS_f32 = tl.cast(HS_val, tl.float32)
        # HS_val has shape [N, D] for this j; we need to multiply with M_f32 scalar
        # We'll accumulate per D: we need HS for each d. But HS_val is a vector [D] in this layout.
        # However, HS is [B, C, S, N, D]; to get HS[b, c, j, n, d] we need to iterate d and load each element.
        # We can load per d: index depends on d.
        # Build a vector acc[D] in registers (static range).
        # We'll accumulate into a vector of size D
        acc = tl.zeros((D,), dtype=tl.float32)
        for d in range(D):
            hs_d_idx = HS_idx + d * HS_stride_d
            m_scalar = M_f32
            acc[d] += m_scalar * tl.load(HS_ptr + hs_d_idx, eviction_policy='evict_last')
    # Store acc
    Y_base = b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n
    # We need to store vector acc to Y for all D
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
        # Validate shapes and device
        assert hidden_states.is_cuda, "Inputs must be CUDA tensors for Triton kernels."
        assert A_cumsum.is_cuda, "A_cumsum must be CUDA tensor."
        assert B.is_cuda and C.is_cuda, "B and C must be CUDA tensors."

        Bsz, Csz, S, N, D = hidden_states.shape
        # Constants
        S_const = self.CHUNK_SIZE
        N_const = self.NUM_HEADS
        K = B.shape[-1]  # state_size of B and C (last dimension)
        assert S == S_const, f"hidden_states.chunk_size must equal CHUNK_SIZE=128, got {S}"
        assert N == N_const, f"hidden_states.num_heads must equal NUM_HEADS=32, got {N}"
        assert B.shape[:-2] == (Bsz, Csz) and C.shape[:-2] == (Bsz, Csz), "B and C batch/num_chunks must match hidden_states"
        assert B.shape[-2] == S_const and C.shape[-2] == S_const, "B and C chunk_size must equal CHUNK_SIZE=128"
        assert B.shape[-1] == K and C.shape[-1] == K, "B and C last dim (state_size) must match"
        assert A_cumsum.shape == (Bsz, N_const, Csz, S_const), "A_cumsum shape must be [B, NUM_HEADS, C, CHUNK_SIZE]"

        # Step 1: Compute L using Triton
        # A_cumsum: [B, N, C, S]
        # Output L: [B, C, S, S, N] bfloat16
        L = torch.empty((Bsz, Csz, S_const, S_const, N_const), device=hidden_states.device, dtype=torch.bfloat16)

        A_ptr = A_cumsum
        L_ptr = L

        # Strides for A: [B, N, C, S]
        A_stride_b = A_cumsum.stride(0)
        A_stride_n = A_cumsum.stride(1)
        A_stride_c = A_cumsum.stride(2)
        A_stride_s = A_cumsum.stride(3)

        # Strides for L: [B, C, I, J, N]
        L_stride_b = L.stride(0)
        L_stride_c = L.stride(1)
        L_stride_i = L.stride(2)
        L_stride_j = L.stride(3)
        L_stride_n = L.stride(4)

        grid = (Bsz, Csz, N_const)
        segment_sum_exp_mask[grid](
            A_ptr, L_ptr,
            Bsz, Csz, N_const,
            S_const,
            A_stride_b, A_stride_n, A_stride_c, A_stride_s,
            L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
            num_warps=1, num_stages=1,
        )

        # Step 2: Expand B and C to include NUM_HEADS (repeat_interleave ratio = NUM_HEADS // N_GROUPS = 4)
        B_expanded = B.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3)  # [B, C, S, N, K]
        C_expanded = C.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3)  # [B, C, S, N, K]

        # Step 3: Compute G using Triton (contraction over K)
        G = torch.empty((Bsz, Csz, S_const, S_const, N_const), device=hidden_states.device, dtype=torch.float32)

        # Strides for B_expanded: [B, C, S, N, K]
        B_stride_b = B_expanded.stride(0)
        B_stride_c = B_expanded.stride(1)
        B_stride_i = B_expanded.stride(2)
        B_stride_n = B_expanded.stride(3)
        B_stride_k = B_expanded.stride(4)

        # Strides for C_expanded: [B, C, S, N, K]
        C_stride_b = C_expanded.stride(0)
        C_stride_c = C_expanded.stride(1)
        C_stride_j = C_expanded.stride(2)
        C_stride_n = C_expanded.stride(3)
        C_stride_k = C_expanded.stride(4)

        # Strides for G: [B, C, I, J, N]
        G_stride_b = G.stride(0)
        G_stride_c = G.stride(1)
        G_stride_i = G.stride(2)
        G_stride_j = G.stride(3)
        G_stride_n = G.stride(4)

        grid2 = (Bsz, Csz, S_const, S_const, N_const)
        contract_BC_G[grid2](
            B_expanded, C_expanded, G,
            Bsz, Csz, N_const,
            S_const, K,
            B_stride_b, B_stride_c, B_stride_i, B_stride_n, B_stride_k,
            C_stride_b, C_stride_c, C_stride_j, C_stride_n, C_stride_k,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            num_warps=1, num_stages=1,
        )

        # Step 4: Compute M = G * L elementwise (PyTorch for simplicity)
        # G: [B, C, S, S, N], L: [B, C, S, S, N]
        M = G * L.to(torch.float32)  # multiply in float32

        # Step 5: Compute Y_diag via Triton diagonal contraction
        # hidden_states: [B, C, S, N, D], Y_out: [B, C, S, N, D]
        Y_out = torch.empty((Bsz, Csz, S_const, N_const, D), device=hidden_states.device, dtype=torch.bfloat16)

        # Strides for M, HS, Y
        M_stride_b = M.stride(0); M_stride_c = M.stride(1); M_stride_i = M.stride(2); M_stride_j = M.stride(3); M_stride_n = M.stride(4)
        HS_stride_b = hidden_states.stride(0); HS_stride_c = hidden_states.stride(1); HS_stride_j = hidden_states.stride(2); HS_stride_n = hidden_states.stride(3); HS_stride_d = hidden_states.stride(4)
        Y_stride_b = Y_out.stride(0); Y_stride_c = Y_out.stride(1); Y_stride_i = Y_out.stride(2); Y_stride_n = Y_out.stride(3); Y_stride_d = Y_out.stride(4)

        grid3 = (Bsz, Csz, S_const, N_const)
        diag_contract_Y[grid3](
            M, hidden_states, Y_out,
            Bsz, Csz, N_const, S_const, D,
            M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
            HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
            Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
            num_warps=1, num_stages=1,
        )

        return Y_out


def run(*args):
    return ModelNew()(*args)
