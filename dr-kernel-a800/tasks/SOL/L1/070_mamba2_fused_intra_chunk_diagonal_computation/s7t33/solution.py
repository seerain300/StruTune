import torch
import triton
import triton.language as tl


@triton.jit
def mask_cumsum_exp(A_ptr, L_ptr,
                     B_size,  # not used, but we keep signature
                     N, C, S,
                     A_stride_b, A_stride_n, A_stride_c, A_stride_s,
                     L_stride_b, L_stride_n, L_stride_c, L_stride_i, L_stride_j,
                     num_warps: tl.constexpr, num_stages: tl.constexpr):
    # Each program handles one (b, n, c, i)
    b = tl.program_id(0)
    n = tl.program_id(1)
    c = tl.program_id(2)
    i = tl.program_id(3)

    # Running sum along j for fixed i
    running = 0.0
    # Loop over j from 0 to S-1
    for j in range(0, S):
        # mask: include only j < i (diagonal=-1)
        include = j < i
        # Load A[b, n, c, j]
        a_ptr = A_ptr + b * A_stride_b + n * A_stride_n + c * A_stride_c + j * A_stride_s
        a = tl.load(a_ptr)
        # If not include, set a to 0
        a = tl.where(include, a, 0.0)
        running += a
        # Store exp(running) to L[b, n, c, i, j]
        l_ptr = L_ptr + b * L_stride_b + n * L_stride_n + c * L_stride_c + i * L_stride_i + j * L_stride_j
        tl.store(l_ptr, tl.exp(running))

    # No need to write j >= i entries since running won't update them in this loop.
    # For j >= i, they should be exp(0) = 1? Actually, the original code zeros them out via tril mask.
    # We need to set j >= i entries to 0.0 after this loop.
    for j in range(0, S):
        if not (j < i):
            l_ptr = L_ptr + b * L_stride_b + n * L_stride_n + c * L_stride_c + i * L_stride_i + j * L_stride_j
            tl.store(l_ptr, 0.0)


@triton.jit
def contract_BC_to_G(B_ptr, C_ptr, G_ptr,
                     B_size, C_size, S, N, K,
                     B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
                     C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
                     G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
                     num_warps: tl.constexpr, num_stages: tl.constexpr):
    # Grid is (B, C, S, S, N)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    acc = 0.0
    # Loop over k from 0 to K-1
    for k in range(0, K):
        b_j_k = tl.load(B_ptr + b * B_stride_b + c * B_stride_c + j * B_stride_j + n * B_stride_n + k * B_stride_k)
        c_i_k = tl.load(C_ptr + b * C_stride_b + c * C_stride_c + i * C_stride_i + n * C_stride_n + k * C_stride_k)
        acc += b_j_k * c_i_k

    # Store G[b, c, i, j, n]
    g_ptr = G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    tl.store(g_ptr, acc)


@triton.jit
def diag_contract_Y(M_ptr, HS_ptr, Y_ptr,
                     B_size, C_size, S, N, D,
                     M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
                     HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
                     Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
                     num_warps: tl.constexpr, num_stages: tl.constexpr):
    # Grid is (B, C, S, N, D)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    # Loop over j from 0 to S-1
    for j in range(0, S):
        m_ptr = M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n
        hs_ptr = HS_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d
        m = tl.load(m_ptr)
        hs = tl.load(hs_ptr)
        acc += m * hs

    y_ptr = Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d
    tl.store(y_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from original signature
        self.NUM_HEADS = 32
        self.N_GROUPS = 8
        # CHUNK_SIZE is defined but not used in original forward; keep for consistency (unused)

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        # Infer shapes
        assert hidden_states.dim() == 5, "hidden_states must be 5D [B, C, S, N, D]"
        Bsz, Csz, S, N, D = hidden_states.shape
        # A_cumsum shape: [B, N, C, S]
        assert A_cumsum.shape[0] == Bsz and A_cumsum.shape[3] == S, "A_cumsum shape mismatch"
        B_size, C_size, G, K = B.shape  # B_size, C_size should match hidden_states, G should be N_GROUPS
        assert B_size == Bsz and C_size == Csz, "B and C batch/num_chunks must match hidden_states"
        assert C.shape == (Bsz, Csz, S, G, K), "C shape mismatch"

        # Ensure tensors are on GPU for Triton; compute in float32 for numeric stability
        device = hidden_states.device
        assert device.type == "cuda", "Triton kernels require CUDA tensors"

        # 1) Compute L in Triton: masked cumsum along source index j for each (b, n, c, i), apply exp
        # A_cumsum: [B, N, C, S]
        A = A_cumsum
        L = torch.empty((Bsz, N, Csz, S, S), device=device, dtype=torch.float32)

        A_stride_b, A_stride_n, A_stride_c, A_stride_s = A.stride()
        L_stride_b, L_stride_n, L_stride_c, L_stride_i, L_stride_j = L.stride()

        grid_L = (Bsz, N, Csz, S)
        mask_cumsum_exp[grid_L](
            A, L,
            Bsz, N, Csz, S,
            A_stride_b, A_stride_n, A_stride_c, A_stride_s,
            L_stride_b, L_stride_n, L_stride_c, L_stride_i, L_stride_j,
            num_warps=1, num_stages=1
        )

        # 2) Expand B and C to num_heads (N=32) via repeat_interleave(4)
        # This matches the original: NUM_HEADS // N_GROUPS = 4
        B_expanded = B.repeat_interleave(4, dim=3)  # -> [B, C, S, N, K]
        C_expanded = C.repeat_interleave(4, dim=3)  # -> [B, C, S, N, K]

        # 3) Compute G in Triton: G[b, c, i, j, n] = sum_k C_expanded[b, c, i, n, k] * B_expanded[b, c, j, n, k]
        G = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k = B_expanded.stride()
        C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k = C_expanded.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()

        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G[grid_G](
            B_expanded, C_expanded, G,
            Bsz, Csz, S, N, K,
            B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
            C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            num_warps=1, num_stages=1
        )

        # 4) Elementwise multiply: M = G * L (PyTorch allowed for elementwise)
        M = G * L  # [B, C, S, S, N], float32

        # 5) Diagonal contraction: Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
        # hidden_states: [B, C, S, N, D], ensure float32 compute
        HS = hidden_states.to(torch.float32)
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

        # Return cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
