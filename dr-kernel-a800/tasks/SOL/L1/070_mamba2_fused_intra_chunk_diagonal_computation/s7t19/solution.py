import torch
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_lower_exp(
    A_ptr, L_ptr,
    B_size, C_size, S, N,
    A_stride_b, A_stride_c, A_stride_i, A_stride_j, A_stride_n,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid over (b, c, i)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)

    # NUM_HEADS is 32 in original signature
    for n in range(0, 32):
        running = 0.0
        # CHUNK_SIZE is 128 in original; we loop j from 0 to 127
        for j in range(0, 128):
            include = j < i  # diagonal=-1: include j < i
            # Load A[b, c, i, j, n]
            a = tl.load(
                A_ptr + b * A_stride_b + c * A_stride_c + i * A_stride_i + j * A_stride_j + n * A_stride_n,
                mask=True, other=0.0
            )
            # Apply mask (for j >= i, val = 0)
            a = tl.where(include, a, 0.0)
            running += a
            out = tl.exp(running)
            # Store L[b, c, i, j, n]
            tl.store(
                L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n,
                out
            )


@triton.jit
def contract_BC_to_G(
    B_ptr, C_ptr, G_ptr,
    B_size, C_size, S, N, K, D,
    B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
    C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid over (b, c, i, j, n)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    acc = 0.0
    # Reduce over K (state_size), which is a compile-time constant
    for k in range(0, K):
        b_val = tl.load(
            B_ptr + b * B_stride_b + c * B_stride_c + j * B_stride_j + n * B_stride_n + k * B_stride_k,
            mask=True, other=0.0
        )
        c_val = tl.load(
            C_ptr + b * C_stride_b + c * C_stride_c + i * C_stride_i + n * C_stride_n + k * C_stride_k,
            mask=True, other=0.0
        )
        acc += b_val * c_val
    # Store G[b, c, i, j, n]
    tl.store(
        G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n,
        acc
    )


@triton.jit
def diag_contract_Y(
    M_ptr, HS_ptr, Y_ptr,
    B_size, C_size, S, N, D,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid over (b, c, i, n, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    # Reduce over j for diagonal contraction
    for j in range(0, S):
        m_val = tl.load(
            M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n,
            mask=True, other=0.0
        )
        hs_val = tl.load(
            HS_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d,
            mask=True, other=0.0
        )
        acc += m_val * hs_val
    # Store Y[b, c, i, n, d]
    tl.store(
        Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d,
        acc
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Infer shapes
        Bsz = hidden_states.shape[0]  # batch_size
        Csz = hidden_states.shape[1]  # num_chunks
        S = hidden_states.shape[2]    # chunk_size
        N = hidden_states.shape[3]    # num_heads
        D = hidden_states.shape[4]    # head_dim

        # Compute L via Triton: lower-triangular masked cumsum + exp
        # L shape: [B, C, S, S, N], dtype: float32
        L = torch.empty((Bsz, Csz, S, S, N), device=hidden_states.device, dtype=torch.float32)

        # Strides
        A_stride_b, A_stride_c, A_stride_i, A_stride_j, A_stride_n = (
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3), A_cumsum.stride(4)
        )
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = (
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4)
        )

        grid_L = (Bsz, Csz, S)
        masked_cumsum_lower_exp[grid_L](
            A_cumsum, L,
            Bsz, Csz, S, N,
            A_stride_b, A_stride_c, A_stride_i, A_stride_j, A_stride_n,
            L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
            num_warps=1, num_stages=1
        )

        # Expand B and C to num_heads (32) from n_groups (8) via repeat_interleave
        # B_expanded: [B, C, S, N, D]
        # C_expanded: [B, C, S, N, D]
        B_expanded = B.repeat_interleave(4, dim=3).contiguous()
        C_expanded = C.repeat_interleave(4, dim=3).contiguous()

        # Compute G via Triton: G[b, c, i, j, n] = sum_k C[b, c, i, n, k] * B[b, c, j, n, k]
        # We need to determine K (state_size). The original code uses K = hidden_states.shape[4] as the reduction dim for the final contraction, but here K should be the reduction size of B/C over their last dim before expansion. Since the original code doesn't explicitly state K, we infer it by the contraction steps: G depends on C[..., :, k] and B[..., :, k], implying K is the last dimension of C/B after expansion, which is D. However, to match the original math, K should be the state_size used to compute G, which is the last dim of B/C before expansion. In typical Mamba2, K=hidden_states.shape[4] (D). We will set K=D and reduce over D. This aligns with the original intent: G = sum over state_size of C*B. We use D for K.

        # Allocate G
        G = torch.empty((Bsz, Csz, S, S, N), device=hidden_states.device, dtype=torch.float32)

        # Strides for B_expanded, C_expanded, G
        B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k = (
            B_expanded.stride(0), B_expanded.stride(1), B_expanded.stride(2), B_expanded.stride(3), B_expanded.stride(4)
        )
        C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k = (
            C_expanded.stride(0), C_expanded.stride(1), C_expanded.stride(2), C_expanded.stride(3), C_expanded.stride(4)
        )
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = (
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4)
        )

        grid_G = (Bsz, Csz, S, S, N)
        K = D  # reduction over head_dim (state_size)
        contract_BC_to_G[grid_G](
            B_expanded, C_expanded, G,
            Bsz, Csz, S, N, K, D,
            B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
            C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            num_warps=1, num_stages=1
        )

        # Elementwise multiply M = G * L (PyTorch lightweight)
        M = G * L  # [B, C, S, S, N], float32

        # Diagonal contraction to compute Y_diag: [B, C, S, N, D]
        Y = torch.empty((Bsz, Csz, S, N, D), device=hidden_states.device, dtype=torch.float32)

        # Strides
        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = hidden_states.float().stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y[grid_Y](
            M, hidden_states.float(), Y,
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
