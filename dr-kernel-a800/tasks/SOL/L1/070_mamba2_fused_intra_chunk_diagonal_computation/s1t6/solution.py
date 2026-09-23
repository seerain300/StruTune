import torch
import triton
import triton.language as tl

# Triton kernel: compute lower-triangular masked cumsum along j axis and store A_masked_cumsum
# Input A: [B, H, N, K, K] float32
# Output A_out: [B, H, N, K, K] float32
@triton.jit
def masked_cumsum_tril_kernel(
    A_ptr, A_out_ptr,
    B_batch, B_heads, B_n, B_K, B_K2,
    A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j,
    out_stride_b, out_stride_n, out_stride_i, out_stride_j, out_stride_h,
    K: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    for i in range(K):
        sum_val = 0.0
        for j in range(K):
            # lower-triangular mask with diagonal=-1: include only if j <= i
            include = j <= i
            a_off = b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_i + j * A_stride_j
            val = tl.load(A_ptr + a_off)
            sum_val += tl.where(include, val, 0.0)
        for jj in range(K):
            a_out_off = b * out_stride_b + h * out_stride_h + n * out_stride_n + i * out_stride_i + jj * out_stride_j
            tl.store(A_out_ptr + a_out_off, sum_val)

# Triton kernel: apply exp to form L
@triton.jit
def exp_kernel(
    in_ptr, out_ptr,
    B_batch, B_heads, B_n, B_K, B_K2,
    in_stride_b, in_stride_h, in_stride_n, in_stride_i, in_stride_j,
    out_stride_b, out_stride_n, out_stride_i, out_stride_j, out_stride_h,
    K: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    for i in range(K):
        for j in range(K):
            in_off = b * in_stride_b + h * in_stride_h + n * in_stride_n + i * in_stride_i + j * in_stride_j
            val = tl.load(in_ptr + in_off)
            out_off = b * out_stride_b + h * out_stride_h + n * out_stride_n + i * out_stride_i + j * out_stride_j
            tl.store(out_ptr + out_off, tl.exp(val))

# Triton kernel: contraction B -> C -> G: G[b, n, i, j, h] = sum over state of C[b,n,i,g,s]*B[b,n,j,g,s]
# B: [B, N, K, groups, state], C: [B, N, K, groups, state], G: [B, N, K, K, H]
@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    B_batch, B_n, B_K, B_groups, B_state,
    C_batch, C_n, C_K, C_groups, C_state,
    G_batch, G_n, G_K, G_heads,
    K: tl.constexpr, STATE: tl.constexpr, REPEAT: tl.constexpr, BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    g = h // REPEAT
    for i in range(K):
        acc = 0.0
        for j in range(K):
            # Accumulate over state dimension in blocks
            for s_start in range(0, STATE, BLOCK_S):
                s = s_start + tl.arange(0, BLOCK_S)
                mask_s = s < STATE
                B_off = b * B_batch + n * B_n + j * B_K + g * B_groups + s * B_state
                C_off = b * C_batch + n * C_n + i * C_K + g * C_groups + s * C_state
                B_vals = tl.load(B_ptr + B_off, mask=mask_s, other=0.0)
                C_vals = tl.load(C_ptr + C_off, mask=mask_s, other=0.0)
                acc += tl.sum(B_vals * C_vals, axis=0)
            G_off = b * G_batch + n * G_n + i * G_K + j * G_K + h * G_heads
            tl.store(G_ptr + G_off, acc)

# Triton kernel: elementwise multiply M = G * L
# G: [B, N, K, K, H], L: [B, H, N, K, K], M: [B, N, K, K, H]
@triton.jit
def elem_mul_GL_kernel(
    G_ptr, L_ptr, M_ptr,
    B_batch, B_n, B_K, B_heads, B_K2,  # B_K2 is K for L
    G_stride_b, G_stride_n, G_stride_k1, G_stride_k2, G_stride_h,
    L_stride_b, L_stride_h, L_stride_n, L_stride_k1, L_stride_k2,
    M_stride_b, M_stride_n, M_stride_k1, M_stride_k2, M_stride_h,
    K: tl.constexpr, H: tl.constexpr
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)
    G_off = b * G_stride_b + n * G_stride_n + i * G_stride_k1 + j * G_stride_k2 + h * G_stride_h
    L_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_k1 + j * L_stride_k2
    M_off = b * M_stride_b + n * M_stride_n + i * M_stride_k1 + j * M_stride_k2 + h * M_stride_h
    G_val = tl.load(G_ptr + G_off)
    L_val = tl.load(L_ptr + L_off)
    tl.store(M_ptr + M_off, G_val * L_val)

# Triton kernel: final reduction to Y_diag: [B, N, K, H, D]
# M: [B, N, K, K, H], hidden: [B, N, K, H, D]
@triton.jit
def reduce_M_hidden_kernel(
    M_ptr, hidden_ptr, out_ptr,
    B_batch, B_n, B_K, B_heads, B_K2, B_D,
    M_stride_b, M_stride_n, M_stride_k1, M_stride_k2, M_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d,
    out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d,
    K: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    for d_start in range(0, B_D, BLOCK_D):
        d = d_start + tl.arange(0, BLOCK_D)
        mask_d = d < B_D
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for j in range(K):
            M_off = b * M_stride_b + n * M_stride_n + i * M_stride_k1 + j * M_stride_k2 + h * M_stride_h
            hidden_off = b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_k + h * hidden_stride_h + d * hidden_stride_d
            M_val = tl.load(M_ptr + M_off)
            hidden_vals = tl.load(hidden_ptr + hidden_off, mask=mask_d, other=0.0)
            acc += M_val * hidden_vals
        out_off = b * out_stride_b + n * out_stride_n + i * out_stride_k + h * out_stride_h + d * out_stride_d
        tl.store(out_ptr + out_off, acc, mask=mask_d)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, N, K, H, D]
        B_batch, B_n, B_K, B_heads, B_D = hidden_states.shape
        # A_cumsum: [B, H, N, K, K]
        A_batch, A_heads, A_n, A_K1, A_K2 = A_cumsum.shape
        # Ensure consistency: N dimension is B_n
        assert A_n == B_n, "num_chunks (N) from A_cumsum must match hidden_states"
        # B and C: [B, N, K, groups, state]
        B_batch_b, B_n_b, B_K_b, B_groups, B_state = B.shape
        C_batch_c, C_n_c, C_K_c, C_groups, C_state = C.shape
        assert B_batch_b == B_batch and B_n_b == B_n and B_K_b == B_K and B_groups == C_groups and C_n_c == B_n and C_K_c == B_K, "B and C must have matching batch, num_chunks, K, and groups"

        # Make tensors contiguous
        hidden = hidden_states.contiguous()
        A = A_cumsum.contiguous()
        B_t = B.contiguous()
        C_t = C.contiguous()

        # Constants
        K = B_K  # chunk size
        H = B_heads  # num heads
        N = B_n  # num chunks
        D = B_D  # head_dim
        REPEAT = H // 8  # map heads to groups as in original (NUM_HEADS=32, N_GROUPS=8)

        # 1) Compute A_masked_cumsum (masked cumsum along j axis)
        A_masked_cumsum = torch.empty((B_batch, A_heads, N, K, K), dtype=torch.float32, device=hidden.device)
        masked_cumsum_tril_kernel[(B_batch, A_heads, N)](
            A, A_masked_cumsum,
            B_batch, A_heads, N, K, K,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3), A.stride(4),
            A_masked_cumsum.stride(0), A_masked_cumsum.stride(2), A_masked_cumsum.stride(3), A_masked_cumsum.stride(4), A_masked_cumsum.stride(1),
            K=K, num_warps=4
        )

        # 2) Compute L = exp(A_masked_cumsum)
        L = torch.empty_like(A_masked_cumsum, dtype=torch.float32, device=hidden.device)
        exp_kernel[(B_batch, A_heads, N)](
            A_masked_cumsum, L,
            B_batch, A_heads, N, K, K,
            A_masked_cumsum.stride(0), A_masked_cumsum.stride(1), A_masked_cumsum.stride(2), A_masked_cumsum.stride(3), A_masked_cumsum.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            K=K, num_warps=4
        )

        # 3) Contraction to G: [B, N, K, K, H]
        G = torch.empty((B_batch, N, K, K, H), dtype=torch.float32, device=hidden.device)
        contract_BC_to_G_kernel[(B_batch, N, H)](
            B_t, C_t, G,
            B_batch, N, K, B_groups, B_state,
            C_batch_c, N, K, C_groups, C_state,
            B_batch, N, K, H,
            K=K, STATE=B_state, REPEAT=REPEAT, BLOCK_S=64, num_warps=4
        )

        # 4) Elementwise M = G * L
        M = torch.empty_like(G, dtype=torch.float32, device=hidden.device)
        elem_mul_GL_kernel[(B_batch, N, K, K, H)](
            G, L, M,
            B_batch, N, K, H, K,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            K=K, H=H, num_warps=4
        )

        # 5) Final reduction to Y_diag: [B, N, K, H, D]
        out = torch.empty((B_batch, N, K, H, D), dtype=torch.float32, device=hidden.device)
        reduce_M_hidden_kernel[(B_batch, N, K, H)](
            M, hidden, out,
            B_batch, N, K, H, K, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            K=K, H=H, D=D, BLOCK_D=64, num_warps=4
        )

        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
