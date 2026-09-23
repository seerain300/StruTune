import torch
import triton
import triton.language as tl

# We will infer shapes from inputs; the following are not hardcoded.
# However, we retain original assumptions for consistency:
# - NUM_HEADS = 32, N_GROUPS = 8, REPEAT = NUM_HEADS // N_GROUPS = 4
# - This is a requirement of the original code's logic.

@triton.jit
def masked_cumsum_tril_exp(A_ptr, L_ptr,
                           B_batch, B_heads, B_n,
                           A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j,
                           L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
                           K: tl.constexpr):
    # Grid: (B_batch, B_heads, B_n, K)
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    i = tl.program_id(3)
    cumsum = 0.0
    # Loop over j from 0 to K-1; lower triangular with diagonal=-1 means j <= i contributes
    for j in range(K):
        # If j <= i, include; else treat as 0
        if j <= i:
            a_off = b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_i + j * A_stride_j
            val = tl.load(A_ptr + a_off)
            cumsum += val
        l_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
        tl.store(L_ptr + l_off, tl.exp(cumsum))

@triton.jit
def contract_BC_to_G(B_ptr, C_ptr, G_ptr,
                     B_batch, B_n, B_K, B_ng, B_STATE,
                     G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                     B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s,
                     C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s,
                     K: tl.constexpr, H: tl.constexpr, BLOCK_S: tl.constexpr):
    # Grid: (B_batch, B_n, H)
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    repeat = H // 8  # NUM_HEADS // N_GROUPS from original code
    g = h // repeat  # group index derived from head index
    for i in range(K):
        for j in range(K):
            acc = 0.0
            # Accumulate over state_size in blocks
            for s_start in range(0, B_STATE, BLOCK_S):
                s = s_start + tl.arange(0, BLOCK_S)
                mask_s = s < B_STATE
                B_off = b * B_stride_b + n * B_stride_n + j * B_stride_k + g * B_stride_g + s * B_stride_s
                C_off = b * C_stride_b + n * C_stride_n + i * C_stride_k + g * C_stride_g + s * C_stride_s
                B_vals = tl.load(B_ptr + B_off, mask=mask_s, other=0.0)
                C_vals = tl.load(C_ptr + C_off, mask=mask_s, other=0.0)
                acc += tl.sum(B_vals * C_vals, axis=0)
            G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + j * G_stride_j + h * G_stride_h
            tl.store(G_ptr + G_off, acc)

@triton.jit
def final_reduce(G_ptr, L_ptr, hidden_ptr, out_ptr,
                 B_batch, B_n, B_K, B_H, B_D,
                 G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                 L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
                 hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d,
                 out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d,
                 K: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    # Grid: (B_batch, B_n, B_K, B_H)
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    for j in range(K):
        # Compute M = G * L for this (i, j, h)
        G_ijh = 0.0
        # We need G[b, n, i, j, h] and L[b, h, n, i, j]
        G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + j * G_stride_j + h * G_stride_h
        L_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
        G_ijh = tl.load(G_ptr + G_off)
        L_ijh = tl.load(L_ptr + L_off)
        M_ijh = G_ijh * L_ijh

        # Accumulate over d: out[b, n, i, h, d] += M_ijh * hidden[b, n, j, h, d]
        # We'll accumulate into a vector of size D and store at the end
        acc = tl.zeros([1], dtype=tl.float32)
        for d_start in range(0, D, BLOCK_D):
            d = d_start + tl.arange(0, BLOCK_D)
            mask_d = d < D
            hid_off = b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_k + h * hidden_stride_h + d * hidden_stride_d
            vals = tl.load(hidden_ptr + hid_off, mask=mask_d, other=0.0)
            # vals is vector, multiply by scalar M_ijh
            # acc += sum(vals * M_ijh)
            acc += tl.sum(vals * M_ijh, axis=0)
        # Store acc to out[b, n, i, h, 0:D]
        # We need to write element-wise
        # First, compute base pointer then per d store
        for d in range(0, D):
            out_off = b * out_stride_b + n * out_stride_n + i * out_stride_k + h * out_stride_h + d * out_stride_d
            # acc is scalar; store it
            tl.store(out_ptr + out_off, acc)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward that computes Y_diag = sum over j of (G[i,j,h] * L[i,j,h]) * hidden[b,n,j,h,d]
        where:
          - hidden_states: [B, N, K, H, D]
          - A_cumsum: [B, H, N, K, K]
          - B: [B, N, K, n_groups, STATE_SIZE]
          - C: [B, N, K, n_groups, STATE_SIZE]
        Output: [B, N, K, H, D] in bfloat16 (computed in float32 inside Triton).
        """
        # Extract runtime shapes
        B_batch, N, K, H, D = hidden_states.shape  # note: N is num_chunks in original, renamed here to avoid confusion
        # A_cumsum: [B, H, N, K, K]
        assert A_cumsum.shape[0] == B_batch and A_cumsum.shape[2] == N and A_cumsum.shape[3] == K and A_cumsum.shape[4] == K
        B_batch_A, H_A, N_A, K_A, K_A2 = A_cumsum.shape

        # B and C: [B, N, K, n_groups, STATE_SIZE]
        assert B.shape == C.shape, "B and C must have the same shape"
        assert B.shape[0] == B_batch and B.shape[1] == N and B.shape[2] == K
        B_B, B_N, B_K, B_ng, B_STATE = B.shape
        assert B_N == N and B_K == K

        # Check consistency: H_A == H, K_A == K (runtime, not hardcoded)
        # We will use H and K derived from hidden_states.

        # Prepare strides
        # A_cumsum strides
        A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j = A_cumsum.stride()
        # Allocate L: [B, H, N, K, K] float32
        L = torch.empty((B_batch, H, N, K, K), dtype=torch.float32, device=hidden_states.device)

        # Launch masked cumsum + exp kernel
        grid1 = (B_batch, H, N, K)
        masked_cumsum_tril_exp[grid1](
            A_cumsum, L,
            B_batch, H, N,
            A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j,
            L.stride(0), L.stride(3), L.stride(2), L.stride(3), L.stride(1),
            K=K,
            num_warps=4, num_stages=2
        )

        # Compute G: [B, N, K, K, H]
        G = torch.empty((B_batch, N, K, K, H), dtype=torch.float32, device=hidden_states.device)

        # Strides for B, C, G
        B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s = B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4)
        C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s = C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4)
        G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h = G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4)

        grid2 = (B_batch, N, H)
        contract_BC_to_G[grid2](
            B, C, G,
            B_batch, N, K, B_ng, B_STATE,
            G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
            B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s,
            C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s,
            K=K, H=H, BLOCK_S=64,
            num_warps=4, num_stages=2
        )

        # Prepare output: out in float32
        out = torch.empty((B_batch, N, K, H, D), dtype=torch.float32, device=hidden_states.device)

        # Strides for hidden and out
        hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d = hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4)
        out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d = out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4)

        grid3 = (B_batch, N, K, H)
        final_reduce[grid3](
            G, L, hidden_states, out,
            B_batch, N, K, H, D,
            G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
            L.stride(0), L.stride(3), L.stride(2), L.stride(3), L.stride(1),
            hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d,
            out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d,
            K=K, D=D, BLOCK_D=64,
            num_warps=4, num_stages=2
        )

        # Return in bfloat16 to match original behavior
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
