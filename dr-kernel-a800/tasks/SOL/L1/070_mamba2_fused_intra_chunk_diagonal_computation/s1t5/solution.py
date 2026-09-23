import torch
import triton
import triton.language as tl

# Constants matching the original code
CHUNK_SIZE = 128  # K
NUM_HEADS = 32
N_GROUPS = 8
REPEAT = NUM_HEADS // N_GROUPS  # 4

@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    B_batch, B_n, B_K, B_groups, B_state,
    C_batch, C_n, C_K, C_groups, C_state,
    G_batch, G_n, G_K, G_heads,  # G: [B, N, K, K, H]
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    g = h // REPEAT  # n_groups index for this head

    # Accumulator for G[i, j, h]
    for i in range(CHUNK_SIZE):
        acc = 0.0
        for j in range(CHUNK_SIZE):
            # Accumulate over state_size in blocks
            for s_start in range(0, B_state, BLOCK_S):
                s = s_start + tl.arange(0, BLOCK_S)
                mask_s = s < B_state
                # B[b, n, j, g, s]
                B_off = b * B_batch + n * B_n + j * B_K + g * B_groups + s * B_state
                B_vals = tl.load(B_ptr + B_off, mask=mask_s, other=0.0)
                # C[b, n, i, g, s]
                C_off = b * C_batch + n * C_n + i * C_K + g * C_groups + s * C_state
                C_vals = tl.load(C_ptr + C_off, mask=mask_s, other=0.0)
                # acc += sum(B * C)
                acc += tl.sum(B_vals * C_vals, axis=0)
            # Store G[b, n, i, j, h] = acc
            G_off = b * G_batch + n * G_n + i * G_K + j * G_K + h * G_heads
            tl.store(G_ptr + G_off, acc)

@triton.jit
def elementwise_mul_LG_kernel(
    G_ptr, L_ptr, M_ptr,
    B_batch, B_n, B_K, B_heads, B_K2,  # B_K2 is K for L
    G_stride_b, G_stride_n, G_stride_k1, G_stride_k2, G_stride_h,
    L_stride_b, L_stride_n, L_stride_k1, L_stride_k2, L_stride_h,
    M_stride_b, M_stride_n, M_stride_k1, M_stride_k2, M_stride_h
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

@triton.jit
def final_reduce_kernel(
    M_ptr, hidden_ptr, out_ptr,
    B_batch, B_n, B_K, B_heads, B_D,
    M_stride_b, M_stride_n, M_stride_k1, M_stride_k2, M_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d,
    out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d,
    BLOCK_D: tl.constexpr
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    # Vectorize over d
    for d_start in range(0, B_D, BLOCK_D):
        d = d_start + tl.arange(0, BLOCK_D)
        mask_d = d < B_D
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        # Loop over j in [0..K-1]
        for j in range(CHUNK_SIZE):
            M_off = b * M_stride_b + n * M_stride_n + i * M_stride_k1 + j * M_stride_k2 + h * M_stride_h
            M_val = tl.load(M_ptr + M_off)
            hidden_off = b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_k + h * hidden_stride_h + d * hidden_stride_d
            hidden_vals = tl.load(hidden_ptr + hidden_off, mask=mask_d, other=0.0)
            acc += M_val * hidden_vals
        out_off = b * out_stride_b + n * out_stride_n + i * out_stride_k + h * out_stride_h + d * out_stride_d
        tl.store(out_ptr + out_off, acc, mask=mask_d)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_diag using Triton kernels. Returns tensor of shape [B, N, K, H, D] in bfloat16.
        """
        device = hidden_states.device
        K = CHUNK_SIZE

        # Compute L in torch: apply lower-tri mask (diagonal=-1), cumsum along j, then exp
        # A_cumsum: [B, H, N, K, K]
        tril_mask = torch.tril(torch.ones((K, K), device=device, dtype=torch.bool), diagonal=-1)
        # Broadcast mask to [1,1,1,K,K]
        tril_mask_broadcast = tril_mask.view(1, 1, 1, K, K)
        A_masked = A_cumsum.masked_fill(~tril_mask_broadcast, 0.0)
        # cumsum along last dim (j) for each fixed (b,h,n,i): cumsum over dim=3
        # Note: this cumsum is over the "source" axis; the original code says "cumsum(A)" without axis,
        # so we interpret it as along j. This matches the intended lower-tri cumsum.
        A_cumsum_masked = torch.cumsum(A_masked, dim=3)
        L = torch.exp(A_cumsum_masked.to(torch.float32))  # [B,H,N,K,K]

        # Prepare B, C for contraction: [B, N, K, groups, state_size]
        B = B.to(torch.float32)
        C = C.to(torch.float32)

        B_batch, B_n, B_K, B_groups, B_state = B.shape
        C_batch, C_n, C_K, C_groups, C_state = C.shape
        assert B_batch == C_batch and B_n == C_n, "B and C batch/num_chunks must match"
        assert B_K == C_K and B_groups == C_groups and B_state == C_state, "B and C shapes must match"
        assert B_K == K and C_K == K, "B/C K must equal CHUNK_SIZE"

        # Output G: [B, N, K, K, H]
        H = NUM_HEADS
        G = torch.empty((B_batch, B_n, K, K, H), dtype=torch.float32, device=device)

        # Launch contraction kernel: grid over (B, N, H)
        grid = (B_batch, B_n, H)
        contract_BC_to_G_kernel[grid](
            B, C, G,
            B_batch, B_n, K, B_groups, B_state,
            C_batch, C_n, K, C_groups, C_state,
            G.shape[0], G.shape[1], G.shape[2], H,
            BLOCK_S=64,
        )

        # Multiply elementwise by L: M = G * L, where L is [B,H,N,K,K] broadcast to [B,N,K,K,H]
        L_perm = L.permute(0, 2, 3, 1, 4).contiguous()  # [B,N,K,K,H]
        M = torch.empty_like(G, dtype=torch.float32, device=device)

        grid_mul = (B_batch, B_n, K, K, H)
        elementwise_mul_LG_kernel[grid_mul](
            G, L_perm, M,
            G.shape[0], G.shape[1], K, H, K,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L_perm.stride(0), L_perm.stride(1), L_perm.stride(2), L_perm.stride(3), L_perm.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
        )

        # Final reduction: hidden_states: [B, N, K, H, D]
        hidden = hidden_states.to(torch.float32)
        B_batch_h, B_n_h, B_K_h, B_heads_h, B_D = hidden.shape
        assert B_batch_h == B_batch and B_n_h == B_n and B_K_h == K and B_heads_h == H, "hidden states must match expected dims"
        out = torch.empty((B_batch, B_n, K, H, B_D), dtype=torch.float32, device=device)

        grid_reduce = (B_batch, B_n, K, H)
        final_reduce_kernel[grid_reduce](
            M, hidden, out,
            B_batch, B_n, K, H, B_D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            BLOCK_D=64,
        )

        # Return in bfloat16 as original
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
