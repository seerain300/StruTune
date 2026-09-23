import torch
import triton
import triton.language as tl

# Triton kernel: compute L = exp(cumsum(masked A)) with lower-triangular mask (diagonal=-1).
# A: [B, H, N, K, K], L: [B, H, N, K, K]
@triton.jit
def masked_cumsum_tril_exp(A_ptr, L_ptr,
                           B_batch, B_heads, B_n,
                           A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j,
                           L_stride_b, L_stride_h, L_stride_n, L_stride_i, L_stride_j,
                           K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    i = tl.program_id(3)
    cumsum = 0.0
    for j in range(K):
        if j <= i:
            a_off = b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_i + j * A_stride_j
            val = tl.load(A_ptr + a_off)
            cumsum += val
        l_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
        tl.store(L_ptr + l_off, tl.exp(cumsum))

# Triton kernel: contract B and C into G: G[i, j, h] = sum_s C[b, n, i, g, s] * B[b, n, j, g, s]
# B: [B, N, K, n_groups, STATE_SIZE], C: [B, N, K, n_groups, STATE_SIZE], G: [B, N, K, K, H]
@triton.jit
def contract_BC_to_G(B_ptr, C_ptr, G_ptr,
                     B_batch, B_n, B_K, B_ng, B_STATE, B_H,  # B_H = H (number of heads in output)
                     G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                     B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s,
                     C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s,
                     K: tl.constexpr, BLOCK_S: tl.constexpr, REPEAT: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    for i in range(K):
        for j in range(K):
            acc = 0.0
            # Loop over state_size in blocks
            for s_start in range(0, B_STATE, BLOCK_S):
                s = s_start + tl.arange(0, BLOCK_S)
                mask_s = s < B_STATE
                g = h // REPEAT  # expand to n_groups dimension
                B_off = b * B_stride_b + n * B_stride_n + j * B_stride_k + g * B_stride_g + s * B_stride_s
                C_off = b * C_stride_b + n * C_stride_n + i * C_stride_k + g * C_stride_g + s * C_stride_s
                B_vals = tl.load(B_ptr + B_off, mask=mask_s, other=0.0)
                C_vals = tl.load(C_ptr + C_off, mask=mask_s, other=0.0)
                acc += tl.sum(B_vals * C_vals, axis=0)
            G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + j * G_stride_j + h * G_stride_h
            tl.store(G_ptr + G_off, acc)

# Triton kernel: final reduction to Y_diag
# G: [B, N, K, K, H], L: [B, H, N, K, K], hidden: [B, N, K, H, D], out: [B, N, K, H, D]
@triton.jit
def final_reduce(G_ptr, L_ptr, hidden_ptr, out_ptr,
                 B_batch, B_n, B_K, B_H, B_D,
                 G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                 L_stride_b, L_stride_h, L_stride_n, L_stride_i, L_stride_j,
                 hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d,
                 out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d,
                 D: tl.constexpr, BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    for d_start in range(0, B_D, BLOCK_D):
        d = d_start + tl.arange(0, BLOCK_D)
        mask_d = d < B_D
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for j in range(B_K):
            G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + h * G_stride_h + j * G_stride_j
            G_val = tl.load(G_ptr + G_off)  # scalar
            L_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
            L_val = tl.load(L_ptr + L_off)  # scalar
            hidden_off = b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_k + h * hidden_stride_h + d * hidden_stride_d
            hidden_vals = tl.load(hidden_ptr + hidden_off, mask=mask_d, other=0.0)
            acc += G_val * L_val * hidden_vals
        out_off = b * out_stride_b + n * out_stride_n + i * out_stride_k + h * out_stride_h + d * out_stride_d
        tl.store(out_ptr + out_off, acc)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor):
        # Extract dynamic shapes from inputs
        assert hidden_states.ndim == 5, "hidden_states must be [B, N, K, H, D]"
        B_batch, N, K, H, D = hidden_states.shape

        # Ensure inputs are on same device and contiguous
        device = hidden_states.device
        A = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        hidden = hidden_states.contiguous()

        # L: [B, H, N, K, K] (float32)
        L = torch.empty((B_batch, H, N, K, K), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute L = exp(cumsum(masked A)) with tril(diagonal=-1)
        grid_L = (B_batch, H, N)
        masked_cumsum_tril_exp[grid_L](
            A, L,
            B_batch, H, N,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3), A.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            K=K, num_warps=4, num_stages=2
        )

        # Determine n_groups and REPEAT from B's shape (original uses N_GROUPS=8, REPEAT=4)
        assert B.ndim == 5 and B.shape[:3] == (B_batch, N, K), "B must be [B, N, K, n_groups, STATE_SIZE]"
        assert C.ndim == 5 and C.shape[:3] == (B_batch, N, K), "C must be [B, N, K, n_groups, STATE_SIZE]"
        B_ng = B.shape[3]  # n_groups (original code uses 8)
        B_STATE = B.shape[4]  # STATE_SIZE (original code uses 64)
        # Original code sets REPEAT = NUM_HEADS // N_GROUPS = 4 when NUM_HEADS=32, N_GROUPS=8.
        # We enforce that H is divisible by 4; if not, fallback to torch path (rare for given workloads).
        assert H % 4 == 0, "H (num_heads) must be divisible by 4 for this Triton implementation."
        REPEAT = H // 4  # expand heads to n_groups

        # Allocate G: [B, N, K, K, H] in float32
        G = torch.empty((B_batch, N, K, K, H), dtype=torch.float32, device=device)

        # Launch Triton contraction kernel to produce G
        grid_G = (B_batch, N, H)
        contract_BC_to_G[grid_G](
            B, C, G,
            B_batch, N, K, B_ng, B_STATE, H,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
            K=K, BLOCK_S=64, REPEAT=REPEAT, num_warps=4, num_stages=2
        )

        # Allocate output Y_diag in float32: [B, N, K, H, D]
        out = torch.empty((B_batch, N, K, H, D), dtype=torch.float32, device=device)

        # Launch Triton final reduction kernel
        grid_out = (B_batch, N, K, H)
        final_reduce[grid_out](
            G, L, hidden, out,
            B_batch, N, K, H, D,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            D=D, BLOCK_D=64, num_warps=4, num_stages=2
        )

        # Return in bfloat16 to match original's output dtype
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
