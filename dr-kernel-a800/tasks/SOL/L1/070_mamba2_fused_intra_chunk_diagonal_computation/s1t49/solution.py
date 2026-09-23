import torch
import triton
import triton.language as tl

# Constants matching the original code's hardcoded logic
CHUNK_SIZE = 128
NUM_HEADS = 32
N_GROUPS = 8
REPEAT = NUM_HEADS // N_GROUPS  # 4
STATE_SIZE = 64
HEAD_DIM = 64

# Triton kernel: create a lower-triangular mask (diagonal=-1) for 2D [K, K]
# mask_out: int8 [K, K], where 1 if j <= i, else 0
@triton.jit
def create_tril_minus1_mask(mask_out_ptr,
                            K: tl.constexpr,
                            mask_stride0, mask_stride1):
    i = tl.program_id(0)
    j = tl.program_id(1)
    val = 1 if j <= i else 0
    tl.store(mask_out_ptr + i * mask_stride0 + j * mask_stride1, val)

# Triton kernel: compute masked cumsum along j for each (b, h, n, i), then exp to produce L
# A: [B, H, N, K, K], L: [B, H, N, K, K]
# We apply lower-triangular mask (diagonal=-1): only j <= i contributes
@triton.jit
def masked_cumsum_tril_exp(A_ptr, L_ptr, mask_2d_ptr,
                           B_batch, B_heads, B_n,
                           A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j,
                           L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
                           mask_stride0, mask_stride1,
                           K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    i = tl.program_id(3)
    cumsum = 0.0
    for j in range(K):
        # Check triangular condition and mask
        if j <= i:
            a_off = b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_i + j * A_stride_j
            val = tl.load(A_ptr + a_off)
            cumsum += val
        # Store exp(cumsum) as L (diagonal=-1: no contribution above the main diagonal)
        l_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
        tl.store(L_ptr + l_off, tl.exp(cumsum))

# Triton kernel: contract B and C into G: [B, N, K, K, H]
# B: [B, N, K, n_groups, STATE_SIZE], C: [B, N, K, n_groups, STATE_SIZE]
# g = h // REPEAT (REPEAT=4), so groups=N_GROUPS=8
@triton.jit
def contract_BC_to_G(B_ptr, C_ptr, G_ptr,
                     B_batch, B_n, B_K, B_ng, B_STATE,
                     G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                     B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s,
                     C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s,
                     K: tl.constexpr, BLOCK_S: tl.constexpr, H: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    g = h // REPEAT  # groups index
    for i in range(K):
        for j in range(K):
            acc = 0.0
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

# Triton kernel: final reduction to produce Y_diag: [B, N, K, H, D]
# G: [B, N, K, K, H], L: [B, H, N, K, K], hidden: [B, N, K, H, D]
@triton.jit
def final_reduce(G_ptr, L_ptr, hidden_ptr, out_ptr,
                 B_batch, B_n, B_K, B_H, B_D,
                 G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                 L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
                 hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d,
                 out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d,
                 K: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    for j in range(K):
        # Load G[i, j, h] and L[h, n, i, j] and compute elementwise M = G * L
        G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + j * G_stride_j + h * G_stride_h
        L_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
        G_val = tl.load(G_ptr + G_off)
        L_val = tl.load(L_ptr + L_off)
        M_val = G_val * L_val
        # Accumulate over hidden dimension d
        acc = 0.0
        for d_start in range(0, D, BLOCK_D):
            d = d_start + tl.arange(0, BLOCK_D)
            mask_d = d < D
            hidden_off = b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_k + h * hidden_stride_h + d * hidden_stride_d
            vals = tl.load(hidden_ptr + hidden_off, mask=mask_d, other=0.0)
            acc += tl.sum(vals, axis=0)
        # Store acc (float32)
        out_off = b * out_stride_b + n * out_stride_n + i * out_stride_k + h * out_stride_h  # d is implicit in outer loop
        tl.store(out_ptr + out_off, acc)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Extract shapes (expect original hardcoded logic)
        B_batch, N, K, H, D = hidden_states.shape
        assert H == NUM_HEADS, f"Expected H={NUM_HEADS}, got {H}"
        assert D == HEAD_DIM, f"Expected D={HEAD_DIM}, got {D}"
        assert K == CHUNK_SIZE, f"Expected K={CHUNK_SIZE}, got {K}"
        assert A_cumsum.shape == (B_batch, H, N, K, K), f"A_cumsum shape mismatch: {A_cumsum.shape} vs {(B_batch, H, N, K, K)}"
        assert B.shape[0] == B_batch and B.shape[1] == N and B.shape[2] == K and B.shape[3] == N_GROUPS and B.shape[4] == STATE_SIZE
        assert C.shape == B.shape, "C shape must match B"

        # Ensure computations are in float32 in Triton
        A = A_cumsum.to(torch.float32)
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)
        hidden_f32 = hidden_states.to(torch.float32)

        # Allocate output [B, N, K, H, D] as float32 for compute
        out = torch.empty((B_batch, N, K, H, D), dtype=torch.float32, device=hidden_states.device)

        # Allocate intermediate L [B, H, N, K, K] and mask [K, K] (int8)
        L = torch.empty((B_batch, H, N, K, K), dtype=torch.float32, device=hidden_states.device)
        mask_2d = torch.empty((K, K), dtype=torch.int8, device=hidden_states.device)

        # Create mask in Triton
        grid_mask = (K, K)
        create_tril_minus1_mask[grid_mask](mask_2d, K, mask_2d.stride(0), mask_2d.stride(1))

        # Compute L via masked cumsum and exp in Triton
        grid_L = (B_batch, H, N, K)
        masked_cumsum_tril_exp[grid_L](
            A, L, mask_2d,
            B_batch, H, N,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3), A.stride(4),
            L.stride(0), L.stride(3), L.stride(2), L.stride(3), L.stride(1),
            mask_2d.stride(0), mask_2d.stride(1),
            K=K
        )

        # Allocate G [B, N, K, K, H] float32
        G = torch.empty((B_batch, N, K, K, H), dtype=torch.float32, device=hidden_states.device)

        # Launch contraction kernel
        grid_G = (B_batch, N, H)
        contract_BC_to_G[grid_G](
            B_f32, C_f32, G,
            B_batch, N, K, N_GROUPS, STATE_SIZE,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            B_f32.stride(0), B_f32.stride(1), B_f32.stride(2), B_f32.stride(3), B_f32.stride(4),
            C_f32.stride(0), C_f32.stride(1), C_f32.stride(2), C_f32.stride(3), C_f32.stride(4),
            K=K, BLOCK_S=64, H=H, num_warps=4, num_stages=2
        )

        # Final reduction to Y_diag
        grid_reduce = (B_batch, N, K, H)
        final_reduce[grid_reduce](
            G, L, hidden_f32, out,
            B_batch, N, K, H, D,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(3), L.stride(2), L.stride(3), L.stride(1),
            hidden_f32.stride(0), hidden_f32.stride(1), hidden_f32.stride(2), hidden_f32.stride(3), hidden_f32.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            K=K, D=D, BLOCK_D=64, num_warps=4, num_stages=2
        )

        # Return as bfloat16, matching original function signature
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
