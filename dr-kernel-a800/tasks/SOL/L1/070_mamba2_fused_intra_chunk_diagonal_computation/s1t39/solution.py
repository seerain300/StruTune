import torch
import triton
import triton.language as tl

# Constants to match original behavior (taken from the provided code)
CHUNK_SIZE = 128  # K
NUM_HEADS = 32    # H
HEAD_DIM = 64     # D
N_GROUPS = 8      # groups per chunk
REPEAT = 4        # H // (N_GROUPS * groups) = 32 // (8 * 1) -> but since B uses n_groups=8 and H=32, REPEAT=4 is correct for original setup
STATE_SIZE = 64   # state_size

# Triton kernel: create lower-triangular mask with diagonal=-1 (keep j <= i), output int8 0/1, shape [K, K]
@triton.jit
def create_tril_mask_int8(mask_ptr, K: tl.constexpr):
    i = tl.program_id(0)
    j = tl.program_id(1)
    # Bounds: K must be CHUNK_SIZE (128), but we still guard
    if i < K and j < K:
        if j <= i:
            tl.store(mask_ptr + i * K + j, 1)
        else:
            tl.store(mask_ptr + i * K + j, 0)

# Triton kernel: masked cumsum along last axis (j) for fixed (b,h,n,i), then exp into L
# A_cumsum: [B, H, N, K, K], L: [B, H, N, K, K], float32
@triton.jit
def masked_cumsum_exp_tril(A_ptr, L_ptr,
                           B_batch, B_heads, B_n,
                           A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j,
                           L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
                           K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    i = tl.program_id(3)
    # Only valid if i < K
    for j in range(K):
        if j <= i:
            a_off = b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_i + j * A_stride_j
            val = tl.load(A_ptr + a_off)  # float32
            cumsum = cumsum + val
        else:
            cumsum = cumsum  # keep as is
        l_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
        tl.store(L_ptr + l_off, tl.exp(cumsum))

# Triton kernel: contract B and C to produce G: [B, N, K, K, H]
# B: [B, N, K, n_groups, STATE_SIZE], C: [B, N, K, n_groups, STATE_SIZE]
@triton.jit
def contract_BC_to_G(B_ptr, C_ptr, G_ptr,
                     B_batch, B_n, B_K, B_ng, B_STATE,
                     G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                     B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s,
                     C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s,
                     K: tl.constexpr, H: tl.constexpr, BLOCK_S: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    # Map group index g = h // REPEAT
    g = h // REPEAT
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

# Triton kernel: final reduction over j for each (b, n, i, h): out[b, n, i, h, d] = sum_j G[b, n, i, j, h] * L[b, h, n, i, j]
# We vectorize over d (HEAD_DIM=64) to accumulate in float32, then store in out as float32. Host will cast to bfloat16.
@triton.jit
def reduce_G_L_to_out(G_ptr, L_ptr, out_ptr,
                      B_batch, B_n, B_K, B_H,
                      G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                      L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
                      out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d,
                      D: tl.constexpr, BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    # Accumulate across j = 0..K-1
    for j in range(B_K):
        G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + j * G_stride_j + h * G_stride_h
        L_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
        g_val = tl.load(G_ptr + G_off)
        l_val = tl.load(L_ptr + L_off)
        contrib = g_val * l_val
        # Add contrib into out[b, n, i, h, d] vector across d
        for d_start in range(0, D, BLOCK_D):
            d = d_start + tl.arange(0, BLOCK_D)
            mask_d = d < D
            out_off = b * out_stride_b + n * out_stride_n + i * out_stride_k + h * out_stride_h + d * out_stride_d
            out_vec = tl.load(out_ptr + out_off, mask=mask_d, other=0.0)
            out_vec = out_vec + contrib
            tl.store(out_ptr + out_off, out_vec, mask=mask_d)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, A_cumsum, B, C):
        # Ensure tensors are on CUDA for Triton
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Tensors must be on CUDA for Triton."
        B_batch, N, K, H, D = hidden_states.shape

        # Enforce constants for correctness (matching the original implementation)
        # If any mismatch occurs, fallback to torch to avoid Triton errors
        if not (K == CHUNK_SIZE and H == NUM_HEADS and D == HEAD_DIM):
            # Fallback to torch compute to avoid Triton OOB/shape issues
            # This mirrors the original logic without Triton for safety
            # Step 1: Create mask (tril with diagonal=-1)
            # Construct A implicitly from hidden (not used in original but for consistency)
            # We'll compute L via cumsum on A using torch where A = hidden[..., h, d] somehow.
            # To keep correctness for general inputs, fallback computes L via exp of cumsum of masked hidden (not available).
            # Since original doesn't provide A, fallback with torch may be incorrect. Therefore, we assert correctness.
            raise RuntimeError("Input shapes do not match the required constants for Triton path.")

        # Prepare A_cumsum in float32 (original expects float32 math)
        A = A_cumsum.to(torch.float32)

        # 1) Create tril mask int8 [K, K]
        mask = torch.empty((K, K), dtype=torch.int8, device=hidden_states.device)
        grid_mask = (K, K)
        create_tril_mask_int8[grid_mask](mask, K=K)

        # 2) Compute L = exp(cumsum(A with tril mask)) in float32: L: [B, H, N, K, K]
        L = torch.empty((B_batch, H, N, K, K), dtype=torch.float32, device=hidden_states.device)
        A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j = A.stride()
        L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h = L.stride()
        grid_L = (B_batch, H, N, K)
        masked_cumsum_exp_tril[grid_L](
            A, L,
            B_batch, H, N,
            A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j,
            L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
            K=K
        )

        # 3) Contract B and C to G: [B, N, K, K, H]
        # Ensure B, C have n_groups=8 and state_size=STATE_SIZE=64
        B_ng = B.size(3)
        if not (B_ng == N_GROUPS and B.size(4) == STATE_SIZE):
            raise RuntimeError("B's shape mismatch: expected [B, N, K, 8, 64].")
        C_ng = C.size(3)
        if not (C_ng == N_GROUPS and C.size(4) == STATE_SIZE):
            raise RuntimeError("C's shape mismatch: expected [B, N, K, 8, 64].")

        G = torch.empty((B_batch, N, K, K, H), dtype=torch.float32, device=hidden_states.device)
        B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s = B.stride()
        C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s = C.stride()
        G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h = G.stride()
        BLOCK_S = 64
        grid_G = (B_batch, N, H)
        contract_BC_to_G[grid_G](
            B, C, G,
            B_batch, N, K, N_GROUPS, STATE_SIZE,
            G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
            B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s,
            C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s,
            K=K, H=H, BLOCK_S=BLOCK_S
        )

        # 4) Final reduction: out float32 [B, N, K, H, D]
        out = torch.empty((B_batch, N, K, H, D), dtype=torch.float32, device=hidden_states.device)
        out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d = out.stride()
        G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h = G.stride()
        L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h = L.stride()
        grid_reduce = (B_batch, N, K, H)
        reduce_G_L_to_out[grid_reduce](
            G, L, out,
            B_batch, N, K, H,
            G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
            L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
            out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d,
            D=D, BLOCK_D=64
        )

        # Cast to bfloat16 to match original output dtype
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
