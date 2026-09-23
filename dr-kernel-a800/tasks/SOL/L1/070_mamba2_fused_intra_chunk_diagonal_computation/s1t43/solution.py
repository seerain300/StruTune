import torch
import triton
import triton.language as tl

# ------------------------------
# Triton kernels
# ------------------------------

# Kernel 1: masked cumsum + exp to produce L: [B, H, N, K, K]
# A: [B, H, N, K, K], L: [B, H, N, K, K]
@triton.jit
def masked_cumsum_tril_exp(A_ptr, L_ptr,
                           B_batch, B_heads, B_n,
                           A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j,
                           L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
                           CHUNK: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    i = tl.program_id(3)
    cumsum = 0.0
    for j in range(CHUNK):
        # tril(diagonal=-1): only contribute if j <= i
        if j <= i:
            a_off = b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_i + j * A_stride_j
            val = tl.load(A_ptr + a_off)
            cumsum += val
        l_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
        tl.store(L_ptr + l_off, tl.exp(cumsum))

# Kernel 2: contract B and C into G: [B, N, K, K, H]
# B: [B, N, K, n_groups, STATE_SIZE], C: [B, N, K, n_groups, STATE_SIZE], H: num_heads
# We expand n_groups -> H via REPEAT (REPEAT = H // n_groups). We assert H % n_groups == 0.
@triton.jit
def contract_BC_to_G(B_ptr, C_ptr, G_ptr,
                     B_batch, B_n, B_K, B_n_groups, B_STATE, B_H,
                     G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                     B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s,
                     C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s,
                     CHUNK: tl.constexpr, BLOCK_S: tl.constexpr, REPEAT: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    g = h // REPEAT  # expand heads from n_groups to H
    for i in range(CHUNK):
        for j in range(CHUNK):
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

# Kernel 3: final reduction to Y_diag: [B, N, K, H, D]
# G: [B, N, K, K, H], L: [B, H, N, K, K], hidden: [B, N, K, H, D], out: [B, N, K, H, D]
@triton.jit
def final_reduce(G_ptr, L_ptr, hidden_ptr, out_ptr,
                 B_batch, B_n, B_K, B_H, B_D,
                 G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                 L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
                 hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d,
                 out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d,
                 CHUNK: tl.constexpr, BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for j in range(CHUNK):
        G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + j * G_stride_j + h * G_stride_h
        L_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
        G_val = tl.load(G_ptr + G_off)
        L_val = tl.load(L_ptr + L_off)
        coeff = G_val * L_val  # scalar
        # hidden[b, n, j, h, :] vector
        d = tl.arange(0, BLOCK_D)
        mask_d = d < B_D
        hidden_off = b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_k + h * hidden_stride_h + d * hidden_stride_d
        hidden_vec = tl.load(hidden_ptr + hidden_off, mask=mask_d, other=0.0)
        acc += coeff * hidden_vec
    # store accumulated result for all D
    out_off_base = b * out_stride_b + n * out_stride_n + i * out_stride_k + h * out_stride_h
    for d in range(0, B_D):
        out_off = out_off_base + d * out_stride_d
        tl.store(out_ptr + out_off, acc[d])

# ------------------------------
# ModelNew: Triton-only forward
# ------------------------------

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, A_cumsum, B, C):
        # hidden_states: [B, N, K, H, D]
        assert hidden_states.dim() == 5, "hidden_states must be [B, N, K, H, D]"
        B_batch, N, K, H, D = hidden_states.shape

        # A_cumsum: [B, H, N, K, K] (note: original code uses [H, N, K, K] in a function, but provided as [B, H, N, K, K] here)
        assert A_cumsum.dim() == 5, "A_cumsum must be [B, H, N, K, K]"
        B_batch_A, H_A, N_A, K_A, K_j = A_cumsum.shape
        assert B_batch == B_batch_A and N == N_A and K == K_A and K_j == K, "A_cumsum shape mismatch"

        # B: [B, N, K, n_groups, STATE_SIZE]
        assert B.dim() == 5, "B must be [B, N, K, n_groups, STATE_SIZE]"
        B_batch_B, N_B, K_B, n_groups, STATE = B.shape
        assert B_batch_B == B_batch and N_B == N and K_B == K, "B shape mismatch"

        # C: [B, N, K, n_groups, STATE_SIZE]
        assert C.dim() == 5, "C must be [B, N, K, n_groups, STATE_SIZE]"
        B_batch_C, N_C, K_C, n_groups_C, STATE_C = C.shape
        assert B_batch_C == B_batch and N_C == N and K_C == K and n_groups_C == n_groups and STATE_C == STATE, "C shape mismatch"

        # Ensure contiguous tensors
        A = A_cumsum.contiguous()
        B_contig = B.contiguous()
        C_contig = C.contiguous()
        hidden = hidden_states.contiguous()

        # Allocate outputs
        # L: [B, H, N, K, K]
        L = torch.empty((B_batch, H, N, K, K), device=hidden.device, dtype=hidden.dtype)
        # G: [B, N, K, K, H]
        G = torch.empty((B_batch, N, K, K, H), device=hidden.device, dtype=hidden.dtype)
        # out: [B, N, K, H, D] accumulate in float32
        out = torch.empty((B_batch, N, K, H, D), device=hidden.device, dtype=torch.float32)

        # REPEAT = NUM_HEADS // N_GROUPS (original code uses NUM_HEADS=32, N_GROUPS=8 => REPEAT=4)
        # We infer H and n_groups from shapes. The original code expands n_groups -> H via repeat_interleave(REPEAT).
        # So REPEAT = H // n_groups, and must be a positive integer. We assert this.
        REPEAT = H // n_groups
        assert H % n_groups == 0, "H must be divisible by n_groups"
        assert REPEAT > 0, "REPEAT must be positive"

        # Launch kernel 1: masked cumsum + exp -> L
        grid_L = (B_batch, H, N, K)
        masked_cumsum_tril_exp[grid_L](
            A, L,
            B_batch, H, N,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3), A.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            CHUNK=K,
            num_warps=4, num_stages=2
        )

        # Launch kernel 2: contract B and C to G
        grid_G = (B_batch, N, H)
        contract_BC_to_G[grid_G](
            B_contig, C_contig, G,
            B_batch, N, K, n_groups, STATE, H,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            B_contig.stride(0), B_contig.stride(1), B_contig.stride(2), B_contig.stride(3), B_contig.stride(4),
            C_contig.stride(0), C_contig.stride(1), C_contig.stride(2), C_contig.stride(3), C_contig.stride(4),
            CHUNK=K, BLOCK_S=64, REPEAT=REPEAT,
            num_warps=4, num_stages=2
        )

        # Launch kernel 3: final reduction to Y_diag
        grid_out = (B_batch, N, K, H)
        final_reduce[grid_out](
            G, L, hidden, out,
            B_batch, N, K, H, D,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            CHUNK=K, BLOCK_D=64,
            num_warps=4, num_stages=2
        )

        # Return in bfloat16 as per original run's return (they cast to bfloat16 at the end)
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
