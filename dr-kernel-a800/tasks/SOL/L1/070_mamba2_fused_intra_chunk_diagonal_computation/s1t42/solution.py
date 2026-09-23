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
        if j <= i:
            a_off = b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_i + j * A_stride_j
            val = tl.load(A_ptr + a_off)
            cumsum += val
        l_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
        tl.store(L_ptr + l_off, tl.exp(cumsum))

# Kernel 2: contract B and C into G: [B, N, K, K, H]
# B: [B, N, K, n_groups, STATE], C: [B, N, K, n_groups, STATE]
# Use g = h // REPEAT (REPEAT=4). REPEAT must divide H.
@triton.jit
def contract_BC_to_G(B_ptr, C_ptr, G_ptr,
                     B_batch, B_n, B_K, B_STATE, REPEAT: tl.constexpr,
                     G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                     B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s,
                     C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s,
                     CHUNK: tl.constexpr, BLOCK_S: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    g = h // REPEAT
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
# G: [B, N, K, K, H], L: [B, N, K, K, H], hidden: [B, N, K, H, D], out: [B, N, K, H, D]
@triton.jit
def final_reduce_matmul_L(G_ptr, L_ptr, hidden_ptr, out_ptr,
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
    acc = tl.zeros((), dtype=tl.float32)
    for j in range(CHUNK):
        G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + j * G_stride_j + h * G_stride_h
        L_off = b * L_stride_b + n * L_stride_n + i * L_stride_i + j * L_stride_j + h * L_stride_h
        G_val = tl.load(G_ptr + G_off)
        L_val = tl.load(L_ptr + L_off)
        prod = G_val * L_val
        # Vectorize over D
        d_vec = tl.arange(0, BLOCK_D)
        mask_d = d_vec < B_D
        hidden_off = b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_k + h * hidden_stride_h + d_vec * hidden_stride_d
        hidden_vals = tl.load(hidden_ptr + hidden_off, mask=mask_d, other=0.0)
        acc += tl.sum(prod * hidden_vals, axis=0)
    # Store result
    out_off = b * out_stride_b + n * out_stride_n + i * out_stride_k + h * out_stride_h
    for d in range(B_D):
        tl.store(out_ptr + out_off + d * out_stride_d, acc)

# ------------------------------
# ModelNew: entry point
# ------------------------------

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, A_cumsum, B, C):
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA for Triton."
        hidden = hidden_states.contiguous()
        A = A_cumsum.contiguous()
        Bc = B.contiguous()
        Cc = C.contiguous()

        # Extract shapes from inputs
        B_batch, B_n, B_K, B_H, B_D = hidden.shape  # [B, N, K, H, D]
        A_shape = A.shape
        assert A_shape == (B_batch, B_H, B_n, B_K, B_K), f"A_cumsum shape must be [B, H, N, K, K], got {A_shape}"
        B_shape = Bc.shape
        C_shape = Cc.shape
        # Assert expected shapes consistent with original code
        assert B_shape[0] == B_batch and B_shape[1] == B_n and B_shape[2] == B_K and B_shape[3] == 8 and B_shape[4] == 64, \
            f"B shape expected [B, N, K, 8, 64], got {B_shape}"
        assert C_shape[0] == B_batch and C_shape[1] == B_n and C_shape[2] == B_K and C_shape[3] == 8 and C_shape[4] == 64, \
            f"C shape expected [B, N, K, 8, 64], got {C_shape}"

        # Constants
        N_GROUPS = 8  # original code
        NUM_HEADS = 32  # original code
        REPEAT = NUM_HEADS // N_GROUPS  # 4
        assert B_H % REPEAT == 0, f"NUM_HEADS={NUM_HEADS} must divide NUM_HEADS//N_GROUPS={REPEAT} into H={B_H}"

        BLOCK_S = 64  # match STATE_SIZE=64
        BLOCK_D = 64  # match D=64

        # Allocate L and G in float32 for computation
        L = torch.empty((B_batch, B_H, B_n, B_K, B_K), device=hidden.device, dtype=torch.float32)
        G = torch.empty((B_batch, B_n, B_K, B_K, B_H), device=hidden.device, dtype=torch.float32)

        # Launch masked cumsum + exp to produce L
        grid_L = (B_batch, B_H, B_n, B_K)  # program over (b, h, n, i)
        masked_cumsum_tril_exp[grid_L](
            A, L,
            B_batch, B_H, B_n,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3), A.stride(4),
            L.stride(0), L.stride(2), L.stride(3), L.stride(4), L.stride(1),
            CHUNK=B_K,
            num_warps=4, num_stages=2
        )

        # Launch contraction BC -> G
        grid_G = (B_batch, B_n, B_H)  # program over (b, n, h)
        contract_BC_to_G[grid_G](
            Bc, Cc, G,
            B_batch, B_n, B_K, 64, REPEAT,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            Bc.stride(0), Bc.stride(1), Bc.stride(2), Bc.stride(3), Bc.stride(4),
            Cc.stride(0), Cc.stride(1), Cc.stride(2), Cc.stride(3), Cc.stride(4),
            CHUNK=B_K, BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        # Final reduction to Y_diag
        out = torch.empty((B_batch, B_n, B_K, B_H, B_D), device=hidden.device, dtype=torch.float32)
        grid_out = (B_batch, B_n, B_K, B_H)  # program over (b, n, i, h)
        final_reduce_matmul_L[grid_out](
            G, L, hidden, out,
            B_batch, B_n, B_K, B_H, B_D,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(2), L.stride(3), L.stride(4), L.stride(1),
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            CHUNK=B_K, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2
        )

        # Return in bfloat16 to match original
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
