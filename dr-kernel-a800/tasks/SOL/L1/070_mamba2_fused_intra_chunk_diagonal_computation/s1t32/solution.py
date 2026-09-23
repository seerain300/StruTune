import torch
import triton
import triton.language as tl

# Host-side maximum constants for Triton compile-time loop bounds.
MAX_K = 128      # max chunk size (sequence length)
MAX_H = 32       # max number of heads
MAX_D = 64       # max head dimension (output last dim)
MAX_STATE = 64   # max state size for contraction

# Triton kernel: compute L = exp(cumsum(masked A)) for lower-triangular (diagonal=-1)
# A: [B, H, N, K, K], L: [B, H, N, K, K]
# We use compile-time MAX_K for loop bounds and mask off work beyond actual K.
@triton.jit
def masked_cumsum_tril_exp(A_ptr, L_ptr,
                           B_batch, B_heads, B_n,
                           A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j,
                           L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
                           K: tl.constexpr, H: tl.constexpr, N: tl.constexpr, MAX_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    i = tl.program_id(3)
    cumsum = 0.0
    for j in range(MAX_K):
        # Only include j <= i for the lower-triangular (diagonal=-1); mask by j < K
        use = (j < K) and (j <= i)
        if use:
            a_off = b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_i + j * A_stride_j
            val = tl.load(A_ptr + a_off)
            cumsum += val
        l_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
        # Store exp(cumsum) only if j < K; otherwise 1.0 as neutral for upper triangle
        tl.store(L_ptr + l_off, tl.exp(cumsum) if (j < K) else 1.0)

# Triton kernel: contract B and C into G with head expansion via REPEAT=4
# B: [B, N, K, n_groups, STATE_SIZE], C: [B, N, K, n_groups, STATE_SIZE]
# G: [B, N, K, K, H]
# We use compile-time constants for loops: K=MAX_K, STATE_SIZE, H=MAX_H, REPEAT=4.
@triton.jit
def contract_BC_to_G(B_ptr, C_ptr, G_ptr,
                     B_batch, B_n,
                     B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s,
                     C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s,
                     G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                     K: tl.constexpr, STATE_SIZE: tl.constexpr, H: tl.constexpr, REPEAT: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    g = h // REPEAT  # map head to group
    for i in range(K):
        for j in range(K):
            acc = 0.0
            # Accumulate over state_size in chunks; loops are compile-time
            for s_start in range(0, STATE_SIZE, 64):
                s = s_start + tl.arange(0, 64)
                mask_s = s < STATE_SIZE
                B_off = b * B_stride_b + n * B_stride_n + j * B_stride_k + g * B_stride_g + s * B_stride_s
                C_off = b * C_stride_b + n * C_stride_n + i * C_stride_k + g * C_stride_g + s * C_stride_s
                B_vals = tl.load(B_ptr + B_off, mask=mask_s, other=0.0)
                C_vals = tl.load(C_ptr + C_off, mask=mask_s, other=0.0)
                acc += tl.sum(B_vals * C_vals, axis=0)
            G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + j * G_stride_j + h * G_stride_h
            tl.store(G_ptr + G_off, acc)

# Triton kernel: final reduction to produce Y_diag
# G: [B, N, K, K, H], L: [B, H, N, K, K], hidden: [B, N, K, H, D], out: [B, N, K, H, D]
@triton.jit
def reduce_G_L_hidden(G_ptr, L_ptr, hidden_ptr, out_ptr,
                      B_batch, B_n,
                      G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                      L_stride_b, L_stride_h, L_stride_n, L_stride_i, L_stride_j,
                      hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d,
                      out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d,
                      K: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    for j in range(K):
        G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + j * G_stride_j + h * G_stride_h
        L_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
        M_val = tl.load(G_ptr + G_off)
        for dd in range(0, D, 32):
            dd_vec = dd + tl.arange(0, 32)
            mask_d = dd_vec < D
            hidden_off = b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_k + h * hidden_stride_h + dd_vec * hidden_stride_d
            hidden_vals = tl.load(hidden_ptr + hidden_off, mask=mask_d, other=0.0)
            out_off = b * out_stride_b + n * out_stride_n + i * out_stride_k + h * out_stride_h + dd_vec * out_stride_d
            acc = tl.sum(M_val * hidden_vals, axis=0)
            tl.store(out_ptr + out_off, acc, mask=mask_d)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor):
        # Expect shapes: hidden_states: [B, N, K, H, D]
        assert hidden_states.ndim == 5, "hidden_states must be [B, N, K, H, D]"
        B_size, N, K, H, D = hidden_states.shape

        device = hidden_states.device
        # Ensure contiguous and dtype float32 for computation
        A = A_cumsum.to(device).contiguous().to(torch.float32)
        Bt = B.to(device).contiguous().to(torch.float32)
        Ct = C.to(device).contiguous().to(torch.float32)
        hidden = hidden_states.to(device).contiguous().to(torch.float32)

        # Original mapping: NUM_HEADS=32, N_GROUPS=8, REPEAT=4
        N_GROUPS = 8
        REPEAT = 4
        # For this implementation, we assume H == N_GROUPS * REPEAT, which is 32. If not, we can still run but results may differ.
        # To be safe, we'll assert to match original behavior.
        assert N_GROUPS * REPEAT == H, "H must equal N_GROUPS * REPEAT (32 = 8 * 4) for this implementation."

        # Allocate outputs
        L = torch.empty((B_size, H, N, K, K), device=device, dtype=torch.float32)
        G = torch.empty((B_size, N, K, K, H), device=device, dtype=torch.float32)
        out = torch.empty((B_size, N, K, H, D), device=device, dtype=torch.float32)

        # Launch Triton kernels with compile-time constants for loop bounds.
        # Grid sizes based on actual runtime shapes.
        # Kernel 1: masked_cumsum_tril_exp to produce L
        grid_L = (B_size, H, N, K)
        masked_cumsum_tril_exp[grid_L](
            A, L,
            B_size, H, N,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3), A.stride(4),
            L.stride(0), L.stride(2), L.stride(3), L.stride(4), L.stride(1),
            K=K, H=H, N=N, MAX_K=MAX_K
        )

        # Kernel 2: contract_BC_to_G to produce G
        grid_G = (B_size, N, H)
        contract_BC_to_G[grid_G](
            Bt, Ct, G,
            B_size, N,
            Bt.stride(0), Bt.stride(1), Bt.stride(2), Bt.stride(3), Bt.stride(4),
            Ct.stride(0), Ct.stride(1), Ct.stride(2), Ct.stride(3), Ct.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            K=MAX_K, STATE_SIZE=Ct.shape[4], H=MAX_H, REPEAT=REPEAT
        )

        # Kernel 3: final reduction to produce out
        grid_out = (B_size, N, K, H)
        reduce_G_L_hidden[grid_out](
            G, L, hidden, out,
            B_size, N,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            K=MAX_K, H=MAX_H, D=D
        )

        # Return in bfloat16 to match original
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
