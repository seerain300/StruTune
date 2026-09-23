import torch
import triton
import triton.language as tl

# Fixed constants matching the original example code
K = 128         # CHUNK_SIZE
H = 32          # NUM_HEADS
N_GROUPS = 8    # hard-coded in original
REPEAT = H // N_GROUPS  # 4
STATE_SIZE = 64
D = 64          # HEAD_DIM

# Triton kernel: compute L = exp(cumsum(masked A)) with lower-triangular mask (diagonal=-1)
# A_cumsum: [B, H, N, K, K] float32
# L: [B, H, N, K, K] float32
@triton.jit
def masked_cumsum_tril_exp(A_ptr, L_ptr,
                           B_batch, B_heads, B_n,
                           A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j,
                           L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
                           K_CONST: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    i = tl.program_id(3)
    cumsum = 0.0
    for j in range(K_CONST):
        if j <= i:
            a_off = b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_i + j * A_stride_j
            val = tl.load(A_ptr + a_off)  # float32
            cumsum += val
        l_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
        tl.store(L_ptr + l_off, tl.exp(cumsum))

# Triton kernel: contract B and C into G: G[i, j, h] = sum_s C[b, n, i, g, s] * B[b, n, j, g, s]
# B: [B, N, K, n_groups, STATE_SIZE], C: [B, N, K, n_groups, STATE_SIZE]
# Write G: [B, N, K, K, H] float32
@triton.jit
def contract_BC_to_G(B_ptr, C_ptr, G_ptr,
                     B_batch, B_n, B_K, B_ng, B_STATE,
                     G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                     B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s,
                     C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s,
                     K_CONST: tl.constexpr, STATE_CONST: tl.constexpr, H_CONST: tl.constexpr, REPEAT_CONST: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    for i in range(K_CONST):
        for j in range(K_CONST):
            acc = 0.0
            g = h // REPEAT_CONST  # group index for this head
            for s in range(0, STATE_CONST):
                B_off = b * B_stride_b + n * B_stride_n + j * B_stride_k + g * B_stride_g + s * B_stride_s
                C_off = b * C_stride_b + n * C_stride_n + i * C_stride_k + g * C_stride_g + s * C_stride_s
                b_val = tl.load(B_ptr + B_off)
                c_val = tl.load(C_ptr + C_off)
                acc += b_val * c_val
            G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + j * G_stride_j + h * G_stride_h
            tl.store(G_ptr + G_off, acc)

# Triton kernel: final reduction Y_diag[b, n, i, h, d] = sum_j M[b, n, i, j, h] * hidden[b, n, j, h, d]
# M: [B, N, K, K, H], hidden: [B, N, K, H, D], out: [B, N, K, H, D] float32
@triton.jit
def final_reduce_M_mul_hidden(M_ptr, hidden_ptr, out_ptr,
                              B_batch, B_n, M_K, M_K2, M_H, hidden_H, hidden_D,
                              out_stride_b, out_stride_n, out_stride_i, out_stride_h, out_stride_d,
                              M_stride_b, M_stride_n, M_stride_i, M_stride_j, M_stride_h,
                              hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d,
                              D_CONST: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    # Vectorize over D
    d = tl.arange(0, D_CONST)
    # Accumulator for each d
    acc = tl.zeros((D_CONST,), dtype=tl.float32)
    for j in range(M_K):  # j loops over K
        M_off = b * M_stride_b + n * M_stride_n + i * M_stride_i + j * M_stride_j + h * M_stride_h
        M_val = tl.load(M_ptr + M_off)  # scalar float32
        hidden_off = b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_k + h * hidden_stride_h + d * hidden_stride_d
        hidden_vec = tl.load(hidden_ptr + hidden_off)  # [D_CONST] float32
        acc += M_val * hidden_vec
    # Store result to out
    out_off = b * out_stride_b + n * out_stride_n + i * out_stride_i + h * out_stride_h + d * out_stride_d
    tl.store(out_ptr + out_off, acc)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized version of the original run, preserving exact semantics and output shape.
        Returns Y_diag with shape [B, N, K, H, D] in bfloat16.
        """
        assert hidden_states.shape[-2:] == (K, D), f"hidden_states last dims must be (K={K}, D={D}), got {hidden_states.shape[-2:]}"
        assert hidden_states.shape[2] == K and hidden_states.shape[3] == H, "hidden_states must have [B, N, K, H, D] with K=128, H=32"
        B_batch, N_chunks, K_k, H_heads, D_dim = hidden_states.shape
        assert K_k == K and H_heads == H and D_dim == D, "Input shapes must match fixed constants (K=128, H=32, D=64)."

        # Ensure all inputs are float32 for Triton kernels
        A = A_cumsum.contiguous().to(torch.float32)
        Bt = B.contiguous().to(torch.float32)
        Ct = C.contiguous().to(torch.float32)
        hidden = hidden_states.contiguous().to(torch.float32)

        # Output tensor for reduction
        out = torch.empty((B_batch, N_chunks, K, H, D), device=hidden.device, dtype=torch.float32)

        # Allocate L: [B, H, N, K, K]
        L = torch.empty((B_batch, H, N_chunks, K, K), device=hidden.device, dtype=torch.float32)

        # Kernel 1: compute L = exp(cumsum(masked A))
        grid_L = (B_batch, H, N_chunks, K)
        masked_cumsum_tril_exp[grid_L](
            A, L,
            B_batch, H, N_chunks,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3), A.stride(4),
            L.stride(0), L.stride(2), L.stride(3), L.stride(4), L.stride(1),
            K_CONST=K
        )

        # Permute L to [B, N, K, K, H] for contraction
        L_perm = L.permute(0, 2, 3, 4, 1)  # [B, N, K, K, H]

        # Kernel 2: contract B and C into G: [B, N, K, K, H]
        # B: [B, N, K, n_groups(=N_GROUPS), STATE_SIZE]
        B_ng = N_GROUPS
        B_STATE = STATE_SIZE
        G = torch.empty((B_batch, N_chunks, K, K, H), device=hidden.device, dtype=torch.float32)

        contract_BC_to_G[(B_batch, N_chunks, H)](
            Bt, Ct, G,
            B_batch, N_chunks, K, B_ng, B_STATE,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            Bt.stride(0), Bt.stride(1), Bt.stride(2), Bt.stride(3), Bt.stride(4),
            Ct.stride(0), Ct.stride(1), Ct.stride(2), Ct.stride(3), Ct.stride(4),
            K_CONST=K, STATE_CONST=B_STATE, H_CONST=H, REPEAT_CONST=REPEAT
        )

        # Kernel 3: final reduction Y = sum_j M * hidden, where M = G * L_perm
        M = G * L_perm  # elementwise multiply
        final_reduce_M_mul_hidden[(B_batch, N_chunks, K, H)](
            M, hidden, out,
            B_batch, N_chunks, K, K, H, H, D,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            D_CONST=D
        )

        # Return in bfloat16 as original run returns
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
