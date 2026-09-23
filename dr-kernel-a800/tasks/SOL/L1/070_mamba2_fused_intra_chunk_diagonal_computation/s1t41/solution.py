import torch
import triton
import triton.language as tl

# Triton kernel: masked cumsum (lower-triangular, diagonal=-1) + exp -> L
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

# Triton kernel: contract B and C into G: [B, N, K, K, H]
# B: [B, N, K, n_groups, STATE], C: [B, N, K, n_groups, STATE]
# We use g = h // REPEAT, where REPEAT = NUM_HEADS // N_GROUPS and original code sets REPEAT=4.
@triton.jit
def contract_BC_to_G(B_ptr, C_ptr, G_ptr,
                     B_batch, B_n, B_K, B_ng, B_STATE,
                     G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                     B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s,
                     C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s,
                     CHUNK: tl.constexpr, STATE: tl.constexpr, REPEAT: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    g = h // REPEAT
    for i in range(CHUNK):
        for j in range(CHUNK):
            acc = 0.0
            # Accumulate over state dimension in blocks
            for s_start in range(0, STATE, 64):
                s = s_start + tl.arange(0, 64)
                mask_s = s < STATE
                B_off = b * B_stride_b + n * B_stride_n + j * B_stride_k + g * B_stride_g + s * B_stride_s
                C_off = b * C_stride_b + n * C_stride_n + i * C_stride_k + g * C_stride_g + s * C_stride_s
                B_vals = tl.load(B_ptr + B_off, mask=mask_s, other=0.0)
                C_vals = tl.load(C_ptr + C_off, mask=mask_s, other=0.0)
                acc += tl.sum(B_vals * C_vals, axis=0)
            G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + j * G_stride_j + h * G_stride_h
            tl.store(G_ptr + G_off, acc)

# Triton kernel: final reduction
# hidden: [B, N, K, H, D], G: [B, N, K, K, H], L: [B, H, N, K, K]
# out: [B, N, K, H, D]
@triton.jit
def reduce_G_L_hidden(hidden_ptr, G_ptr, L_ptr, out_ptr,
                      B_batch, B_n, B_K, B_H, B_D,
                      hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d,
                      G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                      L_stride_b, L_stride_h, L_stride_n, L_stride_i, L_stride_j,
                      out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d,
                      K: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.arange(0, BLOCK_D)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for j in range(K):
        G_val = tl.load(G_ptr + b * G_stride_b + n * G_stride_n + i * G_stride_i + j * G_stride_j + h * G_stride_h)
        L_val = tl.load(L_ptr + b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j)
        G_L = G_val * L_val
        hidden_off = b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_k + h * hidden_stride_h + d * hidden_stride_d
        hidden_vec = tl.load(hidden_ptr + hidden_off, mask=d < D, other=0.0)
        acc += G_L * hidden_vec
    out_off = b * out_stride_b + n * out_stride_n + i * out_stride_k + h * out_stride_h + d * out_stride_d
    tl.store(out_ptr + out_off, acc, mask=d < D)

# Entry point: ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_diag in Triton:
        - hidden_states: [B, N, K, H, D]
        - A_cumsum: [B, H, N, K, K]
        - B: [B, N, K, n_groups, STATE]
        - C: [B, N, K, n_groups, STATE]
        Output: [B, N, K, H, D] (bfloat16)
        """
        # Extract dimensions from inputs
        B_batch, B_n, B_K, B_H, B_D = hidden_states.shape
        assert B_cumsum is not None, "A_cumsum must be provided"
        # Original constants from the provided model
        N_GROUPS = 8
        NUM_HEADS = B_H
        REPEAT = NUM_HEADS // N_GROUPS  # 4
        assert B_H % REPEAT == 0, "NUM_HEADS must be divisible by N_GROUPS * REPEAT"

        # Ensure tensors are CUDA and contiguous
        device = hidden_states.device
        assert device.type == "cuda", "Triton kernels require CUDA tensors"
        A = A_cumsum.contiguous()  # [B, H, N, K, K]
        Bt = B.contiguous()
        Ct = C.contiguous()
        hidden = hidden_states.contiguous()

        # Allocate outputs
        L = torch.empty((B_batch, B_H, B_n, B_K, B_K), dtype=torch.float32, device=device)
        G = torch.empty((B_batch, B_n, B_K, B_K, B_H), dtype=torch.float32, device=device)
        out = torch.empty((B_batch, B_n, B_K, B_H, B_D), dtype=torch.float32, device=device)

        # Launch masked cumsum + exp to produce L
        grid_L = (B_batch, B_H, B_n, B_K)
        masked_cumsum_tril_exp[grid_L](
            A, L,
            B_batch, B_H, B_n,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3), A.stride(4),
            L.stride(0), L.stride(2), L.stride(3), L.stride(4), L.stride(1),
            CHUNK=B_K,
            num_warps=4, num_stages=2
        )

        # Launch contraction of B and C into G
        n_groups = N_GROUPS
        state = Bt.shape[-1]  # assume original state size = 64
        grid_G = (B_batch, B_n, B_H)
        contract_BC_to_G[grid_G](
            Bt, Ct, G,
            B_batch, B_n, B_K, n_groups, state,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            Bt.stride(0), Bt.stride(1), Bt.stride(2), Bt.stride(3), Bt.stride(4),
            Ct.stride(0), Ct.stride(1), Ct.stride(2), Ct.stride(3), Ct.stride(4),
            CHUNK=B_K, STATE=state, REPEAT=REPEAT,
            num_warps=4, num_stages=2
        )

        # Launch final reduction
        grid_out = (B_batch, B_n, B_K, B_H)
        reduce_G_L_hidden[grid_out](
            hidden, G, L, out,
            B_batch, B_n, B_K, B_H, B_D,
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            K=B_K, D=B_D, BLOCK_D=64,
            num_warps=4, num_stages=2
        )

        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
