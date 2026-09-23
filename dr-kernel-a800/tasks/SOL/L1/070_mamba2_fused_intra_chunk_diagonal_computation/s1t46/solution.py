import torch
import triton
import triton.language as tl

# Constants implied by the original PyTorch code
CHUNK_SIZE = 128
NUM_HEADS = 32
N_GROUPS = 8
REPEAT = NUM_HEADS // N_GROUPS  # 4
STATE_SIZE = 64
HEAD_DIM = 64

@triton.jit
def masked_cumsum_tril_exp(A_ptr, L_ptr,
                           B_batch, B_heads, B_n,
                           A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j,
                           L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
                           CHUNK: tl.constexpr):
    # Grid is (B_batch, B_heads, B_n, i in [0..CHUNK-1])
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    i = tl.program_id(3)
    cumsum = 0.0
    for j in range(CHUNK):
        # Lower-triangular with diagonal=-1: only if j <= i
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
    # Grid over (B_batch, B_n, h in [0..H-1])
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    g = h // 4  # REPEAT=4 -> map head to group
    for i in range(K):
        for j in range(K):
            acc = 0.0
            # Accumulate over state dimension in blocks
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
                 B_batch, B_n, B_K, B_h, B_d,
                 G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                 L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
                 hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d,
                 out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d,
                 K: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    # Grid over (B_batch, B_n, i in [0..K-1], h in [0..B_h-1])
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    for j in range(K):
        G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + j * G_stride_j + h * G_stride_h
        L_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
        G_val = tl.load(G_ptr + G_off)
        L_val = tl.load(L_ptr + L_off)
        prod = G_val * L_val  # scalar
        for d_start in range(0, D, BLOCK_D):
            d = d_start + tl.arange(0, BLOCK_D)
            mask_d = d < D
            hidden_off = b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_k + h * hidden_stride_h + d * hidden_stride_d
            out_off = b * out_stride_b + n * out_stride_n + i * out_stride_k + h * out_stride_h + d * out_stride_d
            hidden_vec = tl.load(hidden_ptr + hidden_off, mask=mask_d, other=0.0)
            tl.store(out_ptr + out_off, prod * hidden_vec, mask=mask_d)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward matching the original Model.run semantics for the typical configuration:
          - hidden_states: [B, N, K, H, D] with K=128, H=32, D=64
          - A_cumsum: [B, H, N, K, K]
          - B: [B, N, K, n_groups=8, STATE_SIZE=64]
          - C: [B, N, K, n_groups=8, STATE_SIZE=64]
        Returns: [B, N, K, H, D] in bfloat16.
        """
        # Expected shapes (constants implied by the original PyTorch code)
        B_batch, N, K, H, D = hidden_states.shape
        assert K == CHUNK_SIZE and H == NUM_HEADS and D == HEAD_DIM, "hidden_states must have shape [B, N, 128, 32, 64]"
        assert A_cumsum.shape == (B_batch, H, N, K, K), "A_cumsum must be [B, 32, N, 128, 128]"
        assert B.shape == (B_batch, N, K, N_GROUPS, STATE_SIZE), "B must be [B, N, 128, 8, 64]"
        assert C.shape == (B_batch, N, K, N_GROUPS, STATE_SIZE), "C must be [B, N, 128, 8, 64]"
        # Allocate L as float32
        L = torch.empty((B_batch, H, N, K, K), dtype=torch.float32, device=hidden_states.device)
        # Strides for A_cumsum and L
        A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j = A_cumsum.stride()
        L_stride_b, L_stride_h, L_stride_n, L_stride_i, L_stride_j = L.stride()
        # Launch masked cumsum + exp
        grid1 = (B_batch, H, N, K)
        masked_cumsum_tril_exp[grid1](
            A_cumsum, L,
            B_batch, H, N,
            A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j,
            L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
            CHUNK=K, num_warps=4, num_stages=2
        )
        # Allocate G as float32: [B, N, K, K, H]
        G = torch.empty((B_batch, N, K, K, H), dtype=torch.float32, device=hidden_states.device)
        # Strides for B, C, G
        B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s = B.stride()
        C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s = C.stride()
        G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h = G.stride()
        # Launch contraction kernel
        grid2 = (B_batch, N, H)
        contract_BC_to_G[grid2](
            B, C, G,
            B_batch, N, K, N_GROUPS, STATE_SIZE,
            G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
            B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s,
            C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s,
            K=K, H=H, BLOCK_S=64, num_warps=4, num_stages=2
        )
        # Output buffer as float32: [B, N, K, H, D]
        out = torch.empty((B_batch, N, K, H, D), dtype=torch.float32, device=hidden_states.device)
        # Strides for hidden and out
        hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d = hidden_states.stride()
        out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d = out.stride()
        # Launch final reduction kernel
        grid3 = (B_batch, N, K, H)
        final_reduce[grid3](
            G, L, hidden_states, out,
            B_batch, N, K, H, D,
            G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
            L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
            hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d,
            out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d,
            K=K, D=D, BLOCK_D=64, num_warps=4, num_stages=2
        )
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
