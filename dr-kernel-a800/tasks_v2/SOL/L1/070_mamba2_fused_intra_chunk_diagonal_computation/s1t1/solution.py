import torch
import triton
import triton.language as tl

# Constants consistent with the original example
CHUNK_SIZE = 128
NUM_HEADS = 32
N_GROUPS = 8
REPEAT = NUM_HEADS // N_GROUPS  # 4
STATE_SIZE = 64  # example; adjust if your actual B/C have different state_size
BLOCK_S = 32   # reduction block size for state_size
BLOCK_D = 64   # vectorization block for head_dim

# Triton kernel: Contract B and C per (h) to produce G[b, n, i, h, j]
# G[b, n, i, h, j] = sum_s C[b, n, i, g, s] * B[b, n, j, g, s]
@triton.jit
def contract_BC_per_h_kernel(
    B_ptr, C_ptr, G_ptr,
    B_batch, B_n, B_chunk, B_groups, B_state,
    C_batch, C_n, C_chunk, C_groups, C_state,
    G_batch, G_n, G_chunk, G_heads,
    REPEAT: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    g = h // REPEAT
    # Loop over i and j in [0, CHUNK_SIZE)
    for i in range(CHUNK_SIZE):
        for j in range(CHUNK_SIZE):
            acc = 0.0
            # Reduce over state_size in blocks
            for s_start in range(0, STATE_SIZE, BLOCK_S):
                s_idx = s_start + tl.arange(0, BLOCK_S)
                mask_s = s_idx < STATE_SIZE
                # Compute offsets
                B_off = b * B_batch * B_n * B_chunk * B_groups * B_state + n * B_n * B_chunk * B_groups * B_state + j * B_chunk * B_groups * B_state + g * B_groups * B_state + s_idx * B_state
                C_off = b * C_batch * C_n * C_chunk * C_groups * C_state + n * C_n * C_chunk * C_groups * C_state + i * C_chunk * C_groups * C_state + g * C_groups * C_state + s_idx * C_state
                B_vals = tl.load(B_ptr + B_off, mask=mask_s, other=0.0)
                C_vals = tl.load(C_ptr + C_off, mask=mask_s, other=0.0)
                acc += tl.sum(B_vals * C_vals, axis=0)
            # Store G[b, n, i, h, j]
            G_off = b * G_batch * G_n * G_chunk * G_heads + n * G_n * G_chunk * G_heads + i * G_chunk * G_heads + h * G_heads + j
            tl.store(G_ptr + G_off, acc)

# Triton kernel: Final reduction to compute Y_diag[b, n, i, h, d] = sum_j G[b,n,i,j,h] * L[b,n,i,j,h] * hidden[b,n,j,h,d]
@triton.jit
def final_reduce_kernel(
    G_ptr, L_ptr, hidden_ptr, out_ptr,
    B_batch, B_n, B_chunk, B_heads, B_hidden_dim,
    L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
    G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_j, hidden_stride_h, hidden_stride_d,
    out_stride_b, out_stride_n, out_stride_i, out_stride_h, out_stride_d,
    CHUNK: tl.constexpr, BLOCK_D: tl.constexpr
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    # Vectorize over head_dim in blocks
    for d_start in range(0, B_hidden_dim, BLOCK_D):
        d = d_start + tl.arange(0, BLOCK_D)
        mask_d = d < B_hidden_dim
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        # Loop j over CHUNK_SIZE
        for j in range(CHUNK):
            # Load G[b, n, i, h, j]
            G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + h * G_stride_h + j * G_stride_j
            G_val = tl.load(G_ptr + G_off)
            # Load L[b, n, i, j, h]
            L_off = b * L_stride_b + n * L_stride_n + i * L_stride_i + j * L_stride_j + h * L_stride_h
            L_val = tl.load(L_ptr + L_off)
            # Load hidden[b, n, j, h, d]
            hidden_off = b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_j + h * hidden_stride_h + d * hidden_stride_d
            hidden_vals = tl.load(hidden_ptr + hidden_off, mask=mask_d, other=0.0)
            acc += G_val * L_val * hidden_vals
        # Store to out[b, n, i, h, d]
        out_off = b * out_stride_b + n * out_stride_n + i * out_stride_i + h * out_stride_h + d * out_stride_d
        tl.store(out_ptr + out_off, acc, mask=mask_d)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized version of the original run function.
        Steps:
          1) Build L in torch (GPU) using tril mask, cumsum, and exp. L: [B, H, N, CHUNK_SIZE, CHUNK_SIZE].
          2) Contract B and C to G using Triton kernel: G: [B, N, CHUNK_SIZE, H, CHUNK_SIZE].
          3) Compute final output Y_diag via Triton reduction: [B, N, CHUNK_SIZE, H, head_dim].
        Returns Y_diag cast to bfloat16.
        """
        assert hidden_states.ndim == 5, "hidden_states must be [B, N, chunk_size, H, D]"
        assert A_cumsum.ndim == 4, "A_cumsum must be [B, H, N, chunk_size]"
        assert B.ndim == 5 and C.ndim == 5, "B and C must be [B, N, chunk_size, groups, state_size]"

        device = hidden_states.device
        dtype = torch.float32

        # Shapes
        B_batch, B_n, chunk_size, B_groups, B_state = B.shape
        B_heads, B_chunks, _, _, hidden_dim = hidden_states.shape

        # Expand B and C from groups to heads
        REPEAT = NUM_HEADS // N_GROUPS  # must match original mapping
        B_expanded = B.repeat_interleave(REPEAT, dim=3)  # [B, N, chunk_size, H, state_size]
        C_expanded = C.repeat_interleave(REPEAT, dim=3)  # [B, N, chunk_size, H, state_size]

        # Allocate G [B, N, chunk_size, H, CHUNK_SIZE]
        G = torch.empty((B_batch, B_n, chunk_size, B_heads, CHUNK_SIZE), device=device, dtype=dtype)

        # Build L in torch: [B, H, N, CHUNK_SIZE, CHUNK_SIZE]
        CHUNK = CHUNK_SIZE
        # Expand A_cumsum to [B, H, N, CHUNK, CHUNK]
        A_expanded = A_cumsum.unsqueeze(-1).expand(B_batch, B_heads, B_n, CHUNK, CHUNK).to(torch.float32)
        mask_lower = torch.tril(torch.ones(CHUNK, CHUNK, device=device, dtype=torch.bool), diagonal=-1)
        A_masked = A_expanded.masked_fill(~mask_lower, 0.0)
        A_cumsum_seg = torch.cumsum(A_masked, dim=-1)  # cumsum along last dim (CHUNK)
        mask_upper = torch.tril(torch.ones(CHUNK, CHUNK, device=device, dtype=torch.bool), diagonal=0)
        A_for_exp = A_cumsum_seg.masked_fill(~mask_upper, -float('inf'))
        L = torch.exp(A_for_exp)  # [B, H, N, CHUNK, CHUNK]

        # Launch Triton contraction kernel: grid (B, N, H)
        contract_BC_per_h_kernel[(B_batch, B_n, B_heads)](
            B_expanded, C_expanded, G,
            B_batch, B_n, chunk_size, B_groups, B_state,
            B_batch, B_n, chunk_size, B_groups, B_state,
            B_batch, B_n, chunk_size, B_heads,
            REPEAT=REPEAT, BLOCK_S=BLOCK_S
        )

        # Allocate output Y_diag [B, N, CHUNK_SIZE, H, hidden_dim] in float32
        out = torch.empty((B_batch, B_n, CHUNK_SIZE, B_heads, hidden_dim), device=device, dtype=torch.float32)

        # Launch final reduction Triton kernel: grid (B, N, CHUNK_SIZE, H)
        final_reduce_kernel[(B_batch, B_n, CHUNK_SIZE, B_heads)](
            G, L,
            hidden_states, out,
            B_batch, B_n, CHUNK_SIZE, B_heads, hidden_dim,
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            CHUNK=CHUNK, BLOCK_D=BLOCK_D
        )

        # Return in bfloat16, matching original behavior
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
