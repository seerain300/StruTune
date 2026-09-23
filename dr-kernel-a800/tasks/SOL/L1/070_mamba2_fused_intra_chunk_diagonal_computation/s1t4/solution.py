import torch
import triton
import triton.language as tl

# Constants used in the original code
CHUNK_SIZE = 128
NUM_HEADS = 32
N_GROUPS = 8
REPEAT = NUM_HEADS // N_GROUPS  # 4 in the example
STATE_SIZE = 64  # example; adjust based on actual inputs if needed

# Triton kernel: compute A_masked_cumsum and L = exp(masked_cumsum), store in float32
@triton.jit
def compute_L_and_Cumsum_Exp(
    A_ptr, L_ptr,
    B_batch, B_heads, B_chunks,
    A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j,
    L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
    CHUNK: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    # Base offsets for A and L for fixed (b, h, n)
    A_base = b * A_stride_b + h * A_stride_h + n * A_stride_n
    L_base = b * L_stride_b + h * L_stride_h + n * L_stride_n

    # We need a lower-triangular cumsum along j for each i: i >= j contributes; j > i -> 0
    # Implement 2D tiling on (i, j) of size CHUNK x CHUNK
    for i in range(CHUNK):
        # Compute cumsum along j for fixed i (lower triangle only)
        cum = 0.0
        for j in range(CHUNK):
            if j <= i:
                a_off = A_base + i * A_stride_i + j * A_stride_j
                val = tl.load(A_ptr + a_off)
                cum += val
            else:
                cum += 0.0  # upper triangle contribution is zero
            # Store masked cumsum to A_masked_cumsum
            A_masked_off = L_base + i * L_stride_i + j * L_stride_j
            tl.store(L_ptr + A_masked_off, cum)
    # Now apply exp to get L
    for i in range(CHUNK):
        for j in range(CHUNK):
            L_off = L_base + i * L_stride_i + j * L_stride_j
            val = tl.load(L_ptr + A_base + i * A_stride_i + j * A_stride_j)  # masked cumsum
            tl.store(L_ptr + L_off, tl.exp(val))


# Triton kernel: contract B and C to produce G[i, j, h] = sum_s C[b, n, i, g, s] * B[b, n, j, g, s]
@triton.jit
def contract_BC_to_G(
    B_ptr, C_ptr, G_ptr,
    B_batch, B_n, B_chunk, B_groups, B_state,
    C_batch, C_n, C_chunk, C_groups, C_state,
    G_batch, G_n, G_chunk, G_heads,
    REPEAT: tl.constexpr, BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    g = h // REPEAT
    # Accumulator for G[i, j, h]
    # We'll compute G for all i, j by looping over state_size
    for i in range(B_chunk):
        for j in range(B_chunk):
            acc = 0.0
            # Loop over state_size in blocks
            for s_start in range(0, B_state, BLOCK_S):
                s_idx = s_start + tl.arange(0, BLOCK_S)
                mask_s = s_idx < B_state
                # Compute B indices: [b, n, j, g, s]
                B_off = b * B_batch * B_n * B_chunk * B_groups * B_state + \
                        n * B_n * B_chunk * B_groups * B_state + \
                        j * B_chunk * B_groups * B_state + \
                        g * B_groups * B_state + s_idx * B_state
                # Compute C indices: [b, n, i, g, s]
                C_off = b * C_batch * C_n * C_chunk * C_groups * C_state + \
                        n * C_n * C_chunk * C_groups * C_state + \
                        i * C_chunk * C_groups * C_state + \
                        g * C_groups * C_state + s_idx * C_state
                B_vals = tl.load(B_ptr + B_off, mask=mask_s, other=0.0)
                C_vals = tl.load(C_ptr + C_off, mask=mask_s, other=0.0)
                prod = B_vals * C_vals
                # Reduce across the BLOCK_S vector
                acc += tl.sum(prod, axis=0)
            # Store G[b, n, i, j, h] (float32)
            G_off = b * G_batch * G_n * G_chunk * G_heads + \
                    n * G_n * G_chunk * G_heads + \
                    i * G_chunk * G_heads + \
                    h * G_heads
            tl.store(G_ptr + G_off, acc)


# Triton kernel: final reduction to compute Y_diag[b, n, i, h, d] = sum_j G[b, n, i, j, h] * L[b, n, i, j, h] * hidden[b, n, j, h, d]
@triton.jit
def final_reduce_kernel(
    G_ptr, L_ptr, hidden_ptr, out_ptr,
    B_batch, B_n, B_chunk, B_heads, B_hidden_dim,
    L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
    G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_j, hidden_stride_h, hidden_stride_d,
    out_stride_b, out_stride_n, out_stride_i, out_stride_h, out_stride_d,
    BLOCK_D: tl.constexpr
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    for d_start in range(0, B_hidden_dim, BLOCK_D):
        d = d_start + tl.arange(0, BLOCK_D)
        mask_d = d < B_hidden_dim
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for j in range(B_chunk):
            # Load G[b, n, i, j, h]
            G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + h * G_stride_h
            G_val = tl.load(G_ptr + G_off)  # scalar
            # Load L[b, n, i, j, h]
            L_off = b * L_stride_b + n * L_stride_n + i * L_stride_i + j * L_stride_j + h * L_stride_h
            L_val = tl.load(L_ptr + L_off)  # scalar
            # Load hidden[b, n, j, h, d]
            hidden_off = b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_j + h * hidden_stride_h + d * hidden_stride_d
            hidden_vals = tl.load(hidden_ptr + hidden_off, mask=mask_d, other=0.0)
            acc += G_val * L_val * hidden_vals
        # Store out[b, n, i, h, d]
        out_off = b * out_stride_b + n * out_stride_n + i * out_stride_i + h * out_stride_h + d * out_stride_d
        tl.store(out_ptr + out_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Shapes based on provided signatures
        # hidden_states: [B, N, chunk_size, H, D]
        # A_cumsum: [B, H, N, CHUNK] (we assume CHUNK=128 consistent with example; if not, adapt)
        # B: [B, N, chunk_size, n_groups, state_size]
        # C: [B, N, chunk_size, n_groups, state_size]
        # Output: [B, N, chunk_size, H, D]
        # Ensure tensors are on CUDA and contiguous for Triton
        device = hidden_states.device
        B_batch, B_n, B_chunk, B_heads, B_hidden_dim = hidden_states.shape
        A_batch, A_heads, A_n, _ = A_cumsum.shape
        assert A_batch == B_batch and A_n == B_n and A_heads == B_heads, "Shape mismatch: A_cumsum and hidden_states"

        # Allocate L in float32: [B, H, N, CHUNK, CHUNK]
        L = torch.empty((B_batch, B_heads, B_n, CHUNK_SIZE, CHUNK_SIZE), device=device, dtype=torch.float32)
        # Launch compute_L_and_Cumsum_Exp kernel: grid (B, H, N)
        grid_L = (B_batch, B_heads, B_n)
        compute_L_and_Cumsum_Exp[grid_L](
            A_cumsum, L,
            B_batch, B_heads, B_n,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3), A_cumsum.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            CHUNK=CHUNK_SIZE
        )

        # Allocate G in float32: [B, N, CHUNK, CHUNK, H]
        G = torch.empty((B_batch, B_n, CHUNK_SIZE, CHUNK_SIZE, B_heads), device=device, dtype=torch.float32)

        # Launch contract_BC_to_G kernel: grid (B, N, H)
        grid_G = (B_batch, B_n, B_heads)
        contract_BC_to_G[grid_G](
            B, C, G,
            B_batch, B_n, B_chunk, B_heads // REPEAT, B_heads // REPEAT, B_hidden_dim,  # we use B_hidden_dim as 'state_size' here (tune if needed)
            B_batch, B_n, B_chunk, B_heads // REPEAT, B_heads // REPEAT,
            B_batch, B_n, CHUNK_SIZE, B_heads,
            REPEAT=REPEAT, BLOCK_S=64
        )

        # Allocate output Y_diag in float32: [B, N, CHUNK, H, D]
        out = torch.empty((B_batch, B_n, CHUNK_SIZE, B_heads, B_hidden_dim), device=device, dtype=torch.float32)

        # Launch final_reduce_kernel: grid (B, N, CHUNK, H)
        grid_out = (B_batch, B_n, CHUNK_SIZE, B_heads)
        final_reduce_kernel[grid_out](
            G, L, hidden_states.to(torch.float32), out,
            B_batch, B_n, CHUNK_SIZE, B_heads, B_hidden_dim,
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            hidden_states.float().stride(0), hidden_states.float().stride(1), hidden_states.float().stride(2), hidden_states.float().stride(3), hidden_states.float().stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            BLOCK_D=64
        )

        # Return in bfloat16 as original code
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
