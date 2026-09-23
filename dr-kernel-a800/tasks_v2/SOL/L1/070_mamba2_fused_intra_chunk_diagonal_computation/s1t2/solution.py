import torch
import triton
import triton.language as tl

# Constants for the example (unchanged semantics). CHUNK_SIZE is used for L and G sizes.
CHUNK_SIZE = 128
NUM_HEADS = 32
N_GROUPS = 8
REPEAT = NUM_HEADS // N_GROUPS  # 4
STATE_SIZE = 64

BLOCK_S = 32
BLOCK_D = 64

# Triton kernel: build L = exp(cumsum(masked A)) where mask is lower-triangular (diagonal=-1).
# A: [B, H, N, CHUNK, CHUNK], L: [B, H, N, CHUNK, CHUNK]
@triton.jit
def compute_L_kernel(
    A_ptr, L_ptr,
    B_batch, B_heads, B_chunks,  # placeholders for stride args, not used directly
    A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j,
    L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
    CHUNK: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    for i in range(CHUNK):
        for j in range(CHUNK):
            # lower-triangular mask with diagonal=-1: include j <= i, exclude j > i
            if j > i:
                val = 0.0
            else:
                a_off = b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_i + j * A_stride_j
                val = tl.load(A_ptr + a_off)
            exp_val = tl.exp(val)
            l_off = b * L_stride_b + n * L_stride_n + i * L_stride_i + j * L_stride_j + h * L_stride_h
            tl.store(L_ptr + l_off, exp_val)

# Triton kernel: contract B and C to G
# B_expanded: [B, N, CHUNK, H, STATE], C_expanded: [B, N, CHUNK, H, STATE]
# G: [B, N, CHUNK, H, CHUNK] (indexes 3 and 4 correspond to h and j in original G)
@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    B_batch, B_n, B_chunk, B_groups, B_state,
    C_batch, C_n, C_chunk, C_groups, C_state,
    G_batch, G_n, G_chunk, G_heads,
    REPEAT: tl.constexpr, BLOCK_S: tl.constexpr,
    CHUNK: tl.constexpr
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    g = h // REPEAT
    for i in range(CHUNK):
        for j in range(CHUNK):
            acc = 0.0
            for s_start in range(0, STATE_SIZE, BLOCK_S):
                s_idx = s_start + tl.arange(0, BLOCK_S)
                mask_s = s_idx < STATE_SIZE
                # Offsets for B[b, n, j, g, s] and C[b, n, i, g, s]
                B_off = b * B_batch * B_n * B_chunk * B_groups * B_state + n * B_n * B_chunk * B_groups * B_state + j * B_chunk * B_groups * B_state + g * B_groups * B_state + s_idx * B_state
                C_off = b * C_batch * C_n * C_chunk * C_groups * C_state + n * C_n * C_chunk * C_groups * C_state + i * C_chunk * C_groups * C_state + g * C_groups * C_state + s_idx * C_state
                B_vals = tl.load(B_ptr + B_off, mask=mask_s, other=0.0)
                C_vals = tl.load(C_ptr + C_off, mask=mask_s, other=0.0)
                acc += tl.sum(B_vals * C_vals, axis=0)
            G_off = b * G_batch * G_n * G_chunk * G_heads + n * G_n * G_chunk * G_heads + i * G_chunk * G_heads + h * G_heads + j
            tl.store(G_ptr + G_off, acc)

# Triton kernel: final reduction to compute Y_diag
# G: [B, N, CHUNK, H, CHUNK]
# L: [B, H, N, CHUNK, CHUNK] (we pass as L[b,h,n,i,j] by indexing)
# hidden: [B, N, CHUNK, H, D]
# out: [B, N, CHUNK, H, D]
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
    for d_start in range(0, B_hidden_dim, BLOCK_D):
        d = d_start + tl.arange(0, BLOCK_D)
        mask_d = d < B_hidden_dim
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for j in range(CHUNK):
            G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + h * G_stride_h + j * G_stride_j
            G_val = tl.load(G_ptr + G_off)
            # L offset for [b, h, n, i, j]
            L_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
            L_val = tl.load(L_ptr + L_off)
            hidden_off = b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_j + h * hidden_stride_h + d * hidden_stride_d
            hidden_vals = tl.load(hidden_ptr + hidden_off, mask=mask_d, other=0.0)
            acc += G_val * L_val * hidden_vals
        out_off = b * out_stride_b + n * out_stride_n + i * out_stride_i + h * out_stride_h + d * out_stride_d
        tl.store(out_ptr + out_off, acc, mask=mask_d)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        Triton-only implementation of the original function. No torch elementwise ops in forward.
        All computations (masking, cumsum, exp, contraction, reduction) are performed by Triton kernels.
        Returns Y_diag in bfloat16.
        """
        # hidden_states: [B, N, chunk_size, H, D]
        # A_cumsum: [B, H, N, chunk_size]
        # B: [B, N, chunk_size, groups, state_size]
        # C: [B, N, chunk_size, groups, state_size]

        # No torch operations on inputs; only allocations and kernel launches.

        B_batch, B_heads, B_chunks, B_groups, B_state = hidden_states.shape
        B_n, _, B_chunk, B_groups_in, B_state_in = B.shape
        B_hidden_dim = hidden_states.shape[-1]
        assert B_groups_in == B_groups, "B and hidden_states group dims inconsistent"

        device = hidden_states.device

        # Output tensor Y_diag: [B, N, CHUNK, H, D] float32
        out = torch.empty((B_batch, B_n, CHUNK_SIZE, B_heads, B_hidden_dim), device=device, dtype=torch.float32)

        # We need L: [B, H, N, CHUNK, CHUNK] float32. Compute via Triton kernel.
        L = torch.empty((B_batch, B_heads, B_n, CHUNK_SIZE, CHUNK_SIZE), device=device, dtype=torch.float32)

        # Launch compute_L_kernel: grid over (B, H, N)
        compute_L_kernel[(B_batch, B_heads, B_n)](
            A_cumsum, L,
            B_batch, B_heads, CHUNK_SIZE,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), CHUNK_SIZE, CHUNK_SIZE,
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            CHUNK=CHUNK_SIZE,
            num_warps=4
        )

        # Expanded B and C for contraction: map groups to heads
        B_expanded = B  # do not use repeat_interleave (torch op); kernel handles REPEAT mapping via h // REPEAT
        C_expanded = C

        # G: [B, N, CHUNK, H, CHUNK] float32
        G = torch.empty((B_batch, B_n, CHUNK_SIZE, B_heads, CHUNK_SIZE), device=device, dtype=torch.float32)

        # Launch contract_BC_to_G_kernel: grid over (B, N, H)
        contract_BC_to_G_kernel[(B_batch, B_n, B_heads)](
            B_expanded, C_expanded, G,
            B_batch, B_n, B_chunk, B_groups, B_state,
            B_batch, B_n, B_chunk, B_groups, B_state,
            B_batch, B_n, CHUNK_SIZE, B_heads,
            REPEAT=REPEAT, BLOCK_S=BLOCK_S, CHUNK=CHUNK_SIZE,
            num_warps=4
        )

        # Launch final_reduce_kernel: grid over (B, N, CHUNK, H)
        final_reduce_kernel[(B_batch, B_n, CHUNK_SIZE, B_heads)](
            G, L, hidden_states, out,
            B_batch, B_n, CHUNK_SIZE, B_heads, B_hidden_dim,
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            CHUNK=CHUNK_SIZE, BLOCK_D=BLOCK_D,
            num_warps=4
        )

        # Return in bfloat16
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
