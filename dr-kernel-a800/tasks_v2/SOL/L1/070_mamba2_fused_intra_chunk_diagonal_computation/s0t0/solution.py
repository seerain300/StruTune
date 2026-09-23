import torch
import triton
import triton.language as tl

# Kernel 1: Build L = exp(cumsum(A_masked_lower_tri)) for causal mask
# A: [B, H, C, S] float32
# L: [B, C, S, S, H] float32
@triton.jit
def build_L_kernel(
    A_ptr, L_ptr,
    B: tl.constexpr, C: tl.constexpr, S: tl.constexpr, H: tl.constexpr
):
    # Grid: (B, C, H, S) -> each program computes the row vector for one i
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    i = tl.program_id(3)

    # initialize cumulative sum vector for this (b, h, c, i)
    # we will iterate j from 0..S-1 and only add when j <= i (lower-triangular)
    cumsum = tl.zeros([S], dtype=tl.float32)
    row_start = (b * H * C * S) + (h * C * S) + (c * S)
    for j in range(0, S):
        a_idx = (row_start + j)
        a_val = tl.load(A_ptr + a_idx)
        # include j <= i (lower-triangular)
        if j <= i:
            cumsum = cumsum + a_val
        # L[i,j] = exp(cumsum[i])
        L_idx = (b * C * S * S * H) + (c * S * S * H) + (i * S * H) + (j * H) + h
        L_val = tl.exp(cumsum[j])
        tl.store(L_ptr + L_idx, L_val)


# Kernel 2: Compute G = sum_n C[:, :, :, n, :] * B[:, :, :, n, :] -> [B, C, S, S, H]
# B_expanded: [B, C, S, H, S] float32
# C_expanded: [B, C, S, H, S] float32
# G: [B, C, S, S, H] float32
@triton.jit
def compute_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, H: tl.constexpr
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # row j
    j = tl.program_id(3)  # col i
    h = tl.program_id(4)

    # Compute sum over n (we assume n in [0..H-1] == head_dim)
    sum_val = tl.zeros([1], dtype=tl.float32)
    for n in range(0, H):
        # B[b, c, j, n, i] and C[b, c, i, n, j]
        # linear indexing: B_idx = ((b*Bsz*Csz*S*H) + (c*Csz*S*H) + (j*S*H) + n*H + i)
        #                    C_idx = ((b*Bsz*Csz*S*H) + (c*Csz*S*H) + (i*S*H) + n*H + j)
        # since B/C shapes are [B, C, S, H, S], and H==S, indexing can be constructed accordingly.
        # However, simpler is to rely on the expanded shapes where H matches S. Here we assume S==H.
        # To keep general, let's index using the expanded dimensions assuming last dim is head_dim (S).
        # But since H is constexpr, we can compute:
        # B_idx = b*Bsz*Csz*S*H + c*Csz*S*H + j*S*H + n*H + i
        # C_idx = b*Bsz*Csz*S*H + c*Csz*S*H + i*S*H + n*H + j
        B_idx = (b * (Csz * S * H) + c * (S * H) + j * H + n * H + i)
        C_idx = (b * (Csz * S * H) + c * (S * H) + i * H + n * H + j)
        # Now convert to pointer with base: (b*Bsz + c*Csz + i*S + j*H + n*H) but above indexing suffices.
        # We need to map to the actual linearized memory: given the expanded shapes, each (b,c,i,j,n) is linearized.
        # For simplicity, pass B and C as contiguous [B, C, S, H, S] and compute indices accordingly.
        # The original code expands B and C to H from N_GROUPS via repeat_interleave. Here we assume S==H.
        B_val = tl.load(B_ptr + B_idx)
        C_val = tl.load(C_ptr + C_idx)
        sum_val = sum_val + C_val * B_val

    G_idx = (b * Csz * S * S * H) + (c * S * S * H) + (i * S * H) + (j * H) + h
    tl.store(G_ptr + G_idx, sum_val)


# Kernel 3: Compute Y_diag = sum over hidden_states_j of M[i,j,h] * hidden_states[j,h,d]
# M: [B, C, S, S, H] float32
# hidden_states: [B, C, S, H, HD] float32
# Y: [B, C, S, H, HD] float32
@triton.jit
def compute_Y_diag_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, H: tl.constexpr, HD: tl.constexpr
):
    # Flatten the first three dims into one grid dimension: (B*C*S)
    pid0 = tl.program_id(0)
    d = tl.program_id(1)  # over head_dim (constexpr)

    BC = Bsz * Csz
    T = BC * S
    # Recover b, c, i
    c_i = pid0 // S
    i = pid0 % S
    b = c_i // Csz
    c = c_i % Csz

    h = tl.program_id(2)  # head index

    # Accumulator for this (b, c, i, h, d)
    acc = tl.zeros([1], dtype=tl.float32)

    # Reduce over j in [0..S-1]
    for j in range(0, S):
        M_idx = (b * Csz * S * S * H) + (c * S * S * H) + (i * S * H) + (j * H) + h
        M_val = tl.load(M_ptr + M_idx)

        HS_idx = (b * Csz * S * H * HD) + (c * S * H * HD) + (j * H * HD) + (h * HD) + d
        HS_val = tl.load(hidden_ptr + HS_idx)

        acc = acc + M_val * HS_val

    Y_idx = (b * Csz * S * H * HD) + (c * S * H * HD) + (i * S * HD) + (h * HD) + d
    tl.store(Y_ptr + Y_idx, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA."
        hidden_states = hidden_states.contiguous()
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        Bsz, Csz, S, H, HD = hidden_states.shape  # hidden_states: [Bsz, Csz, S, H, HD]
        # A_cumsum: [B, H, C, S] -> note original code has (num_chunks, num_heads, num_chunks, chunk_size)
        # We need to map to our variables: original A_cumsum has shape [batch, num_heads, num_chunks, chunk_size].
        # The provided run(...) signature uses hidden_states: [batch, num_chunks, chunk_size, num_heads, head_dim],
        # and A_cumsum: [batch, num_heads, num_chunks, chunk_size]. Here our hidden_states has last dim = num_heads,
        # while A_cumsum last dim = num_heads as well. We will assume A_cumsum.shape[-2] == H and shape[-1] == S.
        # Given the original code uses NUM_HEADS=32 and chunk_size=128, and in benchmarks chunk_size varies,
        # we can safely read A_cumsum with H,S. If the shapes do not match, we fall back to a safe permute.
        # However, to keep strict compliance, we require exact shapes: H == A_cumsum.size(-2) and S == A_cumsum.size(-1).
        # If not, we permute to match: A_cumsum.permute(0, 2, 1, 3) -> [B, C, H, S].
        A_shape = A_cumsum.shape
        if A_shape[-2] != H or A_shape[-1] != S:
            A_cumsum = A_cumsum.permute(0, 2, 1, 3).contiguous()

        Bsz_A, Csz_A, H_A, S_A = A_cumsum.shape
        assert Bsz_A == Bsz and Csz_A == Csz, "A_cumsum batch and num_chunks must match hidden_states."

        # Compute L: [B, C, S, S, H]
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        # Launch grid: (B, C, H, S)
        grid_L = (Bsz, Csz, H, S)
        build_L_kernel[grid_L](A_cumsum, L, Bsz, Csz, S, H)

        # Compute G: [B, C, S, S, H]
        # We need to expand B and C along num_heads. Original code does repeat_interleave(NUM_HEADS // N_GROUPS, dim=3).
        # NUM_HEADS=32, N_GROUPS=8 -> repeat_interleave(4). Here H is already 32 (from hidden_states.shape[-1]), so no repeat needed.
        B_expanded = B  # already [B, C, S, H, S], because H == num_heads and S == chunk_size
        C_expanded = C  # already [B, C, S, H, S]
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        grid_G = (Bsz, Csz, S, S, H)
        compute_G_kernel[grid_G](B_expanded, C_expanded, G, Bsz, Csz, S, H)

        # Compute M = G * L
        M = G * L

        # Compute Y_diag: [B, C, S, H, HD]
        Y = torch.empty((Bsz, Csz, S, H, HD), dtype=torch.float32, device=hidden_states.device)
        # Grid: (B*C*S) over first flattened dim, HD over second dim, H over third dim
        grid_Y = (Bsz * Csz * S, HD, H)
        compute_Y_diag_kernel[grid_Y](M, hidden_states, Y, Bsz, Csz, S, H, HD)

        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
